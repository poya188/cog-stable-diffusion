import os
from typing import Optional, List

import torch
from torch import autocast
from diffusers import PNDMScheduler, LMSDiscreteScheduler
from PIL import Image
from cog import BasePredictor, Input, Path

from animate import StableDiffusionAnimationPipeline


MODEL_CACHE = "diffusers-cache"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.pipe = StableDiffusionAnimationPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            scheduler=make_scheduler(100),  # timesteps is arbitrary at this point
            revision="fp16",
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt_start: str = Input(description="Prompt to start the animation with"),
        prompt_end: str = Input(description="Prompt to end the animation with"),
        width: int = Input(
            description="Width of output image",
            choices=[128, 256, 512, 768, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image",
            choices=[128, 256, 512, 768],
            default=512,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        prompt_strength: float = Input(
            description="Lower prompt strength generates more coherent gifs, higher respects prompts more but can be jumpy",
            default=0.8,
        ),
        num_animation_frames: int = Input(
            description="Number of frames to animate", default=10, ge=2, le=50
        ),
        num_interpolation_steps: int = Input(
            description="Number of steps to interpolate between animation frames",
            default=5,
            ge=1,
            le=50,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        gif_frames_per_second: int = Input(
            description="Frames/second in output GIF",
            default=20,
            ge=1,
            le=50,
        ),
        gif_ping_pong: bool = Input(
            description="Whether to reverse the animation and go back to the beginning before looping",
            default=False,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        batch_size = 1

        # Generate initial latents to start to generate animation frames from
        self.pipe.scheduler = make_scheduler(num_inference_steps)
        num_initial_steps = int(num_inference_steps * (1 - prompt_strength))
        print(f"Generating initial latents for {num_initial_steps} steps")
        initial_latents = torch.randn(
            (batch_size, self.pipe.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device="cuda",
        )
        do_classifier_free_guidance = guidance_scale > 1.0
        text_embeddings_start = self.pipe.embed_text(
            prompt_start, do_classifier_free_guidance, batch_size
        )
        text_embeddings_end = self.pipe.embed_text(
            prompt_end, do_classifier_free_guidance, batch_size
        )
        text_embeddings_mid = text_embeddings_start * 0.5 + text_embeddings_end * 0.5
        latents_mid = self.pipe.denoise(
            latents=initial_latents,
            text_embeddings=text_embeddings_mid,
            t_start=1,
            t_end=num_initial_steps,
            guidance_scale=guidance_scale,
        )

        # Generate latents for animation frames
        frames_latents = []
        for i in range(num_animation_frames):
            print(f"Generating frame {i}")
            x = i / (num_animation_frames - 1)
            text_embeddings = text_embeddings_start * (1 - x) + text_embeddings_end * x

            # re-initialize scheduler
            self.pipe.scheduler = make_scheduler(num_inference_steps)

            latents = self.pipe.denoise(
                latents=latents_mid,
                text_embeddings=text_embeddings,
                t_start=num_initial_steps,
                t_end=None,
                guidance_scale=guidance_scale,
            )
            self.pipe.numpy_to_pil(self.pipe.latents_to_image(latents))[0].save("test.png")

            # Run safety check on first and last frame
            if i == 0 or i == num_animation_frames - 1:
                self.pipe.safety_check(self.pipe.latents_to_image(latents))

            # de-noise this frame
            frames_latents.append(latents)

        # Decode images by interpolate between animation frames
        print("Generating images from latents")
        images = []
        for i in range(num_animation_frames - 1):
            latents_start = frames_latents[i]
            latents_end = frames_latents[i + 1]
            for j in range(num_interpolation_steps):
                x = j / num_interpolation_steps
                latents = latents_start * (1 - x) + latents_end * x
                numpy_image = self.pipe.latents_to_image(latents)
                image = self.pipe.numpy_to_pil(numpy_image.astype("float32"))[0]
                images.append(image)

        # Save the gif
        print("Saving GIF")
        output_path = "/tmp/video.gif"

        if gif_ping_pong:
            images += images[-1:1:-1]
        gif_frame_duration = int(1000 / gif_frames_per_second)

        image = images[0]
        with open(output_path, "wb") as f:
            image.save(
                fp=f,
                format="GIF",
                append_images=images[1:],
                save_all=True,
                duration=gif_frame_duration,
                loop=0,
            )

        return Path(output_path)


def make_scheduler(num_inference_steps):
    scheduler = PNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
    )
    scheduler.set_timesteps(num_inference_steps, offset=1)
    return scheduler
