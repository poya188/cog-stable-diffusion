import sys
import os
from typing import Optional, List, Iterator

import cv2
import av
import numpy as np
import torch
from torch import autocast
import tensorflow as tf
from diffusers import PNDMScheduler, LMSDiscreteScheduler
from PIL import Image
from cog import BasePredictor, Input, Path

from animate import StableDiffusionAnimationPipeline

sys.path.append("/frame-interpolation")
from eval import interpolator as film_interpolator, util as film_util

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

        # Stop tensorflow eagerly taking all GPU memory
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("Loading interpolator...")
        self.interpolator = film_interpolator.Interpolator(
            # from https://drive.google.com/drive/folders/1i9Go1YI2qiFWeT5QtywNFmYAA74bhXWj?usp=sharing
            "/src/frame_interpolation_saved_model",
            None,
        )

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
        film_interpolation: bool = Input(
            description="Whether to use FILM for between-frame interpolation (film-net.github.io)",
            default=False,
        ),
        intermediate_output: bool = Input(
            description="Whether to display intermediate outputs during generation",
            default=False,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        output_format: str = Input(
            description="Output file format", choices=["gif", "mp4"], default="gif"
        ),
    ) -> Iterator[Path]:
        """Run a single prediction on the model"""
        with torch.autocast("cuda"), torch.inference_mode():
            if seed is None:
                seed = int.from_bytes(os.urandom(2), "big")
            print(f"Using seed: {seed}")
            generator = torch.Generator("cuda").manual_seed(seed)

            batch_size = 1

            # Generate initial latents to start to generate animation frames from
            initial_scheduler = self.pipe.scheduler = make_scheduler(
                num_inference_steps
            )
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
            text_embeddings_mid = slerp(0.5, text_embeddings_start, text_embeddings_end)
            latents_mid = self.pipe.denoise(
                latents=initial_latents,
                text_embeddings=text_embeddings_mid,
                t_start=1,
                t_end=num_initial_steps,
                guidance_scale=guidance_scale,
            )

            print("Generating first frame")
            # re-initialize scheduler
            self.pipe.scheduler = make_scheduler(num_inference_steps, initial_scheduler)
            latents_start = self.pipe.denoise(
                latents=latents_mid,
                text_embeddings=text_embeddings_start,
                t_start=num_initial_steps,
                t_end=None,
                guidance_scale=guidance_scale,
            )
            image_start = self.pipe.latents_to_image(latents_start)
            self.pipe.safety_check(image_start)

            if intermediate_output:
                yield save_pil_image(
                    self.pipe.numpy_to_pil(image_start)[0], path="/tmp/output-0.png"
                )

            print("Generating last frame")
            # re-initialize scheduler
            self.pipe.scheduler = make_scheduler(num_inference_steps, initial_scheduler)
            latents_end = self.pipe.denoise(
                latents=latents_mid,
                text_embeddings=text_embeddings_end,
                t_start=num_initial_steps,
                t_end=None,
                guidance_scale=guidance_scale,
            )
            image_end = self.pipe.latents_to_image(latents_end)
            self.pipe.safety_check(image_end)

            # Generate latents for animation frames
            frames_latents = []
            for i in range(num_animation_frames):
                if i == 0:
                    latents = latents_start
                elif i == num_animation_frames - 1:
                    latents = latents_end
                else:
                    print(f"Generating frame {i}")
                    text_embeddings = slerp(
                        i / (num_animation_frames - 1),
                        text_embeddings_start,
                        text_embeddings_end,
                    )

                    # re-initialize scheduler
                    self.pipe.scheduler = make_scheduler(
                        num_inference_steps, initial_scheduler
                    )
                    latents = self.pipe.denoise(
                        latents=latents_mid,
                        text_embeddings=text_embeddings,
                        t_start=num_initial_steps,
                        t_end=None,
                        guidance_scale=guidance_scale,
                    )

                # de-noise this frame
                frames_latents.append(latents)
                if intermediate_output and i > 0:
                    image = self.pipe.latents_to_image(latents)
                    yield save_pil_image(
                        self.pipe.numpy_to_pil(image)[0], path=f"/tmp/output-{i}.png"
                    )

            # Decode images by interpolate between animation frames
            if film_interpolation:
                images = self.interpolate_film(frames_latents, num_interpolation_steps)
            else:
                images = self.interpolate_latents(
                    frames_latents, num_interpolation_steps
                )

            # Save the video
            if gif_ping_pong:
                images += images[-1:1:-1]

            if output_format == "gif":
                yield self.save_gif(images, gif_frames_per_second)
            else:
                yield self.save_mp4(images, gif_frames_per_second, width, height)

    def save_mp4(self, images, fps, width, height):
        print("Saving MP4")
        output_path = "/tmp/output.mp4"

        output = av.open(output_path, "w")
        stream = output.add_stream(
            "h264", rate=fps, options={"crf": "17", "tune": "film"}
        )
        # stream.bit_rate = 8000000
        # stream.bit_rate = 16000000
        stream.width = width
        stream.height = height

        for i, image in enumerate(images):
            image = (image * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            packet = stream.encode(frame)
            output.mux(packet)

        # flush
        packet = stream.encode(None)
        output.mux(packet)
        output.close()

        return Path(output_path)

    def save_gif(self, images, fps):
        print("Saving GIF")
        pil_images = [
            self.pipe.numpy_to_pil(img.astype("float32"))[0] for img in images
        ]

        output_path = "/tmp/video.gif"
        gif_frame_duration = int(1000 / fps)

        with open(output_path, "wb") as f:
            pil_images[0].save(
                fp=f,
                format="GIF",
                append_images=pil_images[1:],
                save_all=True,
                duration=gif_frame_duration,
                loop=0,
            )

        return Path(output_path)

    def interpolate_latents(self, frames_latents, num_interpolation_steps):
        print("Interpolating images from latents")
        images = []
        for i in range(len(frames_latents) - 1):
            latents_start = frames_latents[i]
            latents_end = frames_latents[i + 1]
            for j in range(num_interpolation_steps):
                x = j / num_interpolation_steps
                latents = latents_start * (1 - x) + latents_end * x
                image = self.pipe.latents_to_image(latents)
                images.append(image)
        return images

    def interpolate_film(self, frames_latents, num_interpolation_steps):
        print("Interpolating images with FILM")
        images = [
            self.pipe.latents_to_image(lat)[0].astype("float32")
            for lat in frames_latents
        ]
        if num_interpolation_steps == 0:
            return images

        num_recursion_steps = max(int(np.ceil(np.log2(num_interpolation_steps))), 1)
        images = film_util.interpolate_recursively_from_memory(
            images, num_recursion_steps, self.interpolator
        )
        images = [img.clip(0, 1) for img in images]
        return images


def make_scheduler(num_inference_steps, from_scheduler=None):
    scheduler = PNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
    )
    scheduler.set_timesteps(num_inference_steps, offset=1)
    if from_scheduler:
        scheduler.cur_model_output = from_scheduler.cur_model_output
        scheduler.counter = from_scheduler.counter
        scheduler.cur_sample = from_scheduler.cur_sample
        scheduler.ets = from_scheduler.ets[:]
    return scheduler


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""
    # from https://gist.github.com/nateraw/c989468b74c616ebbc6474aa8cdd9e53

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def save_pil_image(image, path):
    image.save(path)
    return Path(path)
