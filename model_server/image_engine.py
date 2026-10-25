import io
import os
import random
import threading
import time

import torch
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)
from PIL import Image

from config import settings

# SD 1.5 reads at most 77 tokens, so this list stays short and the important words come first.
DEFAULT_NEGATIVE = (
    "text, watermark, logo, letters, blurry, lowres, jpeg artifacts, deformed, bad anatomy, "
    "bad hands, extra fingers, mutated hands, poorly drawn face, extra limbs, cartoon, "
    "anime, 3d render, plastic skin, oversaturated, cropped, ugly"
)


class UnsafeImageError(Exception):
    pass


def _check_safe(result) -> None:
    flagged = getattr(result, "nsfw_content_detected", None)

    if flagged and any(flagged):
        raise UnsafeImageError(
            "The image was blocked by the safety filter. Try a different description."
        )


class ImageEngine:
    """Stable Diffusion 1.5 with the optional brand style LoRA."""

    def __init__(self):
        token = settings.HF_TOKEN or None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        kwargs = {"torch_dtype": dtype, "token": token}

        if not settings.IMAGE_SAFETY_CHECKER:
            kwargs.update(safety_checker=None, requires_safety_checker=False)

        if settings.IMAGE_VAE:
            try:
                kwargs["vae"] = AutoencoderKL.from_pretrained(settings.IMAGE_VAE, torch_dtype=dtype, token=token)
            except Exception as exc:
                print("Using the model's own VAE; could not load", settings.IMAGE_VAE, repr(exc))

        try:
            # The fp16 files are half the download; not every SD 1.5 copy has them.
            variant = "fp16" if self.device == "cuda" else None
            pipe = StableDiffusionPipeline.from_pretrained(settings.IMAGE_MODEL, variant=variant, **kwargs)
        except (OSError, ValueError):
            pipe = StableDiffusionPipeline.from_pretrained(settings.IMAGE_MODEL, **kwargs)

        # DPM++ 2M Karras gives clean results in about 25 steps.
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )

        # PyTorch 2 attention (SDPA) is used by default; it is faster than attention
        # slicing and already memory efficient.
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)

        self.lora = None

        if settings.IMAGE_LORA and os.path.isdir(settings.IMAGE_LORA) and any(
            name.endswith(".safetensors") for name in os.listdir(settings.IMAGE_LORA)
        ):
            pipe.load_lora_weights(settings.IMAGE_LORA)
            pipe.fuse_lora(lora_scale=settings.IMAGE_LORA_SCALE)
            self.lora = settings.IMAGE_LORA

        # train_sd_lora.py saves the trigger word next to the weights.
        self.trigger = settings.IMAGE_LORA_TRIGGER
        trigger_file = os.path.join(settings.IMAGE_LORA or "", "trigger.txt")

        if self.lora and not self.trigger and os.path.isfile(trigger_file):
            with open(trigger_file, encoding="utf-8") as handle:
                self.trigger = handle.read().strip()

        self.pipe = pipe

        # Second pass for the hi-res fix. It shares every weight with pipe (no extra memory).
        self.hires = None

        if settings.IMAGE_HIRES:
            self.hires = StableDiffusionImg2ImgPipeline.from_pipe(pipe)
            self.hires.set_progress_bar_config(disable=True)

        self.lock = threading.Lock()

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int | None = None,
        guidance_scale: float | None = None,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
    ) -> dict:
        started = time.time()

        prompt = prompt.strip()

        if self.lora and self.trigger and self.trigger.lower() not in prompt.lower():
            prompt = self.trigger + ", " + prompt

        size = settings.IMAGE_SIZE
        # SD needs sides that are a multiple of 8.
        width = max(256, min(768, (width or size) // 8 * 8))
        height = max(256, min(768, (height or size) // 8 * 8))
        seed = seed if seed is not None else random.randint(0, 2**31 - 1)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        common = dict(
            prompt=prompt,
            negative_prompt=negative_prompt.strip() or DEFAULT_NEGATIVE,
            num_inference_steps=max(10, min(steps or settings.IMAGE_STEPS, 60)),
            guidance_scale=guidance_scale or settings.IMAGE_GUIDANCE,
            generator=generator,
        )

        with self.lock, torch.inference_mode():
            result = self.pipe(width=width, height=height, **common)
            _check_safe(result)
            image: Image.Image = result.images[0]

            if self.hires is not None:
                # Enlarge 1.5x, then let SD redraw the fine detail at that size.
                big = image.resize((width * 3 // 2 // 8 * 8, height * 3 // 2 // 8 * 8), Image.LANCZOS)
                result = self.hires(image=big, strength=settings.IMAGE_HIRES_STRENGTH, **common)
                _check_safe(result)
                image = result.images[0]

        target = (width * settings.IMAGE_UPSCALE, height * settings.IMAGE_UPSCALE)

        if target[0] > image.width:
            image = image.resize(target, Image.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)

        return {
            "png": buffer.getvalue(),
            "seed": seed,
            "seconds": round(time.time() - started, 2),
        }
