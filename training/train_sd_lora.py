"""LoRA fine-tuning of Stable Diffusion 1.5 on your own product or brand photos.

Put 15 to 60 photos in training/data/brand_images/. A caption for photo.jpg goes in
photo.txt next to it (optional). Every caption gets the trigger word, so the model
learns your look under that word; the model server adds the word to every prompt.

On a Colab T4 the default run takes about 10 to 20 minutes.

Usage:
  python training/train_sd_lora.py --trigger "brandstyle"
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_samples(folder: Path, trigger: str, default_caption: str) -> list[tuple[Path, str]]:
    samples = []

    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_TYPES:
            continue

        caption_file = path.with_suffix(".txt")
        caption = caption_file.read_text(encoding="utf-8").strip() if caption_file.exists() else ""
        caption = caption or default_caption

        if trigger and trigger.lower() not in caption.lower():
            caption = f"{trigger}, {caption}"

        samples.append((path, caption))

    return samples


def to_tensor(path: Path, size: int, flip: bool) -> torch.Tensor:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image = ImageOps.fit(image, (size, size), Image.BICUBIC)

    if flip and random.random() < 0.5:
        image = ImageOps.mirror(image)

    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0

    return torch.from_numpy(array).permute(2, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--images", default=str(ROOT / "training/data/brand_images"))
    parser.add_argument("--output", default=str(ROOT / "training/output/sd15-brand-lora"))
    parser.add_argument("--trigger", default="brandstyle")
    parser.add_argument("--default-caption", default="a professional marketing photo")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=0, help="0 = about 80 steps per photo")
    parser.add_argument("--no-flip", action="store_true", help="Keep for photos with text or logos")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    folder = Path(args.images)

    if not folder.is_dir():
        sys.exit(f"Image folder not found: {folder}")

    samples = load_samples(folder, args.trigger.strip(), args.default_caption)

    if len(samples) < 5:
        sys.exit(f"Found {len(samples)} photos in {folder}. Add at least 5 (15 to 60 is best).")

    steps = args.steps or max(400, min(1500, 80 * len(samples)))
    print(f"Training on {len(samples)} photos for {steps} steps")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    weight_dtype = torch.float16 if cuda else torch.float32

    def part(cls, name):
        return cls.from_pretrained(args.model, subfolder=name, token=args.hf_token)

    tokenizer = part(CLIPTokenizer, "tokenizer")
    text_encoder = part(CLIPTextModel, "text_encoder")
    vae = part(AutoencoderKL, "vae")
    unet = part(UNet2DConditionModel, "unet")
    noise_scheduler = part(DDPMScheduler, "scheduler")

    for module in (text_encoder, vae, unet):
        module.requires_grad_(False)
        module.to(device, dtype=weight_dtype)

    unet.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )

    # Trainable LoRA weights stay in fp32 for stable fp16 training.
    lora_params = [param for param in unet.parameters() if param.requires_grad]

    for param in lora_params:
        param.data = param.data.float()

    if cuda:
        unet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-2)
    lr_at = lambda step: args.lr * 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps))  # noqa: E731
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)

    # Captions do not change, so they are encoded once.
    with torch.no_grad():
        prompt_ids = {
            caption: tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(device)
            for _, caption in samples
        }
        embeddings = {caption: text_encoder(ids)[0] for caption, ids in prompt_ids.items()}

    unet.train()
    started = time.time()
    running, logged = 0.0, 0

    for step in range(1, steps + 1):
        path, caption = random.choice(samples)
        pixels = to_tensor(path, args.resolution, not args.no_flip).unsqueeze(0).to(device, dtype=weight_dtype)

        with torch.no_grad():
            latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)

        if noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=cuda):
            prediction = unet(noisy, timesteps, embeddings[caption]).sample

        loss = F.mse_loss(prediction.float(), target.float())

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        scaler.step(optimizer)
        scaler.update()

        running += loss.item()
        logged += 1

        if step % 25 == 0 or step == steps:
            print(f"step {step}/{steps}  loss {running / logged:.4f}  {time.time() - started:.0f}s")
            running, logged = 0.0, 0

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    unet_lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(output),
        unet_lora_layers=unet_lora,
        safe_serialization=True,
    )

    (output / "trigger.txt").write_text(args.trigger.strip(), encoding="utf-8")
    (output / "training_info.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "photos": len(samples),
                "steps": steps,
                "rank": args.rank,
                "trigger": args.trigger.strip(),
                "minutes": round((time.time() - started) / 60, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Saved LoRA to", output)


if __name__ == "__main__":
    main()
