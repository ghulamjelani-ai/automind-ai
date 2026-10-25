import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _path(name: str, default: str) -> str:
    """Relative paths are read from the project folder, so the notebook and a local run agree."""

    value = os.getenv(name, default).strip()

    if not value:
        return ""

    path = Path(value)

    return str(path if path.is_absolute() else PROJECT_DIR / path)


class Settings:
    # Shared secret. The backend sends it in the X-API-Key header.
    API_KEY: str = os.getenv("MODEL_SERVER_API_KEY", "")

    # Text model: Mistral 7B Instruct plus the LoRA adapter from training/train_mistral_lora.py
    TEXT_MODEL: str = os.getenv("TEXT_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    TEXT_ADAPTER: str = _path("TEXT_ADAPTER", "training/output/mistral-marketing-lora")

    # auto: 4-bit on GPUs under 22 GB (for example a Colab T4), full fp16 on bigger GPUs.
    # Set to "4bit" or "fp16" to force one.
    TEXT_PRECISION: str = os.getenv("TEXT_PRECISION", "auto").strip().lower()

    # Prompts longer than this drop their oldest chat turns first.
    MAX_INPUT_TOKENS: int = int(os.getenv("MAX_INPUT_TOKENS", "6000"))
    MAX_NEW_TOKENS_LIMIT: int = int(os.getenv("MAX_NEW_TOKENS_LIMIT", "1600"))

    # Image model: Stable Diffusion 1.5 plus the LoRA from training/train_sd_lora.py
    IMAGE_MODEL: str = os.getenv("IMAGE_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    IMAGE_LORA: str = _path("IMAGE_LORA", "training/output/sd15-brand-lora")
    IMAGE_LORA_SCALE: float = float(os.getenv("IMAGE_LORA_SCALE", "0.8"))

    # The word the image LoRA was trained with. It is added to every image prompt.
    IMAGE_LORA_TRIGGER: str = os.getenv("IMAGE_LORA_TRIGGER", "").strip()

    # Fine-tuned SD 1.5 decoder: sharper detail, cleaner faces and hands. Empty = the model's own.
    IMAGE_VAE: str = os.getenv("IMAGE_VAE", "stabilityai/sd-vae-ft-mse").strip()

    # SD 1.5 draws best at 512 px. The "hi-res fix" redraws it at 1.5x (768 px) to add real
    # detail, then the result is enlarged to IMAGE_SIZE * IMAGE_UPSCALE for the feed.
    IMAGE_SIZE: int = int(os.getenv("IMAGE_SIZE", "512"))
    IMAGE_UPSCALE: int = int(os.getenv("IMAGE_UPSCALE", "2"))
    IMAGE_HIRES: bool = _bool("IMAGE_HIRES", "true")
    IMAGE_HIRES_STRENGTH: float = float(os.getenv("IMAGE_HIRES_STRENGTH", "0.45"))
    IMAGE_STEPS: int = int(os.getenv("IMAGE_STEPS", "28"))
    IMAGE_GUIDANCE: float = float(os.getenv("IMAGE_GUIDANCE", "7.0"))

    # Turn off only if you understand the risk: the safety checker blocks unsafe images.
    IMAGE_SAFETY_CHECKER: bool = _bool("IMAGE_SAFETY_CHECKER", "true")

    # Set to false to run only the text model (for example on a small GPU).
    LOAD_TEXT: bool = _bool("LOAD_TEXT", "true")
    LOAD_IMAGE: bool = _bool("LOAD_IMAGE", "true")

    HF_TOKEN: str = os.getenv("HF_TOKEN", "")


settings = Settings()
