"""GPU model server: Mistral 7B (captions, descriptions, hashtags, plans) and
Stable Diffusion 1.5 (images). The marketing backend calls it over HTTP.

Run from this folder:  uvicorn server:app --host 0.0.0.0 --port 7860
"""

import base64
import secrets
import threading
import traceback

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from config import settings

app = FastAPI(title="Marketing AI Model Server", version="1.0.0")

if not settings.API_KEY:
    settings.API_KEY = secrets.token_urlsafe(24)
    print("=" * 70)
    print("MODEL_SERVER_API_KEY was not set. Generated one for this run:")
    print(settings.API_KEY)
    print("=" * 70)

state = {
    "status": "loading",
    "step": "starting",
    "error": None,
    "text": None,
    "image": None,
    "errors": {},
}


def _load(name: str, label: str, factory):
    """Loads one model. A failure is recorded but never stops the other model."""

    state["step"] = f"loading {label}"

    try:
        state[name] = factory()
    except Exception as exc:
        traceback.print_exc()
        state["errors"][name] = f"{label}: {exc!r}"


def _text_engine():
    from text_engine import TextEngine

    return TextEngine()


def _image_engine():
    # Imported here so a diffusers import error is reported like any other load error.
    from image_engine import ImageEngine

    return ImageEngine()


def _load_models():
    try:
        if settings.LOAD_TEXT:
            _load("text", "Mistral 7B", _text_engine)

            if state["text"]:
                print("Text model ready:", settings.TEXT_MODEL, "adapter:", state["text"].adapter)

        if settings.LOAD_IMAGE:
            _load("image", "Stable Diffusion 1.5", _image_engine)

            if state["image"]:
                print("Image model ready:", settings.IMAGE_MODEL, "lora:", state["image"].lora)
    finally:
        # Always leave "loading", even after an unexpected error, so /health tells the truth.
        state["error"] = "; ".join(state["errors"].values()) or None
        loaded = state["text"] is not None or state["image"] is not None
        state["status"] = "ready" if loaded else "error"
        state["step"] = "ready" if not state["errors"] else "ready with errors" if loaded else "failed"


@app.on_event("startup")
def start_loading():
    # Loading takes a few minutes, so it runs in the background and /health reports progress.
    threading.Thread(target=_load_models, daemon=True).start()


def require_key(x_api_key: str = Header(default="")):
    if not secrets.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(status_code=401, detail="Invalid model server API key")


def _engine(name: str):
    engine = state[name]

    if engine is not None:
        return engine

    if name in state["errors"]:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {state['errors'][name]}")

    if state["status"] in ("ready", "error"):
        raise HTTPException(status_code=501, detail=f"The {name} model is turned off on this server")

    raise HTTPException(status_code=503, detail=f"Models are still loading ({state['step']})")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    max_new_tokens: int = Field(default=800, ge=16, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    # Text the answer must start with (it is part of the returned text).
    prefix: str = Field(default="", max_length=200)


class ImageRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    steps: int | None = Field(default=None, ge=10, le=60)
    guidance_scale: float | None = Field(default=None, ge=1.0, le=15.0)
    width: int | None = Field(default=None, ge=256, le=768)
    height: int | None = Field(default=None, ge=256, le=768)
    seed: int | None = Field(default=None, ge=0)


@app.get("/health")
def health():
    text = state["text"]
    image = state["image"]

    return {
        "status": state["status"],
        "step": state["step"],
        "error": state["error"],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "text_model": settings.TEXT_MODEL if text else None,
        "text_fine_tuned": bool(text and text.adapter),
        "text_4bit": bool(text and text.quantized),
        "image_model": settings.IMAGE_MODEL if image else None,
        "image_fine_tuned": bool(image and image.lora),
    }


@app.post("/v1/chat", dependencies=[Depends(require_key)])
def chat(data: ChatRequest):
    engine = _engine("text")

    try:
        return engine.generate(
            [message.model_dump() for message in data.messages],
            max_new_tokens=data.max_new_tokens,
            temperature=data.temperature,
            top_p=data.top_p,
            prefix=data.prefix,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=507, detail="The GPU ran out of memory. Try a shorter request.")


@app.post("/v1/image", dependencies=[Depends(require_key)])
def image(data: ImageRequest):
    engine = _engine("image")

    from image_engine import UnsafeImageError

    try:
        result = engine.generate(**data.model_dump())
    except UnsafeImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=507, detail="The GPU ran out of memory. Please try again.")

    return {
        "image_base64": base64.b64encode(result["png"]).decode("ascii"),
        "seed": result["seed"],
        "seconds": result["seconds"],
    }
