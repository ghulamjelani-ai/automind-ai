from fastapi import APIRouter

from app.core import model_server

router = APIRouter()


@router.get("/status")
def ai_status():
    """Whether the GPU model server (Mistral 7B + Stable Diffusion 1.5) is ready."""

    return {"success": True, **model_server.health()}
