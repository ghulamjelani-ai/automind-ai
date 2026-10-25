"""HTTP client for the GPU model server (model_server/server.py).

Every failure becomes an HTTPException with a message the user can act on, so a
stopped Colab session or a wrong URL never shows up as a raw stack trace.
"""

import base64
import time

import requests
from fastapi import HTTPException, status

from app.core.config import settings

# Status codes a tunnel (Cloudflare, ngrok) returns when the Colab side is gone or busy.
GATEWAY_CODES = {502, 504, 520, 521, 522, 523, 524, 530}

OFFLINE_FIX = (
    "Start the Colab notebook (colab/MarketingAI_GPU.ipynb), then copy the URL and key "
    "it prints into MODEL_SERVER_URL and MODEL_SERVER_API_KEY in backend/.env and "
    "restart the backend."
)


def _error(code: int, message: str) -> HTTPException:
    return HTTPException(status_code=code, detail=message)


def _post(path: str, payload: dict, timeout: float | None = None) -> dict:
    if not settings.MODEL_SERVER_URL:
        raise _error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The AI model server is not configured. " + OFFLINE_FIX,
        )

    url = settings.MODEL_SERVER_URL + path
    headers = {"X-API-Key": settings.MODEL_SERVER_API_KEY}
    wait = timeout or settings.MODEL_SERVER_TIMEOUT

    response = None

    # One quick retry covers a dropped connection or a tunnel hiccup.
    for attempt in range(2):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=(10, wait))
        except requests.Timeout:
            raise _error(
                status.HTTP_504_GATEWAY_TIMEOUT,
                "The AI model took too long to answer. Please try again with a shorter request.",
            )
        except requests.RequestException as exc:
            print(f"Model server request failed (attempt {attempt + 1}):", repr(exc))
            response = None

            if attempt == 0:
                time.sleep(1.5)
                continue

            raise _error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "Could not reach the AI model server. " + OFFLINE_FIX,
            )

        if response.status_code in GATEWAY_CODES and attempt == 0:
            time.sleep(1.5)
            continue

        break

    if response.status_code == 200:
        try:
            return response.json()
        except ValueError:
            raise _error(status.HTTP_502_BAD_GATEWAY, "The AI model server sent an unreadable answer.")

    detail = ""

    try:
        detail = str(response.json().get("detail") or "")
    except (ValueError, AttributeError):
        pass

    print("Model server error:", response.status_code, detail or response.text[:300])

    if response.status_code == 401:
        raise _error(
            status.HTTP_502_BAD_GATEWAY,
            "The AI model server rejected the API key. Make MODEL_SERVER_API_KEY in "
            "backend/.env match the key printed by the Colab notebook.",
        )

    if response.status_code == 503:
        raise _error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The AI models are still loading on the GPU. Please try again in a minute.",
        )

    if response.status_code == 422 and detail:
        raise _error(status.HTTP_422_UNPROCESSABLE_ENTITY, detail)

    if response.status_code == 507:
        raise _error(status.HTTP_503_SERVICE_UNAVAILABLE, detail or "The GPU is out of memory. Please try again.")

    if response.status_code in GATEWAY_CODES or response.status_code == 404:
        raise _error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The AI model server is offline. " + OFFLINE_FIX,
        )

    raise _error(
        status.HTTP_502_BAD_GATEWAY,
        f"The AI model server returned an error (status {response.status_code}). "
        + (detail or "Please try again."),
    )


def chat(
    messages: list[dict],
    max_new_tokens: int = 800,
    temperature: float = 0.7,
    prefix: str = "",
) -> str:
    data = _post(
        "/v1/chat",
        {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "prefix": prefix,
        },
    )

    return str(data.get("text") or "").strip()


def image(prompt: str, negative_prompt: str = "") -> bytes:
    data = _post(
        "/v1/image",
        {"prompt": prompt, "negative_prompt": negative_prompt},
    )

    try:
        return base64.b64decode(data["image_base64"])
    except (KeyError, TypeError, ValueError):
        raise _error(status.HTTP_502_BAD_GATEWAY, "The AI model server did not return an image. Please try again.")


def health() -> dict:
    """Status for the dashboard badge. Never raises."""

    if not settings.MODEL_SERVER_URL:
        return {"online": False, "status": "not_configured", "message": "MODEL_SERVER_URL is not set"}

    try:
        response = requests.get(settings.MODEL_SERVER_URL + "/health", timeout=8)
        data = response.json() if response.status_code == 200 else {}
    except (requests.RequestException, ValueError):
        data = {}

    if not data:
        return {"online": False, "status": "offline", "message": "The GPU model server is not reachable"}

    return {
        "online": data.get("status") == "ready",
        "status": data.get("status", "unknown"),
        "message": data.get("error") or data.get("step") or "",
        "gpu": data.get("gpu"),
        "text_model": data.get("text_model"),
        "text_fine_tuned": bool(data.get("text_fine_tuned")),
        "image_model": data.get("image_model"),
        "image_fine_tuned": bool(data.get("image_fine_tuned")),
    }
