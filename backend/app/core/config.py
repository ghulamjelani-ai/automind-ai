import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)


class Settings:
    APP_NAME: str = os.getenv(
        "APP_NAME",
        "Marketing AI Backend"
    )

    FRONTEND_URL: str = os.getenv(
        "FRONTEND_URL",
        "http://localhost:5173"
    )

    FIREBASE_CREDENTIALS_PATH: str = os.getenv(
        "FIREBASE_CREDENTIALS_PATH",
        ""
    )

    # AI model server (model_server/ on a Colab GPU): Mistral 7B and Stable Diffusion 1.5.
    # The Colab notebook prints both values when the server starts.
    MODEL_SERVER_URL: str = os.getenv(
        "MODEL_SERVER_URL",
        ""
    ).strip().rstrip("/")

    MODEL_SERVER_API_KEY: str = os.getenv(
        "MODEL_SERVER_API_KEY",
        ""
    ).strip()

    # Longest wait for one model call. Keep it under 100 seconds when the server is
    # reached through a Cloudflare quick tunnel, which closes slower requests.
    MODEL_SERVER_TIMEOUT: float = float(
        os.getenv("MODEL_SERVER_TIMEOUT", "95")
    )

    # Database
    DB_HOST: str = os.getenv(
        "DB_HOST",
        "127.0.0.1"
    )

    DB_PORT: int = int(
        os.getenv("DB_PORT", "5432")
    )

    DB_NAME: str = os.getenv(
        "DB_NAME",
        "marketing_ai_db"
    )

    DB_USER: str = os.getenv(
        "DB_USER",
        "marketing_ai_user"
    )

    DB_PASSWORD: str = os.getenv(
        "DB_PASSWORD",
        "12345"
    )


settings = Settings()


print("ENV FILE LOADED FROM:", ENV_PATH)
print("DB HOST:", settings.DB_HOST)
print("DB PORT:", settings.DB_PORT)
print("DB NAME:", settings.DB_NAME)
print("DB USER:", settings.DB_USER)
print("MODEL SERVER:", settings.MODEL_SERVER_URL or "(not set)")
