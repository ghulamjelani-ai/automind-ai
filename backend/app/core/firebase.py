import os

import firebase_admin
from firebase_admin import auth as firebase_auth
from firebase_admin import credentials
from fastapi import HTTPException, status

from app.core.config import settings


def get_firebase_app():
    """
    Initialize Firebase Admin SDK only when needed.
    This keeps backend running even if service account is not added yet.
    """

    if firebase_admin._apps:
        return firebase_admin.get_app()

    credentials_path = settings.FIREBASE_CREDENTIALS_PATH

    if credentials_path and os.path.exists(credentials_path):
        cred = credentials.Certificate(credentials_path)
        return firebase_admin.initialize_app(cred)

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=(
            "Firebase Admin service account is not configured. "
            "Add FIREBASE_CREDENTIALS_PATH in .env later."
        ),
    )


def verify_firebase_token(id_token: str):
    """
    Verify Firebase ID token from frontend.
    """

    try:
        get_firebase_app()
        decoded_token = firebase_auth.verify_id_token(id_token)

        return {
            "uid": decoded_token.get("uid"),
            "email": decoded_token.get("email"),
            "name": decoded_token.get("name"),
            "picture": decoded_token.get("picture"),
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired Firebase token: {str(error)}",
        )