from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.core.firebase import verify_firebase_token
from app.db.database import get_db
from app.models.models import User

router = APIRouter()


def get_current_firebase_user(authorization: str | None = Header(default=None)):
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must start with Bearer",
        )

    id_token = authorization.replace("Bearer ", "").strip()

    if not id_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Firebase ID token",
        )

    return verify_firebase_token(id_token)


def get_or_create_user(db: Session, firebase_user: dict):
    firebase_uid = firebase_user.get("uid")

    if not firebase_uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Firebase uid missing",
        )

    user = db.query(User).filter(User.firebase_uid == firebase_uid).first()

    if user:
        user.name = firebase_user.get("name") or user.name
        user.email = firebase_user.get("email") or user.email
        user.photo_url = firebase_user.get("picture") or user.photo_url

        db.commit()
        db.refresh(user)

        return user

    user = User(
        firebase_uid=firebase_uid,
        name=firebase_user.get("name"),
        email=firebase_user.get("email"),
        photo_url=firebase_user.get("picture"),
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def get_current_db_user(
    firebase_user: dict = Depends(get_current_firebase_user),
    db: Session = Depends(get_db),
):
    return get_or_create_user(db, firebase_user)


@router.get("/me")
def me(user: User = Depends(get_current_db_user)):
    return {
        "success": True,
        "user": {
            "id": str(user.id),
            "firebase_uid": user.firebase_uid,
            "name": user.name,
            "email": user.email,
            "photo_url": user.photo_url,
        },
    }


@router.post("/sync")
def sync_user(user: User = Depends(get_current_db_user)):
    return {
        "success": True,
        "message": "User synced successfully",
        "user": {
            "id": str(user.id),
            "firebase_uid": user.firebase_uid,
            "name": user.name,
            "email": user.email,
            "photo_url": user.photo_url,
        },
    }