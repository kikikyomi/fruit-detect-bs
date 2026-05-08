from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["users"])


class UserCreate(BaseModel):
    username: str = Field(min_length=1, max_length=32)
    role: Literal["admin", "operator", "viewer"] = "operator"


_users: list[dict[str, object]] = [
    {"id": 1, "username": "admin", "role": "admin"},
    {"id": 2, "username": "inspector", "role": "operator"},
]


@router.get("/users")
def get_users() -> dict[str, object]:
    return {"items": _users}


@router.post("/users")
def create_user(payload: UserCreate) -> dict[str, object]:
    new_user = {
        "id": (max((int(u["id"]) for u in _users), default=0) + 1),
        "username": payload.username,
        "role": payload.role,
    }
    _users.append(new_user)
    return {"item": new_user}


@router.get("/profile")
def get_profile() -> dict[str, object]:
    return {
        "username": "admin",
        "role": "admin",
        "email": "admin@example.com",
    }
