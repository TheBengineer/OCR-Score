"""Pydantic v2 request/response schemas for the authentication module."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from backend.auth.models import UserRole

# ── Login ─────────────────────────────────────────────────────────────────────


class LoginRequest(BaseModel):
    """Schema for user login."""

    username: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    """Schema returned on successful login."""

    access_token: str
    token_type: str = "bearer"
    user: UserRead


# ── User ──────────────────────────────────────────────────────────────────────


class UserRead(BaseModel):
    """Public representation of a user (no password)."""

    id: uuid.UUID
    username: str
    email: str
    role: UserRole
    created_at: datetime

    model_config = {"from_attributes": True}


class UserCreate(BaseModel):
    """Schema for creating a new user (admin only)."""

    username: str = Field(..., min_length=1, max_length=255)
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.VIEWER


# ── API keys ──────────────────────────────────────────────────────────────────


class ApiKeyCreate(BaseModel):
    """Schema for requesting a new API key."""

    name: str = Field(..., min_length=1, max_length=255, description="Human-readable label for this key")


class ApiKeyCreated(BaseModel):
    """Schema returned once when an API key is created.

    The ``key`` field contains the raw key value and is only provided at
    creation time. It cannot be retrieved later.
    """

    id: uuid.UUID
    name: str
    key: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ApiKeyRead(BaseModel):
    """Public representation of an API key (no raw key value)."""

    id: uuid.UUID
    name: str
    created_at: datetime
    last_used_at: datetime | None = None

    model_config = {"from_attributes": True}
