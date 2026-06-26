"""Pydantic schemas for the OCREngine model."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


# ── Create ───────────────────────────────────────────────────────────────────
class OCREngineCreate(BaseModel):
    """Schema for registering a new OCR engine."""

    slug: str = Field(
        ...,
        max_length=127,
        description="Unique machine-readable identifier (e.g. 'tesseract')",
        pattern=r"^[a-z][a-z0-9_-]*$",
    )
    display_name: str = Field(
        ...,
        max_length=255,
        description="Human-readable engine name",
    )
    version: str = Field(
        default="0.0.0",
        max_length=63,
        description="Engine plugin version",
    )
    enabled: bool = True
    config_schema: dict | None = Field(
        default=None,
        description="JSON Schema describing valid engine_config values",
    )
    description: str | None = Field(default=None, description="Engine description")


# ── Read ─────────────────────────────────────────────────────────────────────
class OCREngineRead(BaseModel):
    """Schema for reading an OCR engine record."""

    id: uuid.UUID
    slug: str
    display_name: str
    version: str
    enabled: bool
    config_schema: dict | None = None
    description: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── Update ───────────────────────────────────────────────────────────────────
class OCREngineUpdate(BaseModel):
    """Schema for updating an OCR engine record (partial)."""

    display_name: str | None = Field(default=None, max_length=255)
    version: str | None = Field(default=None, max_length=63)
    enabled: bool | None = None
    config_schema: dict | None = None
    description: str | None = None
