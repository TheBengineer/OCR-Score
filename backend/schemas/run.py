"""Pydantic schemas for the OCRRun model."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from backend.models.enums import RunStatus


# ── Create ───────────────────────────────────────────────────────────────────
class OCRRunCreate(BaseModel):
    """Schema for creating a new OCR run."""

    pdf_id: uuid.UUID
    engine_id: uuid.UUID
    engine_config: dict | None = Field(
        default=None,
        description="Engine-specific configuration for this run",
    )
    engine_version: str | None = Field(
        default=None,
        max_length=63,
        description="Engine version at time of run",
    )
    run_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 of (pdf_id + engine_id + engine_config) for dedup",
    )
    environment: dict | None = Field(
        default=None,
        description="Runtime environment metadata",
    )


# ── Read ─────────────────────────────────────────────────────────────────────
class OCRRunRead(BaseModel):
    """Schema for reading an OCR run record."""

    id: uuid.UUID
    pdf_id: uuid.UUID
    engine_id: uuid.UUID
    status: RunStatus
    engine_config: dict | None = None
    engine_version: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    raw_output_uri: str | None = None
    run_hash: str
    environment: dict | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Update ───────────────────────────────────────────────────────────────────
class OCRRunUpdate(BaseModel):
    """Schema for updating an OCR run record (partial)."""

    status: RunStatus | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    raw_output_uri: str | None = None
