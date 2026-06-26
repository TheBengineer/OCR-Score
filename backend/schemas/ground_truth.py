"""Pydantic schemas for GroundTruthVersion and GTPageResult."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from backend.models.enums import GroundTruthSource


# ── GroundTruthVersion ───────────────────────────────────────────────────────
class GroundTruthVersionCreate(BaseModel):
    """Schema for creating a new ground truth version."""

    pdf_id: uuid.UUID
    version_number: int = Field(default=1, ge=1)
    source: GroundTruthSource = GroundTruthSource.MANUAL
    created_by: str | None = Field(
        default=None,
        max_length=255,
        description="User or system that created this version",
    )
    notes: str | None = None


class GroundTruthVersionRead(BaseModel):
    """Schema for reading a ground truth version."""

    id: uuid.UUID
    pdf_id: uuid.UUID
    version_number: int
    source: GroundTruthSource
    created_at: datetime
    created_by: str | None = None
    notes: str | None = None
    deleted_at: datetime | None = None

    model_config = {"from_attributes": True}


# ── GTPageResult ─────────────────────────────────────────────────────────────
class GTPageResultCreate(BaseModel):
    """Schema for creating a ground truth page result."""

    gt_version_id: uuid.UUID
    page_number: int = Field(..., ge=1, description="1-based page number")
    width: float | None = Field(
        default=None,
        gt=0,
        description="Page width in points (72 DPI)",
    )
    height: float | None = Field(
        default=None,
        gt=0,
        description="Page height in points (72 DPI)",
    )
    data: dict = Field(
        default_factory=dict,
        description="Canonical JSONB page hierarchy (same schema as PageResult)",
    )
    confidence: float | None = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this ground truth (1.0 = certain)",
    )


class GTPageResultRead(BaseModel):
    """Schema for reading a ground truth page result."""

    id: uuid.UUID
    gt_version_id: uuid.UUID
    page_number: int
    width: float | None = None
    height: float | None = None
    data: dict
    confidence: float | None = None

    model_config = {"from_attributes": True}


class GTPageResultUpdate(BaseModel):
    """Schema for updating a ground truth page result (partial)."""

    data: dict | None = None
    width: float | None = Field(default=None, gt=0)
    height: float | None = Field(default=None, gt=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
