"""Pydantic schemas for Score and ScoreSummary models."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from backend.models.enums import ScoreLevel, ScoreMetric


# ── Score ────────────────────────────────────────────────────────────────────
class ScoreCreate(BaseModel):
    """Schema for recording a single score measurement."""

    run_id: uuid.UUID
    gt_version_id: uuid.UUID | None = Field(
        default=None,
        description="NULL if computed via multi-engine consensus or no GT",
    )
    level: ScoreLevel
    page_number: int | None = Field(
        default=None,
        ge=1,
        description="Page number when level is page-specific; NULL for document-level",
    )
    metric: ScoreMetric
    value: float = Field(..., description="Score value (typically 0-1)")
    confidence_weighted: bool = False
    details: dict | None = Field(
        default=None,
        description="Optional breakdown (e.g. substitution/insertion/deletion counts)",
    )


class ScoreRead(BaseModel):
    """Schema for reading a score record."""

    id: uuid.UUID
    run_id: uuid.UUID
    gt_version_id: uuid.UUID | None = None
    level: ScoreLevel
    page_number: int | None = None
    metric: ScoreMetric
    value: float
    confidence_weighted: bool
    details: dict | None = None
    computed_at: datetime

    model_config = {"from_attributes": True}


# ── ScoreSummary ─────────────────────────────────────────────────────────────
class ScoreSummaryRead(BaseModel):
    """Schema for reading a score summary."""

    id: uuid.UUID
    run_id: uuid.UUID
    gt_version_id: uuid.UUID | None = None
    overall_score: float
    breakdown: dict | None = None
    computed_at: datetime

    model_config = {"from_attributes": True}
