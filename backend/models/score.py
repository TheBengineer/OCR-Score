"""Score and ScoreSummary models — granular evaluation results."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Index, Integer, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base
from backend.models.enums import ScoreLevel, ScoreMetric

if TYPE_CHECKING:
    from backend.models.ground_truth import GroundTruthVersion
    from backend.models.run import OCRRun


class Score(Base):
    """A single score measurement at a specific level and metric.

    Scores can be computed at the character, word, line, block, table, page,
    or document level.
    """

    __tablename__ = "scores"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ocr_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    gt_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ground_truth_versions.id", ondelete="SET NULL"),
        nullable=True,
        default=None,
        comment="NULL if computed against multiple engines (consensus) or no GT",
    )
    level: Mapped[ScoreLevel] = mapped_column(
        Enum(ScoreLevel, name="score_level", create_type=True),
        nullable=False,
    )
    page_number: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        default=None,
        comment="Page number when level is page-specific; NULL for document-level",
    )
    metric: Mapped[ScoreMetric] = mapped_column(
        Enum(ScoreMetric, name="score_metric", create_type=True),
        nullable=False,
    )
    value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Score value (typically 0-1, lower CER/WER is better)",
    )
    confidence_weighted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether OCR confidence was factored into this score",
    )
    details: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
        comment="Optional breakdown (e.g. substitution/insertion/deletion counts)",
    )
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # -- relationships -------------------------------------------------------
    run: Mapped[OCRRun] = relationship("OCRRun", back_populates="scores")
    gt_version: Mapped[GroundTruthVersion | None] = relationship(
        "GroundTruthVersion",
        back_populates="scores",
    )

    # -- indexes -------------------------------------------------------------
    __table_args__ = (
        Index("ix_scores_run_id", "run_id"),
        Index("ix_scores_gt_version_id", "gt_version_id"),
        Index("ix_scores_level_metric", "level", "metric"),
    )

    def __repr__(self) -> str:
        return (
            f"<Score id={self.id!r} run_id={self.run_id!r} "
            f"level={self.level!r} metric={self.metric!r} value={self.value:.4f}>"
        )


class ScoreSummary(Base):
    """Aggregate score summary for a run against a ground truth version.

    Provides a quick-lookup row for dashboard and ranking views.
    """

    __tablename__ = "score_summaries"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ocr_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    gt_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ground_truth_versions.id", ondelete="SET NULL"),
        nullable=True,
        default=None,
    )
    overall_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Composite overall score (0-1, higher is better)",
    )
    breakdown: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
        comment=(
            "Full breakdown by level and metric, e.g. "
            '{"character": {"cer": 0.05, "accuracy": 0.95}, '
            '"word": {"wer": 0.08, "f1": 0.93}}'
        ),
    )
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # -- relationships -------------------------------------------------------
    run: Mapped[OCRRun] = relationship("OCRRun", back_populates="score_summaries")
    gt_version: Mapped[GroundTruthVersion | None] = relationship(
        "GroundTruthVersion",
        back_populates="score_summaries",
    )

    # -- indexes -------------------------------------------------------------
    __table_args__ = (
        Index("ix_score_summaries_run_id", "run_id"),
        Index("ix_score_summaries_gt_version_id", "gt_version_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<ScoreSummary id={self.id!r} run_id={self.run_id!r} "
            f"overall={self.overall_score:.4f}>"
        )
