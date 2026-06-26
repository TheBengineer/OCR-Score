"""Ground truth models — manually curated or consensus-built reference for scoring."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base
from backend.models.enums import GroundTruthSource

if TYPE_CHECKING:
    from backend.models.pdf import PDF
    from backend.models.score import Score, ScoreSummary


class GroundTruthVersion(Base):
    """A versioned ground truth for a PDF document.

    Supports soft-delete so old versions can be hidden without data loss.
    """

    __tablename__ = "ground_truth_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    pdf_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("pdfs.id", ondelete="CASCADE"),
        nullable=False,
    )
    version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )
    source: Mapped[GroundTruthSource] = mapped_column(
        Enum(GroundTruthSource, name="ground_truth_source", create_type=True, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=GroundTruthSource.MANUAL,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    created_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        default=None,
        comment="User or system that created this version",
    )
    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        comment="Soft-delete timestamp; NULL means active",
    )

    # -- relationships -------------------------------------------------------
    pdf: Mapped[PDF] = relationship("PDF", back_populates="ground_truth_versions")
    page_results: Mapped[list[GTPageResult]] = relationship(
        "GTPageResult",
        back_populates="gt_version",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    scores: Mapped[list[Score]] = relationship(
        "Score",
        back_populates="gt_version",
    )
    score_summaries: Mapped[list[ScoreSummary]] = relationship(
        "ScoreSummary",
        back_populates="gt_version",
    )

    # -- indexes -------------------------------------------------------------
    __table_args__ = (
        Index("ix_gt_versions_pdf_id", "pdf_id"),
        Index("ix_gt_versions_pdf_version", "pdf_id", "version_number", unique=True),
    )

    def __repr__(self) -> str:
        return (
            f"<GroundTruthVersion id={self.id!r} pdf_id={self.pdf_id!r} "
            f"v{self.version_number}>"
        )


class GTPageResult(Base):
    """Ground truth for a single page — same JSONB shape as PageResult."""

    __tablename__ = "gt_page_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    gt_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ground_truth_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="1-based page number",
    )
    width: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Page width in points (72 DPI)",
    )
    height: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Page height in points (72 DPI)",
    )
    data: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Same canonical JSONB schema as PageResult.data",
    )
    confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        default=1.0,
        comment="Confidence in this ground truth (1.0 = certain)",
    )

    # -- relationships -------------------------------------------------------
    gt_version: Mapped[GroundTruthVersion] = relationship(
        "GroundTruthVersion",
        back_populates="page_results",
    )

    # -- indexes -------------------------------------------------------------
    __table_args__ = (
        Index("ix_gt_page_results_gt_version_id", "gt_version_id"),
        Index(
            "ix_gt_page_results_version_page",
            "gt_version_id",
            "page_number",
            unique=True,
        ),
        Index(
            "ix_gt_page_results_data_gin",
            "data",
            postgresql_using="gin",
            postgresql_ops={"data": "jsonb_path_ops"},
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<GTPageResult id={self.id!r} gt_version_id={self.gt_version_id!r} "
            f"page={self.page_number}>"
        )
