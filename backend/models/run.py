"""OCRRun model — an immutable record of an OCR processing run."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base
from backend.models.enums import RunStatus

if TYPE_CHECKING:
    from backend.models.engine import OCREngine
    from backend.models.page_result import PageResult
    from backend.models.pdf import PDF
    from backend.models.score import Score, ScoreSummary


class OCRRun(Base):
    """An individual OCR processing run.

    Every run is immutable: once created, ``engine_config``, ``engine_version``,
    and the resulting page results are not modified. The ``run_hash`` ensures
    idempotency — identical configurations against the same PDF produce the same hash.
    """

    __tablename__ = "ocr_runs"

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
    engine_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ocr_engines.id", ondelete="RESTRICT"),
        nullable=False,
    )
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, name="run_status", create_type=True),
        nullable=False,
        default=RunStatus.PENDING,
    )
    engine_config: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
        comment="Engine-specific configuration used for this run",
    )
    engine_version: Mapped[str | None] = mapped_column(
        String(63),
        nullable=True,
        default=None,
        comment="Engine version at time of run",
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
    )
    raw_output_uri: Mapped[str | None] = mapped_column(
        String(2048),
        nullable=True,
        default=None,
        comment="URI to the engine's raw (pre-normalization) output",
    )
    run_hash: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        comment="SHA-256 of (pdf_id + engine_id + engine_config) for dedup",
    )
    environment: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
        comment="Runtime environment metadata (OS, package versions, etc.)",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # -- relationships -------------------------------------------------------
    pdf: Mapped[PDF] = relationship("PDF", back_populates="runs")
    engine: Mapped[OCREngine] = relationship("OCREngine", back_populates="runs")
    page_results: Mapped[list[PageResult]] = relationship(
        "PageResult",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    scores: Mapped[list[Score]] = relationship(
        "Score",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    score_summaries: Mapped[list[ScoreSummary]] = relationship(
        "ScoreSummary",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    # -- indexes -------------------------------------------------------------
    __table_args__ = (
        Index("ix_ocr_runs_pdf_id", "pdf_id"),
        Index("ix_ocr_runs_engine_id", "engine_id"),
        Index("ix_ocr_runs_status", "status"),
        Index("ix_ocr_runs_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<OCRRun id={self.id!r} pdf_id={self.pdf_id!r} "
            f"engine_id={self.engine_id!r} status={self.status!r}>"
        )
