"""PageResult model — per-page OCR output stored as a rich JSONB hierarchy."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Index, Integer
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base

if TYPE_CHECKING:
    from backend.models.run import OCRRun


class PageResult(Base):
    """OCR result for a single page of a PDF.

    The full block / line / word / character hierarchy is stored as JSONB in
    ``data``, using a canonical schema (72 DPI, top-left origin coordinates).
    This avoids 5-table JOINs when loading the viewer.
    """

    __tablename__ = "page_results"

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
        comment=(
            "Full page hierarchy in canonical JSONB: "
            "{blocks: [{type, bbox, confidence, order, lines: [{text, bbox, confidence, order, "
            "words: [{text, bbox, confidence, order, chars: [{char, bbox, confidence, order}]}]}]}], "
            "tables: [{bbox, num_rows, num_cols, caption, "
            "cells: [{row, col, row_span, col_span, text, bbox, confidence}]}]"
        ),
    )
    confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Aggregate per-page confidence (0-1)",
    )

    # -- relationships -------------------------------------------------------
    run: Mapped[OCRRun] = relationship("OCRRun", back_populates="page_results")

    # -- indexes -------------------------------------------------------------
    __table_args__ = (
        Index("ix_page_results_run_id", "run_id"),
        Index("ix_page_results_run_page", "run_id", "page_number", unique=True),
        Index(
            "ix_page_results_data_gin",
            "data",
            postgresql_using="gin",
            postgresql_ops={"data": "jsonb_path_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<PageResult id={self.id!r} run_id={self.run_id!r} page={self.page_number}>"
