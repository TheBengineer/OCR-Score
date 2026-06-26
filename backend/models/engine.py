"""OCREngine model — registered OCR engines (Tesseract, Document AI, Textract, etc.)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base

if TYPE_CHECKING:
    from backend.models.run import OCRRun


class OCREngine(Base):
    """A registered OCR engine plugin.

    Each engine is identified by its ``slug`` (e.g. ``tesseract``, ``gcp-document-ai``)
    and carries an optional JSON ``config_schema`` that describes valid engine-specific
    configuration parameters.
    """

    __tablename__ = "ocr_engines"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    slug: Mapped[str] = mapped_column(
        String(127),
        unique=True,
        nullable=False,
        comment="Unique machine-readable identifier (e.g. 'tesseract')",
    )
    display_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Human-readable engine name",
    )
    version: Mapped[str] = mapped_column(
        String(63),
        nullable=False,
        default="0.0.0",
        comment="Engine plugin version",
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )
    config_schema: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
        comment="JSON Schema describing valid engine_config values",
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # -- relationships -------------------------------------------------------
    runs: Mapped[list[OCRRun]] = relationship(
        "OCRRun",
        back_populates="engine",
    )

    def __repr__(self) -> str:
        return f"<OCREngine id={self.id!r} slug={self.slug!r}>"
