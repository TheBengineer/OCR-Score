"""PDF document model — uploaded PDFs with content-addressed hashes and soft delete."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, Enum, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base
from backend.models.enums import PDFStatus

if TYPE_CHECKING:
    from backend.models.ground_truth import GroundTruthVersion
    from backend.models.run import OCRRun


class PDF(Base):
    """An uploaded PDF document.

    Content-addressed via md5_hash and sha256_hash for reproducibility.
    Supports soft-delete via ``deleted_at``.
    """

    __tablename__ = "pdfs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Storage filename (sanitized, unique)",
    )
    original_filename: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        comment="Original uploaded filename",
    )
    file_size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    page_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    md5_hash: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
        comment="MD5 hex digest of file contents",
    )
    sha256_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="SHA-256 hex digest of file contents",
    )
    mime_type: Mapped[str] = mapped_column(
        String(127),
        nullable=False,
        default="application/pdf",
    )
    status: Mapped[PDFStatus] = mapped_column(
        Enum(PDFStatus, name="pdf_status", create_type=True),
        nullable=False,
        default=PDFStatus.UPLOADING,
    )
    upload_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        comment="Soft-delete timestamp; NULL means active",
    )

    # -- relationships -------------------------------------------------------
    runs: Mapped[list[OCRRun]] = relationship(
        "OCRRun",
        back_populates="pdf",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    ground_truth_versions: Mapped[list[GroundTruthVersion]] = relationship(
        "GroundTruthVersion",
        back_populates="pdf",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    # -- indexes -------------------------------------------------------------
    __table_args__ = (
        Index("ix_pdfs_status", "status"),
        Index("ix_pdfs_upload_timestamp", "upload_timestamp"),
        Index(
            "ix_pdfs_md5_sha256",
            "md5_hash",
            "sha256_hash",
            postgresql_where=func.deleted_at.is_(None),
        ),
    )

    def __repr__(self) -> str:
        return f"<PDF id={self.id!r} filename={self.filename!r} status={self.status!r}>"
