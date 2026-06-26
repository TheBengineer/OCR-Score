"""Pydantic schemas for the PDF model."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from backend.models.enums import PDFStatus


# ── Create ───────────────────────────────────────────────────────────────────
class PDFCreate(BaseModel):
    """Schema for uploading / creating a new PDF record."""

    filename: str = Field(
        ...,
        max_length=255,
        description="Sanitized storage filename",
    )
    original_filename: str = Field(
        ...,
        max_length=1024,
        description="Original uploaded filename",
    )
    file_size_bytes: int = Field(..., gt=0, description="File size in bytes")
    page_count: int = Field(..., ge=0, description="Number of pages")
    md5_hash: str = Field(
        ...,
        min_length=32,
        max_length=32,
        description="MD5 hex digest",
    )
    sha256_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hex digest",
    )
    mime_type: str = Field(
        default="application/pdf",
        max_length=127,
        description="MIME type",
    )
    status: PDFStatus = Field(default=PDFStatus.UPLOADING)


# ── Read ─────────────────────────────────────────────────────────────────────
class PDFRead(BaseModel):
    """Schema for reading a PDF record from the API."""

    id: uuid.UUID
    filename: str
    original_filename: str
    file_size_bytes: int
    page_count: int
    md5_hash: str
    sha256_hash: str
    mime_type: str
    status: PDFStatus
    upload_timestamp: datetime
    deleted_at: datetime | None = None

    model_config = {"from_attributes": True}


# ── Update ───────────────────────────────────────────────────────────────────
class PDFUpdate(BaseModel):
    """Schema for updating a PDF record (partial)."""

    status: PDFStatus | None = None
    page_count: int | None = Field(default=None, ge=0)
    filename: str | None = Field(default=None, max_length=255)
    original_filename: str | None = Field(default=None, max_length=1024)
