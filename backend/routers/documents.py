"""PDF document management router — upload, retrieve, list, and soft-delete PDFs.

Content-addressed storage ensures that identical files are stored exactly
once, and soft-delete provides a safety net for accidental removals.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db_session
from backend.models.enums import PDFStatus
from backend.models.pdf import PDF
from backend.schemas.pdf import PDFRead
from backend.settings import settings
from backend.storage import ContentAddressableStorage

# ── Router ────────────────────────────────────────────────────────────────────

documents_router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

# ── Dependencies ──────────────────────────────────────────────────────────────

SessionDep = Annotated[AsyncSession, Depends(get_db_session)]


def get_storage() -> ContentAddressableStorage:
    """Provide the singleton content-addressable storage instance."""
    return ContentAddressableStorage(Path(settings.storage_path))


StorageDep = Annotated[ContentAddressableStorage, Depends(get_storage)]

# ── Helpers ────────────────────────────────────────────────────────────────────


def _compute_hashes(file_bytes: bytes) -> tuple[str, str]:
    """Compute SHA-256 and MD5 hex digests of file contents.

    Args:
        file_bytes: The raw file content.

    Returns:
        A tuple of (sha256_hex, md5_hex).
    """
    sha256 = hashlib.sha256(file_bytes).hexdigest()
    md5 = hashlib.md5(file_bytes).hexdigest()
    return sha256, md5


_PDF_MAGIC = b"%PDF"


def _validate_pdf(file_bytes: bytes) -> None:
    """Validate that the byte content starts with PDF magic bytes.

    Args:
        file_bytes: The raw file content to validate.

    Raises:
        HTTPException: If the content does not start with ``%PDF``.
    """
    if not file_bytes.startswith(_PDF_MAGIC):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded file is not a valid PDF (missing %PDF header)",
        )


async def _pdf_by_sha256(db: AsyncSession, sha256: str) -> PDF | None:
    """Look up an active (non-deleted) PDF record by SHA-256 hash.

    Args:
        db: The async database session.
        sha256: The SHA-256 hex digest to search for.

    Returns:
        The matching PDF record, or ``None``.
    """
    result = await db.execute(
        select(PDF).where(
            PDF.sha256_hash == sha256,
            PDF.deleted_at.is_(None),
        ),
    )
    return result.scalars().one_or_none()


async def _pdf_by_id(db: AsyncSession, pdf_id: uuid.UUID) -> PDF | None:
    """Look up an active (non-deleted) PDF record by its UUID.

    Args:
        db: The async database session.
        pdf_id: The UUID of the PDF record.

    Returns:
        The matching PDF record, or ``None``.
    """
    result = await db.execute(
        select(PDF).where(
            PDF.id == pdf_id,
            PDF.deleted_at.is_(None),
        ),
    )
    return result.scalars().one_or_none()


# ── Endpoints ─────────────────────────────────────────────────────────────────


@documents_router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_pdf(
    file: UploadFile,
    db: SessionDep,
    storage: StorageDep,
) -> JSONResponse:
    """Upload a PDF document.

    The file content is validated for PDF magic bytes, content-addressed by
    SHA-256, and stored to disk. If the same content already exists the
    existing document ID is returned with a 200 status instead.
    """
    file_bytes = await file.read()

    # -- validate -----------------------------------------------------------
    _validate_pdf(file_bytes)

    # -- hash ---------------------------------------------------------------
    sha256, md5 = _compute_hashes(file_bytes)

    # -- check for existing document with same content ----------------------
    existing = await _pdf_by_sha256(db, sha256)
    if existing is not None:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "id": str(existing.id),
                "status": existing.status.value,
                "filename": existing.filename,
                "message": "document already exists",
            },
        )

    # -- create DB record ---------------------------------------------------
    pdf_record = PDF(
        filename=f"{sha256}.pdf",
        original_filename=file.filename or f"{sha256}.pdf",
        file_size_bytes=len(file_bytes),
        md5_hash=md5,
        sha256_hash=sha256,
        mime_type="application/pdf",
        status=PDFStatus.UPLOADED,
    )
    db.add(pdf_record)
    await db.commit()
    await db.refresh(pdf_record)

    # -- store file to disk -------------------------------------------------
    await storage.store(file_bytes, sha256)

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "id": str(pdf_record.id),
            "status": pdf_record.status.value,
            "filename": pdf_record.filename,
        },
    )


@documents_router.get("/{pdf_id}", response_model=PDFRead)
async def get_document(pdf_id: uuid.UUID, db: SessionDep) -> PDF:
    """Retrieve metadata for a single document by its UUID."""
    pdf = await _pdf_by_id(db, pdf_id)
    if pdf is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="document not found",
        )
    return pdf


@documents_router.get("")
async def list_documents(
    db: SessionDep,
    cursor: uuid.UUID | None = Query(default=None, description="UUID of the last item from the previous page"),  # noqa: B008
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of items to return"),  # noqa: B008
) -> dict[str, object]:
    """List active documents with cursor-based pagination.

    Returns a page of documents ordered by UUID, along with a ``next_cursor``
    value that can be passed as the ``cursor`` parameter to fetch the next
    page. When ``next_cursor`` is ``null`` there are no more results.
    """
    query = select(PDF).where(PDF.deleted_at.is_(None))
    if cursor is not None:
        query = query.where(PDF.id > cursor)
    query = query.order_by(PDF.id).limit(limit + 1)

    result = await db.execute(query)
    items: list[PDF] = list(result.scalars().unique().all())

    next_cursor: str | None = None
    if len(items) > limit:
        next_cursor = str(items[-1].id)
        items = items[:limit]

    return {
        "items": [PDFRead.model_validate(p) for p in items],
        "next_cursor": next_cursor,
    }


@documents_router.delete("/{pdf_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(pdf_id: uuid.UUID, db: SessionDep) -> None:
    """Soft-delete a document by setting its ``deleted_at`` timestamp."""
    pdf = await _pdf_by_id(db, pdf_id)
    if pdf is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="document not found",
        )
    pdf.deleted_at = datetime.now(UTC)
    pdf.status = PDFStatus.DELETED
    await db.commit()
