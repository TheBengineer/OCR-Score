"""Batch processing endpoints — create, monitor, and poll batch OCR jobs.

Endpoints
---------
- ``POST /api/v1/batch`` — Start batch processing (returns 202).
- ``GET  /api/v1/batch/{id}`` — Get batch metadata and status.
- ``GET  /api/v1/batch/{id}/progress`` — Per-PDF progress summary.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.batch_processor import BatchProcessor, BatchProcessorError
from backend.database import get_db_session
from backend.storage import ContentAddressableStorage

# ── Router ────────────────────────────────────────────────────────────────────

batch_router = APIRouter(prefix="/api/v1/batch", tags=["batch"])

# ── Dependencies ──────────────────────────────────────────────────────────────

SessionDep = Annotated[AsyncSession, Depends(get_db_session)]


def get_batch_processor(
    db: SessionDep,
) -> BatchProcessor:
    """FastAPI dependency for a fresh ``BatchProcessor`` per request."""
    from pathlib import Path  # noqa: PLC0415

    from backend.settings import settings  # noqa: PLC0415

    storage = ContentAddressableStorage(Path(settings.storage_path))
    return BatchProcessor(db=db, storage=storage)


BatchProcessorDep = Annotated[BatchProcessor, Depends(get_batch_processor)]

# ── Request / response schemas ────────────────────────────────────────────────


class BatchCreateRequest(BaseModel):
    """Request body for ``POST /api/v1/batch``."""

    pdf_ids: list[uuid.UUID] = Field(
        ...,
        min_length=1,
        description="UUIDs of the PDF documents to process",
    )
    engine_slugs: list[str] = Field(
        ...,
        min_length=1,
        description="Engine slugs to run (e.g. ['mock', 'tesseract'])",
    )
    config: dict | None = Field(
        default=None,
        description="Shared engine configuration",
    )


class BatchCreateResponse(BaseModel):
    """Response body returned after creating a batch."""

    id: uuid.UUID
    status: str
    total_items: int
    message: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────


@batch_router.post("", status_code=status.HTTP_202_ACCEPTED)
async def create_batch(
    body: BatchCreateRequest,
    processor: BatchProcessorDep,
) -> dict:
    """Start batch processing.

    Creates a batch record and launches background processing for all
    PDF×engine combinations sequentially.
    """
    try:
        batch = await processor.create_batch(
            pdf_ids=body.pdf_ids,
            engine_slugs=body.engine_slugs,
            config=body.config,
        )
    except BatchProcessorError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    # Launch background processing
    import asyncio  # noqa: PLC0415

    asyncio.create_task(processor.process_batch(batch.id))

    return {
        "id": str(batch.id),
        "status": batch.status,
        "total_items": len(batch.items),
        "message": f"Batch {batch.id} created and processing started",
    }


@batch_router.get("/{batch_id}")
async def get_batch(
    batch_id: uuid.UUID,
    processor: BatchProcessorDep,
) -> dict:
    """Get batch metadata and overall status."""
    batch = processor.get_batch(batch_id)
    if batch is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="batch not found",
        )

    total = len(batch.items)
    completed = sum(1 for item in batch.items if item.status == "completed")
    failed = sum(1 for item in batch.items if item.status == "failed")

    return {
        "id": str(batch.id),
        "pdf_ids": [str(pid) for pid in batch.pdf_ids],
        "engine_slugs": batch.engine_slugs,
        "config": batch.config,
        "status": batch.status,
        "created_at": batch.created_at.isoformat(),
        "total_items": total,
        "completed": completed,
        "failed": failed,
        "error_message": batch.error_message,
    }


@batch_router.get("/{batch_id}/progress")
async def get_batch_progress(
    batch_id: uuid.UUID,
    processor: BatchProcessorDep,
) -> dict:
    """Get per-PDF progress for a batch.

    Returns a progress summary with ``total``, ``completed``, ``failed``,
    ``percent``, and a detailed ``items`` array showing each PDF×engine
    combination's status.
    """
    progress = processor.get_batch_progress(batch_id)
    if progress is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="batch not found",
        )
    return progress
