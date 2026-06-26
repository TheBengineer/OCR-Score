"""Ground truth management router — create, read, update, version, and delete.

Endpoints
---------
- ``POST   /api/v1/ground-truth`` — Create a new GT version
- ``GET    /api/v1/ground-truth`` — List GT versions (filterable by pdf_id)
- ``GET    /api/v1/ground-truth/{id}`` — Get full GT data with pages
- ``PUT    /api/v1/ground-truth/{id}/pages/{page}`` — Edit a single page's GT
- ``PUT    /api/v1/ground-truth/{id}/pages/{page}/words/{word_idx}`` — Edit a word
- ``DELETE /api/v1/ground-truth/{id}`` — Soft-delete a GT version
- ``POST   /api/v1/ground-truth/{id}/promote`` — Promote a version to "current"
- ``GET    /api/v1/ground-truth/current/{pdf_id}`` — Get current GT for a PDF
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db_session
from backend.ground_truth_manager import GroundTruthManager, GroundTruthManagerError
from backend.models.enums import GroundTruthSource
from backend.schemas.ground_truth import (
    GroundTruthVersionRead,
    GTPageResultRead,
)

# ── Router ────────────────────────────────────────────────────────────────────

gt_router = APIRouter(prefix="/api/v1/ground-truth", tags=["ground-truth"])

# ── Dependencies ──────────────────────────────────────────────────────────────

SessionDep = Annotated[AsyncSession, Depends(get_db_session)]


def get_gt_manager(db: SessionDep) -> GroundTruthManager:
    """FastAPI dependency for a fresh ``GroundTruthManager`` per request."""
    return GroundTruthManager(db=db)


ManagerDep = Annotated[GroundTruthManager, Depends(get_gt_manager)]

# ── Request / response schemas ────────────────────────────────────────────────


class GTCreateRequest(BaseModel):
    """Request body for ``POST /api/v1/ground-truth``."""

    pdf_id: uuid.UUID
    source: GroundTruthSource = GroundTruthSource.MANUAL
    engine_ids: list[uuid.UUID] | None = Field(
        default=None,
        description="OCR run UUIDs for consensus-based GT generation",
    )
    notes: str | None = None
    created_by: str | None = None


class GTCreateResponse(BaseModel):
    """Response body after creating a ground truth version."""

    id: uuid.UUID
    pdf_id: uuid.UUID
    version_number: int
    source: GroundTruthSource
    created_at: str
    message: str | None = None


class GTListResponse(BaseModel):
    """Response body for listing GT versions."""

    items: list[GroundTruthVersionRead]


class GTDetailResponse(BaseModel):
    """Full GT version detail with page results."""

    version: GroundTruthVersionRead
    pages: list[GTPageResultRead]


class GTPageUpdateRequest(BaseModel):
    """Request body for ``PUT .../pages/{page}``."""

    data: dict = Field(..., description="Canonical JSONB page hierarchy")


class GTWordUpdateRequest(BaseModel):
    """Request body for ``PUT .../pages/{page}/words/{word_idx}``."""

    text: str = Field(..., description="Corrected text for the word")


# ── Endpoints ─────────────────────────────────────────────────────────────────


@gt_router.post("", status_code=status.HTTP_201_CREATED)
async def create_gt_version(
    body: GTCreateRequest,
    manager: ManagerDep,
) -> GTCreateResponse:
    """Create a new ground truth version.

    When *source* is ``consensus`` and *engine_ids* are provided, the system
    runs Consensus Entropy across the engine outputs to auto-generate the GT.
    """
    try:
        gt = await manager.create_gt_version(
            pdf_id=body.pdf_id,
            source=body.source,
            engine_ids=body.engine_ids,
            notes=body.notes,
            created_by=body.created_by,
        )
    except GroundTruthManagerError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    page_count = len(gt.page_results) if gt.page_results else 0
    return GTCreateResponse(
        id=gt.id,
        pdf_id=gt.pdf_id,
        version_number=gt.version_number,
        source=gt.source,
        created_at=gt.created_at.isoformat() if gt.created_at else "",
        message=(
            f"Created v{gt.version_number} with {page_count} page(s)"
            if page_count > 0
            else f"Created empty v{gt.version_number} — add pages manually"
        ),
    )


@gt_router.get("")
async def list_gt_versions(
    manager: ManagerDep,
    pdf_id: uuid.UUID | None = Query(default=None, description="Filter by PDF UUID"),  # noqa: B008
) -> GTListResponse:
    """List ground truth versions, optionally filtered by PDF.

    Soft-deleted versions are excluded. Ordered by version number descending.
    """
    versions = await manager.list_gt_versions(pdf_id=pdf_id)
    return GTListResponse(
        items=[GroundTruthVersionRead.model_validate(v) for v in versions],
    )


@gt_router.get("/{gt_id}")
async def get_gt_version(
    gt_id: uuid.UUID,
    manager: ManagerDep,
) -> GTDetailResponse:
    """Retrieve a full ground truth version with all page results."""
    gt = await manager.get_gt_version(gt_id)
    if gt is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ground truth version not found",
        )

    pages: list[GTPageResultRead] = []
    if gt.page_results:
        pages = [
            GTPageResultRead.model_validate(pr)
            for pr in sorted(gt.page_results, key=lambda p: p.page_number)
        ]

    return GTDetailResponse(
        version=GroundTruthVersionRead.model_validate(gt),
        pages=pages,
    )


@gt_router.put("/{gt_id}/pages/{page_num}")
async def update_gt_page(
    gt_id: uuid.UUID,
    page_num: int,
    body: GTPageUpdateRequest,
    manager: ManagerDep,
) -> GTPageResultRead:
    """Edit a single page's ground truth data.

    Creates a **new version** with the updated page; the existing version
    is preserved unchanged for audit.
    """
    try:
        page_result = await manager.update_gt_page(
            gt_id=gt_id,
            page_num=page_num,
            page_data=body.data,
        )
    except GroundTruthManagerError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return GTPageResultRead.model_validate(page_result)


@gt_router.put("/{gt_id}/pages/{page_num}/words/{word_idx}")
async def update_gt_word(
    gt_id: uuid.UUID,
    page_num: int,
    word_idx: int,
    body: GTWordUpdateRequest,
    manager: ManagerDep,
) -> GTPageResultRead:
    """Correct a single word in ground truth.

    Navigates the canonical JSONB hierarchy (blocks → lines → words) to
    find and correct the word at *word_idx*. Creates a **new version**.
    """
    try:
        page_result = await manager.update_gt_word(
            gt_id=gt_id,
            page_num=page_num,
            word_idx=word_idx,
            new_text=body.text,
        )
    except GroundTruthManagerError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return GTPageResultRead.model_validate(page_result)


@gt_router.delete("/{gt_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_gt_version(
    gt_id: uuid.UUID,
    manager: ManagerDep,
) -> None:
    """Soft-delete a ground truth version.

    Sets ``deleted_at`` — the data is preserved but excluded from normal queries.
    """
    try:
        await manager.soft_delete_gt(gt_id)
    except GroundTruthManagerError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@gt_router.post("/{gt_id}/promote", status_code=status.HTTP_200_OK)
async def promote_gt_version(
    gt_id: uuid.UUID,
    manager: ManagerDep,
) -> dict[str, object]:
    """Promote a ground truth version to be the "current" one for its PDF.

    After promotion, :meth:`get_current_gt` will return this version.
    """
    try:
        await manager.promote_gt_version(gt_id)
    except GroundTruthManagerError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return {"id": str(gt_id), "message": "Promoted to current version"}


@gt_router.get("/current/{pdf_id}")
async def get_current_gt(
    pdf_id: uuid.UUID,
    manager: ManagerDep,
) -> GTDetailResponse | dict[str, object]:
    """Get the current (most recently promoted) ground truth for a PDF."""
    gt = await manager.get_current_gt(pdf_id)
    if gt is None:
        return {"pdf_id": str(pdf_id), "message": "No ground truth versions found"}

    pages: list[GTPageResultRead] = []
    if gt.page_results:
        pages = [
            GTPageResultRead.model_validate(pr)
            for pr in sorted(gt.page_results, key=lambda p: p.page_number)
        ]

    return GTDetailResponse(
        version=GroundTruthVersionRead.model_validate(gt),
        pages=pages,
    )
