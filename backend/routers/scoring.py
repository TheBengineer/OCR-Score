"""Scoring router — per-run scores, per-page breakdowns, cross-engine comparison.

Endpoints
---------
- ``GET /api/v1/runs/{run_id}/scores`` — Aggregate scores for a run.
- ``GET /api/v1/runs/{run_id}/scores/by-page`` — Per-page score breakdown.
- ``GET /api/v1/documents/{pdf_id}/runs/comparison`` — Cross-engine comparison.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db_session
from backend.evaluation.scoring_service import (
    compare_engines,
    compute_page_scores,
    compute_run_scores,
)

# ── Router ────────────────────────────────────────────────────────────────────

scoring_router = APIRouter(tags=["scoring"])

# ── Dependencies ──────────────────────────────────────────────────────────────

SessionDep = Annotated[AsyncSession, Depends(get_db_session)]


# ── Endpoints ─────────────────────────────────────────────────────────────────


@scoring_router.get("/api/v1/runs/{run_id}/scores")
async def get_run_scores(
    run_id: uuid.UUID,
    db: SessionDep,
    gt_version_id: uuid.UUID = Query(  # noqa: B008
        ...,
        description="Ground-truth version UUID to score against",
    ),
) -> dict:
    """Aggregate evaluation scores for a completed OCR run."""
    try:
        result = await compute_run_scores(
            run_id=run_id,
            gt_version_id=gt_version_id,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    if result.get("overall") is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=result.get("message", "run not completed"),
        )

    return result


@scoring_router.get("/api/v1/runs/{run_id}/scores/by-page")
async def get_run_scores_by_page(
    run_id: uuid.UUID,
    db: SessionDep,
    gt_version_id: uuid.UUID = Query(  # noqa: B008
        ...,
        description="Ground-truth version UUID to score against",
    ),
) -> dict:
    """Compute per-page score breakdown for a completed OCR run.

    Returns a list of per-page CER, WER, char/word F1.
    """
    # First get the overall result to know how many pages
    try:
        overall = await compute_run_scores(
            run_id=run_id,
            gt_version_id=gt_version_id,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    if overall.get("overall") is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=overall.get("message", "run not completed"),
        )

    # Compute per-page scores
    pages: list[dict] = []
    for page_num in range(1, overall["pages"] + 1):
        page_result = await compute_page_scores(
            run_id=run_id,
            page_number=page_num,
            gt_version_id=gt_version_id,
            db=db,
        )
        if page_result.get("cer") is not None:
            pages.append(page_result)

    return {
        "run_id": str(run_id),
        "gt_version_id": str(gt_version_id),
        "pages": pages,
    }


@scoring_router.get("/api/v1/documents/{pdf_id}/runs/comparison")
async def get_engine_comparison(
    pdf_id: uuid.UUID,
    db: SessionDep,
    engine_ids: str = Query(  # noqa: B008
        ...,
        description="Comma-separated list of engine UUIDs to compare",
    ),
    gt_version_id: uuid.UUID = Query(  # noqa: B008
        ...,
        description="Ground-truth version UUID to score against",
    ),
) -> dict:
    """Cross-engine comparison for a document.

    Scores each engine's most recent completed run against the same ground
    truth version.
    """
    try:
        engine_uuid_list = [uuid.UUID(e.strip()) for e in engine_ids.split(",")]
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid engine_ids format: {exc}",
        ) from exc

    try:
        result = await compare_engines(
            pdf_id=pdf_id,
            engine_ids=engine_uuid_list,
            gt_version_id=gt_version_id,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return result
