"""Cross-run and cross-engine comparison endpoints.

Endpoints
---------
- ``GET /api/v1/comparison/runs?run_ids=X,Y,Z`` — Compare specific runs
  side-by-side with their scores.
- ``GET /api/v1/comparison/engines?engine_ids=X,Y&pdf_ids=A,B,C`` — Compare
  engines across PDFs.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from backend.database import get_db_session
from backend.evaluation._evaluators import evaluate_run
from backend.evaluation.bootstrap import bootstrap_ci
from backend.evaluation.scoring_service import (
    _build_gt_data,
    _build_run_data,
)
from backend.models.enums import RunStatus
from backend.models.run import OCRRun

# ── Router ────────────────────────────────────────────────────────────────────

comparison_router = APIRouter(prefix="/api/v1/comparison", tags=["comparison"])

# ── Dependencies ──────────────────────────────────────────────────────────────

SessionDep = Annotated[AsyncSession, Depends(get_db_session)]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _compute_run_scores(run_data: dict, gt_data: dict) -> dict:
    """Compute scores dict for a single completed run."""
    scores = evaluate_run(run_data, gt_data)
    per_page_cer = [p["cer"] for p in scores.get("per_page", [])]
    ci: dict | None = None
    if per_page_cer and len(per_page_cer) >= 2:
        try:
            ci_result = bootstrap_ci(per_page_cer, metric_name="cer")
            ci = {
                "cer_lower": ci_result["ci_lower"],
                "cer_upper": ci_result["ci_upper"],
                "ci_level": ci_result["ci_level"],
            }
        except ValueError:
            ci = None

    return {
        "cer": scores["cer"],
        "wer": scores["wer"],
        "char_f1": scores["char_f1"],
        "word_f1": scores["word_f1"],
        "pages": scores["num_pages"],
        "bootstrap_ci": ci,
    }


async def _run_to_comparison_entry(
    run: OCRRun,
    gt_version_id: uuid.UUID | None,
    db: AsyncSession,
) -> dict:
    """Convert a run into a comparison entry dict with scores."""
    if run.status != RunStatus.COMPLETED:
        return {
            "run_id": str(run.id),
            "pdf_id": str(run.pdf_id),
            "engine_slug": str(run.engine_id),
            "status": run.status.value,
            "scores": None,
            "message": "Run is not completed",
        }

    run_data = await _build_run_data(run.id, db)

    entry: dict = {
        "run_id": str(run.id),
        "pdf_id": str(run.pdf_id),
        "engine_slug": str(run.engine_id),
        "status": run.status.value,
        "scores": None,
    }

    if gt_version_id is not None:
        try:
            gt_data = await _build_gt_data(gt_version_id, db)
            entry["scores"] = _compute_run_scores(run_data, gt_data)
        except ValueError:
            entry["scores"] = None
            entry["message"] = "Ground truth version not found"

    return entry


async def _get_latest_completed_run(
    db: AsyncSession,
    pdf_id: uuid.UUID,
    engine_id: uuid.UUID,
) -> OCRRun | None:
    """Find the most recent completed run for a PDF and engine."""
    result = await db.execute(
        select(OCRRun)
        .where(
            OCRRun.pdf_id == pdf_id,
            OCRRun.engine_id == engine_id,
            OCRRun.status == RunStatus.COMPLETED,
        )
        .order_by(OCRRun.completed_at.desc())
        .limit(1),
    )
    return result.scalars().one_or_none()


# ── Endpoints ─────────────────────────────────────────────────────────────────


@comparison_router.get("/runs")
async def compare_runs(
    db: SessionDep,
    run_ids: str = Query(  # noqa: B008
        ...,
        description="Comma-separated run UUIDs to compare (e.g. 'id1,id2,id3')",
    ),
    gt_version_id: uuid.UUID | None = Query(  # noqa: B008
        default=None,
        description="Optional ground-truth version UUID for scoring",
    ),
) -> dict:
    """Compare specific runs side-by-side.

    Accepts up to 5 run IDs. Returns each run's metadata and scores
    (if computed) for side-by-side comparison.
    """
    # Parse run_ids
    parsed_ids: list[uuid.UUID] = []
    for raw in run_ids.split(","):
        piece = raw.strip()
        if not piece:
            continue
        try:
            parsed_ids.append(uuid.UUID(piece))
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid run UUID: '{piece}'",
            ) from exc

    if len(parsed_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 run IDs are required for comparison",
        )
    if len(parsed_ids) > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At most 5 run IDs are supported",
        )

    # Fetch runs
    result = await db.execute(
        select(OCRRun)
        .options(joinedload(OCRRun.engine))
        .where(OCRRun.id.in_(parsed_ids)),
    )
    runs = list(result.scalars().unique().all())

    if len(runs) != len(parsed_ids):
        found = {r.id for r in runs}
        missing = [str(rid) for rid in parsed_ids if rid not in found]
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Runs not found: {', '.join(missing)}",
        )

    # Build entries
    entries: list[dict] = []
    for run in runs:
        entry = await _run_to_comparison_entry(run, gt_version_id, db)
        entries.append(entry)

    return {
        "run_ids": [str(rid) for rid in parsed_ids],
        "gt_version_id": str(gt_version_id) if gt_version_id else None,
        "entries": entries,
    }


@comparison_router.get("/engines")
async def compare_engines_across_pdfs(
    db: SessionDep,
    engine_ids: str = Query(  # noqa: B008
        ...,
        description="Comma-separated engine UUIDs to compare (e.g. 'id1,id2')",
    ),
    pdf_ids: str = Query(  # noqa: B008
        ...,
        description="Comma-separated PDF UUIDs to compare across",
    ),
    gt_version_id: uuid.UUID | None = Query(  # noqa: B008
        default=None,
        description="Optional ground-truth version UUID for scoring",
    ),
) -> dict:
    """Compare engines across PDFs.

    For each engine×PDF pair, finds the most recent completed run and
    returns scores side-by-side.
    """
    # Parse engine_ids
    parsed_engine_ids: list[uuid.UUID] = []
    for raw in engine_ids.split(","):
        piece = raw.strip()
        if not piece:
            continue
        try:
            parsed_engine_ids.append(uuid.UUID(piece))
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid engine UUID: '{piece}'",
            ) from exc

    # Parse pdf_ids
    parsed_pdf_ids: list[uuid.UUID] = []
    for raw in pdf_ids.split(","):
        piece = raw.strip()
        if not piece:
            continue
        try:
            parsed_pdf_ids.append(uuid.UUID(piece))
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid PDF UUID: '{piece}'",
            ) from exc

    if not parsed_engine_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 1 engine ID is required",
        )
    if not parsed_pdf_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 1 PDF ID is required",
        )

    # Gather entries per engine
    engine_entries: list[dict] = []
    for engine_id in parsed_engine_ids:
        pdf_entries: list[dict] = []
        for pdf_id in parsed_pdf_ids:
            run = await _get_latest_completed_run(db, pdf_id, engine_id)
            if run is None:
                pdf_entries.append({
                    "pdf_id": str(pdf_id),
                    "run_id": None,
                    "scores": None,
                    "message": "No completed run found",
                })
            else:
                entry = await _run_to_comparison_entry(
                    run, gt_version_id, db
                )
                pdf_entries.append({
                    "pdf_id": str(pdf_id),
                    "run_id": str(run.id),
                    "scores": entry.get("scores"),
                    "message": entry.get("message"),
                })

        engine_entries.append({
            "engine_id": str(engine_id),
            "pdfs": pdf_entries,
        })

    return {
        "engine_ids": [str(eid) for eid in parsed_engine_ids],
        "pdf_ids": [str(pid) for pid in parsed_pdf_ids],
        "gt_version_id": str(gt_version_id) if gt_version_id else None,
        "engines": engine_entries,
    }
