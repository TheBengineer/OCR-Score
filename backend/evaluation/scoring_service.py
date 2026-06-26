"""Scoring service — orchestrates score computation from DB data.

Provides three main operations:
- ``compute_run_scores`` — Aggregate scores for a completed run.
- ``compute_page_scores`` — Per-page scores for a single page in a run.
- ``compare_engines`` — Cross-engine comparison for a document.

All functions load data from the database, delegate to the evaluation
pipeline (:func:`evaluate_run` / :func:`evaluate_page`), and optionally
attach bootstrap confidence intervals.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.evaluation import bootstrap_ci, evaluate_run
from backend.evaluation._evaluators import _extract_words, evaluate_page
from backend.models.enums import RunStatus
from backend.models.ground_truth import GTPageResult
from backend.models.page_result import PageResult
from backend.models.run import OCRRun

# ── Public API ─────────────────────────────────────────────────────────────


async def compute_run_scores(
    run_id: uuid.UUID,
    gt_version_id: uuid.UUID,
    db: AsyncSession,
) -> dict[str, Any]:
    """Compute aggregate scores for a completed run against a GT version.

    Returns
    -------
    dict
        ``{
            "run_id": ...,
            "gt_version_id": ...,
            "overall": {"cer": ..., "wer": ..., "char_f1": ..., "word_f1": ...},
            "bootstrap_ci": {"cer_lower": ..., "cer_upper": ...},
            "pages": ...,
            "evaluated_pages": ...,
        }``
    """
    run = await _load_run(run_id, db)
    if run.status != RunStatus.COMPLETED:
        return {
            "run_id": str(run_id),
            "gt_version_id": str(gt_version_id),
            "overall": None,
            "bootstrap_ci": None,
            "pages": 0,
            "evaluated_pages": 0,
            "message": f"Run status is '{run.status.value}', not 'completed'",
        }

    run_data = await _build_run_data(run_id, db)
    gt_data = await _build_gt_data(gt_version_id, db)

    result = evaluate_run(run_data, gt_data)

    per_page_cer = [p["cer"] for p in result["per_page"]]
    ci: dict[str, Any] | None = None
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
        "run_id": str(run_id),
        "gt_version_id": str(gt_version_id),
        "overall": {
            "cer": result["cer"],
            "wer": result["wer"],
            "char_f1": result["char_f1"],
            "word_f1": result["word_f1"],
        },
        "bootstrap_ci": ci,
        "pages": result["num_pages"],
        "evaluated_pages": result["num_pages"],
    }


async def compute_page_scores(
    run_id: uuid.UUID,
    page_number: int,
    gt_version_id: uuid.UUID,
    db: AsyncSession,
) -> dict[str, Any]:
    """Compute scores for a single page within a run.

    Returns
    -------
    dict
        ``{"page": ..., "cer": ..., "wer": ..., "char_f1": ..., "word_f1": ...}``
    """
    run = await _load_run(run_id, db)
    if run.status != RunStatus.COMPLETED:
        return {
            "page": page_number,
            "cer": None,
            "wer": None,
            "char_f1": None,
            "word_f1": None,
            "message": f"Run status is '{run.status.value}', not 'completed'",
        }

    # Load the specific page result
    result = await db.execute(
        select(PageResult).where(
            PageResult.run_id == run_id,
            PageResult.page_number == page_number,
        ),
    )
    page_result = result.scalars().one_or_none()
    if page_result is None:
        return {
            "page": page_number,
            "cer": None,
            "wer": None,
            "char_f1": None,
            "word_f1": None,
            "message": f"Page {page_number} not found in run",
        }

    # Load the corresponding GT page
    gt_result = await db.execute(
        select(GTPageResult).where(
            GTPageResult.gt_version_id == gt_version_id,
            GTPageResult.page_number == page_number,
        ),
    )
    gt_page = gt_result.scalars().one_or_none()
    if gt_page is None:
        return {
            "page": page_number,
            "cer": None,
            "wer": None,
            "char_f1": None,
            "word_f1": None,
            "message": f"Ground truth page {page_number} not found",
        }

    run_words = _extract_words({"data": page_result.data})
    gt_words = _extract_words({"data": gt_page.data})
    scores = evaluate_page(run_words, gt_words)

    return {
        "page": page_number,
        "cer": scores["cer"],
        "wer": scores["wer"],
        "char_f1": scores["char_f1"],
        "word_f1": scores["word_f1"],
    }


async def compare_engines(
    pdf_id: uuid.UUID,
    engine_ids: list[uuid.UUID],
    gt_version_id: uuid.UUID,
    db: AsyncSession,
) -> dict[str, Any]:
    """Cross-engine comparison for a document.

    For each engine ID, finds the most recent completed run against the given
    PDF, computes aggregate scores, and returns them side-by-side.

    Returns
    -------
    dict
        ``{
            "pdf_id": ...,
            "gt_version_id": ...,
            "engines": [
                {"engine_id": ..., "run_id": ..., "scores": {...}},
                ...
            ],
        }``
    """
    gt_data = await _build_gt_data(gt_version_id, db)

    engine_results: list[dict[str, Any]] = []
    for engine_id in engine_ids:
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
        run = result.scalars().one_or_none()
        if run is None:
            engine_results.append({
                "engine_id": str(engine_id),
                "run_id": None,
                "scores": None,
                "message": "No completed run found for this engine",
            })
            continue

        run_data = await _build_run_data(run.id, db)
        run_scores = evaluate_run(run_data, gt_data)
        engine_results.append({
            "engine_id": str(engine_id),
            "run_id": str(run.id),
            "scores": {
                "cer": run_scores["cer"],
                "wer": run_scores["wer"],
                "char_f1": run_scores["char_f1"],
                "word_f1": run_scores["word_f1"],
                "pages": run_scores["num_pages"],
            },
        })

    return {
        "pdf_id": str(pdf_id),
        "gt_version_id": str(gt_version_id),
        "engines": engine_results,
    }


# ── Internal helpers ───────────────────────────────────────────────────────


async def _load_run(run_id: uuid.UUID, db: AsyncSession) -> OCRRun:
    """Load a run by ID, raising if not found."""
    result = await db.execute(select(OCRRun).where(OCRRun.id == run_id))
    run = result.scalars().one_or_none()
    if run is None:
        msg = f"Run {run_id} not found"
        raise ValueError(msg)
    return run


async def _build_run_data(run_id: uuid.UUID, db: AsyncSession) -> dict[str, list[dict]]:
    """Build the ``{"pages": [...]}`` dict expected by ``evaluate_run``.

    Each page entry has the ``{"data": {...}}`` shape that
    :func:`_extract_words` can consume.
    """
    result = await db.execute(
        select(PageResult)
        .where(PageResult.run_id == run_id)
        .order_by(PageResult.page_number),
    )
    page_results = list(result.scalars().unique().all())
    return {
        "pages": [
            {"data": pr.data} for pr in page_results
        ],
    }


async def _build_gt_data(
    gt_version_id: uuid.UUID,
    db: AsyncSession,
) -> dict[str, list[dict]]:
    """Build the ``{"pages": [...]}`` dict from a GT version."""
    result = await db.execute(
        select(GTPageResult)
        .where(GTPageResult.gt_version_id == gt_version_id)
        .order_by(GTPageResult.page_number),
    )
    gt_pages = list(result.scalars().unique().all())
    return {
        "pages": [
            {"data": gp.data} for gp in gt_pages
        ],
    }
