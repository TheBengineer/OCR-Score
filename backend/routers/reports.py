"""Reports router — aggregate summaries, engine rankings, and CSV/JSON/HTML export.

Endpoints
---------
- ``GET /api/v1/reports/summary`` — Aggregate scores across all completed runs.
- ``GET /api/v1/reports/engines``  — Per-engine rankings across all PDFs.
- ``GET /api/v1/reports/export``   — Export data as CSV, JSON, or HTML.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db_session
from backend.models.enums import RunStatus
from backend.models.run import OCRRun
from backend.report_generator import (
    aggregate_engine_rankings,
    compute_summary_statistics,
    generate_csv_report,
    generate_html_report,
    generate_json_report,
)

# ── Router ────────────────────────────────────────────────────────────────────

reports_router = APIRouter(prefix="/api/v1/reports", tags=["reports"])

# ── Dependencies ──────────────────────────────────────────────────────────────

SessionDep = Annotated[AsyncSession, Depends(get_db_session)]


async def _resolve_run_ids(
    db: AsyncSession,
    run_ids_param: str | None,
) -> list[uuid.UUID]:
    """Parse the optional ``run_ids`` query parameter into UUIDs.

    If the parameter is ``None`` or empty, return **all** completed run UUIDs.
    """
    if run_ids_param:
        ids: list[uuid.UUID] = []
        for raw in run_ids_param.split(","):
            piece = raw.strip()
            if not piece:
                continue
            try:
                ids.append(uuid.UUID(piece))
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid run UUID: '{piece}'",
                ) from exc
        return ids

    # Default: all completed runs
    result = await db.execute(
        select(OCRRun.id).where(OCRRun.status == RunStatus.COMPLETED),
    )
    return [row[0] for row in result.all()]


def _disposition_filename(format: str) -> str:
    """Return a Content-Disposition filename for the given format."""
    return f"ocrscore_report.{format}"


# ── Endpoints ─────────────────────────────────────────────────────────────────


@reports_router.get("/summary")
async def summary(db: SessionDep) -> JSONResponse:
    """Aggregate summary statistics across all completed runs.

    Returns counts and averages for PDFs, runs, scores, and identifies the
    best-performing engine.
    """
    stats = await compute_summary_statistics(db)
    return JSONResponse(content=stats)


@reports_router.get("/engines")
async def engines(db: SessionDep) -> JSONResponse:
    """Per-engine rankings sorted by average CER (best first).

    Each entry includes ``avg_cer``, ``avg_wer``, ``avg_f1``, and the number
    of completed runs for that engine.
    """
    rankings = await aggregate_engine_rankings(db)
    return JSONResponse(content=rankings)


@reports_router.get("/export")
async def export(
    db: SessionDep,
    format: str = Query(  # noqa: A002
        ...,
        description="Export format: 'csv', 'json', or 'html'",
    ),
    run_ids: str | None = Query(  # noqa: B008
        default=None,
        description="Comma-separated list of run UUIDs to export (default: all completed runs)",
    ),
) -> Response:
    """Export OCR evaluation data in the requested format.

    Supports three formats:
    - **csv** — Tabular data with one row per run per page.
    - **json** — Full structured data dump with engine summaries.
    - **html** — Self-contained report page with inline CSS.
    """
    normalized_format = format.strip().lower()

    if normalized_format not in ("csv", "json", "html"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{format}'. Use 'csv', 'json', or 'html'.",
        )

    parsed_run_ids = await _resolve_run_ids(db, run_ids)

    if not parsed_run_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No completed runs found to export.",
        )

    if normalized_format == "json":
        data = await generate_json_report(parsed_run_ids, db)
        return JSONResponse(
            content=data,
            media_type="application/json",
            headers={
                "Content-Disposition": (
                    f"attachment; filename={_disposition_filename('json')}"
                ),
            },
        )

    # CSV and HTML are generated to a temporary file then served
    suffix = f".{normalized_format}"
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if normalized_format == "csv":
            await generate_csv_report(parsed_run_ids, db, tmp_path)
        else:  # html
            await generate_html_report(parsed_run_ids, db, tmp_path)

        content = Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    media_types = {
        "csv": "text/csv",
        "html": "text/html",
    }

    return Response(
        content=content,
        media_type=media_types[normalized_format],
        headers={
            "Content-Disposition": (
                f"attachment; filename={_disposition_filename(normalized_format)}"
            ),
        },
    )
