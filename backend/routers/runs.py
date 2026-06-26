"""OCR run management — start, list, retrieve, and cancel OCR processing runs.

Endpoints
---------
- ``POST /api/v1/runs`` — Start a run (returns 202, or 200 if already completed).
- ``GET  /api/v1/runs`` — List runs, filterable by pdf_id / engine_id / status.
- ``GET  /api/v1/runs/{id}`` — Get run metadata + status.
- ``GET  /api/v1/runs/{id}/results`` — List page results (paginated).
- ``GET  /api/v1/runs/{id}/results/{page}`` — Get a single page result.
- ``GET  /api/v1/runs/{id}/raw`` — Download raw engine output.
- ``DELETE /api/v1/runs/{id}`` — Cancel or acknowledge terminal run.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from backend.alignment.comparator import align_multiple_engine_pages, build_comparison_grid
from backend.database import get_db_session
from backend.models.enums import RunStatus
from backend.models.page_result import PageResult
from backend.models.run import OCRRun
from backend.run_orchestrator import RunOrchestrator, RunOrchestratorError
from backend.schemas.page_result import PageResultRead
from backend.schemas.run import OCRRunRead
from backend.settings import settings
from backend.storage import ContentAddressableStorage

# ── Router ────────────────────────────────────────────────────────────────────

runs_router = APIRouter(prefix="/api/v1/runs", tags=["runs"])

# ── Dependencies ──────────────────────────────────────────────────────────────

SessionDep = Annotated[AsyncSession, Depends(get_db_session)]


def get_storage() -> ContentAddressableStorage:
    """Provide the singleton content-addressable storage instance."""
    return ContentAddressableStorage(Path(settings.storage_path))


StorageDep = Annotated[ContentAddressableStorage, Depends(get_storage)]


def get_orchestrator(
    db: SessionDep,
    storage: StorageDep,
) -> RunOrchestrator:
    """FastAPI dependency for a fresh ``RunOrchestrator`` per request."""
    return RunOrchestrator(db=db, storage=storage)


OrchestratorDep = Annotated[RunOrchestrator, Depends(get_orchestrator)]

# ── Request / response schemas ────────────────────────────────────────────────


class RunCreateRequest(BaseModel):
    """Request body for ``POST /api/v1/runs``."""

    pdf_id: uuid.UUID
    engine_id: str = Field(
        ...,
        description="Engine slug (e.g. 'mock', 'tesseract', 'gcp-document-ai')",
    )
    config: dict | None = Field(
        default=None,
        description="Engine-specific configuration parameters",
    )


class RunCreateResponse(BaseModel):
    """Response body returned after creating (or finding) a run."""

    id: uuid.UUID
    status: RunStatus
    message: str | None = None


class RunListResponse(BaseModel):
    """Paginated list of runs."""

    items: list[OCRRunRead]
    total: int


class PageResultListResponse(BaseModel):
    """Paginated list of page results for a run."""

    items: list[PageResultRead]
    page: int
    page_size: int
    total: int


# ── Endpoints ─────────────────────────────────────────────────────────────────


@runs_router.post("", status_code=status.HTTP_202_ACCEPTED)
async def create_run(
    body: RunCreateRequest,
    orchestrator: OrchestratorDep,
) -> JSONResponse:
    """Start an OCR run.

    If a **completed** run with the exact same parameters already exists
    (same PDF, engine, config, and date), the existing run is returned with
    a ``200`` status instead (idempotent dedup).

    A new run starts in the ``pending`` state and transitions through
    ``queued → running → completed|failed`` via a background task.
    """
    try:
        run = await orchestrator.create_run(body.pdf_id, body.engine_id, body.config)
    except RunOrchestratorError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    # Existing completed run → 200 (idempotent dedup)
    if run.status == RunStatus.COMPLETED:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=RunCreateResponse(
                id=run.id,
                status=run.status,
                message="run already completed",
            ).model_dump(mode="json"),
        )

    # Launch background execution
    import asyncio  # noqa: PLC0415

    asyncio.create_task(orchestrator.execute_run(run.id))

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=RunCreateResponse(id=run.id, status=run.status).model_dump(mode="json"),
    )


@runs_router.get("")
async def list_runs(
    db: SessionDep,
    pdf_id: uuid.UUID | None = Query(default=None),  # noqa: B008
    engine_id: uuid.UUID | None = Query(default=None),  # noqa: B008
    status_filter: RunStatus | None = Query(default=None, alias="status"),  # noqa: B008
    limit: int = Query(default=50, ge=1, le=100),  # noqa: B008
    offset: int = Query(default=0, ge=0),  # noqa: B008
) -> RunListResponse:
    """List OCR runs with optional filters and pagination.

    Results are ordered by most recently created first.
    """
    query = select(OCRRun).order_by(OCRRun.created_at.desc())

    if pdf_id is not None:
        query = query.where(OCRRun.pdf_id == pdf_id)
    if engine_id is not None:
        query = query.where(OCRRun.engine_id == engine_id)
    if status_filter is not None:
        query = query.where(OCRRun.status == status_filter)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total: int = total_result.scalar() or 0

    query = query.offset(offset).limit(limit)
    result = await db.execute(query)
    runs = list(result.scalars().unique().all())

    return RunListResponse(
        items=[OCRRunRead.model_validate(r) for r in runs],
        total=total,
    )


@runs_router.get("/{run_id}")
async def get_run(
    run_id: uuid.UUID,
    db: SessionDep,
) -> OCRRunRead:
    """Retrieve metadata and status for a single run."""
    result = await db.execute(select(OCRRun).where(OCRRun.id == run_id))
    run = result.scalars().one_or_none()
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="run not found",
        )
    return OCRRunRead.model_validate(run)


@runs_router.get("/{run_id}/results")
async def list_results(
    run_id: uuid.UUID,
    db: SessionDep,
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),  # noqa: B008
    page_size: int = Query(default=50, ge=1, le=100, description="Results per page"),  # noqa: B008
) -> PageResultListResponse:
    """List page results for a run, paginated."""
    # Verify run exists
    run_check = await db.execute(
        select(OCRRun.id).where(OCRRun.id == run_id),
    )
    if run_check.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="run not found",
        )

    # Count total
    count_result = await db.execute(
        select(func.count()).select_from(
            select(PageResult).where(PageResult.run_id == run_id).subquery(),
        ),
    )
    total: int = count_result.scalar() or 0

    result = await db.execute(
        select(PageResult)
        .where(PageResult.run_id == run_id)
        .order_by(PageResult.page_number)
        .offset((page - 1) * page_size)
        .limit(page_size),
    )
    items = list(result.scalars().unique().all())

    return PageResultListResponse(
        items=[PageResultRead.model_validate(i) for i in items],
        page=page,
        page_size=page_size,
        total=total,
    )


@runs_router.get("/{run_id}/results/{page_number}")
async def get_page_result(
    run_id: uuid.UUID,
    page_number: int,
    db: SessionDep,
) -> PageResultRead:
    """Retrieve a single page's normalised OCR result."""
    result = await db.execute(
        select(PageResult).where(
            PageResult.run_id == run_id,
            PageResult.page_number == page_number,
        ),
    )
    pr = result.scalars().one_or_none()
    if pr is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="page result not found",
        )
    return PageResultRead.model_validate(pr)


@runs_router.get("/{run_id}/results/{page_number}/compare")
async def compare_page_across_engines(
    run_id: uuid.UUID,
    page_number: int,
    db: SessionDep,
    engine_ids: str = Query(  # noqa: B008
        default="",
        description="Comma-separated run UUIDs to compare against this run (e.g. "
        "'uuid1,uuid2,uuid3')",
    ),
) -> dict:
    """Compare a single page's OCR results across multiple engine runs.

    Returns an aligned character grid with consensus information for all
    specified runs, enabling the frontend to composite overlay layers
    without N+1 API calls.

    The ``run_id`` in the path is the **primary** run; ``engine_ids``
    are additional **run** UUIDs (not engine slugs).  The response
    includes per-aligned-word consensus, per-engine status, and overall
    statistics.

    Requires at least 2 runs (primary + at least 1 extra).  All runs
    must have status ``completed``.  Returns ``404`` if the run or page
    is not found, or ``400`` if fewer than 2 engine IDs are provided.
    """
    # ── Parse engine_ids ──────────────────────────────────────────────
    if not engine_ids or not engine_ids.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 runs are required for comparison; "
            "provide engine_ids query parameter with additional run UUIDs",
        )

    extra_run_ids: list[uuid.UUID] = []
    for raw in engine_ids.split(","):
        piece = raw.strip()
        if not piece:
            continue
        try:
            extra_run_ids.append(uuid.UUID(piece))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid run UUID: '{piece}'",
            ) from None

    all_run_ids = list(set([run_id] + extra_run_ids))
    n_runs = len(all_run_ids)
    if n_runs < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"At least 2 runs are required for comparison; got {n_runs}",
        )

    # ── Fetch runs with engine info ───────────────────────────────────
    result = await db.execute(
        select(OCRRun)
        .options(joinedload(OCRRun.engine))
        .where(OCRRun.id.in_(all_run_ids)),
    )
    runs = list(result.scalars().unique().all())

    if len(runs) != n_runs:
        found = {r.id for r in runs}
        missing = [str(rid) for rid in all_run_ids if rid not in found]
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Runs not found: {', '.join(missing)}",
        )

    # ── Verify all runs are completed ─────────────────────────────────
    for run in runs:
        if run.status != RunStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Run {run.id} has status '{run.status.value}', expected 'completed'",
            )

    # ── Fetch page results ─────────────────────────────────────────────
    pr_result = await db.execute(
        select(PageResult).where(
            PageResult.run_id.in_(all_run_ids),
            PageResult.page_number == page_number,
        ),
    )
    page_results = list(pr_result.scalars().unique().all())
    pr_map: dict[uuid.UUID, PageResult] = {pr.run_id: pr for pr in page_results}

    for rid in all_run_ids:
        if rid not in pr_map:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Page {page_number} not found for run {rid}",
            )

    # ── Build engine results for alignment ────────────────────────────
    first_pr = pr_map[all_run_ids[0]]
    dimensions = {"width": first_pr.width or 0.0, "height": first_pr.height or 0.0}

    engine_results: list[dict] = []
    for run in runs:
        engine_slug = run.engine.slug if run.engine else str(run.engine_id)
        engine_results.append({
            "engine": engine_slug,
            "data": pr_map[run.id].data,
        })

    aligned = align_multiple_engine_pages(engine_results)
    return build_comparison_grid(aligned, page_number=page_number, dimensions=dimensions)


# ── Alternative compare across runs (PDF-scoped) ──────────────────────────────


# ── Pages Router (cross-run comparison) ───────────────────────────────────────

pages_router = APIRouter(prefix="/api/v1/pages", tags=["pages"])


@pages_router.get("/compare")
async def compare_pages_across_runs(
    pdf_id: uuid.UUID,
    db: SessionDep,
    page: int = Query(default=1, ge=1),  # noqa: B008
    engine_ids: str = Query(  # noqa: B008
        ...,
        description="Comma-separated run UUIDs to compare (e.g. 'uuid1,uuid2,uuid3')",
    ),
) -> dict:
    """Compare OCR results for a page across multiple runs by PDF.

    Unlike the run-scoped compare endpoint (:meth:`compare_page_across_engines`),
    this accepts an explicit list of **run** UUIDs and a PDF ID for
    verification.  All specified runs must belong to the same PDF.

    Requires at least 2 runs.  Returns ``404`` if any run or the requested
    page is not found.  Returns ``400`` if fewer than 2 runs are specified,
    if a run does not belong to the PDF, or if a run is not completed.
    """
    # ── Parse engine_ids ──────────────────────────────────────────────
    run_ids: list[uuid.UUID] = []
    for raw in engine_ids.split(","):
        piece = raw.strip()
        if not piece:
            continue
        try:
            run_ids.append(uuid.UUID(piece))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid run UUID: '{piece}'",
            ) from None

    n_runs = len(run_ids)
    if n_runs < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"At least 2 runs are required for comparison; got {n_runs}",
        )

    # ── Fetch runs with engine info ───────────────────────────────────
    result = await db.execute(
        select(OCRRun)
        .options(joinedload(OCRRun.engine))
        .where(OCRRun.id.in_(run_ids)),
    )
    runs = list(result.scalars().unique().all())

    if len(runs) != n_runs:
        found = {r.id for r in runs}
        missing = [str(rid) for rid in run_ids if rid not in found]
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Runs not found: {', '.join(missing)}",
        )

    # ── Verify all runs belong to the same PDF ────────────────────────
    for run in runs:
        if run.pdf_id != pdf_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Run {run.id} does not belong to PDF {pdf_id}",
            )

    # ── Verify all runs are completed ─────────────────────────────────
    for run in runs:
        if run.status != RunStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Run {run.id} has status '{run.status.value}', expected 'completed'",
            )

    # ── Fetch page results ─────────────────────────────────────────────
    pr_result = await db.execute(
        select(PageResult).where(
            PageResult.run_id.in_(run_ids),
            PageResult.page_number == page,
        ),
    )
    page_results = list(pr_result.scalars().unique().all())
    pr_map: dict[uuid.UUID, PageResult] = {pr.run_id: pr for pr in page_results}

    for rid in run_ids:
        if rid not in pr_map:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Page {page} not found for run {rid}",
            )

    # ── Build engine results for alignment ────────────────────────────
    first_pr = pr_map[run_ids[0]]
    dimensions = {"width": first_pr.width or 0.0, "height": first_pr.height or 0.0}

    engine_results: list[dict] = []
    for run in runs:
        engine_slug = run.engine.slug if run.engine else str(run.engine_id)
        engine_results.append({
            "engine": engine_slug,
            "data": pr_map[run.id].data,
        })

    aligned = align_multiple_engine_pages(engine_results)
    return build_comparison_grid(aligned, page_number=page, dimensions=dimensions)


@runs_router.get("/{run_id}/raw")
async def get_raw_output(
    run_id: uuid.UUID,
    db: SessionDep,
) -> JSONResponse:
    """Download the engine's raw (pre-normalisation) output as JSON."""
    result = await db.execute(
        select(OCRRun).where(OCRRun.id == run_id),
    )
    run = result.scalars().one_or_none()
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="run not found",
        )
    if run.raw_output_uri is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="raw output not yet available",
        )

    raw_path = Path(run.raw_output_uri)
    if not raw_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="raw output file not found on disk",
        )

    raw_bytes = raw_path.read_bytes()
    raw_data = json.loads(raw_bytes)
    return JSONResponse(content=raw_data)


@runs_router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_run(
    run_id: uuid.UUID,
    orchestrator: OrchestratorDep,
) -> Response:
    """Cancel a run (if in-flight) or acknowledge an already-terminal run.

    Returns ``204`` in all cases (idempotent).
    """
    run = await orchestrator.cancel_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="run not found",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
