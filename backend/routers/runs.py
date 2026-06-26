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
