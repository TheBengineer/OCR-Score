"""Batch OCR processing — runs multiple PDF×engine combinations sequentially.

The ``BatchProcessor`` manages the lifecycle of batch processing jobs:

1. **create_batch** — Validates all PDFs and engines exist, creates a batch
   record with one item per ``(pdf_id, engine_id)`` combination.
2. **process_batch** — Iterates through items sequentially, creating and
   executing an ``OCRRun`` for each combination via ``RunOrchestrator``.
3. **get_batch_progress** — Returns completion statistics.

Each item tracks its individual run ID and status, so the frontend can poll
for granular per-PDF progress without real-time WebSocket updates.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import async_session_factory
from backend.engine.registry import EngineRegistry
from backend.engine.registry import registry as _global_registry
from backend.models.enums import RunStatus
from backend.models.pdf import PDF
from backend.models.run import OCRRun
from backend.run_orchestrator import RunOrchestrator, RunOrchestratorError
from backend.storage import ContentAddressableStorage

# ── Types ───────────────────────────────────────────────────────────────────


class BatchStatus:
    """Possible batch lifecycle statuses."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchItem:
    """A single PDF×engine combination within a batch."""

    pdf_id: uuid.UUID
    engine_slug: str
    run_id: uuid.UUID | None = None
    status: str = BatchStatus.PENDING
    message: str | None = None


@dataclass
class Batch:
    """Represents a batch processing job."""

    id: uuid.UUID
    pdf_ids: list[uuid.UUID]
    engine_slugs: list[str]
    config: dict[str, Any]
    status: str = BatchStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    items: list[BatchItem] = field(default_factory=list)
    error_message: str | None = None


# ── In-memory store ────────────────────────────────────────────────────────

_batches: dict[uuid.UUID, Batch] = {}


def _clear_batches() -> None:
    """Clear all in-memory batches (used in testing)."""
    _batches.clear()


# ── BatchProcessor ──────────────────────────────────────────────────────────


class BatchProcessorError(Exception):
    """Raised when a batch operation precondition fails."""


class BatchProcessor:
    """Creates and processes batches of OCR runs sequentially.

    Attributes:
        db: An async SQLAlchemy session (for the request-scoped ``create_batch``).
        storage: Content-addressable storage for raw outputs.
        _registry: Engine plugin registry.
    """

    def __init__(
        self,
        db: AsyncSession,
        storage: ContentAddressableStorage,
        registry: EngineRegistry | None = None,
    ) -> None:
        self._db = db
        self._storage = storage
        self._registry = registry if registry is not None else _global_registry

    async def create_batch(
        self,
        pdf_ids: list[uuid.UUID],
        engine_slugs: list[str],
        config: dict[str, Any] | None = None,
    ) -> Batch:
        """Validate inputs and create a new batch record.

        Args:
            pdf_ids: UUIDs of the PDFs to process.
            engine_slugs: Engine identifiers to run (e.g. ``"mock"``).
            config: Shared engine configuration (merged with engine defaults).

        Returns:
            A ``Batch`` with all items initialised as ``pending``.

        Raises:
            BatchProcessorError: If any PDF or engine is not found.
        """
        if not pdf_ids:
            msg = "pdf_ids must not be empty"
            raise BatchProcessorError(msg)
        if not engine_slugs:
            msg = "engine_slugs must not be empty"
            raise BatchProcessorError(msg)

        # Validate all PDFs exist in a single query
        result = await self._db.execute(
            select(PDF).where(
                PDF.id.in_(pdf_ids),
                PDF.deleted_at.is_(None),
            ),
        )
        found_pdfs: list[PDF] = list(result.scalars().all())
        found: set[uuid.UUID] = {p.id for p in found_pdfs}
        missing = [str(pid) for pid in pdf_ids if pid not in found]
        if missing:
            msg = f"PDF(s) not found or have been deleted: {', '.join(missing)}"
            raise BatchProcessorError(msg)

        # Validate all engines exist (in-memory registry)
        from backend.engine.registry import registry as engine_registry
        engine_registry.discover()
        available = {e.engine_id for e in engine_registry.list()}
        for slug in engine_slugs:
            if slug not in available:
                msg = f"Engine '{slug}' is not registered"
                raise BatchProcessorError(msg)

        merged_config = config or {}

        batch = Batch(
            id=uuid.uuid4(),
            pdf_ids=pdf_ids,
            engine_slugs=engine_slugs,
            config=merged_config,
        )

        # Create items for every PDF×engine combination
        for pdf_id in pdf_ids:
            for slug in engine_slugs:
                batch.items.append(BatchItem(pdf_id=pdf_id, engine_slug=slug))

        _batches[batch.id] = batch
        return batch

    async def process_batch(
        self,
        batch_id: uuid.UUID,
    ) -> Batch:
        """Execute all items in a batch sequentially.

        Each item creates and executes an OCR run. Progress is tracked via
        the batch's items list. The method runs each item in sequence to
        avoid overloading OCR engines.

        **Session isolation**: This method opens its own database session
        instead of using the request-injected ``self._db``, because it runs
        as a background task (``asyncio.create_task``) after the HTTP response
        is sent — at which point FastAPI will have closed the DI session.

        Args:
            batch_id: UUID of the batch to process.

        Returns:
            The updated ``Batch`` with final statuses.

        Raises:
            BatchProcessorError: If the batch is not found.
        """
        batch = _batches.get(batch_id)
        if batch is None:
            msg = f"Batch {batch_id} not found"
            raise BatchProcessorError(msg)

        batch.status = BatchStatus.RUNNING

        async with async_session_factory() as batch_db:
            orchestrator = RunOrchestrator(
                db=batch_db,
                storage=self._storage,
                registry=self._registry,
            )
            completed, failed = await self._run_items(
                batch, orchestrator, batch_db,
            )

        batch.status = (
            BatchStatus.COMPLETED
            if failed == 0 or completed > 0
            else BatchStatus.FAILED
        )
        return batch

    async def _run_items(
        self,
        batch: Batch,
        orchestrator: RunOrchestrator,
        db: AsyncSession,
    ) -> tuple[int, int]:
        """Execute every item in *batch* through *orchestrator*.

        Exposed as a separate method so tests can inject a fake session
        by calling this directly with a ``FakeSession`` instead of going
        through ``process_batch`` (which opens its own real session).
        """
        completed = 0
        failed = 0
        total = len(batch.items)
        t_start = time.monotonic()

        for idx, item in enumerate(batch.items, start=1):
            item.status = "processing"
            t_item = time.monotonic()
            try:
                run = await orchestrator.create_run(
                    pdf_id=item.pdf_id,
                    engine_slug=item.engine_slug,
                    config=batch.config,
                )
                item.run_id = run.id

                if run.status == RunStatus.COMPLETED:
                    item.status = BatchStatus.COMPLETED
                    completed += 1
                else:
                    # Verbose: run-level details are logged inside execute_run
                    await orchestrator.execute_run(run.id)
                    result = await db.execute(
                        select(OCRRun).where(OCRRun.id == run.id),
                    )
                    updated_run = result.scalars().one_or_none()
                    if updated_run and updated_run.status == RunStatus.COMPLETED:
                        item.status = BatchStatus.COMPLETED
                        completed += 1
                    elif updated_run and updated_run.status == RunStatus.FAILED:
                        item.status = BatchStatus.FAILED
                        item.message = updated_run.error_message
                        failed += 1
                    else:
                        item.status = BatchStatus.COMPLETED
                        completed += 1

                elapsed = time.monotonic() - t_item
                slug = item.engine_slug
                logging.info(
                    "Batch %s [%d/%d] %s pdf=%s %s in %.1fs — %d done, %d failed",
                    str(batch.id)[:8], idx, total, slug, str(item.pdf_id)[:8],
                    "completed" if item.status == BatchStatus.COMPLETED else "failed",
                    elapsed, completed, failed,
                )

            except RunOrchestratorError as exc:
                item.status = BatchStatus.FAILED
                item.message = str(exc)
                failed += 1
                logging.warning(
                    "Batch %s [%d/%d] %s pdf=%s failed: %s",
                    str(batch.id)[:8], idx, total, item.engine_slug,
                    str(item.pdf_id)[:8], exc,
                )

        t_total = time.monotonic() - t_start
        logging.info(
            "Batch %s finished: %d/%d done, %d failed in %.1fs",
            str(batch.id)[:8], completed, total, failed, t_total,
        )
        return completed, failed

    def get_batch(self, batch_id: uuid.UUID) -> Batch | None:
        """Retrieve a batch by ID.

        Args:
            batch_id: UUID of the batch.

        Returns:
            The ``Batch`` or ``None`` if not found.
        """
        return _batches.get(batch_id)

    def get_batch_progress(
        self,
        batch_id: uuid.UUID,
    ) -> dict[str, Any] | None:
        """Return a progress summary for a batch.

        Args:
            batch_id: UUID of the batch.

        Returns:
            A dict with ``total``, ``completed``, ``failed``, ``percent``,
            and ``items``, or ``None`` if the batch is not found.
        """
        batch = _batches.get(batch_id)
        if batch is None:
            return None

        total = len(batch.items)
        completed = sum(
            1 for item in batch.items if item.status == BatchStatus.COMPLETED
        )
        failed = sum(
            1 for item in batch.items if item.status == BatchStatus.FAILED
        )
        pending = total - completed - failed

        return {
            "batch_id": str(batch.id),
            "status": batch.status,
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "percent": (completed / total * 100) if total > 0 else 0.0,
            "items": [
                {
                    "pdf_id": str(item.pdf_id),
                    "engine_slug": item.engine_slug,
                    "run_id": str(item.run_id) if item.run_id else None,
                    "status": item.status,
                    "message": item.message,
                }
                for item in batch.items
            ],
        }
