"""OCR run orchestration — creates, executes, and manages processing runs.

The ``RunOrchestrator`` is the central coordinator for OCR processing:

1. **create_run** — Validates inputs, computes a content-based run hash for
   deduplication, and persists a new ``OCRRun`` record.
2. **execute_run** — Fetches the engine plugin, runs ``process_pdf`` with
   progress tracking, normalises the output, and stores both raw and
   normalised results.  Progress updates are broadcast to any connected
   WebSocket subscribers via the global :data:`manager`.
3. **cancel_run** — Gracefully transitions a cancellable run to ``cancelled``.

Run hash dedup
--------------
The run hash is a SHA-256 of
``{pdf_sha256}|{engine_slug}|{engine_version}|{canonical_json(config)}|{today_utc}``.
A **completed** run with the same hash is returned directly (idempotent).
A **failed** run with the same hash is treated as stale — a new run is
always created.  In-progress runs (pending/queued/running) raise a conflict.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import anyio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.engine.registry import EngineRegistry
from backend.engine.registry import registry as _global_registry
from backend.models.enums import RunStatus
from backend.models.page_result import PageResult
from backend.models.pdf import PDF
from backend.models.run import OCRRun
from backend.storage import ContentAddressableStorage
from backend.websocket_manager import manager

_IN_FLIGHT: frozenset[RunStatus] = frozenset({
    RunStatus.PENDING,
    RunStatus.QUEUED,
    RunStatus.RUNNING,
})


class RunOrchestratorError(Exception):
    """Raised when a run creation/execution precondition fails."""


class RunOrchestrator:
    """Coordinates OCR run creation, execution, and lifecycle management.

    Attributes:
        db: An async SQLAlchemy session.
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

    # ── Public API ─────────────────────────────────────────────────────────

    async def create_run(
        self,
        pdf_id: uuid.UUID,
        engine_slug: str,
        config: dict | None = None,
    ) -> OCRRun:
        """Validate inputs, compute hash, check dedup, and persist a new run.

        Args:
            pdf_id: UUID of the PDF document to process.
            engine_slug: Engine identifier (e.g. ``"mock"``, ``"tesseract"``).
            config: Engine-specific configuration dict (merged with defaults).

        Returns:
            The ``OCRRun`` record — either newly created (status ``PENDING``)
            or an existing **completed** run with the same hash.

        Raises:
            RunOrchestratorError: If the PDF or engine is not found, or if a
                run with the same hash is already in progress.
        """
        # ── 1. Validate PDF ───────────────────────────────────────────────
        pdf = await self._get_active_pdf(pdf_id)
        if pdf is None:
            msg = f"PDF {pdf_id} not found or has been deleted"
            raise RunOrchestratorError(msg)

        # ── 2. Validate engine in DB ───────────────────────────────────────
        engine_record = await self._get_engine_by_slug(engine_slug)
        if engine_record is None:
            msg = f"Engine '{engine_slug}' is not registered"
            raise RunOrchestratorError(msg)

        # ── 3. Resolve engine version (plugin > DB record) ─────────────────
        try:
            engine_plugin = self._registry.get(engine_slug)
            engine_version = engine_plugin.version
        except Exception:  # noqa: BLE001 — fallback to DB version
            engine_version = engine_record.version

        # ── 4. Merge provided config with engine defaults ──────────────────
        merged_config = self._extract_engine_config(engine_record, config)

        # ── 5. Compute run hash ────────────────────────────────────────────
        run_hash = self._compute_run_hash(
            pdf.sha256_hash,
            engine_slug,
            engine_version,
            merged_config,
        )

        # ── 6. Dedup check: completed → return existing ───────────────────
        existing = await self._find_run_by_hash(run_hash, RunStatus.COMPLETED)
        if existing is not None:
            return existing

        # ── 7. Dedup check: in-flight → conflict ──────────────────────────
        existing = await self._find_run_by_hash_in_flight(run_hash)
        if existing is not None:
            msg = (
                f"A run with the same parameters is already in progress "
                f"(run_id={existing.id})"
            )
            raise RunOrchestratorError(msg)

        # ── 8. Create run record ───────────────────────────────────────────
        run = OCRRun(
            pdf_id=pdf_id,
            engine_id=engine_record.id,
            status=RunStatus.PENDING,
            engine_config=merged_config,
            engine_version=engine_version,
            run_hash=run_hash,
        )
        self._db.add(run)
        await self._db.commit()
        await self._db.refresh(run)

        return run

    async def execute_run(self, run_id: uuid.UUID) -> None:
        """Execute an OCR run end-to-end.

        Status flow: ``pending → queued → running → completed|failed``.

        This method is designed to be launched as a background task via
        ``asyncio.create_task()``.  It never raises — errors are captured
        in the ``OCRRun.error_message`` field and the run is marked failed.

        Args:
            run_id: UUID of the run to execute.
        """
        run = await self._get_run(run_id)
        if run is None:
            return

        run_id_str = str(run.id)
        self._add_log(run, "INFO", "Run queued")

        try:
            run.status = RunStatus.QUEUED
            await self._db.commit()
            await manager.broadcast_status_change(run_id_str, "queued", 0)

            # Look up engine by ID (avoids relationship lazy-load issues)
            engine_record = await self._get_engine_by_id(run.engine_id)
            if engine_record is None:
                msg = f"Engine record {run.engine_id} not found in database"
                raise RuntimeError(msg)
            engine_slug = getattr(engine_record, "slug", "unknown")
            engine_plugin = self._registry.get(engine_slug)
            self._add_log(run, "INFO", f"Engine: {engine_slug} v{engine_plugin.version}")

            # Look up PDF by ID (avoids relationship lazy-load issues)
            pdf_record = await self._get_active_pdf(run.pdf_id)
            if pdf_record is None:
                msg = f"PDF record {run.pdf_id} not found or deleted"
                raise RuntimeError(msg)
            pdf_path = self._get_pdf_path(pdf_record.sha256_hash)
            if not await anyio.to_thread.run_sync(pdf_path.exists):
                msg = f"PDF file not found on disk: {pdf_path}"
                raise FileNotFoundError(msg)

            run.status = RunStatus.RUNNING
            run.started_at = datetime.now(UTC)
            run.engine_version = engine_plugin.version
            await self._db.commit()
            await manager.broadcast_status_change(run_id_str, "running", 0)
            self._add_log(run, "INFO", "Processing started")

            # Create a sync progress callback that bridges to the async
            # broadcast — this is called from engine.process_pdf which
            # runs in the same event loop.
            def _progress_callback(pct: int) -> None:
                asyncio.create_task(
                    manager.broadcast_progress(
                        run_id_str,
                        pct,
                        "running",
                        f"Processing... {pct}%",
                    ),
                )

            raw = await engine_plugin.process_pdf(
                str(pdf_path),
                run.engine_config or {},
                progress=_progress_callback,
            )
            self._add_log(run, "INFO", "Engine processing complete, normalising output")

            normalized = engine_plugin.normalize(raw)

            self._add_log(run, "INFO", "Run completed successfully")
            await self._store_results(run, raw, normalized)
            await manager.broadcast_status_change(run_id_str, "completed", 100)

        except Exception as exc:  # noqa: BLE001
            run.status = RunStatus.FAILED
            run.error_message = str(exc)
            run.completed_at = datetime.now(UTC)
            self._add_log(run, "ERROR", str(exc))
            await self._db.commit()
            await manager.broadcast_error(run_id_str, str(exc))

    async def cancel_run(self, run_id: uuid.UUID) -> OCRRun | None:
        """Cancel a run if it is still in a cancellable state.

        Args:
            run_id: UUID of the run to cancel.

        Returns:
            The updated ``OCRRun`` record, or ``None`` if not found.
        """
        run = await self._get_run(run_id)
        if run is None:
            return None
        if run.status not in _IN_FLIGHT:
            return run  # already terminal
        run.status = RunStatus.CANCELLED
        run.completed_at = datetime.now(UTC)
        await self._db.commit()
        return run

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _compute_run_hash(
        pdf_sha256: str,
        engine_slug: str,
        engine_version: str,
        config: dict | None,
    ) -> str:
        """SHA-256 of normalised run parameters for deduplication.

        The hash includes today's UTC date so that runs on different days
        produce different hashes (the same run re-submitted tomorrow gets
        a new execution).
        """
        canonical = json.dumps(config or {}, sort_keys=True, separators=(",", ":"))
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        raw = f"{pdf_sha256}|{engine_slug}|{engine_version}|{canonical}|{today}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _extract_engine_config(
        engine_record: object,
        provided_config: dict | None,
    ) -> dict:
        """Merge user-provided config with engine-level defaults from the DB.

        Args:
            engine_record: The ``OCREngine`` DB record with ``config_schema``.
            provided_config: User-supplied configuration dict.

        Returns:
            A merged dict of defaults overridden by user values.
        """
        schema: dict = getattr(engine_record, "config_schema", None) or {}
        defaults: dict[str, object] = {}
        for prop_name, prop_schema in schema.get("properties", {}).items():
            if isinstance(prop_schema, dict) and "default" in prop_schema:
                defaults[prop_name] = prop_schema["default"]
        return {**defaults, **(provided_config or {})}

    async def _store_results(
        self,
        run: OCRRun,
        raw_data: dict,
        normalized: dict,
    ) -> None:
        """Persist raw engine output and normalised page results.

        The raw output is stored in the content-addressable storage under the
        ``raw/`` prefix.  Each page of the normalised output becomes a
        ``PageResult`` record linked to the run.
        """
        # -- Raw output (content-addressed) ----------------------------------
        raw_bytes = json.dumps(raw_data, ensure_ascii=False).encode("utf-8")
        raw_hash = hashlib.sha256(raw_bytes).hexdigest()
        raw_path = self._storage.get_path(raw_hash, prefix="raw", ext="json")
        await anyio.to_thread.run_sync(
            partial(raw_path.parent.mkdir, parents=True, exist_ok=True),
        )
        await anyio.to_thread.run_sync(raw_path.write_bytes, raw_bytes)
        run.raw_output_uri = str(raw_path)

        # -- Page results ----------------------------------------------------
        for page_data in normalized.get("pages", []):
            page_result = PageResult(
                run_id=run.id,
                page_number=page_data["page_number"],
                width=page_data.get("width"),
                height=page_data.get("height"),
                data=page_data,
                confidence=page_data.get("confidence"),
            )
            self._db.add(page_result)

        run.status = RunStatus.COMPLETED
        run.completed_at = datetime.now(UTC)
        await self._db.commit()

    # ── Query helpers ────────────────────────────────────────────────────

    def _get_pdf_path(self, sha256: str) -> Path:
        """Return the on-disk path for a PDF given its SHA-256 hash."""
        return self._storage.get_path(sha256, prefix="pdfs", ext="pdf")

    async def _get_active_pdf(self, pdf_id: uuid.UUID) -> PDF | None:
        result = await self._db.execute(
            select(PDF).where(PDF.id == pdf_id, PDF.deleted_at.is_(None)),
        )
        return result.scalars().one_or_none()

    async def _get_engine_by_id(self, engine_id: uuid.UUID) -> object | None:
        from backend.models.engine import OCREngine  # noqa: PLC0415

        result = await self._db.execute(
            select(OCREngine).where(OCREngine.id == engine_id),
        )
        return result.scalars().one_or_none()

    async def _get_engine_by_slug(self, slug: str) -> object | None:
        from backend.models.engine import OCREngine  # noqa: PLC0415

        result = await self._db.execute(
            select(OCREngine).where(OCREngine.slug == slug),
        )
        return result.scalars().one_or_none()

    async def _get_run(self, run_id: uuid.UUID) -> OCRRun | None:
        result = await self._db.execute(
            select(OCRRun).where(OCRRun.id == run_id),
        )
        return result.scalars().one_or_none()

    async def _find_run_by_hash(
        self,
        run_hash: str,
        status: RunStatus,
    ) -> OCRRun | None:
        result = await self._db.execute(
            select(OCRRun).where(
                OCRRun.run_hash == run_hash,
                OCRRun.status == status,
            ),
        )
        return result.scalars().one_or_none()

    async def _find_run_by_hash_in_flight(self, run_hash: str) -> OCRRun | None:
        result = await self._db.execute(
            select(OCRRun).where(
                OCRRun.run_hash == run_hash,
                OCRRun.status.in_(_IN_FLIGHT),  # type: ignore[arg-type]
            ),
        )
        return result.scalars().one_or_none()

    @staticmethod
    def _add_log(run: OCRRun, level: str, message: str) -> None:
        """Append a structured log entry to the run's logs list.

        Reassigns the attribute (instead of mutating in-place) so that
        SQLAlchemy's change tracker detects the modification and persists
        it on the next ``commit()``.
        """
        entry: dict[str, str] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "message": message,
        }
        current = list(run.logs or [])
        current.append(entry)
        run.logs = current
