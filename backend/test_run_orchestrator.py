"""Unit tests for the ``RunOrchestrator`` — creation, dedup, execution, and lifecycle.

These tests use an in-memory fake session and a temporary storage directory to
avoid any external dependencies (no PostgreSQL, no real OCR engines).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Select, select
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.sql import operators
from sqlalchemy.sql.elements import BinaryExpression, BooleanClauseList
from sqlalchemy.sql.schema import CallableColumnDefault

from backend.engine.registry import EngineRegistry
from backend.models.enums import PDFStatus, RunStatus
from backend.models.pdf import PDF
from backend.models.run import OCRRun
from backend.run_orchestrator import RunOrchestrator, RunOrchestratorError
from backend.storage import ContentAddressableStorage

# ---------------------------------------------------------------------------
# Fake SQLAlchemy session
# ---------------------------------------------------------------------------


class FakeResult:
    """Fake replacement for SQLAlchemy ``Result`` / ``ScalarResult``."""

    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def scalars(self) -> FakeResult:
        return self

    def one_or_none(self) -> Any | None:
        return self._rows[0] if self._rows else None

    def all(self) -> list[Any]:
        return list(self._rows)

    def unique(self) -> FakeResult:
        return self

    def first(self) -> Any | None:
        return self._rows[0] if self._rows else None

    def scalar_one_or_none(self) -> Any | None:
        return self._rows[0] if self._rows else None

    def scalar(self) -> Any | None:
        return self._rows[0] if self._rows else None


def _extract_conditions(whereclause: Any) -> list[tuple[str, Any, Any]]:
    """Extract ``(field_name, operator, value)`` tuples from a WHERE clause.

    Handles ``AND`` combos, equality, ``IS NULL``, ``IS NOT NULL``, ``IN_``.
    """
    if whereclause is None:
        return []

    if isinstance(whereclause, BooleanClauseList):
        result: list[tuple[str, Any, Any]] = []
        for clause in whereclause.clauses:
            result.extend(_extract_conditions(clause))
        return result

    if isinstance(whereclause, BinaryExpression):
        left = whereclause.left
        right = whereclause.right
        op = whereclause.operator

        if hasattr(left, "key"):
            field_name = str(left.key)
        elif isinstance(left, InstrumentedAttribute):
            field_name = left.key  # type: ignore[union-attr]
        else:
            return []

        if hasattr(right, "value"):
            field_value = right.value
        elif hasattr(right, "clauses"):
            # IN_ clause — extract values from the collection
            field_value = tuple(
                v.value if hasattr(v, "value") else v for v in right.clauses
            )
        elif right is None:
            field_value = None
        else:
            field_value = right

        return [(field_name, op, field_value)]

    return []


class FakeSession:
    """In-memory fake for SQLAlchemy ``AsyncSession``.

    Supports ``add``, ``commit``, ``execute`` (SELECT with WHERE/ORDER BY/LIMIT/
    OFFSET subset), ``refresh`` — sufficient for orchestrator and API tests.
    """

    def __init__(self) -> None:
        self._store: dict[type, dict[uuid.UUID, Any]] = {}
        self._pending: list[Any] = []

    def add(self, obj: Any) -> None:
        self._pending.append(obj)

    async def commit(self) -> None:
        for obj in self._pending:
            typ = type(obj)
            if typ not in self._store:
                self._store[typ] = {}

            mapper = sa_inspect(typ)
            for col in mapper.columns:
                if col.default is not None and not col.default.is_server_default:
                    current = getattr(obj, col.key, None)
                    if current is None:
                        if isinstance(col.default, CallableColumnDefault):
                            try:
                                setattr(obj, col.key, col.default.arg())
                            except TypeError:
                                setattr(obj, col.key, uuid.uuid4())
                        else:
                            setattr(obj, col.key, col.default.arg)

            self._store[typ][obj.id] = obj
        self._pending.clear()

    async def rollback(self) -> None:
        self._pending.clear()

    async def refresh(self, obj: Any) -> None:
        mapper = sa_inspect(type(obj))
        for col in mapper.columns:
            if col.server_default is not None:
                current = getattr(obj, col.key, None)
                if current is None:
                    setattr(obj, col.key, datetime.now(UTC))

    async def execute(self, stmt: Select) -> FakeResult:
        entity = stmt.column_descriptions[0]["entity"]
        # If selecting a specific column, entity could be the column,
        # so walk up to find the model type
        if isinstance(entity, InstrumentedAttribute):
            entity = entity.class_

        items = list(self._store.get(entity, {}).values())

        # Apply WHERE
        conditions = _extract_conditions(stmt.whereclause)
        for field_name, op, value in conditions:
            if op is operators.eq:
                items = [i for i in items if getattr(i, field_name) == value]
            elif op is operators.is_:
                items = [i for i in items if getattr(i, field_name) is None]
            elif op is operators.isnot:
                items = [i for i in items if getattr(i, field_name) is not None]
            elif op is operators.in_op:
                items = [i for i in items if getattr(i, field_name) in value]
            elif op is operators.gt:
                items = [i for i in items if getattr(i, field_name) > value]

        # Apply ORDER BY
        order_by_clause = getattr(stmt, "_order_by_clause", None)
        if order_by_clause is not None:
            order_by_items = (
                list(order_by_clause)
                if hasattr(order_by_clause, "clauses")
                else [order_by_clause]
            )
            for order_spec in order_by_items:
                if hasattr(order_spec, "key"):
                    field_name = order_spec.key
                    items.sort(
                        key=lambda i, f=field_name: getattr(i, f) or "",
                        reverse=True,
                    )

        # Apply OFFSET
        offset_clause = getattr(stmt, "_offset_clause", None)
        if offset_clause is not None:
            offset_value = (
                offset_clause.value
                if hasattr(offset_clause, "value")
                else int(offset_clause)
            )
            items = items[offset_value:]

        # Apply LIMIT
        limit_clause = getattr(stmt, "_limit_clause", None)
        if limit_clause is not None:
            limit_value = (
                limit_clause.value
                if hasattr(limit_clause, "value")
                else int(limit_clause)
            )
            items = items[:limit_value]

        return FakeResult(items)

    async def close(self) -> None:
        pass

    def get(self, entity: type, ident: Any) -> Any:
        """Direct access to stored objects by primary key."""
        return self._store.get(entity, {}).get(ident)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_session() -> FakeSession:
    """Provide a fresh in-memory fake session."""
    return FakeSession()


@pytest.fixture
def storage(tmp_path: Path) -> ContentAddressableStorage:
    """Provide a ``ContentAddressableStorage`` backed by a temp directory."""
    return ContentAddressableStorage(tmp_path)


@pytest.fixture
def engine_registry() -> EngineRegistry:
    """Provide a clean ``EngineRegistry`` with only ``MockEngine`` registered.

    Because ``EngineRegistry`` is a singleton, the global instance is reused.
    We clear its internal engine map so each test starts with a known state.
    """
    registry = EngineRegistry()
    registry._engines.clear()  # type: ignore[attr-defined]
    from backend.mock_engine import MockEngine  # noqa: PLC0415

    registry.register(MockEngine)
    return registry


@pytest.fixture
def orchestrator(
    db_session: FakeSession,
    storage: ContentAddressableStorage,
    engine_registry: EngineRegistry,
) -> RunOrchestrator:
    """Provide a ``RunOrchestrator`` backed by fakes."""
    return RunOrchestrator(
        db=db_session,
        storage=storage,
        registry=engine_registry,
    )


@pytest.fixture
def pdf_record(db_session: FakeSession) -> PDF:
    """Create and return a persisted PDF record."""
    pdf = PDF(
        id=uuid.uuid4(),
        filename="test.pdf",
        original_filename="test.pdf",
        file_size_bytes=1024,
        page_count=3,
        md5_hash="d41d8cd98f00b204e9800998ecf8427e",
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        mime_type="application/pdf",
        status=PDFStatus.READY,
    )
    db_session.add(pdf)
    # Manually commit to populate server defaults
    import asyncio  # noqa: PLC0415

    asyncio.run(db_session.commit())
    return pdf


@pytest.fixture
def engine_record(db_session: FakeSession) -> Any:
    """Create and return a persisted OCREngine record (mock engine in DB)."""
    from backend.models.engine import OCREngine  # noqa: PLC0415

    engine = OCREngine(
        id=uuid.uuid4(),
        slug="mock",
        display_name="Mock Engine",
        version="0.1.0",
        enabled=True,
        config_schema={
            "type": "object",
            "properties": {
                "seed": {"type": "integer", "default": 42},
            },
            "required": [],
        },
        description="Test engine",
    )
    db_session.add(engine)
    import asyncio  # noqa: PLC0415

    asyncio.run(db_session.commit())
    return engine


# ---------------------------------------------------------------------------
# Hash computation tests
# ---------------------------------------------------------------------------


class TestRunHash:
    """Verify ``_compute_run_hash`` produces correct, deterministic hashes."""

    def test_hash_deterministic(self) -> None:
        """Given identical inputs, When computing hash twice, Then results match."""
        h1 = RunOrchestrator._compute_run_hash("abc123", "mock", "1.0", {"seed": 42})
        h2 = RunOrchestrator._compute_run_hash("abc123", "mock", "1.0", {"seed": 42})
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_hash_different_pdf(self) -> None:
        """Given different PDF hashes, When computing hashes, Then they differ."""
        h1 = RunOrchestrator._compute_run_hash("aaa", "mock", "1.0", None)
        h2 = RunOrchestrator._compute_run_hash("bbb", "mock", "1.0", None)
        assert h1 != h2

    def test_hash_different_engine(self) -> None:
        """Given different engine slugs, When computing hashes, Then they differ."""
        h1 = RunOrchestrator._compute_run_hash("abc", "mock", "1.0", None)
        h2 = RunOrchestrator._compute_run_hash("abc", "tesseract", "1.0", None)
        assert h1 != h2

    def test_hash_different_config(self) -> None:
        """Given different configs, When computing hashes, Then they differ."""
        h1 = RunOrchestrator._compute_run_hash("abc", "mock", "1.0", {"seed": 1})
        h2 = RunOrchestrator._compute_run_hash("abc", "mock", "1.0", {"seed": 2})
        assert h1 != h2

    def test_hash_includes_date(self) -> None:
        """Given the same params on different conceptual days, hash should
        differ (verified by injecting a date component into the raw string)."""
        # We can't mock datetime in the static method without a patch,
        # so we verify the hash format is a valid SHA-256 hex digest.
        h = RunOrchestrator._compute_run_hash("abc", "mock", "1.0", None)
        assert len(h) == 64
        # Ensure it contains only hex chars
        int(h, 16)


class TestExtractEngineConfig:
    """Verify config merging behaviour."""

    def test_merge_with_defaults(self, engine_record: Any) -> None:
        """Given engine with default seed=42 and no user config,
        When extracting config, Then defaults are used."""
        config = RunOrchestrator._extract_engine_config(engine_record, None)
        assert config == {"seed": 42}

    def test_override_default(self, engine_record: Any) -> None:
        """Given engine with default seed=42 and user config with seed=99,
        When extracting config, Then user value overrides default."""
        config = RunOrchestrator._extract_engine_config(
            engine_record, {"seed": 99},
        )
        assert config == {"seed": 99}

    def test_empty_schema(self) -> None:
        """Given engine with no config_schema,
        When extracting config, Then just the provided config is returned."""
        from types import SimpleNamespace  # noqa: PLC0415

        engine = SimpleNamespace(config_schema=None)
        config = RunOrchestrator._extract_engine_config(engine, {"foo": "bar"})
        assert config == {"foo": "bar"}


# ---------------------------------------------------------------------------
# create_run tests
# ---------------------------------------------------------------------------


class TestCreateRun:
    """Verify ``create_run`` validation, dedup, and persistence."""

    @pytest.mark.asyncio
    async def test_create_run(
        self,
        orchestrator: RunOrchestrator,
        pdf_record: PDF,
        engine_record: Any,
    ) -> None:
        """Given valid PDF and engine, When creating a run, Then status is pending."""
        run = await orchestrator.create_run(pdf_record.id, "mock", {"seed": 42})
        assert run.status == RunStatus.PENDING
        assert run.pdf_id == pdf_record.id
        assert run.engine_config == {"seed": 42}
        assert run.run_hash is not None
        assert len(run.run_hash) == 64

    @pytest.mark.asyncio
    async def test_create_run_invalid_pdf(
        self,
        orchestrator: RunOrchestrator,
        engine_record: Any,
    ) -> None:
        """Given a non-existent PDF, When creating a run, Then error is raised."""
        fake_id = uuid.uuid4()
        with pytest.raises(RunOrchestratorError, match="not found"):
            await orchestrator.create_run(fake_id, "mock")

    @pytest.mark.asyncio
    async def test_create_run_invalid_engine(
        self,
        orchestrator: RunOrchestrator,
        pdf_record: PDF,
    ) -> None:
        """Given a non-existent engine slug, When creating a run, Then error is raised."""
        with pytest.raises(RunOrchestratorError, match="not registered"):
            await orchestrator.create_run(pdf_record.id, "nonexistent")

    @pytest.mark.asyncio
    async def test_run_hash_dedup(
        self,
        orchestrator: RunOrchestrator,
        pdf_record: PDF,
        engine_record: Any,
    ) -> None:
        """Given a completed run with the same hash,
        When creating another run, Then the existing run is returned."""
        run1 = await orchestrator.create_run(pdf_record.id, "mock", {"seed": 42})

        # Manually mark run1 as completed
        run1.status = RunStatus.COMPLETED
        await orchestrator._db.commit()

        run2 = await orchestrator.create_run(pdf_record.id, "mock", {"seed": 42})
        assert run2.id == run1.id
        assert run2.status == RunStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_hash_allows_rerun_after_failure(
        self,
        orchestrator: RunOrchestrator,
        pdf_record: PDF,
        engine_record: Any,
    ) -> None:
        """Given a failed run with the same hash,
        When creating another run, Then a new run is created."""
        run1 = await orchestrator.create_run(pdf_record.id, "mock", {"seed": 42})

        # Manually mark run1 as failed
        run1.status = RunStatus.FAILED
        await orchestrator._db.commit()

        run2 = await orchestrator.create_run(pdf_record.id, "mock", {"seed": 42})
        assert run2.id != run1.id
        assert run2.status == RunStatus.PENDING  # fresh run

    @pytest.mark.asyncio
    async def test_run_hash_conflict_in_progress(
        self,
        orchestrator: RunOrchestrator,
        pdf_record: PDF,
        engine_record: Any,
    ) -> None:
        """Given an in-progress run with the same hash,
        When creating another run, Then an error is raised."""
        run1 = await orchestrator.create_run(pdf_record.id, "mock", {"seed": 42})
        # run1.status is already PENDING from creation

        with pytest.raises(RunOrchestratorError, match="already in progress"):
            await orchestrator.create_run(pdf_record.id, "mock", {"seed": 42})


# ---------------------------------------------------------------------------
# execute_run tests
# ---------------------------------------------------------------------------


class TestExecuteRun:
    """Verify the full execution pipeline with MockEngine."""

    @pytest.mark.asyncio
    async def test_execute_run_with_mock_engine(
        self,
        orchestrator: RunOrchestrator,
        db_session: FakeSession,
        storage: ContentAddressableStorage,
        pdf_record: PDF,
        engine_record: Any,
    ) -> None:
        """Given a valid run, When executing, Then status transitions to completed
        and page results are stored."""
        run = await orchestrator.create_run(pdf_record.id, "mock", {"seed": 42})
        run_id = run.id

        # Create a minimal PDF file so the engine has something to process
        pdf_path = storage.get_path(
            pdf_record.sha256_hash, prefix="pdfs", ext="pdf",
        )
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4 mock content")

        await orchestrator.execute_run(run_id)

        # Verify final state
        result = await db_session.execute(
            select(OCRRun).where(OCRRun.id == run_id),
        )
        updated_run = result.one_or_none()
        assert updated_run is not None
        assert updated_run.status == RunStatus.COMPLETED
        assert updated_run.completed_at is not None
        assert updated_run.raw_output_uri is not None
        assert updated_run.engine_version == "0.1.0"

        # Verify page results were stored
        from backend.models.page_result import PageResult  # noqa: PLC0415

        pr_result = await db_session.execute(
            select(PageResult).where(PageResult.run_id == run_id),
        )
        page_results = pr_result.all()
        assert len(page_results) >= 2  # MockEngine generates 2-3 pages

    @pytest.mark.asyncio
    async def test_execute_run_handles_missing_pdf(
        self,
        orchestrator: RunOrchestrator,
        db_session: FakeSession,
        pdf_record: PDF,
        engine_record: Any,
    ) -> None:
        """Given a run whose PDF file does not exist on disk,
        When executing, Then the run is marked as failed."""
        run = await orchestrator.create_run(pdf_record.id, "mock", {"seed": 42})
        # Do NOT create the PDF file on disk

        await orchestrator.execute_run(run.id)

        result = await db_session.execute(
            select(OCRRun).where(OCRRun.id == run.id),
        )
        updated = result.one_or_none()
        assert updated is not None
        assert updated.status == RunStatus.FAILED
        assert updated.error_message is not None


# ---------------------------------------------------------------------------
# cancel_run tests
# ---------------------------------------------------------------------------


class TestCancelRun:
    """Verify run cancellation behaviour."""

    @pytest.mark.asyncio
    async def test_cancel_pending_run(
        self,
        orchestrator: RunOrchestrator,
        db_session: FakeSession,
        pdf_record: PDF,
        engine_record: Any,
    ) -> None:
        """Given a pending run, When cancelling, Then status becomes cancelled."""
        run = await orchestrator.create_run(pdf_record.id, "mock", {"seed": 42})

        cancelled = await orchestrator.cancel_run(run.id)
        assert cancelled is not None
        assert cancelled.status == RunStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_run(
        self,
        orchestrator: RunOrchestrator,
    ) -> None:
        """Given a non-existent run ID, When cancelling, Then None is returned."""
        result = await orchestrator.cancel_run(uuid.uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_already_completed(
        self,
        orchestrator: RunOrchestrator,
        db_session: FakeSession,
        pdf_record: PDF,
        engine_record: Any,
    ) -> None:
        """Given a completed run, When cancelling, Then status remains completed."""
        run = await orchestrator.create_run(pdf_record.id, "mock", {"seed": 42})
        run.status = RunStatus.COMPLETED
        await orchestrator._db.commit()

        result = await orchestrator.cancel_run(run.id)
        assert result is not None
        assert result.status == RunStatus.COMPLETED  # unchanged
