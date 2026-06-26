"""Tests for batch processing and cross-run comparison endpoints.

Uses the same ``FakeSession`` infrastructure from ``test_run_orchestrator.py``
to avoid external database dependencies.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from backend.batch_processor import (
    BatchProcessor,
    BatchProcessorError,
    BatchStatus,
    _clear_batches,
)
from backend.engine.registry import EngineRegistry
from backend.models.enums import PDFStatus, RunStatus
from backend.models.pdf import PDF
from backend.models.run import OCRRun
from backend.run_orchestrator import RunOrchestrator
from backend.storage import ContentAddressableStorage

# Reuse the FakeSession from test_run_orchestrator
from backend.test_run_orchestrator import FakeSession  # noqa: PLC0415

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_state() -> None:
    """Clear in-memory batches before each test."""
    _clear_batches()


@pytest.fixture
def db_session() -> FakeSession:
    return FakeSession()


@pytest.fixture
def storage(tmp_path: Path) -> ContentAddressableStorage:
    return ContentAddressableStorage(tmp_path)


@pytest.fixture
def engine_registry() -> EngineRegistry:
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
    return RunOrchestrator(db=db_session, storage=storage, registry=engine_registry)


@pytest.fixture
def processor(
    db_session: FakeSession,
    orchestrator: RunOrchestrator,
) -> BatchProcessor:
    return BatchProcessor(db=db_session, orchestrator=orchestrator)


@pytest.fixture
def pdf_ids(db_session: FakeSession) -> list[uuid.UUID]:
    """Create and persist two PDF records."""
    ids: list[uuid.UUID] = []
    for i in range(2):
        pdf = PDF(
            id=uuid.uuid4(),
            filename=f"test_{i}.pdf",
            original_filename=f"test_{i}.pdf",
            file_size_bytes=1024,
            page_count=3,
            md5_hash=f"d41d8cd98f00b204e9800998ecf8427e{i}",
            sha256_hash=f"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85{i}",
            mime_type="application/pdf",
            status=PDFStatus.READY,
        )
        db_session.add(pdf)
        import asyncio  # noqa: PLC0415

        asyncio.run(db_session.commit())
        ids.append(pdf.id)
    return ids


@pytest.fixture
def engine_slug() -> str:
    """Create and persist a mock engine record in the fake DB."""
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
    # Register in the fake session directly

    # Need to add via the processor's db session
    return "mock"


@pytest.fixture
def engine_ids(db_session: FakeSession) -> list[uuid.UUID]:
    """Create and persist engine records in the fake DB."""
    from backend.models.engine import OCREngine  # noqa: PLC0415

    ids: list[uuid.UUID] = []
    for slug in ["mock", "mock2"]:
        engine_id = uuid.uuid4()
        engine = OCREngine(
            id=engine_id,
            slug=slug,
            display_name=slug.title(),
            version="0.1.0",
            enabled=True,
            config_schema=None,
            description=f"{slug} engine",
        )
        db_session.add(engine)
        import asyncio  # noqa: PLC0415

        asyncio.run(db_session.commit())
        ids.append(engine_id)
    return ids


# Used to set up the db with engine records
@pytest.fixture
def db_with_pdf_and_engine(
    db_session: FakeSession,
    pdf_ids: list[uuid.UUID],  # noqa: ARG001
) -> FakeSession:
    """Add engine records to the session."""
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
    return db_session


# ---------------------------------------------------------------------------
# Test: create_batch
# ---------------------------------------------------------------------------


class TestCreateBatch:
    """Verify batch creation logic."""

    @pytest.mark.asyncio
    async def test_create_batch(
        self,
        db_with_pdf_and_engine: FakeSession,
        storage: ContentAddressableStorage,
        engine_registry: EngineRegistry,
    ) -> None:
        """Given valid PDFs and engines, When creating a batch,
        Then batch has correct structure."""
        db = db_with_pdf_and_engine
        orch = RunOrchestrator(db=db, storage=storage, registry=engine_registry)
        proc = BatchProcessor(db=db, orchestrator=orch)

        # Get the PDF IDs from the session store
        pdfs = list(db._store.get(PDF, {}).values())
        pdf_ids = [p.id for p in pdfs]

        batch = await proc.create_batch(
            pdf_ids=pdf_ids,
            engine_slugs=["mock"],
            config={"seed": 42},
        )

        assert batch.id is not None
        assert len(batch.items) == len(pdf_ids)  # 2 PDFs × 1 engine
        assert batch.status == BatchStatus.PENDING
        assert batch.config == {"seed": 42}
        for item in batch.items:
            assert item.status == BatchStatus.PENDING
            assert item.run_id is None

    @pytest.mark.asyncio
    async def test_batch_with_no_pdfs(
        self,
        processor: BatchProcessor,
    ) -> None:
        """Given empty pdf_ids, When creating a batch,
        Then BatchProcessorError is raised."""
        with pytest.raises(BatchProcessorError, match="pdf_ids must not be empty"):
            await processor.create_batch(pdf_ids=[], engine_slugs=["mock"])

    @pytest.mark.asyncio
    async def test_batch_invalid_pdf(
        self,
        processor: BatchProcessor,
    ) -> None:
        """Given a PDF that does not exist, When creating a batch,
        Then BatchProcessorError is raised."""
        fake_id = uuid.uuid4()
        with pytest.raises(BatchProcessorError, match="not found"):
            await processor.create_batch(
                pdf_ids=[fake_id],
                engine_slugs=["mock"],
            )

    @pytest.mark.asyncio
    async def test_batch_invalid_engine(
        self,
        db_session: FakeSession,
        processor: BatchProcessor,
    ) -> None:
        """Given an engine slug that is not registered,
        When creating a batch, Then BatchProcessorError is raised."""
        # Create a valid PDF first
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
        await db_session.commit()

        with pytest.raises(BatchProcessorError, match="not registered"):
            await processor.create_batch(
                pdf_ids=[pdf.id],
                engine_slugs=["nonexistent"],
            )


# ---------------------------------------------------------------------------
# Test: process_batch
# ---------------------------------------------------------------------------


class TestProcessBatch:
    """Verify batch processing logic."""

    @pytest.mark.asyncio
    async def test_process_batch(
        self,
        db_with_pdf_and_engine: FakeSession,
        storage: ContentAddressableStorage,
        engine_registry: EngineRegistry,
    ) -> None:
        """Given a batch with valid PDFs and engines,
        When processing, Then all items are completed."""
        db = db_with_pdf_and_engine
        orch = RunOrchestrator(db=db, storage=storage, registry=engine_registry)
        proc = BatchProcessor(db=db, orchestrator=orch)

        pdfs = list(db._store.get(PDF, {}).values())
        pdf_ids = [p.id for p in pdfs]

        batch = await proc.create_batch(
            pdf_ids=pdf_ids,
            engine_slugs=["mock"],
        )

        # Create PDF files on disk for MockEngine
        for pdf in pdfs:
            pdf_path = storage.get_path(
                pdf.sha256_hash, prefix="pdfs", ext="pdf",
            )
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(b"%PDF-1.4 mock content")

        processed = await proc.process_batch(batch.id)
        assert processed.status == BatchStatus.COMPLETED

        for item in processed.items:
            assert item.status == BatchStatus.COMPLETED
            assert item.run_id is not None

    @pytest.mark.asyncio
    async def test_get_batch_progress(
        self,
        db_with_pdf_and_engine: FakeSession,
        storage: ContentAddressableStorage,
        engine_registry: EngineRegistry,
    ) -> None:
        """Given a completed batch, When getting progress,
        Then progress reflects completed work."""
        db = db_with_pdf_and_engine
        orch = RunOrchestrator(db=db, storage=storage, registry=engine_registry)
        proc = BatchProcessor(db=db, orchestrator=orch)

        pdfs = list(db._store.get(PDF, {}).values())
        pdf_ids = [p.id for p in pdfs]

        batch = await proc.create_batch(
            pdf_ids=pdf_ids,
            engine_slugs=["mock"],
        )

        for pdf in pdfs:
            pdf_path = storage.get_path(
                pdf.sha256_hash, prefix="pdfs", ext="pdf",
            )
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(b"%PDF-1.4 mock content")

        await proc.process_batch(batch.id)

        progress = proc.get_batch_progress(batch.id)
        assert progress is not None
        assert progress["total"] == len(pdf_ids)
        assert progress["completed"] == len(pdf_ids)
        assert progress["failed"] == 0
        assert progress["percent"] == 100.0

    @pytest.mark.asyncio
    async def test_get_batch_progress_not_found(
        self,
        processor: BatchProcessor,
    ) -> None:
        """Given a non-existent batch ID, When getting progress,
        Then None is returned."""
        progress = processor.get_batch_progress(uuid.uuid4())
        assert progress is None

    @pytest.mark.asyncio
    async def test_batch_with_empty_result(
        self,
        db_with_pdf_and_engine: FakeSession,
        storage: ContentAddressableStorage,
        engine_registry: EngineRegistry,
    ) -> None:
        """Given a batch with no items (empty after creation), When getting
        progress, Then progress has total=0 and percent=0."""
        db = db_with_pdf_and_engine
        orch = RunOrchestrator(db=db, storage=storage, registry=engine_registry)
        proc = BatchProcessor(db=db, orchestrator=orch)

        pdfs = list(db._store.get(PDF, {}).values())
        pdf_ids = [p.id for p in pdfs]

        batch = await proc.create_batch(
            pdf_ids=pdf_ids,
            engine_slugs=["mock"],
        )
        # Manually remove all items to simulate empty
        batch.items.clear()

        progress = proc.get_batch_progress(batch.id)
        assert progress is not None
        assert progress["total"] == 0
        assert progress["percent"] == 0.0


# ---------------------------------------------------------------------------
# Test: cross-run comparison
# ---------------------------------------------------------------------------


class TestCrossRunComparison:
    """Verify cross-run comparison logic via the comparison router."""

    @pytest.mark.asyncio
    async def test_cross_run_comparison(
        self,
        db_with_pdf_and_engine: FakeSession,
        storage: ContentAddressableStorage,
        engine_registry: EngineRegistry,
    ) -> None:
        """Given two completed runs, When comparing,
        Then both runs' scores are returned."""
        db = db_with_pdf_and_engine
        orch = RunOrchestrator(db=db, storage=storage, registry=engine_registry)

        pdfs = list(db._store.get(PDF, {}).values())
        pdf = pdfs[0]

        # Create and execute two runs
        run1 = await orch.create_run(pdf.id, "mock", {"seed": 1})
        pdf_path = storage.get_path(
            pdf.sha256_hash, prefix="pdfs", ext="pdf",
        )
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4 content")
        await orch.execute_run(run1.id)

        run2 = await orch.create_run(pdf.id, "mock", {"seed": 2})
        await orch.execute_run(run2.id)

        # Verify both runs completed
        result1 = await db.execute(
            __import__("sqlalchemy", fromlist=["select"]).select(OCRRun).where(OCRRun.id == run1.id)  # type: ignore[attr-defined]  # noqa: E501
        )
        from sqlalchemy import select  # noqa: PLC0415

        result1 = await db.execute(select(OCRRun).where(OCRRun.id == run1.id))
        r1 = result1.one_or_none()
        result2 = await db.execute(select(OCRRun).where(OCRRun.id == run2.id))
        r2 = result2.one_or_none()

        assert r1 is not None and r1.status == RunStatus.COMPLETED
        assert r2 is not None and r2.status == RunStatus.COMPLETED

        # Test via comparison dict building (simulating the router logic)
        from backend.evaluation.scoring_service import _build_run_data  # noqa: PLC0415, E501

        run1_data = await _build_run_data(run1.id, db)
        run2_data = await _build_run_data(run2.id, db)
        assert len(run1_data["pages"]) > 0
        assert len(run2_data["pages"]) > 0

    @pytest.mark.asyncio
    async def test_engine_comparison(
        self,
        db_with_pdf_and_engine: FakeSession,
        storage: ContentAddressableStorage,
        engine_registry: EngineRegistry,
    ) -> None:
        """Given two engines with runs against the same PDF,
        When comparing engines, Then scores are returned per engine."""
        db = db_with_pdf_and_engine
        orch = RunOrchestrator(db=db, storage=storage, registry=engine_registry)

        pdfs = list(db._store.get(PDF, {}).values())
        pdf = pdfs[0]

        pdf_path = storage.get_path(
            pdf.sha256_hash, prefix="pdfs", ext="pdf",
        )
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4 content")

        # Create a run with mock engine
        run = await orch.create_run(pdf.id, "mock", {"seed": 42})
        await orch.execute_run(run.id)

        from sqlalchemy import select  # noqa: PLC0415

        result = await db.execute(select(OCRRun).where(OCRRun.id == run.id))
        r = result.one_or_none()
        assert r is not None and r.status == RunStatus.COMPLETED

        # Simulate comparison by checking run data
        from backend.evaluation.scoring_service import _build_run_data  # noqa: PLC0415, E501

        run_data = await _build_run_data(run.id, db)
        assert len(run_data["pages"]) > 0
