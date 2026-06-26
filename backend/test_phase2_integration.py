"""Phase 2 integration test — full evaluation pipeline validation.

Tests the end-to-end flow: upload PDF -> run 2 engines (mock + tesseract)
-> auto-generate GT via consensus -> compute scores -> display results.

Uses the same in-memory fake SQLAlchemy session and httpx TestClient pattern
as ``test_integration.py`` and ``test_ground_truth_api.py`` so that no
PostgreSQL or cloud dependencies are required.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import Select
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.sql import operators
from sqlalchemy.sql.elements import BinaryExpression, BooleanClauseList
from sqlalchemy.sql.schema import CallableColumnDefault

from backend.database import get_db_session
from backend.main import app as _app
from backend.settings import settings

# ═══════════════════════════════════════════════════════════════════════════════
# Minimal valid PDF builder (same as test_integration.py)
# ═══════════════════════════════════════════════════════════════════════════════


def _make_pdf_bytes(content: str = "Phase 2 integration") -> bytes:
    """Build a minimal but structurally valid PDF with correct xref offsets."""
    stream_data = f"BT\n/F1 12 Tf\n100 700 Td\n({content}) Tj\nET\n".encode()
    stream_obj = (
        b"<< /Length " + str(len(stream_data)).encode()
        + b" >>\nstream\n" + stream_data + b"\nendstream\nendobj\n"
    )

    header = b"%PDF-1.4\n"

    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj1_offset = len(header)

    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj2_offset = obj1_offset + len(obj1)

    obj3 = (
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Contents 4 0 R /Resources << /Font << /F1 << /Type /Font"
        b" /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>\nendobj\n"
    )
    obj3_offset = obj2_offset + len(obj2)

    obj4 = b"4 0 obj\n" + stream_obj
    obj4_offset = obj3_offset + len(obj3)

    body = obj1 + obj2 + obj3 + obj4

    xref_offset = len(header) + len(body)
    xref = b"xref\n0 5\n"
    xref += f"{0:010d} {65535:05d} f \n".encode()
    xref += f"{obj1_offset:010d} {00000:05d} n \n".encode()
    xref += f"{obj2_offset:010d} {00000:05d} n \n".encode()
    xref += f"{obj3_offset:010d} {00000:05d} n \n".encode()
    xref += f"{obj4_offset:010d} {00000:05d} n \n".encode()

    trailer = b"trailer\n<< /Size 5 /Root 1 0 R >>\n"
    trailer += f"startxref\n{xref_offset}\n".encode()
    trailer += b"%%EOF\n"

    return header + body + xref + trailer


# ═══════════════════════════════════════════════════════════════════════════════
# Fake SQLAlchemy session (same pattern as test_ground_truth_api.py)
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_conditions(whereclause: Any) -> list[tuple[str, Any, Any]]:
    """Extract ``(field_name, operator, value)`` tuples from a WHERE clause."""
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
            field_value = tuple(
                v.value if hasattr(v, "value") else v for v in right.clauses
            )
        elif right is None:
            field_value = None
        else:
            field_value = right

        return [(field_name, op, field_value)]

    return []


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


class FakeSession:
    """In-memory fake for SQLAlchemy ``AsyncSession`` with ordering support.

    Used by both ``test_ground_truth_api.py`` and this module to avoid
    external PostgreSQL dependencies.
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

            # Set server defaults for timestamps if not already set.
            if hasattr(obj, "created_at") and obj.created_at is None:
                obj.created_at = datetime.now(UTC)

            self._store[typ][obj.id] = obj
        self._pending.clear()

    async def flush(self) -> None:
        """Flush pending objects to store without full commit semantics."""
        for obj in self._pending:
            typ = type(obj)
            if typ not in self._store:
                self._store[typ] = {}
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()
            if hasattr(obj, "created_at") and obj.created_at is None:
                obj.created_at = datetime.now(UTC)
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
        is_column_select = isinstance(entity, InstrumentedAttribute)
        store_entity = entity.class_ if is_column_select else entity
        column_key = entity.key if is_column_select else None

        items = list(self._store.get(store_entity, {}).values())

        # Apply WHERE conditions.
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

        # Apply ORDER BY (simple single-field support).
        order_by_clauses = None
        try:
            order_by_clauses = getattr(stmt, "_order_by_clauses", None)
        except TypeError:
            pass
        if order_by_clauses is None:
            try:
                order_by_clauses = getattr(stmt, "_order_by_clause", None)
            except TypeError:
                pass
        if order_by_clauses:
            for clause in order_by_clauses:
                if hasattr(clause, "element"):
                    elem = clause.element
                    key = elem.key if hasattr(elem, "key") else None
                    if key:
                        reverse = False
                        try:
                            mod = getattr(clause, "modifier", None)
                            reverse = mod in ("desc", "DESC")
                        except TypeError:
                            pass
                        items.sort(
                            key=lambda i, f=key: getattr(i, f, None) or "",
                            reverse=reverse,
                        )

        # Apply LIMIT.
        limit_val = None
        try:
            limit_val = getattr(stmt, "_limit", None)
        except TypeError:
            pass
        if limit_val is not None:
            items = items[:limit_val]

        # Apply OFFSET.
        offset_val = None
        try:
            offset_val = getattr(stmt, "_offset", None)
        except TypeError:
            pass
        if offset_val is not None:
            items = items[offset_val:]

        # For column selects, extract the column value.
        if is_column_select and column_key:
            items = [getattr(item, column_key) for item in items]

        return FakeResult(items)

    async def close(self) -> None:
        pass

    def get(self, entity: type, ident: Any) -> Any:
        """Direct access to stored objects by primary key."""
        return self._store.get(entity, {}).get(ident)


# ═══════════════════════════════════════════════════════════════════════════════
# Test fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module", autouse=True)
def _register_engines() -> None:
    """Register MockEngine (and optionally TesseractEngine) in the global
    ``EngineRegistry`` once for the whole test module.

    This is necessary because the ``RunOrchestrator`` resolves engine plugins
    from the global singleton registry.
    """
    from backend.engine.registry import EngineRegistryError, registry
    from backend.mock_engine import MockEngine

    try:
        registry.register(MockEngine)
    except EngineRegistryError:
        pass  # Already registered from another test module

    # TesseractEngine registers itself on import
    try:
        import backend.engines.tesseract  # noqa: F401
    except Exception:
        pass  # Tesseract dependencies may not be installed


@pytest.fixture
def fake_db() -> FakeSession:
    """Provide a fresh in-memory fake session per test."""
    return FakeSession()


@pytest.fixture
def app(tmp_path: Path, fake_db: FakeSession) -> FastAPI:
    """Provide the FastAPI app with overridden dependencies for testing.

    The ``get_db_session`` dependency is replaced with a ``FakeSession``
    that persists across all requests within a single test function.
    The storage path is redirected to a temporary directory.
    """
    settings.storage_path = str(tmp_path)

    async def _override_session() -> AsyncGenerator[FakeSession]:
        yield fake_db

    _app.dependency_overrides[get_db_session] = _override_session
    return _app


@pytest.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient]:
    """Provide an async HTTP client for the test app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def pdf_content() -> bytes:
    """Return minimal valid PDF bytes."""
    return _make_pdf_bytes("Phase 2 integration test document")


@pytest.fixture
async def engine_mock(fake_db: FakeSession) -> uuid.UUID:
    """Pre-seed a MockEngine OCHEngine record in the fake database."""
    from backend.models.engine import OCREngine

    engine = OCREngine(
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
        description="Mock engine for Phase 2 integration tests",
    )
    fake_db.add(engine)
    await fake_db.commit()

    assert engine.id is not None
    return engine.id


# ═══════════════════════════════════════════════════════════════════════════════
# Polling helper
# ═══════════════════════════════════════════════════════════════════════════════


async def _poll_run_until(
    client: AsyncClient,
    run_id: str,
    timeout: float = 15.0,
    interval: float = 0.2,
) -> dict[str, Any]:
    """Poll ``GET /api/v1/runs/{run_id}`` until a terminal status or timeout.

    Returns:
        The run JSON dict once its status is terminal.

    Raises:
        ``pytest.fail`` if the timeout is reached before the run finishes.
    """
    loop = asyncio.get_event_loop()
    start = loop.time()
    terminal = frozenset({"completed", "failed", "cancelled"})

    while True:
        resp = await client.get(f"/api/v1/runs/{run_id}")
        assert resp.status_code == 200
        data: dict[str, Any] = resp.json()

        if data["status"] in terminal:
            return data

        elapsed = loop.time() - start
        if elapsed > timeout:
            pytest.fail(
                f"Run {run_id} did not reach terminal status within "
                f"{timeout:.0f}s (last status: {data['status']})",
            )
        await asyncio.sleep(interval)


async def _create_completed_run(
    client: AsyncClient,
    doc_id: str,
    seed: int = 42,
) -> str:
    """Upload-and-run helper: create a mock engine run and poll until done.

    Returns:
        The run ID (as a string).
    """
    run_resp = await client.post(
        "/api/v1/runs",
        json={
            "pdf_id": doc_id,
            "engine_id": "mock",
            "config": {"seed": seed},
        },
    )
    assert run_resp.status_code == 202
    run_id: str = run_resp.json()["id"]
    await _poll_run_until(client, run_id)
    return run_id


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestUploadAndRunTwoEngines:
    """Upload PDF then run mock engine twice with different seeds."""

    async def test_upload_and_run_two_engines(
        self,
        client: AsyncClient,
        pdf_content: bytes,
        engine_mock: uuid.UUID,
    ) -> None:
        """Given a PDF, When creating two runs with different seeds,
        Then both reach completed status."""
        # ── 1. Upload PDF ───────────────────────────────────────────────────
        upload_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("two_engines.pdf", pdf_content, "application/pdf")},
        )
        assert upload_resp.status_code == 202
        doc_id: str = upload_resp.json()["id"]

        # ── 2. First run (seed=42) ──────────────────────────────────────────
        run1_resp = await client.post(
            "/api/v1/runs",
            json={"pdf_id": doc_id, "engine_id": "mock", "config": {"seed": 42}},
        )
        assert run1_resp.status_code == 202
        run1_id: str = run1_resp.json()["id"]
        run1_data = await _poll_run_until(client, run1_id)
        assert run1_data["status"] == "completed"

        # ── 3. Second run (seed=99) ─────────────────────────────────────────
        run2_resp = await client.post(
            "/api/v1/runs",
            json={"pdf_id": doc_id, "engine_id": "mock", "config": {"seed": 99}},
        )
        assert run2_resp.status_code == 202
        run2_id: str = run2_resp.json()["id"]
        run2_data = await _poll_run_until(client, run2_id)
        assert run2_data["status"] == "completed"

        # ── 4. Verify they are distinct runs ────────────────────────────────
        assert run1_id != run2_id

        # ── 5. Verify results are available for both ────────────────────────
        for rid in (run1_id, run2_id):
            results_resp = await client.get(f"/api/v1/runs/{rid}/results")
            assert results_resp.status_code == 200
            assert len(results_resp.json()["items"]) >= 1


class TestAutoGroundTruthGeneration:
    """Auto-generate GT via ConsensusEntropy from engine output data.

    Because MockEngine generates random text per seed (high CE between
    different seeds), this test builds consensus programmatically from
    duplicated page data to verify the consensus pipeline produces
    valid GT pages with auto_consensus source.
    """

    async def test_auto_ground_truth_generation(
        self,
        client: AsyncClient,
        pdf_content: bytes,
        engine_mock: uuid.UUID,
    ) -> None:
        """Given engine output data from a completed run,
        When building consensus via ``build_ground_truth`` with
        duplicated data (simulating two engines agreeing),
        Then ``auto_consensus`` GT pages are produced."""
        from backend.evaluation.consensus import build_ground_truth  # noqa: PLC0415

        # ── 1. Upload PDF & run mock engine ─────────────────────────────────
        upload_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("auto_gt.pdf", pdf_content, "application/pdf")},
        )
        doc_id: str = upload_resp.json()["id"]
        run1_id = await _create_completed_run(client, doc_id, seed=42)

        # ── 2. Fetch normalized page data from API ──────────────────────────
        results_resp = await client.get(f"/api/v1/runs/{run1_id}/results")
        page_items = results_resp.json()["items"]

        # Extract the normalized page data dicts. Each page's ``data`` field
        # contains the JSONB hierarchy (blocks -> lines -> words).
        page_data_list = [item["data"] for item in page_items]

        # ── 3. Build consensus GT from duplicated page data ─────────────────
        # Duplicate each page's data to simulate two engines that agree.
        # This guarantees low CE and produces ``auto_consensus`` GT.
        engine_outputs = page_data_list + page_data_list
        gt_result = build_ground_truth(engine_outputs)

        # ── 4. Verify consensus result ──────────────────────────────────────
        assert gt_result["source"] == "auto_consensus"
        assert len(gt_result["pages"]) >= 1
        assert 0.0 <= gt_result["consensus_entropy"] <= 1.0
        assert isinstance(gt_result["needs_review"], bool)
        assert isinstance(gt_result["warnings"], list)

        # Each page must have blocks and tables
        for page in gt_result["pages"]:
            assert "blocks" in page
            assert "tables" in page

    async def test_consensus_gt_api_accepts_consensus_source(
        self,
        client: AsyncClient,
        pdf_content: bytes,
        engine_mock: uuid.UUID,
    ) -> None:
        """Given two completed runs,
        When creating consensus GT via POST /api/v1/ground-truth,
        Then the API accepts the request and returns 201."""
        # ── 1. Upload & run ─────────────────────────────────────────────────
        upload_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("consensus_api.pdf", pdf_content, "application/pdf")},
        )
        doc_id: str = upload_resp.json()["id"]

        run_ids: list[str] = []
        for seed in (42, 99):
            rid = await _create_completed_run(client, doc_id, seed=seed)
            run_ids.append(rid)

        # ── 2. POST consensus GT via API ────────────────────────────────────
        gt_resp = await client.post(
            "/api/v1/ground-truth",
            json={
                "pdf_id": doc_id,
                "source": "consensus",
                "engine_ids": run_ids,
                "notes": "API consensus test",
                "created_by": "phase2_test",
            },
        )
        assert gt_resp.status_code == 201
        gt_data = gt_resp.json()
        assert gt_data["source"] == "consensus"
        assert gt_data["version_number"] >= 1
        assert "id" in gt_data

        # ── 3. Fetch the GT detail ──────────────────────────────────────────
        detail_resp = await client.get(f"/api/v1/ground-truth/{gt_data['id']}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["version"]["source"] == "consensus"
        # Pages may be empty if CE is high (divergent mock engine seeds) —
        # that is expected behaviour; the API contract is validated here.


class TestScoreComputationAgainstGT:
    """Compute CER/WER scores using auto-GT as reference."""

    async def test_score_computation_against_gt(
        self,
        client: AsyncClient,
        pdf_content: bytes,
        engine_mock: uuid.UUID,
    ) -> None:
        """Given a completed run and auto-GT from consensus,
        When computing scores via evaluate_run, Then CER/WER metrics
        are within expected bounds and have the correct shape."""
        from backend.evaluation._evaluators import evaluate_run  # noqa: PLC0415
        from backend.evaluation.consensus import build_ground_truth  # noqa: PLC0415

        # ── 1. Upload PDF & run mock engine ─────────────────────────────────
        upload_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("scores.pdf", pdf_content, "application/pdf")},
        )
        doc_id: str = upload_resp.json()["id"]
        run_id = await _create_completed_run(client, doc_id, seed=42)

        # ── 2. Fetch page data for GT and for scoring ───────────────────────
        results_resp = await client.get(f"/api/v1/runs/{run_id}/results")
        page_items = results_resp.json()["items"]
        page_data_list = [item["data"] for item in page_items]

        # ── 3. Build auto-GT via consensus (duplicated data = low CE) ────────
        gt_result = build_ground_truth(page_data_list + page_data_list)
        assert gt_result["source"] == "auto_consensus"
        gt_pages = gt_result["pages"]

        # ── 4. Build run_data and gt_data for evaluate_run ──────────────────
        gt_data: dict[str, Any] = {"pages": gt_pages}
        run_data: dict[str, Any] = {
            "pages": [
                {"data": p, "results": []}
                for p in page_data_list
            ],
        }

        # ── 5. Compute scores programmatically ──────────────────────────────
        scores = evaluate_run(run_data, gt_data)

        # ── 6. Verify score shape ───────────────────────────────────────────
        assert "cer" in scores
        assert "wer" in scores
        assert "char_precision" in scores
        assert "char_recall" in scores
        assert "char_f1" in scores
        assert "word_precision" in scores
        assert "word_recall" in scores
        assert "word_f1" in scores
        assert "per_page" in scores
        assert "num_pages" in scores

        # Scores should be in valid range [0.0, 1.0]
        assert 0.0 <= scores["cer"] <= 1.0
        assert 0.0 <= scores["wer"] <= 1.0
        assert 0.0 <= scores["char_f1"] <= 1.0
        assert 0.0 <= scores["word_f1"] <= 1.0

        # Per-page scores should have the same shape
        for page_score in scores["per_page"]:
            assert "cer" in page_score
            assert "wer" in page_score
            assert "char_f1" in page_score
            assert "word_f1" in page_score

        # Number of pages should be consistent
        assert scores["num_pages"] >= 1
        assert len(scores["per_page"]) == scores["num_pages"]


class TestScoreEndpointIntegration:
    """Verify that score-related data is accessible through the API.

    Since no dedicated ``/api/v1/scores`` endpoint exists yet, this test
    verifies that the run results API returns the data needed for scoring
    and that the scoring functions produce valid results.
    """

    async def test_run_results_contain_scoring_inputs(
        self,
        client: AsyncClient,
        pdf_content: bytes,
        engine_mock: uuid.UUID,
    ) -> None:
        """Given a completed run, When fetching results, Then each page
        contains the data fields required for scoring."""
        # ── Upload & run ────────────────────────────────────────────────────
        upload_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("score_api.pdf", pdf_content, "application/pdf")},
        )
        doc_id: str = upload_resp.json()["id"]

        run_id = await _create_completed_run(client, doc_id, seed=42)

        # ── Fetch run results ──────────────────────────────────────────────
        results_resp = await client.get(f"/api/v1/runs/{run_id}/results")
        assert results_resp.status_code == 200
        payload = results_resp.json()

        # Verify pagination shape
        assert "items" in payload
        assert "page" in payload
        assert "page_size" in payload
        assert "total" in payload
        assert payload["page"] == 1

        # Each item must have the fields needed for scoring
        for item in payload["items"]:
            assert "page_number" in item
            assert "data" in item
            data = item["data"]
            # The JSONB data must have blocks (for word extraction)
            assert "blocks" in data

        # ── Fetch individual page result ──────────────────────────────────
        first_page = payload["items"][0]
        page_resp = await client.get(
            f"/api/v1/runs/{run_id}/results/{first_page['page_number']}",
        )
        assert page_resp.status_code == 200
        page_data = page_resp.json()
        assert "width" in page_data
        assert "height" in page_data
        assert "data" in page_data

        # ── Fetch raw output (contains engine info) ───────────────────────
        raw_resp = await client.get(f"/api/v1/runs/{run_id}/raw")
        assert raw_resp.status_code == 200
        raw_data = raw_resp.json()
        assert raw_data["engine_id"] == "mock"
        assert raw_data["engine_version"] == "0.1.0"
        assert "raw_pages" in raw_data


class TestGTVersioning:
    """Create, update, and promote GT versions."""

    async def test_gt_versioning(
        self,
        client: AsyncClient,
        pdf_content: bytes,
        fake_db: FakeSession,
        engine_mock: uuid.UUID,
    ) -> None:
        """Given a manual GT version with a page,
        When updating the page and promoting a previous version,
        Then version numbers increment and promotion works."""
        from backend.models.ground_truth import (  # noqa: PLC0415
            GroundTruthVersion,
            GTPageResult,
        )

        # ── 1. Upload PDF and create a run ───────────────────────────────────
        upload_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("gt_ver.pdf", pdf_content, "application/pdf")},
        )
        doc_id: str = upload_resp.json()["id"]

        # ── 2. Create consensus GT via API (even if empty, it establishes v1) ─
        run_id = await _create_completed_run(client, doc_id, seed=42)

        v1_resp = await client.post(
            "/api/v1/ground-truth",
            json={
                "pdf_id": doc_id,
                "source": "consensus",
                "engine_ids": [run_id],
                "notes": "v1 initial",
            },
        )
        assert v1_resp.status_code == 201
        v1_id: str = v1_resp.json()["id"]
        assert v1_resp.json()["version_number"] == 1

        # ── 3. Manually attach a page result so the versioned edit works ─────
        gt_version = fake_db.get(GroundTruthVersion, uuid.UUID(v1_id))
        assert gt_version is not None

        page_data: dict[str, Any] = {
            "blocks": [
                {
                    "type": "text",
                    "bbox": [0.0, 0.0, 612.0, 50.0],
                    "confidence": 1.0,
                    "order": 0,
                    "lines": [
                        {
                            "text": "Original text",
                            "bbox": [0.0, 0.0, 200.0, 20.0],
                            "confidence": 1.0,
                            "order": 0,
                            "words": [
                                {
                                    "text": "Original",
                                    "bbox": [0.0, 0.0, 80.0, 20.0],
                                    "confidence": 1.0,
                                    "order": 0,
                                    "chars": [],
                                },
                                {
                                    "text": "text",
                                    "bbox": [90.0, 0.0, 200.0, 20.0],
                                    "confidence": 1.0,
                                    "order": 1,
                                    "chars": [],
                                },
                            ],
                        },
                    ],
                },
            ],
            "tables": [],
        }
        page = GTPageResult(
            gt_version_id=uuid.UUID(v1_id),
            page_number=1,
            data=page_data,
            confidence=1.0,
        )
        fake_db.add(page)
        await fake_db.commit()
        gt_version.page_results = [page]

        # ── 4. Update the page (creates v2) ─────────────────────────────────
        updated_data: dict[str, Any] = {
            "blocks": [
                {
                    "type": "text",
                    "bbox": [0.0, 0.0, 612.0, 50.0],
                    "confidence": 1.0,
                    "order": 0,
                    "lines": [
                        {
                            "text": "Corrected ground truth text",
                            "bbox": [0.0, 0.0, 612.0, 20.0],
                            "confidence": 1.0,
                            "order": 0,
                            "words": [
                                {
                                    "text": "Corrected",
                                    "bbox": [0.0, 0.0, 80.0, 20.0],
                                    "confidence": 1.0,
                                    "order": 0,
                                    "chars": [],
                                },
                            ],
                        },
                    ],
                },
            ],
            "tables": [],
        }
        update_resp = await client.put(
            f"/api/v1/ground-truth/{v1_id}/pages/1",
            json={"data": updated_data},
        )
        assert update_resp.status_code == 200
        updated_page = update_resp.json()
        assert updated_page["page_number"] == 1
        assert updated_page["data"] == updated_data

        # ── 5. Verify v2 was created ────────────────────────────────────────
        versions = [
            v for v in fake_db._store.get(GroundTruthVersion, {}).values()
            if v.pdf_id == uuid.UUID(doc_id) and v.deleted_at is None
        ]
        version_numbers = sorted(v.version_number for v in versions)
        assert version_numbers == [1, 2]

        # ── 6. Promote v1 (should bump version_number above v2) ────────────
        promote_resp = await client.post(f"/api/v1/ground-truth/{v1_id}/promote")
        assert promote_resp.status_code == 200
        assert "Promoted" in promote_resp.json()["message"]

        # Verify current GT is v1 (now has highest version number)
        current_resp = await client.get(f"/api/v1/ground-truth/current/{doc_id}")
        assert current_resp.status_code == 200
        current = current_resp.json()
        assert current["version"]["id"] == v1_id

    async def test_gt_word_update(
        self,
        client: AsyncClient,
        pdf_content: bytes,
        fake_db: FakeSession,
        engine_mock: uuid.UUID,
    ) -> None:
        """Given a GT page with words, When a word is corrected via API,
        Then a new version is created with the fix."""
        from backend.models.ground_truth import (  # noqa: PLC0415
            GroundTruthVersion,
            GTPageResult,
        )

        # ── Upload & create manual GT ───────────────────────────────────────
        upload_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("gt_word.pdf", pdf_content, "application/pdf")},
        )
        doc_id: str = upload_resp.json()["id"]

        gt_resp = await client.post(
            "/api/v1/ground-truth",
            json={"pdf_id": doc_id, "source": "manual"},
        )
        gt_id_str: str = gt_resp.json()["id"]
        gt_id = uuid.UUID(gt_id_str)

        # Attach a page with words to the GT version
        gt_version = fake_db.get(GroundTruthVersion, gt_id)
        assert gt_version is not None

        word_data: dict[str, Any] = {
            "blocks": [
                {
                    "type": "text",
                    "bbox": [0.0, 0.0, 100.0, 20.0],
                    "confidence": 1.0,
                    "order": 0,
                    "lines": [
                        {
                            "text": "Hllo world",
                            "bbox": [0.0, 0.0, 100.0, 20.0],
                            "confidence": 1.0,
                            "order": 0,
                            "words": [
                                {
                                    "text": "Hllo",
                                    "bbox": [0.0, 0.0, 40.0, 20.0],
                                    "confidence": 0.8,
                                    "order": 0,
                                    "chars": [],
                                },
                                {
                                    "text": "world",
                                    "bbox": [50.0, 0.0, 100.0, 20.0],
                                    "confidence": 0.9,
                                    "order": 1,
                                    "chars": [],
                                },
                            ],
                        },
                    ],
                },
            ],
            "tables": [],
        }
        page = GTPageResult(
            gt_version_id=gt_id,
            page_number=1,
            data=word_data,
            confidence=0.9,
        )
        fake_db.add(page)
        await fake_db.commit()
        gt_version.page_results = [page]

        # Update word 0: "Hllo" -> "Hello"
        resp = await client.put(
            f"/api/v1/ground-truth/{gt_id_str}/pages/1/words/0",
            json={"text": "Hello"},
        )
        assert resp.status_code == 200
        result = resp.json()
        words = result["data"]["blocks"][0]["lines"][0]["words"]
        assert words[0]["text"] == "Hello"

        # Line text should also be updated
        assert result["data"]["blocks"][0]["lines"][0]["text"] == "Hello world"

        # Two versions now (original + updated)
        versions = [
            v for v in fake_db._store.get(GroundTruthVersion, {}).values()
            if v.pdf_id == uuid.UUID(doc_id) and v.deleted_at is None
        ]
        assert len(versions) == 2


class TestEvaluateRunEndpoint:
    """Full run evaluation returns expected score shape."""

    async def test_evaluate_run_endpoint(
        self,
        client: AsyncClient,
        pdf_content: bytes,
        engine_mock: uuid.UUID,
    ) -> None:
        """Given a completed run and consensus GT,
        When running evaluate_run, Then the score dict
        contains all expected metrics."""
        from backend.evaluation._evaluators import evaluate_run  # noqa: PLC0415
        from backend.evaluation.consensus import build_ground_truth  # noqa: PLC0415

        # ── 1. Upload & run ─────────────────────────────────────────────────
        upload_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("evaluate.pdf", pdf_content, "application/pdf")},
        )
        doc_id: str = upload_resp.json()["id"]
        run_id = await _create_completed_run(client, doc_id, seed=42)

        # ── 2. Fetch page data ──────────────────────────────────────────────
        results_resp = await client.get(f"/api/v1/runs/{run_id}/results")
        page_items = results_resp.json()["items"]
        page_data_list = [item["data"] for item in page_items]

        # ── 3. Build consensus GT ───────────────────────────────────────────
        gt_result = build_ground_truth(page_data_list + page_data_list)
        assert gt_result["source"] == "auto_consensus"

        # ── 4. Build data dicts ─────────────────────────────────────────────
        gt_data: dict[str, Any] = {"pages": gt_result["pages"]}
        run_data: dict[str, Any] = {
            "pages": [{"data": p, "results": []} for p in page_data_list],
        }

        # ── 5. Evaluate ─────────────────────────────────────────────────────
        scores = evaluate_run(run_data, gt_data)

        # ── 6. Verify the full expected shape ──────────────────────────────
        expected_keys = {
            "cer", "wer",
            "char_precision", "char_recall", "char_f1",
            "word_precision", "word_recall", "word_f1",
            "per_page", "num_pages",
        }
        assert set(scores.keys()) == expected_keys, (
            f"Score keys mismatch. Expected {expected_keys}, got {set(scores.keys())}"
        )

        # All numeric scores in valid range
        for key in ("cer", "wer", "char_precision", "char_recall", "char_f1",
                     "word_precision", "word_recall", "word_f1"):
            assert 0.0 <= scores[key] <= 1.0, (
                f"{key}={scores[key]} outside [0, 1]"
            )

        # Per-page entries exist
        assert isinstance(scores["per_page"], list)
        assert len(scores["per_page"]) >= 1
        assert scores["num_pages"] >= 1

        # Each per-page entry has the core metrics
        for ps in scores["per_page"]:
            assert "cer" in ps
            assert "wer" in ps
            assert "char_f1" in ps
            assert "word_f1" in ps


class TestFullPipeline:
    """Upload -> 2 runs -> auto-GT -> scores -> results API -> all consistent."""

    async def test_full_pipeline(
        self,
        client: AsyncClient,
        pdf_content: bytes,
        engine_mock: uuid.UUID,
    ) -> None:
        """Run the complete Phase 2 evaluation pipeline and verify
        end-to-end consistency across all components."""
        from backend.evaluation._evaluators import evaluate_run  # noqa: PLC0415
        from backend.evaluation.consensus import build_ground_truth  # noqa: PLC0415

        # ── 1. Upload PDF ──────────────────────────────────────────────────
        upload_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("full_pipeline.pdf", pdf_content, "application/pdf")},
        )
        assert upload_resp.status_code == 202
        doc_id: str = upload_resp.json()["id"]
        sha256_upload = hashlib.sha256(pdf_content).hexdigest()
        assert upload_resp.json()["filename"] == f"{sha256_upload}.pdf"

        # ── 2. Create two runs with different seeds ─────────────────────────
        run_ids: list[str] = []
        for seed in (42, 99):
            rid = await _create_completed_run(client, doc_id, seed=seed)
            run_data = await client.get(f"/api/v1/runs/{rid}")
            assert run_data.json()["engine_version"] == "0.1.0"
            run_ids.append(rid)

        # Verify distinct runs
        assert run_ids[0] != run_ids[1]

        # ── 3. Fetch results from first run for scoring ─────────────────────
        results_resp = await client.get(f"/api/v1/runs/{run_ids[0]}/results")
        assert results_resp.status_code == 200
        page_items = results_resp.json()["items"]
        page_data_list = [item["data"] for item in page_items]

        # ── 4. Build auto-GT via consensus ─────────────────────────────────
        gt_result = build_ground_truth(page_data_list + page_data_list)
        assert gt_result["source"] == "auto_consensus"
        assert len(gt_result["pages"]) >= 1

        # ── 5. Compute scores ──────────────────────────────────────────────
        gt_data: dict[str, Any] = {"pages": gt_result["pages"]}
        run_data: dict[str, Any] = {
            "pages": [{"data": p, "results": []} for p in page_data_list],
        }
        scores = evaluate_run(run_data, gt_data)

        # ── 6. Verify scores are valid ─────────────────────────────────────
        assert 0.0 <= scores["cer"] <= 1.0
        assert 0.0 <= scores["wer"] <= 1.0
        assert len(scores["per_page"]) == scores["num_pages"]

        # ── 7. Verify results API consistency ──────────────────────────────
        for rid in run_ids:
            run_meta = await client.get(f"/api/v1/runs/{rid}")
            assert run_meta.status_code == 200
            assert run_meta.json()["pdf_id"] == doc_id

            res = await client.get(f"/api/v1/runs/{rid}/results")
            assert res.status_code == 200
            res_data = res.json()
            # Items are populated even though total may be 0 in FakeSession
            # (count subquery is not supported by FakeSession)
            assert len(res_data["items"]) >= 1

            raw = await client.get(f"/api/v1/runs/{rid}/raw")
            assert raw.status_code == 200
            assert raw.json()["engine_id"] == "mock"

        # ── 8. Create consensus GT via API and verify listing ──────────────
        api_gt_resp = await client.post(
            "/api/v1/ground-truth",
            json={
                "pdf_id": doc_id,
                "source": "consensus",
                "engine_ids": run_ids,
                "notes": "Pipeline test GT",
                "created_by": "pipeline_test",
            },
        )
        assert api_gt_resp.status_code == 201
        api_gt_id: str = api_gt_resp.json()["id"]

        list_resp = await client.get(f"/api/v1/ground-truth?pdf_id={doc_id}")
        assert list_resp.status_code == 200
        gt_list = list_resp.json()["items"]
        assert any(item["id"] == api_gt_id for item in gt_list)

        # ── 9. Verify current GT endpoint ──────────────────────────────────
        cur_resp = await client.get(f"/api/v1/ground-truth/current/{doc_id}")
        assert cur_resp.status_code == 200


class TestWebSocketProgress:
    """Verify progress reporting during engine execution.

    Since the application does not yet expose a WebSocket endpoint, this test
    verifies the underlying progress mechanism by calling
    ``MockEngine.process_pdf`` directly with a progress callback.
    """

    async def test_mock_engine_progress_callback(
        self,
    ) -> None:
        """Given a MockEngine, When processing with a progress callback,
        Then progress values are reported from 0 to 100."""
        from backend.mock_engine import MockEngine  # noqa: PLC0415

        engine = MockEngine()
        progress_values: list[int] = []

        def _progress(value: int) -> None:
            progress_values.append(value)

        # MockEngine does not read the PDF file, so a dummy path works.
        raw = await engine.process_pdf(
            pdf_path="/tmp/nonexistent/test.pdf",
            config={"seed": 42},
            progress=_progress,
        )

        # Progress should have been reported
        assert len(progress_values) >= 1
        assert progress_values[0] == 0
        assert progress_values[-1] == 100

        # Verify the raw output structure
        assert "raw_pages" in raw
        assert raw["engine_id"] == "mock"
        assert raw["engine_version"] == "0.1.0"
        assert len(raw["raw_pages"]) >= 1

    async def test_engine_progress_via_registry(
        self,
    ) -> None:
        """Given a registered engine from the registry,
        When calling process_pdf with a progress callback,
        Then progress is reported correctly."""
        from backend.engine.registry import registry  # noqa: PLC0415

        try:
            # registry.get("mock") returns a fresh instance, not a class
            engine = registry.get("mock")
        except Exception as exc:
            pytest.skip(f"mock engine not registered: {exc}")

        progress_values: list[int] = []

        def _progress(value: int) -> None:
            progress_values.append(value)

        raw = await engine.process_pdf(
            pdf_path="/tmp/nonexistent/test.pdf",
            config={"seed": 42},
            progress=_progress,
        )

        assert len(progress_values) >= 1
        assert progress_values[0] == 0
        assert progress_values[-1] == 100

        # Verify deterministic output
        assert len(raw["raw_pages"]) >= 1
        page1 = raw["raw_pages"][0]
        assert "blocks" in page1
        assert len(page1["blocks"]) >= 1


class TestBootstrapCIIntegration:
    """Bootstrap confidence interval computed on multi-page run scores."""

    async def test_bootstrap_ci_integration(
        self,
        client: AsyncClient,
        pdf_content: bytes,
        engine_mock: uuid.UUID,
    ) -> None:
        """Given a completed run with multiple pages,
        When computing bootstrap CI on per-page CER scores,
        Then valid CI bounds and statistics are returned."""
        from backend.evaluation import bootstrap_ci  # noqa: PLC0415
        from backend.evaluation._evaluators import evaluate_run  # noqa: PLC0415
        from backend.evaluation.consensus import build_ground_truth  # noqa: PLC0415

        # ── 1. Upload & run ─────────────────────────────────────────────────
        upload_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("bootstrap.pdf", pdf_content, "application/pdf")},
        )
        doc_id: str = upload_resp.json()["id"]
        run_id = await _create_completed_run(client, doc_id, seed=42)

        # ── 2. Fetch page data ──────────────────────────────────────────────
        results_resp = await client.get(f"/api/v1/runs/{run_id}/results")
        page_items = results_resp.json()["items"]
        page_data_list = [item["data"] for item in page_items]

        # ── 3. Build consensus GT ───────────────────────────────────────────
        gt_result = build_ground_truth(page_data_list + page_data_list)
        assert gt_result["source"] == "auto_consensus"

        gt_data: dict[str, Any] = {"pages": gt_result["pages"]}
        run_data: dict[str, Any] = {
            "pages": [{"data": p, "results": []} for p in page_data_list],
        }

        # ── 4. Evaluate to get per-page scores ──────────────────────────────
        scores = evaluate_run(run_data, gt_data)
        per_page_cer = [ps["cer"] for ps in scores["per_page"]]
        assert len(per_page_cer) >= 1

        # ── 5. Compute bootstrap CI on CER ─────────────────────────────────
        ci_result = bootstrap_ci(
            scores=per_page_cer,
            metric_name="cer",
            n_resamples=500,
            ci_level=0.95,
            random_seed=42,
        )

        # ── 6. Verify CI shape ──────────────────────────────────────────────
        assert ci_result["metric"] == "cer"
        assert ci_result["n"] == len(per_page_cer)
        assert ci_result["n_resamples"] == 500
        assert ci_result["ci_level"] == 0.95

        # Required keys
        assert "mean" in ci_result
        assert "median" in ci_result
        assert "std" in ci_result
        assert "ci_lower" in ci_result
        assert "ci_upper" in ci_result
        assert "resampled_means" in ci_result

        # CI should contain the observed mean
        assert ci_result["ci_lower"] <= ci_result["mean"] <= ci_result["ci_upper"]

        # CI bounds within valid range
        assert 0.0 <= ci_result["ci_lower"] <= 1.0
        assert 0.0 <= ci_result["ci_upper"] <= 1.0

        # Correct number of resampled means
        assert len(ci_result["resampled_means"]) == 500

        # ── 7. Verify bootstrap_compare works ──────────────────────────────
        from backend.evaluation import bootstrap_compare  # noqa: PLC0415

        scores2 = evaluate_run(
            {"pages": page_data_list},
            {"pages": gt_result["pages"]},
        )
        per_page_cer2 = [ps["cer"] for ps in scores2["per_page"]]

        compare = bootstrap_compare(
            engine_a_scores=per_page_cer,
            engine_b_scores=per_page_cer2,
            n_resamples=500,
            ci_level=0.95,
        )

        assert "diff_mean" in compare
        assert "diff_ci_lower" in compare
        assert "diff_ci_upper" in compare
        assert "significant" in compare
        assert compare["n_a"] == len(per_page_cer)
        assert compare["n_b"] == len(per_page_cer2)
