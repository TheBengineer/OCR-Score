"""End-to-end integration tests for the OCRScore pipeline.

Tests the full flow: upload PDF → run engine → store results → retrieve results.
Uses an in-memory fake SQLAlchemy session and temporary storage to avoid any
external dependencies.  Both MockEngine (always available) and TesseractEngine
(if the ``tesseract`` binary is installed) are exercised.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
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
# Minimal valid PDF builder
# ═══════════════════════════════════════════════════════════════════════════════


def _make_pdf_bytes(content: str = "Integration test") -> bytes:
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
# Fake SQLAlchemy session  (same pattern as test_run_orchestrator.py)
# ═══════════════════════════════════════════════════════════════════════════════


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
    OFFSET subset), ``refresh``, ``merge``, and ``get`` — sufficient for
    integration testing of the full pipeline.
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
        """Simulate refresh from database — populates server-side defaults."""
        mapper = sa_inspect(type(obj))
        for col in mapper.columns:
            if col.server_default is not None:
                current = getattr(obj, col.key, None)
                if current is None:
                    setattr(obj, col.key, datetime.now(UTC))

    async def execute(self, stmt: Select) -> FakeResult:
        descriptions = stmt.column_descriptions
        if not descriptions:
            return FakeResult([])

        entity = descriptions[0].get("entity")
        if entity is None:
            # Aggregate queries (func.count) — return empty so the caller
            # gets ``None`` from ``.scalar()``.
            return FakeResult([])

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
        import backend.engines.tesseract  # noqa: F401 — triggers import-time registration
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
    return _make_pdf_bytes("Integration test document")


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
        description="Mock engine for integration tests",
    )
    fake_db.add(engine)
    await fake_db.commit()

    # Ensure the generated UUID is valid
    assert engine.id is not None
    return engine.id


@pytest.fixture
async def engine_tesseract(fake_db: FakeSession) -> uuid.UUID | None:
    """Pre-seed a TesseractEngine OCHEngine record if the tesseract binary is
    installed.  Returns ``None`` when tesseract is unavailable (callers should
    ``@pytest.mark.skipif`` accordingly)."""
    if shutil.which("tesseract") is None:
        return None

    from backend.models.engine import OCREngine

    engine = OCREngine(
        slug="tesseract",
        display_name="Tesseract OCR",
        version="0.1.0",
        enabled=True,
        config_schema={
            "type": "object",
            "properties": {
                "lang": {"type": "string", "default": "eng"},
                "psm": {"type": "integer", "default": 3},
            },
            "required": [],
        },
        description="Tesseract OCR engine for integration tests",
    )
    fake_db.add(engine)
    await fake_db.commit()
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


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestUploadPDF:
    """Verify that PDF upload works and returns a document ID."""

    async def test_upload_pdf(
        self,
        client: AsyncClient,
        pdf_content: bytes,
    ) -> None:
        """Given a valid PDF, When uploaded, Then a 202 with document ID is returned."""
        response = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("integration.pdf", pdf_content, "application/pdf")},
        )
        assert response.status_code == 202
        data = response.json()
        assert "id" in data
        assert data["status"] == "uploaded"
        sha256 = hashlib.sha256(pdf_content).hexdigest()
        assert data["filename"] == f"{sha256}.pdf"


class TestRunEngine:
    """Verify that engine runs can be created via the API."""

    async def test_run_engine(
        self,
        client: AsyncClient,
        pdf_content: bytes,
        engine_mock: uuid.UUID,
    ) -> None:
        """Given an uploaded PDF and a registered engine,
        When creating a run, Then a 202 with the run ID is returned."""
        # Upload
        upload_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("run_engine.pdf", pdf_content, "application/pdf")},
        )
        assert upload_resp.status_code == 202
        doc_id = upload_resp.json()["id"]

        # Create run
        run_resp = await client.post(
            "/api/v1/runs",
            json={
                "pdf_id": doc_id,
                "engine_id": "mock",
                "config": {"seed": 42},
            },
        )
        assert run_resp.status_code == 202
        data = run_resp.json()
        assert "id" in data
        assert data["status"] == "pending"


class TestRunExecution:
    """Verify that a created run transitions to completed."""

    async def test_run_execution(
        self,
        client: AsyncClient,
        pdf_content: bytes,
        engine_mock: uuid.UUID,
    ) -> None:
        """Given a created run, When polling, Then status becomes completed."""
        # Upload
        upload_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("execution.pdf", pdf_content, "application/pdf")},
        )
        doc_id = upload_resp.json()["id"]

        # Create run
        run_resp = await client.post(
            "/api/v1/runs",
            json={
                "pdf_id": doc_id,
                "engine_id": "mock",
                "config": {"seed": 42},
            },
        )
        run_id = run_resp.json()["id"]

        # Poll until terminal
        run_data = await _poll_run_until(client, run_id)
        assert run_data["status"] == "completed"
        assert run_data["completed_at"] is not None
        assert run_data["engine_version"] == "0.1.0"


class TestGetResults:
    """Verify that normalized page results are returned after a run completes."""

    async def test_get_results(
        self,
        client: AsyncClient,
        pdf_content: bytes,
        engine_mock: uuid.UUID,
    ) -> None:
        """Given a completed run, When fetching results, Then normalized page
        results with valid schema are returned."""
        # Upload + create + complete
        upload_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("results.pdf", pdf_content, "application/pdf")},
        )
        doc_id = upload_resp.json()["id"]

        run_resp = await client.post(
            "/api/v1/runs",
            json={"pdf_id": doc_id, "engine_id": "mock"},
        )
        run_id = run_resp.json()["id"]
        await _poll_run_until(client, run_id)

        # Fetch paginated results
        results_resp = await client.get(f"/api/v1/runs/{run_id}/results")
        assert results_resp.status_code == 200
        results_data = results_resp.json()
        assert "items" in results_data
        assert len(results_data["items"]) >= 2  # MockEngine produces 2-3 pages

        # Check schema of first page result
        first_item = results_data["items"][0]
        assert "page_number" in first_item
        assert "data" in first_item
        assert "blocks" in first_item["data"]
        assert first_item["page_number"] >= 1

        # Fetch a single page result
        page_resp = await client.get(
            f"/api/v1/runs/{run_id}/results/{first_item['page_number']}",
        )
        assert page_resp.status_code == 200
        page_data = page_resp.json()
        assert page_data["page_number"] == first_item["page_number"]
        assert page_data["width"] == 612.0
        assert page_data["height"] == 792.0
        assert "data" in page_data
        assert "blocks" in page_data["data"]

        # Results page results for a non-existent page → 404
        missing_page_resp = await client.get(
            f"/api/v1/runs/{run_id}/results/999",
        )
        assert missing_page_resp.status_code == 404


class TestGetRawOutput:
    """Verify that raw engine output is retrievable after a run completes."""

    async def test_get_raw_output(
        self,
        client: AsyncClient,
        pdf_content: bytes,
        engine_mock: uuid.UUID,
    ) -> None:
        """Given a completed run, When fetching raw output, Then the
        engine's pre-normalisation JSON is returned."""
        # Upload + create + complete
        upload_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("raw.pdf", pdf_content, "application/pdf")},
        )
        doc_id = upload_resp.json()["id"]

        run_resp = await client.post(
            "/api/v1/runs",
            json={"pdf_id": doc_id, "engine_id": "mock"},
        )
        run_id = run_resp.json()["id"]
        await _poll_run_until(client, run_id)

        # Fetch raw output
        raw_resp = await client.get(f"/api/v1/runs/{run_id}/raw")
        assert raw_resp.status_code == 200
        raw_data = raw_resp.json()
        assert raw_data["engine_id"] == "mock"
        assert raw_data["engine_version"] == "0.1.0"
        assert "raw_pages" in raw_data
        assert len(raw_data["raw_pages"]) >= 2
        assert "config_snapshot" in raw_data


class TestRunWithMockEngine:
    """Full pipeline smoke test using MockEngine (no external dependencies)."""

    async def test_run_with_mock_engine(
        self,
        client: AsyncClient,
        pdf_content: bytes,
        engine_mock: uuid.UUID,
    ) -> None:
        """Full flow: upload → create run → poll → verify results."""
        # 1. Upload
        upload_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("mock_test.pdf", pdf_content, "application/pdf")},
        )
        assert upload_resp.status_code == 202
        doc_id = upload_resp.json()["id"]

        # 2. Create run
        run_resp = await client.post(
            "/api/v1/runs",
            json={
                "pdf_id": doc_id,
                "engine_id": "mock",
                "config": {"seed": 42},
            },
        )
        assert run_resp.status_code == 202
        run_id = run_resp.json()["id"]

        # 3. Poll for completion
        run_data = await _poll_run_until(client, run_id)
        assert run_data["status"] == "completed"
        assert run_data["engine_version"] == "0.1.0"

        # 4. Verify page results exist
        results_resp = await client.get(f"/api/v1/runs/{run_id}/results")
        assert results_resp.status_code == 200
        assert len(results_resp.json()["items"]) >= 2

        # 5. Verify raw output is non-empty
        raw_resp = await client.get(f"/api/v1/runs/{run_id}/raw")
        assert raw_resp.status_code == 200
        assert raw_resp.json()["engine_id"] == "mock"


class TestRunWithTesseract:
    """Full pipeline smoke test using TesseractEngine (requires tesseract binary)."""

    @pytest.mark.skipif(
        shutil.which("tesseract") is None,
        reason="tesseract binary not installed on this system",
    )
    async def test_run_with_tesseract(
        self,
        client: AsyncClient,
        pdf_content: bytes,
        engine_tesseract: uuid.UUID | None,
    ) -> None:
        """Full flow with Tesseract engine (same assertions as MockEngine)."""
        # Guard: if the fixture returned None (binary missing), skip gracefully
        if engine_tesseract is None:
            pytest.skip("engine_tesseract fixture returned None")

        # 1. Upload
        upload_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("tesseract_test.pdf", pdf_content, "application/pdf")},
        )
        assert upload_resp.status_code == 202
        doc_id = upload_resp.json()["id"]

        # 2. Create run with tesseract engine
        run_resp = await client.post(
            "/api/v1/runs",
            json={
                "pdf_id": doc_id,
                "engine_id": "tesseract",
                "config": {"lang": "eng", "psm": 3, "oem": 3, "dpi": 300},
            },
        )
        assert run_resp.status_code == 202
        run_id = run_resp.json()["id"]

        # 3. Poll for completion (allow extra time for Tesseract)
        run_data = await _poll_run_until(client, run_id, timeout=30.0)
        assert run_data["status"] == "completed"

        # 4. Verify results
        results_resp = await client.get(f"/api/v1/runs/{run_id}/results")
        assert results_resp.status_code == 200
        items = results_resp.json()["items"]
        assert len(items) >= 1

        # 5. Verify raw output
        raw_resp = await client.get(f"/api/v1/runs/{run_id}/raw")
        assert raw_resp.status_code == 200
        assert raw_resp.json()["engine_id"] == "tesseract"


class TestRunHashDedupIntegration:
    """Verify that re-running the same parameters returns the same completed run."""

    async def test_run_hash_dedup_integration(
        self,
        client: AsyncClient,
        pdf_content: bytes,
        engine_mock: uuid.UUID,
    ) -> None:
        """Given the same PDF/engine/config submitted twice,
        When creating the second run, Then ``200`` (not ``202``) is returned
        with the same run ID and a "run already completed" message."""
        # Upload
        upload_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("dedup.pdf", pdf_content, "application/pdf")},
        )
        doc_id = upload_resp.json()["id"]

        # First run
        run1_resp = await client.post(
            "/api/v1/runs",
            json={
                "pdf_id": doc_id,
                "engine_id": "mock",
                "config": {"seed": 42},
            },
        )
        assert run1_resp.status_code == 202
        run1_id = run1_resp.json()["id"]

        # Wait for completion
        await _poll_run_until(client, run1_id)

        # Second run (identical parameters) → 200, same ID
        run2_resp = await client.post(
            "/api/v1/runs",
            json={
                "pdf_id": doc_id,
                "engine_id": "mock",
                "config": {"seed": 42},
            },
        )
        assert run2_resp.status_code == 200
        run2_data = run2_resp.json()
        assert run2_data["id"] == run1_id
        assert run2_data["message"] == "run already completed"


class TestErrorHandling:
    """Verify that the API returns appropriate error codes for invalid inputs."""

    async def test_error_handling(
        self,
        client: AsyncClient,
        engine_mock: uuid.UUID,
    ) -> None:
        """Given various invalid inputs, When calling endpoints, Then proper
        error status codes and messages are returned."""
        # ── 1. Create run with non-existent PDF → 400 ──────────────────────
        fake_pdf_id = "00000000-0000-0000-0000-000000000001"
        resp = await client.post(
            "/api/v1/runs",
            json={
                "pdf_id": fake_pdf_id,
                "engine_id": "mock",
            },
        )
        assert resp.status_code == 400

        # ── 2. Create run with unknown engine slug → 400 ────────────────────
        pdf_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("err_test.pdf", _make_pdf_bytes("err"), "application/pdf")},
        )
        doc_id = pdf_resp.json()["id"]

        resp = await client.post(
            "/api/v1/runs",
            json={
                "pdf_id": doc_id,
                "engine_id": "nonexistent",
            },
        )
        assert resp.status_code == 400

        # ── 3. Get non-existent run → 404 ──────────────────────────────────
        fake_run_id = "00000000-0000-0000-0000-000000000002"
        resp = await client.get(f"/api/v1/runs/{fake_run_id}")
        assert resp.status_code == 404

        # ── 4. Results for non-existent run → 404 ──────────────────────────
        resp = await client.get(f"/api/v1/runs/{fake_run_id}/results")
        assert resp.status_code == 404

        # ── 5. Raw output for non-existent run → 404 ───────────────────────
        resp = await client.get(f"/api/v1/runs/{fake_run_id}/raw")
        assert resp.status_code == 404

        # ── 6. Upload non-PDF content → 422 ────────────────────────────────
        resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("not_a_pdf.txt", b"not pdf content", "text/plain")},
        )
        assert resp.status_code == 422
