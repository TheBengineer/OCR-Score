"""Integration tests for the OCR run management API endpoints.

Uses the same in-memory fake session and dependency-override pattern as
``test_upload.py`` to avoid external database and OCR engine dependencies.
"""

from __future__ import annotations

import hashlib
import json
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
from backend.models.enums import PDFStatus, RunStatus
from backend.models.page_result import PageResult
from backend.models.pdf import PDF
from backend.models.run import OCRRun
from backend.settings import settings

# ---------------------------------------------------------------------------
# Fake SQLAlchemy session (same pattern as test_upload.py)
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
            # IN_ clause
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
    """In-memory fake for SQLAlchemy ``AsyncSession``."""

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
        if isinstance(entity, InstrumentedAttribute):
            entity = entity.class_

        items = list(self._store.get(entity, {}).values())

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

        return FakeResult(items)

    async def close(self) -> None:
        pass

    def get(self, entity: type, ident: Any) -> Any:
        """Direct access to stored objects by primary key."""
        return self._store.get(entity, {}).get(ident)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_db() -> FakeSession:
    """Provide a fresh fake session per test."""
    return FakeSession()


@pytest.fixture
def app(tmp_path: Path, fake_db: FakeSession) -> FastAPI:
    """Provide the FastAPI app with overridden DB and storage dependencies."""
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
    """Return a minimal valid PDF."""
    return _make_pdf_bytes()


@pytest.fixture
def pdf_id(pdf_content: bytes, fake_db: FakeSession) -> uuid.UUID:
    """Pre-seed a PDF record into the fake database and return its ID."""
    sha256 = hashlib.sha256(pdf_content).hexdigest()
    md5 = hashlib.md5(pdf_content).hexdigest()
    pdf = PDF(
        filename=f"{sha256}.pdf",
        original_filename="test.pdf",
        file_size_bytes=len(pdf_content),
        page_count=3,
        md5_hash=md5,
        sha256_hash=sha256,
        mime_type="application/pdf",
        status=PDFStatus.READY,
    )
    fake_db.add(pdf)
    # Commit synchronously via an event loop
    import asyncio  # noqa: PLC0415

    asyncio.run(fake_db.commit())

    # Also store the PDF file so the engine can read it
    pdf_path = Path(settings.storage_path) / "pdfs" / sha256[:2] / sha256[2:4] / f"{sha256}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(pdf_content)

    return pdf.id


@pytest.fixture
def engine_id(fake_db: FakeSession) -> uuid.UUID:
    """Pre-seed a MockEngine record into the fake database."""
    from backend.models.engine import OCREngine  # noqa: PLC0415

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
        description="Test engine for API tests",
    )
    fake_db.add(engine)
    import asyncio  # noqa: PLC0415

    asyncio.run(fake_db.commit())
    return engine.id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pdf_bytes(content: str = "Hello world") -> bytes:
    """Build a minimal but structurally valid PDF (same as test_upload.py)."""
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateRunAPI:
    """Test ``POST /api/v1/runs``."""

    async def test_create_run_returns_202(
        self,
        client: AsyncClient,
        pdf_id: uuid.UUID,
        engine_id: uuid.UUID,
    ) -> None:
        """Given a valid request, When creating a run, Then 202 is returned."""
        response = await client.post(
            "/api/v1/runs",
            json={
                "pdf_id": str(pdf_id),
                "engine_id": "mock",
                "config": {"seed": 42},
            },
        )
        assert response.status_code == 202
        data = response.json()
        assert "id" in data
        assert data["status"] == "pending"

    async def test_create_run_invalid_pdf(
        self,
        client: AsyncClient,
        engine_id: uuid.UUID,
    ) -> None:
        """Given a non-existent PDF ID, When creating a run, Then 400 is returned."""
        response = await client.post(
            "/api/v1/runs",
            json={
                "pdf_id": "00000000-0000-0000-0000-000000000001",
                "engine_id": "mock",
            },
        )
        assert response.status_code == 400


class TestGetRun:
    """Test ``GET /api/v1/runs/{id}``."""

    async def test_get_run(
        self,
        client: AsyncClient,
        pdf_id: uuid.UUID,
        engine_id: uuid.UUID,
    ) -> None:
        """Given an existing run, When fetched by ID, Then metadata is returned."""
        # Create run first
        create_resp = await client.post(
            "/api/v1/runs",
            json={
                "pdf_id": str(pdf_id),
                "engine_id": "mock",
            },
        )
        run_id = create_resp.json()["id"]

        resp = await client.get(f"/api/v1/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == run_id
        assert data["status"] == "pending"
        assert "created_at" in data

    async def test_get_run_not_found(
        self,
        client: AsyncClient,
    ) -> None:
        """Given a non-existent run ID, When fetched, Then 404 is returned."""
        resp = await client.get(
            "/api/v1/runs/00000000-0000-0000-0000-000000000001",
        )
        assert resp.status_code == 404


class TestGetResults:
    """Test ``GET /api/v1/runs/{id}/results``."""

    async def test_get_results_after_execution(
        self,
        client: AsyncClient,
        pdf_id: uuid.UUID,
        engine_id: uuid.UUID,
        fake_db: FakeSession,
    ) -> None:
        """Given a completed run, When fetching results, Then page results are returned."""
        # Create run
        create_resp = await client.post(
            "/api/v1/runs",
            json={
                "pdf_id": str(pdf_id),
                "engine_id": "mock",
            },
        )
        run_id = uuid.UUID(create_resp.json()["id"])

        # Manually execute run (the background task may not have run yet)
        run = fake_db.get(OCRRun, run_id)
        assert run is not None

        # Manually populate page results to simulate execution
        for pn in range(1, 3):
            pr = PageResult(
                run_id=run_id,
                page_number=pn,
                width=612.0,
                height=792.0,
                data={"page_number": pn, "width": 612.0, "height": 792.0, "blocks": []},
                confidence=0.95,
            )
            fake_db.add(pr)
        run.status = RunStatus.COMPLETED
        run.raw_output_uri = "/tmp/test_raw.json"
        await fake_db.commit()

        # Fetch results
        resp = await client.get(f"/api/v1/runs/{run_id}/results")
        assert resp.status_code == 200
        data = resp.json()
        # The fake session may not support the count subquery,
        # so check the items list directly
        assert len(data["items"]) == 2
        assert data["items"][0]["page_number"] == 1


class TestGetRawOutput:
    """Test ``GET /api/v1/runs/{id}/raw``."""

    async def test_get_raw_output(
        self,
        client: AsyncClient,
        pdf_id: uuid.UUID,
        engine_id: uuid.UUID,
        fake_db: FakeSession,
        tmp_path: Path,
    ) -> None:
        """Given a run with raw output, When fetching raw, Then the raw JSON is returned."""
        # Create run
        create_resp = await client.post(
            "/api/v1/runs",
            json={
                "pdf_id": str(pdf_id),
                "engine_id": "mock",
            },
        )
        run_id = uuid.UUID(create_resp.json()["id"])

        # Manually set raw output
        run = fake_db.get(OCRRun, run_id)
        assert run is not None

        raw_output = {"raw_pages": [], "engine_id": "mock", "engine_version": "0.1.0"}
        raw_path = tmp_path / "test_raw.json"
        raw_path.write_text(json.dumps(raw_output))

        run.raw_output_uri = str(raw_path)
        run.status = RunStatus.COMPLETED
        await fake_db.commit()

        resp = await client.get(f"/api/v1/runs/{run_id}/raw")
        assert resp.status_code == 200
        data = resp.json()
        assert data["engine_id"] == "mock"
        assert data["engine_version"] == "0.1.0"


class TestCancelRun:
    """Test ``DELETE /api/v1/runs/{id}``."""

    async def test_cancel_run(
        self,
        client: AsyncClient,
        pdf_id: uuid.UUID,
        engine_id: uuid.UUID,
        fake_db: FakeSession,
    ) -> None:
        """Given a pending run, When cancelling, Then 204 is returned and status changes."""
        # Register MockEngine in the global registry so the background
        # task doesn't fail before we cancel
        from backend.engine import registry as global_registry  # noqa: PLC0415
        from backend.mock_engine import MockEngine  # noqa: PLC0415

        try:
            global_registry.register(MockEngine)
        except Exception:
            pass  # Already registered

        create_resp = await client.post(
            "/api/v1/runs",
            json={
                "pdf_id": str(pdf_id),
                "engine_id": "mock",
            },
        )
        run_id = create_resp.json()["id"]

        resp = await client.delete(f"/api/v1/runs/{run_id}")
        assert resp.status_code == 204

        # Verify status changed
        run = fake_db.get(OCRRun, uuid.UUID(run_id))
        assert run is not None
        assert run.status == RunStatus.CANCELLED
