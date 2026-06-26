"""Integration tests for the PDF document upload and management API endpoints.

These tests use a fake in-memory database session and a temporary storage
directory to avoid external dependencies.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import Select, UnaryExpression
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.sql import operators
from sqlalchemy.sql.elements import BinaryExpression, BooleanClauseList
from sqlalchemy.sql.schema import CallableColumnDefault

from backend.database import get_db_session
from backend.main import app as _app
from backend.settings import settings

# ── Minimal valid PDF ─────────────────────────────────────────────────────────


def _make_pdf_bytes(content: str = "Hello world") -> bytes:
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


# ── Fake SQLAlchemy session ───────────────────────────────────────────────────


class FakeResult:
    """Fake replacement for SQLAlchemy ``Result`` / ``ScalarResult``.

    Wraps a list of rows and exposes ``scalars()``, ``one_or_none()``,
    ``all()``, ``first()``, and ``unique()``.
    """

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


def _extract_conditions(
    whereclause: Any,
) -> list[tuple[str, Any, Any]]:
    """Extract ``(field_name, operator, value)`` tuples from a WHERE clause.

    Handles ``AND`` combos, equality (``==``), ``is_`` (``IS NULL``),
    ``isnot`` (``IS NOT NULL``), and greater-than (``>``) comparisons.
    """
    conditions: list[tuple[str, Any, Any]] = []

    if whereclause is None:
        return conditions

    if isinstance(whereclause, BooleanClauseList):
        for clause in whereclause.clauses:
            conditions.extend(_extract_conditions(clause))
        return conditions

    if isinstance(whereclause, BinaryExpression):
        left = whereclause.left
        right = whereclause.right
        op = whereclause.operator

        # Get field name from the left side
        field_name = str(left.key) if hasattr(left, "key") else None
        if field_name is None:
            return conditions

        # Get value from the right side
        if hasattr(right, "value"):
            field_value = right.value
        elif right is None or (hasattr(right, "is_arithmetic") and op in (operators.is_, operators.isnot)):
            field_value = None
        else:
            field_value = right

        conditions.append((field_name, op, field_value))

    return conditions


class FakeSession:
    """In-memory fake for SQLAlchemy ``AsyncSession``.

    Supports ``add``, ``commit``, ``rollback``, ``execute`` (with limited
    SELECT support), ``refresh``, and ``merge`` — sufficient for testing
    the document router's query patterns.
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
        """Simulate refreshing from database — populates server-side defaults."""
        mapper = sa_inspect(type(obj))
        for col in mapper.columns:
            if col.server_default is not None:
                current = getattr(obj, col.key, None)
                if current is None:
                    setattr(obj, col.key, datetime.now(UTC))

    async def execute(self, stmt: Select) -> FakeResult:
        model = stmt.column_descriptions[0]["entity"]
        items = list(self._store.get(model, {}).values())

        # Apply WHERE conditions
        conditions = _extract_conditions(stmt.whereclause)
        for field_name, op, value in conditions:
            if op is operators.eq:
                items = [i for i in items if getattr(i, field_name) == value]
            elif op is operators.is_:
                items = [i for i in items if getattr(i, field_name) is None]
            elif op is operators.isnot:
                items = [i for i in items if getattr(i, field_name) is not None]
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
                if isinstance(order_spec, UnaryExpression):
                    field_name = order_spec.element.key
                    descending = order_spec.direction == operators.desc
                elif hasattr(order_spec, "key"):
                    field_name = order_spec.key
                    descending = False
                else:
                    continue
                items.sort(key=lambda i, f=field_name: getattr(i, f), reverse=descending)

        # Apply LIMIT
        limit_clause = getattr(stmt, "_limit_clause", None)
        if limit_clause is not None:
            limit_value = limit_clause.value if hasattr(limit_clause, "value") else int(limit_clause)
            items = items[:limit_value]

        return FakeResult(items)

    async def close(self) -> None:
        pass


# ── Test fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    """Provide the FastAPI app with overridden dependencies for testing.

    The ``get_db_session`` dependency is replaced with a ``FakeSession``
    that persists across all requests within a single test function.
    The storage path is redirected to a temporary directory.
    """
    settings.storage_path = str(tmp_path)

    fake_session = FakeSession()

    async def _override_session() -> AsyncGenerator[FakeSession]:
        yield fake_session

    _app.dependency_overrides[get_db_session] = _override_session
    return _app


@pytest.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient]:
    """Provide an async HTTP client for the test app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def valid_pdf_content() -> bytes:
    """Provide minimal valid PDF bytes for upload tests."""
    return _make_pdf_bytes("Test document")


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestDocumentUpload:
    """Test the ``POST /api/v1/documents/upload`` endpoint."""

    async def test_upload_valid_pdf(self, client: AsyncClient, valid_pdf_content: bytes) -> None:
        """Given a valid PDF file,
        When uploaded, Then a 202 response with the document ID is returned."""
        response = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.pdf", valid_pdf_content, "application/pdf")},
        )
        assert response.status_code == 202
        data = response.json()
        assert "id" in data
        assert data["status"] == "uploaded"
        sha256 = hashlib.sha256(valid_pdf_content).hexdigest()
        assert data["filename"] == f"{sha256}.pdf"

    async def test_upload_invalid_file(self, client: AsyncClient) -> None:
        """Given a file without PDF magic bytes,
        When uploaded, Then a 422 response is returned."""
        response = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("notapdf.txt", b"this is not a PDF", "text/plain")},
        )
        assert response.status_code == 422

    async def test_upload_duplicate(self, client: AsyncClient, valid_pdf_content: bytes) -> None:
        """Given the same PDF content uploaded twice,
        When uploading the second time, Then a 200 response with the
        existing document ID is returned."""
        resp1 = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("first.pdf", valid_pdf_content, "application/pdf")},
        )
        assert resp1.status_code == 202
        first_id = resp1.json()["id"]

        resp2 = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("second.pdf", valid_pdf_content, "application/pdf")},
        )
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["id"] == first_id
        assert data["message"] == "document already exists"


class TestDocumentRetrieval:
    """Test the ``GET /api/v1/documents/{id}`` endpoint."""

    async def test_get_document(self, client: AsyncClient, valid_pdf_content: bytes) -> None:
        """Given an uploaded document,
        When retrieved by its ID, Then the document metadata is returned."""
        upload_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.pdf", valid_pdf_content, "application/pdf")},
        )
        doc_id = upload_resp.json()["id"]

        resp = await client.get(f"/api/v1/documents/{doc_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == doc_id
        assert data["status"] == "uploaded"
        assert data["mime_type"] == "application/pdf"

    async def test_get_nonexistent_document(self, client: AsyncClient) -> None:
        """Given a non-existent document ID,
        When retrieved, Then a 404 response is returned."""
        fake_id = "00000000-0000-0000-0000-000000000001"
        resp = await client.get(f"/api/v1/documents/{fake_id}")
        assert resp.status_code == 404


class TestDocumentListing:
    """Test the ``GET /api/v1/documents`` endpoint."""

    async def test_list_documents(self, client: AsyncClient) -> None:
        """Given multiple uploaded documents,
        When listing all documents, Then a paginated list is returned."""
        for i in range(3):
            await client.post(
                "/api/v1/documents/upload",
                files={"file": (f"doc{i}.pdf", _make_pdf_bytes(f"Doc {i}"), "application/pdf")},
            )

        resp = await client.get("/api/v1/documents?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "next_cursor" in data
        assert len(data["items"]) == 3

    async def test_list_documents_pagination(self, client: AsyncClient) -> None:
        """Given more documents than the page limit,
        When listing with a small limit, Then the response contains at most
        ``limit`` items and a ``next_cursor`` pointing to the next page."""
        for i in range(5):
            await client.post(
                "/api/v1/documents/upload",
                files={"file": (f"doc{i}.pdf", _make_pdf_bytes(f"Doc {i}"), "application/pdf")},
            )

        resp = await client.get("/api/v1/documents?limit=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 2
        assert data["next_cursor"] is not None

        # Fetch next page using the cursor
        cursor = data["next_cursor"]
        resp2 = await client.get(f"/api/v1/documents?limit=2&cursor={cursor}")
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert len(data2["items"]) >= 1
        # Ensure no duplicate items across pages
        first_page_ids = {item["id"] for item in data["items"]}
        second_page_ids = {item["id"] for item in data2["items"]}
        assert first_page_ids.isdisjoint(second_page_ids)


class TestDocumentDeletion:
    """Test the ``DELETE /api/v1/documents/{id}`` endpoint."""

    async def test_delete_document(self, client: AsyncClient, valid_pdf_content: bytes) -> None:
        """Given an existing document,
        When deleted, Then the response is 204 and subsequent GET returns 404."""
        upload_resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.pdf", valid_pdf_content, "application/pdf")},
        )
        doc_id = upload_resp.json()["id"]

        delete_resp = await client.delete(f"/api/v1/documents/{doc_id}")
        assert delete_resp.status_code == 204

        get_resp = await client.get(f"/api/v1/documents/{doc_id}")
        assert get_resp.status_code == 404

    async def test_delete_nonexistent_document(self, client: AsyncClient) -> None:
        """Given a non-existent document ID,
        When deleted, Then a 404 response is returned."""
        fake_id = "00000000-0000-0000-0000-000000000001"
        resp = await client.delete(f"/api/v1/documents/{fake_id}")
        assert resp.status_code == 404
