"""Tests for ground truth CRUD API and manager versioning.

Uses the same in-memory fake session and dependency-override pattern as
``test_runs_api.py`` to avoid external database dependencies.
"""

from __future__ import annotations

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
from backend.ground_truth_manager import GroundTruthManager, GroundTruthManagerError
from backend.main import app as _app
from backend.models.enums import GroundTruthSource, PDFStatus
from backend.models.ground_truth import GroundTruthVersion, GTPageResult
from backend.models.pdf import PDF
from backend.settings import settings

# ---------------------------------------------------------------------------
# Enhanced Fake SQLAlchemy session
# ---------------------------------------------------------------------------


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
    """In-memory fake for SQLAlchemy ``AsyncSession`` with ordering support."""

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
            if hasattr(obj, "deleted_at") and getattr(obj, "deleted_at", None) is None:
                pass  # deleted_at stays None unless explicitly set

            self._store[typ][obj.id] = obj
        self._pending.clear()

    async def flush(self) -> None:
        """Flush pending objects to store without full commit semantics."""
        for obj in self._pending:
            typ = type(obj)
            if typ not in self._store:
                self._store[typ] = {}
            # Assign ID if not already set
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
                        items.sort(key=lambda i: getattr(i, key, None) or 0, reverse=reverse)

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


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_db() -> FakeSession:
    """Provide a fresh fake session per test."""
    return FakeSession()


@pytest.fixture
def app(tmp_path: Path, fake_db: FakeSession) -> FastAPI:
    """Provide the FastAPI app with overridden DB dependency."""
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
def pdf_id(fake_db: FakeSession) -> uuid.UUID:
    """Pre-seed a PDF record into the fake database."""
    pdf = PDF(
        filename="test.pdf",
        original_filename="test.pdf",
        file_size_bytes=1024,
        page_count=3,
        md5_hash="d41d8cd98f00b204e9800998ecf8427e",
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        mime_type="application/pdf",
        status=PDFStatus.READY,
    )
    fake_db.add(pdf)
    import asyncio  # noqa: PLC0415

    asyncio.run(fake_db.commit())
    return pdf.id


# ---------------------------------------------------------------------------
# Manager unit tests
# ---------------------------------------------------------------------------


class TestCreateGTVersion:
    """Test ``GroundTruthManager.create_gt_version``."""

    async def test_create_manual_gt(
        self, fake_db: FakeSession, pdf_id: uuid.UUID
    ) -> None:
        """Given a PDF, When creating a manual GT, Then it is stored with correct source."""
        manager = GroundTruthManager(db=fake_db)
        gt = await manager.create_gt_version(
            pdf_id=pdf_id,
            source=GroundTruthSource.MANUAL,
            notes="Test manual GT",
        )
        assert gt.id is not None
        assert gt.pdf_id == pdf_id
        assert gt.source == GroundTruthSource.MANUAL
        assert gt.version_number == 1
        assert gt.notes == "Test manual GT"
        assert gt.deleted_at is None

    async def test_create_gt_increments_version(
        self, fake_db: FakeSession, pdf_id: uuid.UUID
    ) -> None:
        """Given an existing GT v1, When creating another, Then version is incremented."""
        manager = GroundTruthManager(db=fake_db)
        gt1 = await manager.create_gt_version(
            pdf_id=pdf_id,
            source=GroundTruthSource.MANUAL,
        )
        assert gt1.version_number == 1

        gt2 = await manager.create_gt_version(
            pdf_id=pdf_id,
            source=GroundTruthSource.MANUAL,
            notes="Second version",
        )
        assert gt2.version_number == 2


class TestGetGTVersion:
    """Test ``GroundTruthManager.get_gt_version``."""

    async def test_get_existing_gt(
        self, fake_db: FakeSession, pdf_id: uuid.UUID
    ) -> None:
        """Given a stored GT, When fetched, Then all fields are returned."""
        manager = GroundTruthManager(db=fake_db)
        created = await manager.create_gt_version(
            pdf_id=pdf_id,
            source=GroundTruthSource.MANUAL,
            notes="Fetch me",
        )
        fetched = await manager.get_gt_version(created.id)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.version_number == 1

    async def test_get_nonexistent_gt(self, fake_db: FakeSession) -> None:
        """Given no GT with the ID, When fetched, Then None is returned."""
        manager = GroundTruthManager(db=fake_db)
        result = await manager.get_gt_version(uuid.uuid4())
        assert result is None


class TestListGTVersions:
    """Test ``GroundTruthManager.list_gt_versions``."""

    async def test_list_all_gt_versions(
        self, fake_db: FakeSession, pdf_id: uuid.UUID
    ) -> None:
        """Given multiple GT versions, When listed, Then all non-deleted are returned."""
        manager = GroundTruthManager(db=fake_db)
        await manager.create_gt_version(pdf_id=pdf_id, source=GroundTruthSource.MANUAL)
        await manager.create_gt_version(
            pdf_id=pdf_id,
            source=GroundTruthSource.MANUAL,
            notes="Second",
        )
        versions = await manager.list_gt_versions()
        assert len(versions) == 2

    async def test_list_filtered_by_pdf(
        self, fake_db: FakeSession, pdf_id: uuid.UUID
    ) -> None:
        """Given GTs for multiple PDFs, When filtering by pdf_id, Then only matching returned."""
        # Create a second PDF.
        pdf2 = PDF(
            filename="other.pdf",
            original_filename="other.pdf",
            file_size_bytes=512,
            page_count=1,
            md5_hash="a" * 32,
            sha256_hash="b" * 64,
            mime_type="application/pdf",
            status=PDFStatus.READY,
        )
        fake_db.add(pdf2)
        await fake_db.commit()

        manager = GroundTruthManager(db=fake_db)
        await manager.create_gt_version(pdf_id=pdf_id, source=GroundTruthSource.MANUAL)
        await manager.create_gt_version(pdf_id=pdf2.id, source=GroundTruthSource.MANUAL)

        versions = await manager.list_gt_versions(pdf_id=pdf_id)
        assert len(versions) == 1
        assert versions[0].pdf_id == pdf_id


class TestSoftDeleteGT:
    """Test ``GroundTruthManager.soft_delete_gt``."""

    async def test_soft_delete_sets_deleted_at(
        self, fake_db: FakeSession, pdf_id: uuid.UUID
    ) -> None:
        """Given an active GT, When soft-deleted, Then deleted_at is set."""
        manager = GroundTruthManager(db=fake_db)
        gt = await manager.create_gt_version(
            pdf_id=pdf_id, source=GroundTruthSource.MANUAL
        )
        await manager.soft_delete_gt(gt.id)

        # Should not appear in normal queries.
        fetched = await manager.get_gt_version(gt.id)
        assert fetched is None

    async def test_delete_nonexistent_gt(self, fake_db: FakeSession) -> None:
        """Given no GT with the ID, When soft-deleted, Then an error is raised."""
        manager = GroundTruthManager(db=fake_db)
        with pytest.raises(GroundTruthManagerError, match="not found"):
            await manager.soft_delete_gt(uuid.uuid4())


class TestPromoteGTVersion:
    """Test ``GroundTruthManager.promote_gt_version``."""

    async def test_promote_makes_current(
        self, fake_db: FakeSession, pdf_id: uuid.UUID
    ) -> None:
        """Given two GT versions, When the first is promoted, Then it becomes current."""
        manager = GroundTruthManager(db=fake_db)
        gt1 = await manager.create_gt_version(
            pdf_id=pdf_id, source=GroundTruthSource.MANUAL
        )
        gt2 = await manager.create_gt_version(
            pdf_id=pdf_id, source=GroundTruthSource.MANUAL
        )
        # gt2 is v2, gt1 is v1. Promote gt1.
        await manager.promote_gt_version(gt1.id)

        current = await manager.get_current_gt(pdf_id)
        assert current is not None
        assert current.id == gt1.id
        assert current.version_number == 3  # bumped above v2


class TestGetCurrentGT:
    """Test ``GroundTruthManager.get_current_gt``."""

    async def test_get_current_returns_highest_version(
        self, fake_db: FakeSession, pdf_id: uuid.UUID
    ) -> None:
        """Given multiple versions, When getting current, Then highest version is returned."""
        manager = GroundTruthManager(db=fake_db)
        await manager.create_gt_version(pdf_id=pdf_id, source=GroundTruthSource.MANUAL)
        gt2 = await manager.create_gt_version(
            pdf_id=pdf_id, source=GroundTruthSource.MANUAL
        )
        current = await manager.get_current_gt(pdf_id)
        assert current is not None
        assert current.id == gt2.id

    async def test_get_current_no_gt(self, fake_db: FakeSession) -> None:
        """Given a PDF with no GT, When getting current, Then None is returned."""
        manager = GroundTruthManager(db=fake_db)
        result = await manager.get_current_gt(uuid.uuid4())
        assert result is None


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestCreateGTAPI:
    """Test ``POST /api/v1/ground-truth``."""

    async def test_create_gt_returns_201(
        self, client: AsyncClient, pdf_id: uuid.UUID
    ) -> None:
        """Given a valid request, When creating GT, Then 201 is returned."""
        response = await client.post(
            "/api/v1/ground-truth",
            json={
                "pdf_id": str(pdf_id),
                "source": "manual",
                "notes": "API test",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["version_number"] == 1
        assert data["source"] == "manual"

    async def test_create_gt_invalid_pdf(
        self, client: AsyncClient
    ) -> None:
        """Given a non-existent PDF ID, When creating GT, Then 200 is returned (no FK check)."""
        response = await client.post(
            "/api/v1/ground-truth",
            json={
                "pdf_id": "00000000-0000-0000-0000-000000000001",
                "source": "manual",
            },
        )
        # Manager does not validate PDF existence — GT is created anyway.
        assert response.status_code == 201


class TestGetGTAPI:
    """Test ``GET /api/v1/ground-truth/{id}``."""

    async def test_get_gt_with_pages(
        self, client: AsyncClient, pdf_id: uuid.UUID, fake_db: FakeSession
    ) -> None:
        """Given a GT with page results, When fetched, Then pages are included."""
        # Create GT via API.
        create_resp = await client.post(
            "/api/v1/ground-truth",
            json={"pdf_id": str(pdf_id), "source": "manual"},
        )
        gt_id = create_resp.json()["id"]

        # Manually add a page result to the fake DB.
        gt_version = fake_db.get(GroundTruthVersion, uuid.UUID(gt_id))
        assert gt_version is not None

        page = GTPageResult(
            gt_version_id=uuid.UUID(gt_id),
            page_number=1,
            data={"blocks": [], "tables": []},
            confidence=1.0,
        )
        fake_db.add(page)
        await fake_db.commit()
        # Attach page to the GT relationship for the response.
        gt_version.page_results = [page]

        resp = await client.get(f"/api/v1/ground-truth/{gt_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"]["id"] == gt_id
        assert len(data["pages"]) == 1
        assert data["pages"][0]["page_number"] == 1

    async def test_get_gt_not_found(self, client: AsyncClient) -> None:
        """Given a non-existent GT ID, When fetched, Then 404 is returned."""
        resp = await client.get(
            "/api/v1/ground-truth/00000000-0000-0000-0000-000000000001"
        )
        assert resp.status_code == 404


class TestListGTAPI:
    """Test ``GET /api/v1/ground-truth``."""

    async def test_list_gt_versions(
        self, client: AsyncClient, pdf_id: uuid.UUID
    ) -> None:
        """Given two GT versions, When listed, Then both are returned."""
        await client.post(
            "/api/v1/ground-truth",
            json={"pdf_id": str(pdf_id), "source": "manual"},
        )
        await client.post(
            "/api/v1/ground-truth",
            json={"pdf_id": str(pdf_id), "source": "manual", "notes": "second"},
        )
        resp = await client.get("/api/v1/ground-truth")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 2

    async def test_list_filtered_by_pdf(
        self, client: AsyncClient, pdf_id: uuid.UUID, fake_db: FakeSession
    ) -> None:
        """Given GTs for two PDFs, When filtering by pdf_id, Then only matching."""
        # Create second PDF.
        pdf2 = PDF(
            filename="other.pdf",
            original_filename="other.pdf",
            file_size_bytes=512,
            page_count=1,
            md5_hash="c" * 32,
            sha256_hash="d" * 64,
            mime_type="application/pdf",
            status=PDFStatus.READY,
        )
        fake_db.add(pdf2)
        await fake_db.commit()

        await client.post(
            "/api/v1/ground-truth",
            json={"pdf_id": str(pdf_id), "source": "manual"},
        )
        await client.post(
            "/api/v1/ground-truth",
            json={"pdf_id": str(pdf2.id), "source": "manual"},
        )

        resp = await client.get(f"/api/v1/ground-truth?pdf_id={pdf_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 1


class TestUpdateGTPageAPI:
    """Test ``PUT /api/v1/ground-truth/{id}/pages/{page}``."""

    async def test_update_gt_page(
        self, client: AsyncClient, pdf_id: uuid.UUID, fake_db: FakeSession
    ) -> None:
        """Given a GT with a page, When updating page data, Then new version is created."""
        # Create GT and manually attach a page.
        create_resp = await client.post(
            "/api/v1/ground-truth",
            json={"pdf_id": str(pdf_id), "source": "manual"},
        )
        gt_id = uuid.UUID(create_resp.json()["id"])

        gt_version = fake_db.get(GroundTruthVersion, gt_id)
        assert gt_version is not None
        minimal_data: dict = {
            "blocks": [
                {
                    "type": "text",
                    "bbox": [0, 0, 100, 20],
                    "confidence": 1.0,
                    "order": 0,
                    "lines": [],
                }
            ],
            "tables": [],
        }
        page = GTPageResult(
            gt_version_id=gt_id,
            page_number=1,
            data=minimal_data,
            confidence=1.0,
        )
        fake_db.add(page)
        await fake_db.commit()
        gt_version.page_results = [page]

        # Get all pages for the source so _create_versioned_edit can copy them.
        all_pages_q = [
            p for p in fake_db._store.get(GTPageResult, {}).values()
            if p.gt_version_id == gt_id
        ]
        assert len(all_pages_q) == 1

        new_data: dict = {
            "blocks": [
                {
                    "type": "text",
                    "bbox": [0, 0, 200, 40],
                    "confidence": 0.9,
                    "order": 0,
                    "lines": [],
                }
            ],
            "tables": [],
        }
        resp = await client.put(
            f"/api/v1/ground-truth/{gt_id}/pages/1",
            json={"data": new_data},
        )
        assert resp.status_code == 200
        result = resp.json()
        assert result["page_number"] == 1
        assert result["data"] == new_data

        # Verify new version was created.
        versions = [
            v for v in fake_db._store.get(GroundTruthVersion, {}).values()
            if v.pdf_id == pdf_id and v.deleted_at is None
        ]
        assert len(versions) == 2  # original + new version

    async def test_update_nonexistent_page(
        self, client: AsyncClient, pdf_id: uuid.UUID, fake_db: FakeSession
    ) -> None:
        """Given no page in source GT, When updating, Then 400 is returned."""
        create_resp = await client.post(
            "/api/v1/ground-truth",
            json={"pdf_id": str(pdf_id), "source": "manual"},
        )
        gt_id = create_resp.json()["id"]

        resp = await client.put(
            f"/api/v1/ground-truth/{gt_id}/pages/99",
            json={"data": {"blocks": [], "tables": []}},
        )
        assert resp.status_code == 400


class TestUpdateGTWordAPI:
    """Test ``PUT /api/v1/ground-truth/{id}/pages/{page}/words/{word_idx}``."""

    async def test_update_gt_word(
        self, client: AsyncClient, pdf_id: uuid.UUID, fake_db: FakeSession
    ) -> None:
        """Given a GT page with words, When a word is corrected, Then new version has the fix."""
        create_resp = await client.post(
            "/api/v1/ground-truth",
            json={"pdf_id": str(pdf_id), "source": "manual"},
        )
        gt_id = uuid.UUID(create_resp.json()["id"])

        gt_version = fake_db.get(GroundTruthVersion, gt_id)
        assert gt_version is not None

        word_data = {
            "blocks": [
                {
                    "type": "text",
                    "bbox": [0, 0, 100, 20],
                    "confidence": 1.0,
                    "order": 0,
                    "lines": [
                        {
                            "text": "Hllo world",
                            "bbox": [0, 0, 100, 20],
                            "confidence": 1.0,
                            "order": 0,
                            "words": [
                                {
                                    "text": "Hllo",
                                    "bbox": [0, 0, 40, 20],
                                    "confidence": 0.8,
                                    "order": 0,
                                    "chars": [],
                                },
                                {
                                    "text": "world",
                                    "bbox": [50, 0, 100, 20],
                                    "confidence": 0.9,
                                    "order": 1,
                                    "chars": [],
                                },
                            ],
                        }
                    ],
                }
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

        resp = await client.put(
            f"/api/v1/ground-truth/{gt_id}/pages/1/words/0",
            json={"text": "Hello"},
        )
        assert resp.status_code == 200
        result = resp.json()
        assert result["page_number"] == 1

        # Check the word text was corrected.
        words = result["data"]["blocks"][0]["lines"][0]["words"]
        assert words[0]["text"] == "Hello"

        # Verify line text was also updated.
        assert result["data"]["blocks"][0]["lines"][0]["text"] == "Hello world"

        # Verify new version was created (original + new).
        versions = [
            v for v in fake_db._store.get(GroundTruthVersion, {}).values()
            if v.pdf_id == pdf_id and v.deleted_at is None
        ]
        assert len(versions) == 2


class TestDeleteGTAPI:
    """Test ``DELETE /api/v1/ground-truth/{id}``."""

    async def test_delete_gt_returns_204(
        self, client: AsyncClient, pdf_id: uuid.UUID
    ) -> None:
        """Given an active GT, When deleted, Then 204 is returned."""
        create_resp = await client.post(
            "/api/v1/ground-truth",
            json={"pdf_id": str(pdf_id), "source": "manual"},
        )
        gt_id = create_resp.json()["id"]

        resp = await client.delete(f"/api/v1/ground-truth/{gt_id}")
        assert resp.status_code == 204

    async def test_delete_nonexistent_gt_returns_404(
        self, client: AsyncClient
    ) -> None:
        """Given a non-existent GT ID, When deleted, Then 404 is returned."""
        resp = await client.delete(
            "/api/v1/ground-truth/00000000-0000-0000-0000-000000000001"
        )
        assert resp.status_code == 404


class TestPromoteGTAPI:
    """Test ``POST /api/v1/ground-truth/{id}/promote``."""

    async def test_promote_gt_returns_200(
        self, client: AsyncClient, pdf_id: uuid.UUID
    ) -> None:
        """Given two GT versions, When promoting v1, Then 200 is returned."""
        resp1 = await client.post(
            "/api/v1/ground-truth",
            json={"pdf_id": str(pdf_id), "source": "manual"},
        )
        gt_id = resp1.json()["id"]
        # Create a second version so v1 is not already current.
        await client.post(
            "/api/v1/ground-truth",
            json={"pdf_id": str(pdf_id), "source": "manual"},
        )

        resp = await client.post(f"/api/v1/ground-truth/{gt_id}/promote")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == gt_id
        assert "Promoted" in data["message"]


class TestGetCurrentGTAPI:
    """Test ``GET /api/v1/ground-truth/current/{pdf_id}``."""

    async def test_get_current_gt_returns_promoted(
        self, client: AsyncClient, pdf_id: uuid.UUID, fake_db: FakeSession
    ) -> None:
        """Given a promoted GT, When fetching current, Then it's returned with pages."""
        resp1 = await client.post(
            "/api/v1/ground-truth",
            json={"pdf_id": str(pdf_id), "source": "manual"},
        )
        gt_id = resp1.json()["id"]

        # Attach a page to the GT for the response.
        gt_version = fake_db.get(GroundTruthVersion, uuid.UUID(gt_id))
        assert gt_version is not None
        page = GTPageResult(
            gt_version_id=uuid.UUID(gt_id),
            page_number=1,
            data={"blocks": [], "tables": []},
            confidence=1.0,
        )
        fake_db.add(page)
        await fake_db.commit()
        gt_version.page_results = [page]

        resp = await client.get(f"/api/v1/ground-truth/current/{pdf_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"]["id"] == gt_id

    async def test_get_current_no_gt(self, client: AsyncClient) -> None:
        """Given a PDF with no GT, When fetching current, Then message is returned."""
        resp = await client.get(
            "/api/v1/ground-truth/current/00000000-0000-0000-0000-000000000001"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "No ground truth versions found" in data["message"]


class TestGTVersioning:
    """Test that edits correctly create new versions and increment version_number."""

    async def test_edit_jsonb_increments_version(
        self, client: AsyncClient, pdf_id: uuid.UUID, fake_db: FakeSession
    ) -> None:
        """Given a GT page, When editing, Then a new version with incremented number is created."""
        create_resp = await client.post(
            "/api/v1/ground-truth",
            json={"pdf_id": str(pdf_id), "source": "manual"},
        )
        gt_id = uuid.UUID(create_resp.json()["id"])
        assert create_resp.json()["version_number"] == 1

        gt_version = fake_db.get(GroundTruthVersion, gt_id)
        assert gt_version is not None
        page = GTPageResult(
            gt_version_id=gt_id,
            page_number=1,
            data={"blocks": [], "tables": []},
            confidence=1.0,
        )
        fake_db.add(page)
        await fake_db.commit()
        gt_version.page_results = [page]

        # Edit the page.
        edit_data: dict = {
            "blocks": [
                {
                    "type": "text",
                    "bbox": [0, 0, 10, 10],
                    "confidence": 1.0,
                    "order": 0,
                    "lines": [],
                }
            ],
            "tables": [],
        }
        resp = await client.put(
            f"/api/v1/ground-truth/{gt_id}/pages/1",
            json={"data": edit_data},
        )
        assert resp.status_code == 200

        # Check that a new version exists with v2.
        versions = [
            v for v in fake_db._store.get(GroundTruthVersion, {}).values()
            if v.pdf_id == pdf_id and v.deleted_at is None
        ]
        version_numbers = sorted(v.version_number for v in versions)
        assert version_numbers == [1, 2]  # original v1 + new v2
