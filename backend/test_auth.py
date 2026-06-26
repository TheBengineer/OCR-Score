"""Integration tests for the authentication and role-based access control system.

Test coverage
-------------
- Login with valid / invalid credentials (``test_login_valid``, ``test_login_invalid``).
- JWT-based access to protected endpoints.
- API key authentication (``test_api_key_auth``, ``test_api_key_invalid``).
- Role-based access control (``test_role_access_admin``, ``test_role_access_viewer``).
- API key lifecycle (``test_create_api_key``, list, revoke).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
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

from backend.auth.auth import (
    create_access_token,
    hash_api_key,
    hash_password,
)
from backend.auth.models import ApiKey, User, UserRole
from backend.database import get_db_session
from backend.main import app as _app

# ---------------------------------------------------------------------------
# Fake SQLAlchemy session (same pattern as test_upload.py / test_runs_api.py)
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


def _extract_conditions(
    whereclause: Any,
) -> list[tuple[str, Any, Any]]:
    """Extract ``(field_name, operator, value)`` tuples from a WHERE clause.

    Supports both AND and OR ``BooleanClauseList`` groupings. For OR clauses
    the extracted conditions from each sub-clause are concatenated into a
    single flat list and applied with OR semantics downstream.
    """
    if whereclause is None:
        return []

    if isinstance(whereclause, BooleanClauseList):
        # Determine if this is an OR or AND clause by inspecting the first
        # operator's behaviour.
        is_or = False
        if hasattr(whereclause, "operator"):
            op_func = whereclause.operator
            is_or = op_func is operators.or_ or (
                hasattr(op_func, "__name__") and "or_" in str(op_func.__name__)
            )

        all_conds: list[tuple[str, Any, Any]] = []
        for clause in whereclause.clauses:
            sub = _extract_conditions(clause)
            if is_or:
                # Mark sub-conditions with a sentinel to distinguish OR groups.
                all_conds.extend(("__or_group__",) + c for c in sub)
            else:
                all_conds.extend(sub)
        return all_conds

    if isinstance(whereclause, BinaryExpression):
        left = whereclause.left
        right = whereclause.right
        op = whereclause.operator

        if hasattr(left, "key"):
            field_name = str(left.key)
        elif isinstance(left, InstrumentedAttribute):
            field_name = left.key
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

        # Split conditions into AND groups and OR groups.
        # Tuples starting with "__or_group__" form OR clauses — an item passes
        # if ANY condition in the same OR group matches.
        and_conds: list[tuple[str, Any, Any]] = []
        or_groups: list[list[tuple[str, Any, Any]]] = []
        current_or: list[tuple[str, Any, Any]] | None = None

        for cond in conditions:
            if len(cond) == 4 and cond[0] == "__or_group__":
                if current_or is None:
                    current_or = []
                    or_groups.append(current_or)
                current_or.append(cond[1:])
            else:
                current_or = None
                and_conds.append(cond)

        def _matches(item: Any, field: str, op: Any, val: Any) -> bool:
            attr = getattr(item, field, None)
            if op is operators.eq:
                return attr == val
            elif op is operators.is_:
                return attr is None
            elif op is operators.isnot:
                return attr is not None
            elif op is operators.in_op:
                return attr in val
            return True

        # Apply AND conditions (all must match)
        for field_name, op, value in and_conds:
            items = [i for i in items if _matches(i, field_name, op, value)]

        # Apply OR groups (any must match)
        for or_group in or_groups:
            items = [
                i for i in items
                if any(_matches(i, f, o, v) for f, o, v in or_group)
            ]

        return FakeResult(items)

    async def close(self) -> None:
        pass

    async def delete(self, obj: Any) -> None:
        """Remove an object from the store."""
        typ = type(obj)
        if typ in self._store:
            self._store[typ] = {
                k: v for k, v in self._store[typ].items() if v is not obj
            }


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

TEST_USERNAME = "testuser"
TEST_PASSWORD = "testpassword123"
TEST_EMAIL = "test@example.com"


@pytest.fixture
def fake_db() -> FakeSession:
    """Provide a fresh fake session per test."""
    return FakeSession()


@pytest.fixture
def app(fake_db: FakeSession) -> FastAPI:
    """Provide the FastAPI app with overridden DB dependency."""
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
def admin_user(fake_db: FakeSession) -> User:
    """Pre-seed an admin user and return it."""
    user = User(
        username="admin",
        email="admin@example.com",
        hashed_password=hash_password("adminpass123"),
        role=UserRole.ADMIN,
        created_at=datetime.now(UTC),
    )
    fake_db.add(user)
    import asyncio  # noqa: PLC0415

    asyncio.run(fake_db.commit())
    return user


@pytest.fixture
def viewer_user(fake_db: FakeSession) -> User:
    """Pre-seed a viewer user and return it."""
    user = User(
        username=TEST_USERNAME,
        email=TEST_EMAIL,
        hashed_password=hash_password(TEST_PASSWORD),
        role=UserRole.VIEWER,
        created_at=datetime.now(UTC),
    )
    fake_db.add(user)
    import asyncio  # noqa: PLC0415

    asyncio.run(fake_db.commit())
    return user


@pytest.fixture
def reviewer_user(fake_db: FakeSession) -> User:
    """Pre-seed a reviewer user and return it."""
    user = User(
        username="reviewer",
        email="reviewer@example.com",
        hashed_password=hash_password("reviewerpass123"),
        role=UserRole.REVIEWER,
        created_at=datetime.now(UTC),
    )
    fake_db.add(user)
    import asyncio  # noqa: PLC0415

    asyncio.run(fake_db.commit())
    return user


@pytest.fixture
def admin_token(admin_user: User) -> str:
    """Return a valid JWT for the admin user."""
    return create_access_token({
        "user_id": str(admin_user.id),
        "role": admin_user.role.value,
    })


@pytest.fixture
def viewer_token(viewer_user: User) -> str:
    """Return a valid JWT for the viewer user."""
    return create_access_token({
        "user_id": str(viewer_user.id),
        "role": viewer_user.role.value,
    })


@pytest.fixture
def viewer_api_key(viewer_user: User, fake_db: FakeSession) -> tuple[str, str]:
    """Pre-seed an API key for the viewer user.

    Returns a tuple of ``(raw_key, api_key_id)``.
    """
    raw_key = "sk-test-viewer-key-12345"
    key_hash = hash_api_key(raw_key)
    api_key = ApiKey(
        user_id=viewer_user.id,
        key_hash=key_hash,
        name="test-key",
        created_at=datetime.now(UTC),
    )
    fake_db.add(api_key)
    import asyncio  # noqa: PLC0415

    asyncio.run(fake_db.commit())
    return raw_key, str(api_key.id)


# ---------------------------------------------------------------------------
# Tests — Login
# ---------------------------------------------------------------------------


class TestLogin:
    """Tests for ``POST /api/v1/auth/login``."""

    async def test_login_valid(
        self,
        client: AsyncClient,
        viewer_user: User,
    ) -> None:
        """Given valid credentials, When logging in, Then a JWT token is returned."""
        response = await client.post(
            "/api/v1/auth/login",
            json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["username"] == TEST_USERNAME
        assert data["user"]["role"] == "viewer"
        assert "id" in data["user"]

    async def test_login_invalid(
        self,
        client: AsyncClient,
    ) -> None:
        """Given wrong password, When logging in, Then 401 is returned."""
        response = await client.post(
            "/api/v1/auth/login",
            json={"username": TEST_USERNAME, "password": "wrongpassword"},
        )
        assert response.status_code == 401

    async def test_login_nonexistent_user(
        self,
        client: AsyncClient,
    ) -> None:
        """Given a non-existent username, When logging in, Then 401 is returned."""
        response = await client.post(
            "/api/v1/auth/login",
            json={"username": "nobody", "password": "irrelevant"},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Tests — API Key Authentication
# ---------------------------------------------------------------------------


class TestApiKeyAuth:
    """Tests for API key authentication on protected endpoints."""

    async def test_api_key_auth(
        self,
        client: AsyncClient,
        viewer_api_key: tuple[str, str],
    ) -> None:
        """Given a valid API key, When accessing a protected endpoint, Then access is granted."""
        raw_key, _ = viewer_api_key
        response = await client.post(
            "/api/v1/auth/api-keys",
            json={"name": "second-key"},
            headers={"Authorization": f"Apikey {raw_key}"},
        )
        assert response.status_code == 201

    async def test_api_key_invalid(
        self,
        client: AsyncClient,
    ) -> None:
        """Given an invalid API key, When accessing a protected endpoint, Then 401 is returned."""
        response = await client.post(
            "/api/v1/auth/api-keys",
            json={"name": "should-fail"},
            headers={"Authorization": "Apikey sk-invalid-key"},
        )
        assert response.status_code == 401

    async def test_no_auth_header(
        self,
        client: AsyncClient,
    ) -> None:
        """Given no auth header, When accessing a protected endpoint, Then 401 is returned."""
        response = await client.post(
            "/api/v1/auth/api-keys",
            json={"name": "should-fail"},
        )
        assert response.status_code == 401

    async def test_malformed_auth_header(
        self,
        client: AsyncClient,
    ) -> None:
        """Given a malformed Authorization header, When accessing a protected endpoint, Then 401 is returned."""
        response = await client.post(
            "/api/v1/auth/api-keys",
            json={"name": "should-fail"},
            headers={"Authorization": "BadFormatNoSpace"},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Tests — JWT Authentication (on protected endpoints)
# ---------------------------------------------------------------------------


class TestJWTAuth:
    """Tests for JWT authentication on protected endpoints."""

    async def test_jwt_auth(
        self,
        client: AsyncClient,
        viewer_token: str,
    ) -> None:
        """Given a valid JWT, When creating an API key, Then it succeeds."""
        response = await client.post(
            "/api/v1/auth/api-keys",
            json={"name": "my-key"},
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "my-key"
        assert data["key"].startswith("sk-")

    async def test_jwt_invalid_token(
        self,
        client: AsyncClient,
    ) -> None:
        """Given an invalid JWT, When accessing a protected endpoint, Then 401 is returned."""
        response = await client.post(
            "/api/v1/auth/api-keys",
            json={"name": "should-fail"},
            headers={"Authorization": "Bearer invalid.jwt.token"},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Tests — Role-Based Access Control
# ---------------------------------------------------------------------------


class TestRoleAccess:
    """Tests for role-based access control on admin endpoints."""

    async def test_role_access_admin(
        self,
        client: AsyncClient,
        admin_token: str,
    ) -> None:
        """Given an admin token, When creating a user, Then 201 is returned."""
        response = await client.post(
            "/api/v1/auth/users",
            json={
                "username": "newuser",
                "email": "new@example.com",
                "password": "newuserpass",
                "role": "viewer",
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == "newuser"
        assert data["role"] == "viewer"

    async def test_role_access_viewer(
        self,
        client: AsyncClient,
        viewer_token: str,
    ) -> None:
        """Given a viewer token, When creating a user (admin), Then 403 is returned."""
        response = await client.post(
            "/api/v1/auth/users",
            json={
                "username": "should-fail",
                "email": "fail@example.com",
                "password": "shouldnotwork",
                "role": "viewer",
            },
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert response.status_code == 403

    async def test_role_access_reviewer_denied_admin(
        self,
        client: AsyncClient,
        reviewer_user: User,
    ) -> None:
        """Given a reviewer token, When creating a user (admin), Then 403 is returned."""
        token = create_access_token({
            "user_id": str(reviewer_user.id),
            "role": reviewer_user.role.value,
        })
        response = await client.post(
            "/api/v1/auth/users",
            json={
                "username": "should-fail-2",
                "email": "fail2@example.com",
                "password": "shouldnotwork",
                "role": "viewer",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403


# ---------------------------------------------------------------------------
# Tests — API Key Lifecycle
# ---------------------------------------------------------------------------


class TestApiKeyLifecycle:
    """Tests for API key creation, listing, and revocation."""

    async def test_create_api_key(
        self,
        client: AsyncClient,
        viewer_token: str,
    ) -> None:
        """Given a valid token, When creating an API key, Then the raw key is returned once."""
        response = await client.post(
            "/api/v1/auth/api-keys",
            json={"name": "ci-key"},
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "ci-key"
        assert data["key"].startswith("sk-")
        assert len(data["key"]) > 10
        assert "created_at" in data

    async def test_list_api_keys(
        self,
        client: AsyncClient,
        viewer_token: str,
        viewer_user: User,
        fake_db: FakeSession,
    ) -> None:
        """Given a user has API keys, When listing, Then they are returned."""
        # Seed a couple of keys
        for i in range(2):
            api_key = ApiKey(
                user_id=viewer_user.id,
                key_hash=hash_api_key(f"sk-raw-{i}"),
                name=f"key-{i}",
                created_at=datetime.now(UTC),
            )
            fake_db.add(api_key)

        await fake_db.commit()

        response = await client.get(
            "/api/v1/auth/api-keys",
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    async def test_revoke_api_key(
        self,
        client: AsyncClient,
        viewer_token: str,
        viewer_api_key: tuple[str, str],
    ) -> None:
        """Given an existing API key, When revoked, Then it is deleted."""
        _, key_id = viewer_api_key
        response = await client.delete(
            f"/api/v1/auth/api-keys/{key_id}",
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert response.status_code == 204

    async def test_revoke_nonexistent_key(
        self,
        client: AsyncClient,
        viewer_token: str,
    ) -> None:
        """Given a non-existent API key ID, When revoking, Then 404 is returned."""
        response = await client.delete(
            "/api/v1/auth/api-keys/00000000-0000-0000-0000-000000000001",
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Tests — Password Hashing Utilities
# ---------------------------------------------------------------------------


class TestPasswordHashing:
    """Tests for password hashing (bcrypt via passlib)."""

    def test_hash_and_verify(self) -> None:
        """Given a password, When hashed, Then it can be verified."""
        hashed = hash_password("mypassword")
        assert hashed != "mypassword"
        assert hashed.startswith("$2b$")  # bcrypt prefix
        from backend.auth.auth import verify_password  # noqa: PLC0415

        assert verify_password("mypassword", hashed)
        assert not verify_password("wrongpassword", hashed)


# ---------------------------------------------------------------------------
# Tests — Duplicate User Creation
# ---------------------------------------------------------------------------


class TestCreateUserValidation:
    """Tests for user creation validation."""

    async def test_create_duplicate_username(
        self,
        client: AsyncClient,
        admin_token: str,
        viewer_user: User,
    ) -> None:
        """Given an existing username, When creating a user, Then 409 is returned."""
        response = await client.post(
            "/api/v1/auth/users",
            json={
                "username": TEST_USERNAME,
                "email": "different@example.com",
                "password": "somepassword",
                "role": "viewer",
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 409
