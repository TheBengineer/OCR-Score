"""Authentication utilities — password hashing, JWT handling, API key verification,
and FastAPI dependencies for role-based access control.

Public API
----------
- ``hash_password`` / ``verify_password`` — bcrypt-based credential management.
- ``create_access_token`` — issue a signed JWT with 24-hour expiry.
- ``verify_api_key`` — SHA-256 hash an API key for storage or lookup.
- ``get_current_user`` — FastAPI dependency that resolves the authenticated user.
- ``require_role`` — FastAPI dependency factory that enforces a minimum role level.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.models import ROLE_HIERARCHY, ApiKey, User
from backend.database import get_db_session
from backend.settings import settings

# ── Password hashing ──────────────────────────────────────────────────────────

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Return a bcrypt hash of *password*."""
    return _pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Return ``True`` if *plain_password* matches the bcrypt *hashed_password*."""
    return _pwd_context.verify(plain_password, hashed_password)


# ── JWT token management ──────────────────────────────────────────────────────


def create_access_token(data: dict[str, Any]) -> str:
    """Create a signed JWT containing *data* with a 24-hour expiry.

    The token includes standard ``exp`` and ``iat`` claims in addition to the
    caller-supplied payload.
    """
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415

    to_encode = data.copy()
    to_encode.update({
        "exp": datetime.now(UTC) + timedelta(hours=24),
        "iat": datetime.now(UTC),
    })
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Decode and validate a JWT. Returns the payload or ``None`` on failure."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError:
        return None


# ── API key hashing ────────────────────────────────────────────────────────────


def hash_api_key(raw_key: str) -> str:
    """Return the SHA-256 hex digest of a raw API key."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate a cryptographically random API key prefixed with ``sk-``.

    Returns
    -------
    str
        The raw API key (e.g. ``sk-a1b2c3d4e5f6...``). This value should be
        returned to the caller exactly once and then **discarded** — only the
        SHA-256 hash is stored.
    """
    raw = secrets.token_hex(32)
    return f"sk-{raw}"


# ── FastAPI dependencies ──────────────────────────────────────────────────────


async def get_current_user(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> User | None:
    """Resolve the authenticated user from the request.

    Supports two authentication schemes in the ``Authorization`` header:

    * ``Bearer <JWT>`` — decode the JWT and look up the user by ID.
    * ``Apikey <key>`` — hash the key and look up the matching ``ApiKey`` record.

    Returns ``None`` when no valid credentials are provided. This makes the
    dependency suitable for **optional** use on GET endpoints — the endpoint
    can behave differently for authenticated and unauthenticated callers.

    Raises
    ------
    HTTPException 401
        If the Authorization header is present but the credentials are invalid.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None

    try:
        scheme, credentials = auth_header.split(" ", 1)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format",
        ) from None

    if scheme.lower() == "bearer":
        return await _resolve_jwt_user(credentials, db)
    elif scheme.lower() == "apikey":
        return await _resolve_api_key_user(credentials, db)
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Unsupported authentication scheme: {scheme}",
        )


async def _resolve_jwt_user(token: str, db: AsyncSession) -> User | None:
    """Decode a JWT and return the corresponding User or None."""
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    user_id_str = payload.get("user_id")
    if user_id_str is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing user_id claim",
        )

    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user_id in token",
        ) from None

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


async def _resolve_api_key_user(raw_key: str, db: AsyncSession) -> User | None:
    """Hash the raw API key and return the owning User or None."""
    key_hash = hash_api_key(raw_key)
    result = await db.execute(
        select(ApiKey).where(ApiKey.key_hash == key_hash),
    )
    api_key = result.scalar_one_or_none()
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # Update last_used_at
    from datetime import UTC, datetime  # noqa: PLC0415

    api_key.last_used_at = datetime.now(UTC)

    result = await db.execute(select(User).where(User.id == api_key.user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key owner not found",
        )
    return user


def require_role(min_role: str) -> Any:
    """Return a FastAPI dependency that requires at least *min_role* privilege.

    The role hierarchy is ``viewer (1) < reviewer (2) < admin (3)``.

    Usage::

        @router.get("/admin-only")
        async def admin_endpoint(
            current_user: User = Depends(require_role("admin")),
        ):
            ...

    Raises
    ------
    HTTPException 401
        If the request is unauthenticated.
    HTTPException 403
        If the authenticated user's role is below *min_role*.
    """

    async def _check_role(
        current_user: User | None = Depends(get_current_user),  # noqa: B008
    ) -> User:
        if current_user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )

        user_level = ROLE_HIERARCHY.get(current_user.role.value, 0)
        required_level = ROLE_HIERARCHY.get(min_role, 0)

        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Insufficient permissions. Required role: {min_role}, "
                    f"actual role: {current_user.role.value}"
                ),
            )

        return current_user

    return _check_role
