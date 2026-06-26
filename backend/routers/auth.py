"""Authentication router — login, API key management, and user registration.

Endpoints
---------
- ``POST /api/v1/auth/login`` — Authenticate with username/password, receive a JWT.
- ``POST /api/v1/auth/users`` — Register a new user (admin only).
- ``POST /api/v1/auth/api-keys`` — Create a new API key (authenticated).
- ``GET /api/v1/auth/api-keys`` — List the current user's API keys (authenticated).
- ``DELETE /api/v1/auth/api-keys/{id}`` — Revoke an API key (authenticated).
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.auth import (
    create_access_token,
    generate_api_key,
    hash_api_key,
    hash_password,
    require_role,
    verify_password,
)
from backend.auth.models import ApiKey, User
from backend.auth.schemas import (
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyRead,
    LoginRequest,
    LoginResponse,
    UserCreate,
    UserRead,
)
from backend.database import get_db_session

# ── Router ────────────────────────────────────────────────────────────────────

auth_router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# ── Dependencies ──────────────────────────────────────────────────────────────

SessionDep = Annotated[AsyncSession, Depends(get_db_session)]
AdminDep = Annotated[User, Depends(require_role("admin"))]
AuthDep = Annotated[User, Depends(require_role("viewer"))]


# ── Endpoints ─────────────────────────────────────────────────────────────────


@auth_router.post("/login", response_model=LoginResponse)
async def login(
    body: LoginRequest,
    db: SessionDep,
) -> LoginResponse:
    """Authenticate with username and password, returning a JWT.

    The returned token expires in 24 hours and includes the user's ID and role.
    """
    result = await db.execute(
        select(User).where(User.username == body.username),
    )
    user = result.scalar_one_or_none()

    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    access_token = create_access_token({
        "user_id": str(user.id),
        "role": user.role.value,
    })

    return LoginResponse(
        access_token=access_token,
        user=UserRead.model_validate(user),
    )


@auth_router.post("/users", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user(
    body: UserCreate,
    db: SessionDep,
    _: AdminDep,
) -> User:
    """Create a new user. Requires ``admin`` role."""
    # Check for existing username / email
    existing = await db.execute(
        select(User).where(
            (User.username == body.username) | (User.email == body.email),
        ),
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this username or email already exists",
        )

    user = User(
        username=body.username,
        email=body.email,
        hashed_password=hash_password(body.password),
        role=body.role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@auth_router.post("/api-keys", response_model=ApiKeyCreated, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    body: ApiKeyCreate,
    db: SessionDep,
    current_user: AuthDep,
) -> ApiKeyCreated:
    """Create a new API key for the current user.

    The raw key value is returned **only once** in the response. It cannot be
    retrieved again later.
    """
    raw_key = generate_api_key()
    key_hash = hash_api_key(raw_key)

    api_key = ApiKey(
        user_id=current_user.id,
        key_hash=key_hash,
        name=body.name,
    )
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    return ApiKeyCreated(
        id=api_key.id,
        name=api_key.name,
        key=raw_key,
        created_at=api_key.created_at,
    )


@auth_router.get("/api-keys", response_model=list[ApiKeyRead])
async def list_api_keys(
    db: SessionDep,
    current_user: AuthDep,
) -> list[ApiKey]:
    """List all API keys belonging to the current user."""
    result = await db.execute(
        select(ApiKey)
        .where(ApiKey.user_id == current_user.id)
        .order_by(ApiKey.created_at.desc()),
    )
    return list(result.scalars().all())


@auth_router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: uuid.UUID,
    db: SessionDep,
    current_user: AuthDep,
) -> None:
    """Revoke (delete) an API key. Users can only revoke their own keys."""
    result = await db.execute(
        select(ApiKey).where(
            ApiKey.id == key_id,
            ApiKey.user_id == current_user.id,
        ),
    )
    api_key = result.scalar_one_or_none()
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    await db.delete(api_key)
    await db.commit()
