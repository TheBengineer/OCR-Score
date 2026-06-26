"""Authentication package — users, API keys, JWT tokens, and role-based permissions.

Sub-modules
-----------
- ``models`` — SQLAlchemy ``User`` and ``ApiKey`` models.
- ``auth`` — Password hashing, JWT creation/validation, FastAPI dependencies.
- ``schemas`` — Pydantic v2 request/response schemas for auth endpoints.

Public API (re-exported)
------------------------
- ``User``, ``ApiKey``, ``UserRole`` — data models.
- ``hash_password``, ``verify_password`` — password management.
- ``create_access_token`` — JWT issuance.
- ``verify_api_key`` / ``hash_api_key`` / ``generate_api_key`` — API key utilities.
- ``get_current_user`` — optional auth resolver (FastAPI dependency).
- ``require_role`` — role-gated access (FastAPI dependency factory).
"""

from backend.auth.auth import (
    create_access_token,
    decode_access_token,
    generate_api_key,
    get_current_user,
    hash_api_key,
    hash_password,
    require_role,
    verify_password,
)
from backend.auth.models import ROLE_HIERARCHY, ApiKey, User, UserRole

__all__: list[str] = [
    "User",
    "UserRole",
    "ApiKey",
    "ROLE_HIERARCHY",
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_access_token",
    "hash_api_key",
    "generate_api_key",
    "get_current_user",
    "require_role",
]
