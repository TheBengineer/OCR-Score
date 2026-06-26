"""SecretStore — read/write engine secrets through a simple async CRUD API.

Usage::

    store = SecretStore(db_session)
    await store.set("tesseract", "some_key", "some_value")
    val = await store.get("tesseract", "some_key")   # "some_value"
    all = await store.list("tesseract")               # {"some_key": "some_value"}
"""

from __future__ import annotations

from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.engine_secret import EngineSecret


class SecretStore:
    """Read/write access to engine secrets backed by the ``engine_secrets`` table."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get(self, engine_slug: str, key: str) -> str | None:
        """Return the value for *(engine_slug, key)* or ``None``."""
        result = await self._db.execute(
            select(EngineSecret).where(
                EngineSecret.engine_slug == engine_slug,
                EngineSecret.key == key,
            ),
        )
        record = result.scalars().one_or_none()
        return record.value if record is not None else None

    async def set(self, engine_slug: str, key: str, value: str) -> None:
        """Upsert a secret value for *(engine_slug, key)*."""
        result = await self._db.execute(
            select(EngineSecret).where(
                EngineSecret.engine_slug == engine_slug,
                EngineSecret.key == key,
            ),
        )
        record = result.scalars().one_or_none()
        if record is not None:
            record.value = value
        else:
            self._db.add(EngineSecret(engine_slug=engine_slug, key=key, value=value))
        await self._db.commit()

    async def list(self, engine_slug: str) -> dict[str, str]:
        """Return all secrets for *engine_slug* as a ``{key: value}`` dict."""
        result = await self._db.execute(
            select(EngineSecret).where(
                EngineSecret.engine_slug == engine_slug,
            ),
        )
        return {r.key: r.value for r in result.scalars().all()}

    async def delete(self, engine_slug: str, key: str) -> None:
        """Remove a single secret."""
        await self._db.execute(
            sa_delete(EngineSecret).where(
                EngineSecret.engine_slug == engine_slug,
                EngineSecret.key == key,
            ),
        )
        await self._db.commit()
