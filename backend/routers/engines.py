"""API router for listing available OCR engine plugins and their secrets."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db_session
from backend.engine.registry import registry as engine_registry
from backend.engine.secret_store import SecretStore

engine_router = APIRouter(prefix="/api/v1/engines", tags=["engines"])


def _discover_engines() -> None:
    """Import known engine modules so they self-register (ignoring missing deps)."""
    engine_modules = [
        "backend.engines.tesseract",
        "backend.engines.gcp_document_ai",
        "backend.engines.textract",
        "backend.engines.vlm_olmocr",
        "backend.engines.vlm_deepseek",
    ]
    for mod_name in engine_modules:
        try:
            __import__(mod_name)
        except ImportError:
            pass
    engine_registry.discover()


@engine_router.get("")
async def list_engines() -> list[dict]:
    """List all registered OCR engines with their metadata, config schemas,
    and secret schemas."""
    _discover_engines()
    return [
        {
            "engine_id": engine.engine_id,
            "display_name": engine.display_name,
            "version": engine.version,
            "config_schema": engine.get_config_schema(),
            "secret_schema": [
                {"key": s.key, "env_var": s.env_var, "display_name": s.display_name, "description": s.description}
                for s in engine.get_secret_schema()
            ],
        }
        for engine in engine_registry.list()
    ]


@engine_router.get("/{slug}/secrets")
async def get_engine_secrets(
    slug: str,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, str]:
    """Return all stored secrets for an engine as a ``{key: value}`` dict.

    Only includes keys declared in the engine's ``secret_schema``.
    Values are returned for the UI to populate the settings page.
    """
    _discover_engines()
    store = SecretStore(db)
    return await store.list(slug)


@engine_router.put("/{slug}/secrets")
async def set_engine_secrets(
    slug: str,
    body: dict[str, str],
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, str]:
    """Upsert secrets for an engine.

    Accepts a ``{key: value}`` dict.  Keys not in the engine's
    ``secret_schema`` are silently ignored.
    """
    _discover_engines()
    store = SecretStore(db)

    # Only allow keys the engine has declared
    try:
        engine = engine_registry.get(slug)
    except Exception:
        from fastapi import HTTPException, status
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Engine '{slug}' not found")

    allowed = {s.key for s in engine.get_secret_schema()}
    for key, value in body.items():
        if key in allowed:
            await store.set(slug, key, value)

    return await store.list(slug)
