from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from sqlalchemy import select

from backend.database import async_session_factory, engine
from backend.engine.registry import registry as engine_registry
from backend.models.engine import OCREngine
from backend.routers import (
    auth_router,
    batch_router,
    codegen_router,
    comparison_router,
    documents_router,
    engine_router,
    gt_router,
    pages_router,
    reports_router,
    runs_router,
    ws_router,
)
from backend.settings import settings


async def _seed_engines() -> None:
    """Sync the in-memory engine registry into the ``ocr_engines`` DB table.

    On every startup, registered engine plugins are upserted into the database
    so that ``RunOrchestrator.create_run`` can find them by slug via the FK
    relationship.  Existing records are updated (version, description, config
    schema); removed plugins are not auto-deleted (conservative).
    """
    engine_registry.discover()
    async with async_session_factory() as session:
        for eng in engine_registry.list():
            result = await session.execute(
                select(OCREngine).where(OCREngine.slug == eng.engine_id),
            )
            existing = result.scalars().one_or_none()
            if existing is not None:
                existing.version = eng.version
                existing.display_name = eng.display_name
                existing.description = getattr(eng, "description", None)
                existing.config_schema = eng.get_config_schema()
            else:
                session.add(
                    OCREngine(
                        slug=eng.engine_id,
                        display_name=eng.display_name,
                        version=eng.version,
                        description=getattr(eng, "description", None),
                        config_schema=eng.get_config_schema(),
                    ),
                )
        await session.commit()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan: ensure storage directory exists on startup,
    seed engine records into the database, dispose of the engine on shutdown."""
    Path(settings.storage_path).mkdir(parents=True, exist_ok=True)
    await _seed_engines()
    yield
    await engine.dispose()


app = FastAPI(
    title="OCRScore",
    version="0.1.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan,
)

app.include_router(auth_router)
app.include_router(batch_router)
app.include_router(codegen_router)
app.include_router(comparison_router)
app.include_router(documents_router)
app.include_router(engine_router)
app.include_router(runs_router)
app.include_router(pages_router)  # /api/v1/pages/compare
app.include_router(gt_router)
app.include_router(reports_router)
app.include_router(ws_router)


@app.get("/api/v1/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}
