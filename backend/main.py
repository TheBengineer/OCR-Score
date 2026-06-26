from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from backend.database import engine
from backend.routers import (
    auth_router,
    batch_router,
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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan: ensure storage directory exists on startup,
    dispose of the database engine on shutdown."""
    Path(settings.storage_path).mkdir(parents=True, exist_ok=True)
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
