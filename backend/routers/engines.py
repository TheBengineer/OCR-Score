"""API router for listing available OCR engine plugins."""

from fastapi import APIRouter

from backend.engine.registry import registry as engine_registry

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
    """List all registered OCR engines with their metadata and config schemas."""
    _discover_engines()
    return [
        {
            "engine_id": engine.engine_id,
            "display_name": engine.display_name,
            "version": engine.version,
            "config_schema": engine.get_config_schema(),
        }
        for engine in engine_registry.list()
    ]
