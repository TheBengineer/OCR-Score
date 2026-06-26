"""OCRScore API routers — grouped by domain resource."""

from backend.routers.documents import documents_router

__all__: list[str] = [
    "documents_router",
]
