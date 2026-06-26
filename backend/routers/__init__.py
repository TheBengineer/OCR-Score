"""OCRScore API routers — grouped by domain resource."""

from backend.routers.documents import documents_router
from backend.routers.ground_truth import gt_router
from backend.routers.runs import runs_router
from backend.routers.ws import ws_router

__all__: list[str] = [
    "documents_router",
    "gt_router",
    "runs_router",
    "ws_router",
]
