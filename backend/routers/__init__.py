"""OCRScore API routers — grouped by domain resource."""

from backend.routers.auth import auth_router
from backend.routers.batch import batch_router
from backend.routers.comparison import comparison_router
from backend.routers.documents import documents_router
from backend.routers.engines import engine_router
from backend.routers.ground_truth import gt_router
from backend.routers.reports import reports_router
from backend.routers.runs import pages_router, runs_router
from backend.routers.scoring import scoring_router
from backend.routers.ws import ws_router

__all__: list[str] = [
    "auth_router",
    "batch_router",
    "comparison_router",
    "documents_router",
    "engine_router",
    "gt_router",
    "pages_router",
    "reports_router",
    "runs_router",
    "scoring_router",
    "ws_router",
]
