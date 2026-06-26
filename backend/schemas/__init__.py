"""Pydantic v2 schemas for OCRScore API request/response models."""

from backend.schemas.engine import (
    OCREngineCreate,
    OCREngineRead,
    OCREngineUpdate,
)
from backend.schemas.ground_truth import (
    GroundTruthVersionCreate,
    GroundTruthVersionRead,
    GTPageResultCreate,
    GTPageResultRead,
    GTPageResultUpdate,
)
from backend.schemas.page_result import (
    PageResultChar,
    PageResultCreate,
    PageResultLine,
    PageResultRead,
    PageResultWord,
    PageTable,
    PageTableBlock,
    PageTableCell,
)
from backend.schemas.pdf import PDFCreate, PDFRead, PDFUpdate
from backend.schemas.run import OCRRunCreate, OCRRunRead, OCRRunUpdate
from backend.schemas.score import ScoreCreate, ScoreRead, ScoreSummaryRead

__all__: list[str] = [
    # PDF
    "PDFCreate",
    "PDFRead",
    "PDFUpdate",
    # Engine
    "OCREngineCreate",
    "OCREngineRead",
    "OCREngineUpdate",
    # Run
    "OCRRunCreate",
    "OCRRunRead",
    "OCRRunUpdate",
    # PageResult
    "PageResultChar",
    "PageResultWord",
    "PageResultLine",
    "PageTableBlock",
    "PageTableCell",
    "PageTable",
    "PageResultCreate",
    "PageResultRead",
    # Ground Truth
    "GroundTruthVersionCreate",
    "GroundTruthVersionRead",
    "GTPageResultCreate",
    "GTPageResultRead",
    "GTPageResultUpdate",
    # Score
    "ScoreCreate",
    "ScoreRead",
    "ScoreSummaryRead",
]
