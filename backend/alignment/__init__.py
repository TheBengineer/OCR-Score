"""Sequence alignment and spatial clustering for OCR evaluation.

Provides Needleman-Wunsch (global) and Smith-Waterman (local) word-level
and character-level alignment, plus the full ``align_ocr_texts`` pipeline
for comparing OCR outputs against ground truth.

The spatial clustering sub-module groups word bounding boxes into logical
text blocks (paragraphs, columns, headings) based on spatial proximity,
and orders them in reading order for downstream sequence alignment.

The comparator sub-module provides multi-engine alignment for comparing
OCR outputs across engines side-by-side.
"""

from backend.alignment.aligner import (
    align_ocr_texts,
    character_level_align,
    needleman_wunsch,
    smith_waterman,
)
from backend.alignment.clustering import (
    cluster_words_to_blocks,
    detect_columns,
    estimate_reading_order,
)
from backend.alignment.comparator import (
    align_multiple_engine_pages,
    build_comparison_grid,
)

__all__ = [
    "needleman_wunsch",
    "smith_waterman",
    "align_ocr_texts",
    "character_level_align",
    "cluster_words_to_blocks",
    "detect_columns",
    "estimate_reading_order",
    "align_multiple_engine_pages",
    "build_comparison_grid",
]
