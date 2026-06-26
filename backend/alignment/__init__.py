"""Sequence alignment and spatial clustering for OCR evaluation.

Provides Needleman-Wunsch (global) and Smith-Waterman (local) word-level
and character-level alignment, plus the full ``align_ocr_texts`` pipeline
for comparing OCR outputs against ground truth.

The spatial clustering sub-module groups word bounding boxes into logical
text blocks (paragraphs, columns, headings) based on spatial proximity,
and orders them in reading order for downstream sequence alignment.
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

__all__ = [
    "needleman_wunsch",
    "smith_waterman",
    "align_ocr_texts",
    "character_level_align",
    "cluster_words_to_blocks",
    "detect_columns",
    "estimate_reading_order",
]
