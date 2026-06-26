"""Evaluation scoring pipeline for OCRScore.

Provides character-level and word-level evaluation metrics by comparing OCR
output against ground truth using the alignment algorithms from
:mod:`backend.alignment.aligner`, plus bootstrap confidence intervals.
"""

from backend.evaluation._evaluators import evaluate_page, evaluate_run
from backend.evaluation.bootstrap import (
    add_ci_to_scores,
    bootstrap_ci,
    bootstrap_compare,
    resample_statistic,
)
from backend.evaluation.scoring import (
    compute_cer,
    compute_char_metrics,
    compute_precision_recall_f1,
    compute_wer,
    compute_word_metrics,
)

__all__ = [
    "add_ci_to_scores",
    "bootstrap_ci",
    "bootstrap_compare",
    "compute_cer",
    "compute_char_metrics",
    "compute_precision_recall_f1",
    "compute_wer",
    "compute_word_metrics",
    "evaluate_page",
    "evaluate_run",
    "resample_statistic",
]
