"""Evaluation scoring pipeline for OCRScore.

Provides character-level and word-level evaluation metrics by comparing OCR
output against ground truth using the alignment algorithms from
:mod:`backend.alignment.aligner`, plus bootstrap confidence intervals and
consensus-entropy based automatic ground truth generation.
"""

from backend.evaluation._evaluators import evaluate_page, evaluate_run
from backend.evaluation.bootstrap import (
    add_ci_to_scores,
    bootstrap_ci,
    bootstrap_compare,
    resample_statistic,
)
from backend.evaluation.consensus import (
    build_ground_truth,
    compute_confidence_weighted_consensus,
    compute_consensus_entropy,
)
from backend.evaluation.scoring import (
    compute_cer,
    compute_char_metrics,
    compute_precision_recall_f1,
    compute_wer,
    compute_word_metrics,
)
from backend.evaluation.table_scoring import (
    compute_table_structure_metrics,
    grits_con,
    grits_loc,
    grits_top,
)

__all__ = [
    "add_ci_to_scores",
    "bootstrap_ci",
    "bootstrap_compare",
    "build_ground_truth",
    "compute_cer",
    "compute_char_metrics",
    "compute_confidence_weighted_consensus",
    "compute_consensus_entropy",
    "compute_precision_recall_f1",
    "compute_table_structure_metrics",
    "compute_wer",
    "compute_word_metrics",
    "evaluate_page",
    "evaluate_run",
    "grits_con",
    "grits_loc",
    "grits_top",
    "resample_statistic",
]
