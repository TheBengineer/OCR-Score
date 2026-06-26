"""Evaluation scoring pipeline for OCRScore.

Provides character-level and word-level evaluation metrics by comparing OCR
output against ground truth using the alignment algorithms from
:mod:`backend.alignment.aligner`, plus bootstrap confidence intervals,
consensus-entropy based automatic ground truth generation, and semantic
plausibility scoring for OCR output readability.
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
from backend.evaluation.novel_metrics import (
    compute_all_novel_metrics,
    compute_confidence_calibration_error,
    compute_imagination_rate,
    compute_noise_sensitivity_index,
    validate_metrics_on_fixtures,
)
from backend.evaluation.scoring import (
    compute_cer,
    compute_char_metrics,
    compute_precision_recall_f1,
    compute_wer,
    compute_word_metrics,
)
from backend.evaluation.semantic import (
    compute_fluency_score,
    compute_grammaticality,
    compute_semantic_plausibility,
    compute_semantic_similarity,
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
    "compute_all_novel_metrics",
    "compute_cer",
    "compute_char_metrics",
    "compute_confidence_calibration_error",
    "compute_confidence_weighted_consensus",
    "compute_consensus_entropy",
    "compute_fluency_score",
    "compute_grammaticality",
    "compute_imagination_rate",
    "compute_noise_sensitivity_index",
    "compute_precision_recall_f1",
    "compute_semantic_plausibility",
    "compute_semantic_similarity",
    "compute_table_structure_metrics",
    "compute_wer",
    "compute_word_metrics",
    "evaluate_page",
    "evaluate_run",
    "grits_con",
    "grits_loc",
    "grits_top",
    "resample_statistic",
    "validate_metrics_on_fixtures",
]
