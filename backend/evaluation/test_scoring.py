"""Tests for the evaluation scoring module — CER, WER, precision/recall/F1,
page evaluation, and run-level aggregation."""

from __future__ import annotations

import pytest

from backend.evaluation._evaluators import evaluate_page, evaluate_run
from backend.evaluation.scoring import (
    compute_cer,
    compute_char_metrics,
    compute_precision_recall_f1,
    compute_wer,
    compute_word_metrics,
)


class TestComputeCER:
    """Character Error Rate — core metric."""

    def test_cer_exact_match(self) -> None:
        """Identical strings → CER = 0.0."""
        result = compute_cer("hello world", "hello world")
        assert result == 0.0

    def test_cer_complete_mismatch(self) -> None:
        """Totally different strings → CER = 1.0."""
        result = compute_cer("abcde", "vwxyz")
        assert result == 1.0

    def test_cer_partial(self) -> None:
        """Some correct characters → 0.0 < CER < 1.0."""
        result = compute_cer("kitten", "sitting")
        # "kitten" (6 chars) vs "sitting" (7 chars):
        # k→s (sub), i→i (match), t→t (match), t→t (match), e→i (sub),
        # n→n (match), gap→g (ins)
        # S=2, I=1, D=0 → CER = 3/6 = 0.5
        assert result == 0.5

    def test_cer_empty_reference(self) -> None:
        """Empty reference → 0.0 if hypothesis also empty, 1.0 if not."""
        assert compute_cer("", "") == 0.0
        assert compute_cer("", "hello") == 1.0

    def test_cer_empty_hypothesis(self) -> None:
        """Empty hypothesis → 1.0 (every char is a deletion)."""
        result = compute_cer("hello", "")
        assert result == 1.0

    def test_cer_single_character_match(self) -> None:
        """Single identical char → CER = 0.0."""
        assert compute_cer("a", "a") == 0.0

    def test_cer_single_character_substitution(self) -> None:
        """Single different char → CER = 1.0."""
        assert compute_cer("a", "b") == 1.0

    def test_cer_extra_characters(self) -> None:
        """Hypothesis longer than reference → insertions only."""
        result = compute_cer("abc", "abcdef")
        # 3 matches, 3 insertions → CER = 3/3 = 1.0
        assert result == 1.0

    def test_cer_missing_characters(self) -> None:
        """Hypothesis shorter than reference → deletions only."""
        result = compute_cer("abcdef", "abc")
        # 3 matches, 3 deletions → CER = 3/6 = 0.5
        assert result == 0.5


class TestComputeWER:
    """Word Error Rate — word-level metric."""

    def test_wer_exact_match(self) -> None:
        """Identical word lists → WER = 0.0."""
        result = compute_wer(
            ["the", "quick", "brown", "fox"],
            ["the", "quick", "brown", "fox"],
        )
        assert result == 0.0

    def test_wer_with_substitutions(self) -> None:
        """Substitutions counted correctly."""
        result = compute_wer(
            ["the", "quick", "brown", "fox"],
            ["the", "slow", "brown", "dog"],
        )
        # 2 matches, 2 substitutions → WER = 2/4 = 0.5
        assert result == 0.5

    def test_wer_with_insertions(self) -> None:
        """Insertions counted correctly."""
        result = compute_wer(
            ["hello", "world"],
            ["hello", "beautiful", "world"],
        )
        # 2 matches, 1 insertion → WER = 1/2 = 0.5
        assert result == 0.5

    def test_wer_with_deletions(self) -> None:
        """Deletions counted correctly."""
        result = compute_wer(
            ["the", "quick", "brown", "fox"],
            ["the", "fox"],
        )
        # 2 matches, 2 deletions → WER = 2/4 = 0.5
        assert result == 0.5

    def test_wer_empty_reference(self) -> None:
        """Empty reference → 0.0 if both empty, 1.0 if hypothesis non-empty."""
        assert compute_wer([], []) == 0.0
        assert compute_wer([], ["hello"]) == 1.0

    def test_wer_empty_hypothesis(self) -> None:
        """Empty hypothesis → 1.0 (all ref words are deletions)."""
        result = compute_wer(["hello", "world"], [])
        assert result == 1.0

    def test_wer_complete_mismatch(self) -> None:
        """All words different → WER = 1.0."""
        result = compute_wer(
            ["alpha", "beta"],
            ["gamma", "delta"],
        )
        assert result == 1.0

    def test_wer_mixed_errors(self) -> None:
        """Mix of substitutions, insertions, and deletions."""
        result = compute_wer(
            ["I", "love", "programming"],
            ["I", "enjoy", "coding", "alot"],
        )
        # I→I (match), love→enjoy (sub), programming→coding (sub),
        # gap→alot (ins) → S=2, I=1, D=0 → WER = 3/3 = 1.0
        assert result == 1.0


class TestPrecisionRecallF1:
    """Precision, recall, F1 computation."""

    def test_precision_recall_f1(self) -> None:
        """Known TP/FP/FN → correct values."""
        tp, fp, fn = 80, 10, 20
        precision, recall, f1 = compute_precision_recall_f1(tp, fp, fn)
        assert precision == pytest.approx(80.0 / 90.0)
        assert recall == pytest.approx(80.0 / 100.0)
        expected_f1 = 2.0 * (80.0 / 90.0) * (80.0 / 100.0) / ((80.0 / 90.0) + (80.0 / 100.0))
        assert f1 == pytest.approx(expected_f1)

    def test_perfect_scores(self) -> None:
        """No errors → all = 1.0."""
        p, r, f = compute_precision_recall_f1(100, 0, 0)
        assert (p, r, f) == (1.0, 1.0, 1.0)

    def test_zero_true_positives(self) -> None:
        """No TP → all = 0.0."""
        p, r, f = compute_precision_recall_f1(0, 10, 10)
        assert (p, r, f) == (0.0, 0.0, 0.0)

    def test_zero_false_positives_and_negatives(self) -> None:
        """TP > 0, zero FP/FN → P=R=F1=1.0."""
        p, r, f = compute_precision_recall_f1(42, 0, 0)
        assert (p, r, f) == (1.0, 1.0, 1.0)

    def test_zero_false_positives_only(self) -> None:
        """FP=0, FN>0 → precision=1.0, recall<1.0."""
        p, r, f = compute_precision_recall_f1(50, 0, 50)
        assert p == 1.0
        assert r == 0.5
        assert f == pytest.approx(2.0 / 3.0)

    def test_zero_false_negatives_only(self) -> None:
        """FN=0, FP>0 → recall=1.0, precision<1.0."""
        p, r, f = compute_precision_recall_f1(50, 50, 0)
        assert r == 1.0
        assert p == 0.5
        assert f == pytest.approx(2.0 / 3.0)


class TestComputeCharMetrics:
    """Aggregate character-level metrics."""

    def test_char_confusion_matrix(self) -> None:
        """Confusion matrix built correctly from substitutions."""
        result = compute_char_metrics("abc", "abd")
        matrix = result["confusion_matrix"]
        # a=a (match), b=b (match), c→d (substitution)
        assert matrix == {"c": {"d": 1}}

    def test_char_metrics_identical(self) -> None:
        """Perfect match → CER=0, P=R=F1=1.0, empty confusion."""
        result = compute_char_metrics("hello", "hello")
        assert result["cer"] == 0.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["confusion_matrix"] == {}
        assert result["breakdown"]["matches"] == 5
        assert result["breakdown"]["substitutions"] == 0

    def test_char_metrics_empty(self) -> None:
        """Both empty → CER=0, P=R=F1=0 (no TP)."""
        result = compute_char_metrics("", "")
        assert result["cer"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["confusion_matrix"] == {}

    def test_char_metrics_half_correct(self) -> None:
        """Half the chars correct → metrics proportional."""
        result = compute_char_metrics("abc", "xbc")
        # a→x (sub), b=b (match), c=c (match)
        assert result["cer"] == pytest.approx(1.0 / 3.0)
        assert result["breakdown"]["substitutions"] == 1
        assert result["breakdown"]["matches"] == 2
        # Precision = 2/(2+1+0) = 2/3
        assert result["precision"] == pytest.approx(2.0 / 3.0)
        # Recall = 2/(2+1+0) = 2/3
        assert result["recall"] == pytest.approx(2.0 / 3.0)

    def test_char_confusion_multiple_substitutions(self) -> None:
        """Multiple substitutions accumulate correctly."""
        result = compute_char_metrics("abcde", "abxye")
        # a=a (match), b=b (match), c→x (sub), d→y (sub), e=e (match)
        matrix = result["confusion_matrix"]
        assert matrix["c"]["x"] == 1
        assert matrix["d"]["y"] == 1
        assert len(matrix) == 2

    def test_char_confusion_repeated_substitution(self) -> None:
        """Same (ref, hyp) pair counted multiple times."""
        result = compute_char_metrics("aa", "bb")
        # a→b (sub), a→b (sub)
        matrix = result["confusion_matrix"]
        assert matrix["a"]["b"] == 2


class TestComputeWordMetrics:
    """Aggregate word-level metrics."""

    def test_word_metrics_breakdown(self) -> None:
        """I/D/S breakdown accurate."""
        result = compute_word_metrics(
            ["a", "b", "c"],
            ["a", "x", "c", "d"],
        )
        # a=a (match), b→x (sub), c=c (match), gap→d (ins)
        # S=1, I=1, D=0
        assert result["breakdown"]["substitutions"] == 1
        assert result["breakdown"]["insertions"] == 1
        assert result["breakdown"]["deletions"] == 0
        assert result["breakdown"]["matches"] == 2
        assert result["wer"] == pytest.approx(2.0 / 3.0)

    def test_word_metrics_empty(self) -> None:
        """Empty lists → WER=0, all P/R/F1=0."""
        result = compute_word_metrics([], [])
        assert result["wer"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        for v in result["breakdown"].values():
            assert v == 0

    def test_word_metrics_all_identical(self) -> None:
        """Perfect match → WER=0, P=R=F1=1.0."""
        result = compute_word_metrics(
            ["hello", "world"],
            ["hello", "world"],
        )
        assert result["wer"] == 0.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0


class TestEvaluatePage:
    """Full page evaluation."""

    def test_evaluate_page(self) -> None:
        """Full page evaluation with blocks returns correct metrics."""
        page_results = [
            {"text": "hello", "bbox": [0, 0, 30, 15], "confidence": 0.9},
            {"text": "world", "bbox": [30, 0, 70, 15], "confidence": 0.85},
        ]
        ground_truth = [
            {"text": "hello", "bbox": [0, 0, 30, 15], "confidence": 1.0},
            {"text": "world", "bbox": [30, 0, 70, 15], "confidence": 1.0},
        ]

        result = evaluate_page(page_results, ground_truth)

        assert result["cer"] == 0.0
        assert result["wer"] == 0.0
        assert result["char_precision"] == 1.0
        assert result["char_recall"] == 1.0
        assert result["char_f1"] == 1.0
        assert result["word_precision"] == 1.0
        assert result["word_recall"] == 1.0
        assert result["word_f1"] == 1.0
        assert result["char_confusion"] == {}
        assert result["word_breakdown"]["matches"] == 2
        assert "alignment_stats" in result

    def test_evaluate_page_with_errors(self) -> None:
        """Page evaluation with OCR errors."""
        page_results = [
            {"text": "helpo"},
            {"text": "w0rld"},
            {"text": "extra"},
        ]
        ground_truth = [
            {"text": "hello"},
            {"text": "world"},
        ]

        result = evaluate_page(page_results, ground_truth)

        # CER: "helpo w0rld extra" vs "hello world"
        # WER: S=2 (hello→helpo is substitution at word level since !=),
        #      I=1 (extra is extra word)
        assert result["cer"] > 0.0
        assert result["wer"] > 0.0
        assert result["char_f1"] < 1.0
        assert result["word_f1"] < 1.0
        assert result["word_breakdown"]["substitutions"] >= 2
        assert result["word_breakdown"]["insertions"] >= 1

    def test_evaluate_page_empty(self) -> None:
        """Both empty → zero metrics."""
        result = evaluate_page([], [])
        assert result["cer"] == 0.0
        assert result["wer"] == 0.0
        assert result["char_f1"] == 0.0
        assert result["word_f1"] == 0.0

    def test_evaluate_page_with_config(self) -> None:
        """Config dict forwarded correctly to align_ocr_texts."""
        page_results = [
            {"text": "hello", "confidence": 0.5},
            {"text": "world", "confidence": 0.5},
        ]
        ground_truth = [
            {"text": "hello", "confidence": 0.5},
            {"text": "world", "confidence": 0.5},
        ]
        # similarity_threshold=0.0 makes everything a "match".
        result = evaluate_page(page_results, ground_truth, config={"similarity_threshold": 0.0})
        assert result["alignment_stats"]["matches"] == 2


class TestEvaluateRun:
    """Run-level aggregation."""

    def test_evaluate_run(self) -> None:
        """Aggregate across pages produces correct averages."""
        run_data = {
            "pages": [
                {
                    "results": [
                        {"text": "hello", "confidence": 0.9},
                    ]
                },
                {
                    "results": [
                        {"text": "world", "confidence": 0.8},
                    ]
                },
            ],
        }
        gt_data = {
            "pages": [
                {
                    "results": [
                        {"text": "hello", "confidence": 1.0},
                    ]
                },
                {
                    "results": [
                        {"text": "world", "confidence": 1.0},
                    ]
                },
            ],
        }

        result = evaluate_run(run_data, gt_data)

        assert result["cer"] == 0.0
        assert result["wer"] == 0.0
        assert result["num_pages"] == 2
        assert len(result["per_page"]) == 2
        assert result["char_f1"] == 1.0
        assert result["word_f1"] == 1.0

    def test_evaluate_run_mixed_errors(self) -> None:
        """Mix of good and bad pages → aggregate between extremes."""
        run_data = {
            "pages": [
                {
                    "results": [
                        {"text": "hello"},
                        {"text": "world"},
                    ]
                },
                {
                    "results": [
                        {"text": "xxxxx"},
                    ]
                },
            ],
        }
        gt_data = {
            "pages": [
                {
                    "results": [
                        {"text": "hello"},
                        {"text": "world"},
                    ]
                },
                {
                    "results": [
                        {"text": "hello"},
                        {"text": "world"},
                    ]
                },
            ],
        }

        result = evaluate_run(run_data, gt_data)

        # Page 0: perfect → CER=0, WER=0 (ref "hello world" = 11 chars)
        # Page 1: all wrong → CER=1, WER=1 (ref "hello world", hyp "xxxxx")
        # Weighted by text length: page0 has 11 chars, page1 has 11 chars
        # Expected CER = (0*11 + 1*11) / (11+11) = 11/22 = 0.5
        assert result["cer"] == 0.5
        assert result["num_pages"] == 2
        assert 0.0 < result["wer"] < 1.0

    def test_evaluate_run_single_page(self) -> None:
        """Single page run produces same results as evaluate_page."""
        run_data = {
            "pages": [
                {
                    "results": [
                        {"text": "hello", "bbox": [0, 0, 30, 15], "confidence": 0.9},
                    ]
                },
            ],
        }
        gt_data = {
            "pages": [
                {
                    "results": [
                        {"text": "hello", "bbox": [0, 0, 30, 15], "confidence": 1.0},
                    ]
                },
            ],
        }

        result = evaluate_run(run_data, gt_data)
        assert result["cer"] == 0.0
        assert result["wer"] == 0.0
        assert result["num_pages"] == 1

    def test_evaluate_run_empty_pages(self) -> None:
        """No pages → zero metrics, num_pages=0."""
        result = evaluate_run({"pages": []}, {"pages": []})
        assert result["num_pages"] == 0
        assert result["cer"] == 0.0
        assert result["wer"] == 0.0


class TestEdgeCases:
    """Edge cases and integration scenarios."""

    def test_edge_case_all_identical(self) -> None:
        """Perfect match at both levels → all error metrics = 0, all accuracy = 1."""
        ref_text = "The quick brown fox jumps over the lazy dog"
        hyp_text = ref_text
        ref_words = ref_text.split()
        hyp_words = ref_text.split()

        # Character-level
        char_result = compute_char_metrics(ref_text, hyp_text)
        assert char_result["cer"] == 0.0
        assert char_result["precision"] == 1.0
        assert char_result["recall"] == 1.0
        assert char_result["f1"] == 1.0

        # Word-level
        word_result = compute_word_metrics(ref_words, hyp_words)
        assert word_result["wer"] == 0.0
        assert word_result["precision"] == 1.0
        assert word_result["recall"] == 1.0
        assert word_result["f1"] == 1.0

        # Full page evaluation
        page_dicts = [{"text": w} for w in ref_words]
        page_result = evaluate_page(page_dicts, page_dicts)
        assert page_result["cer"] == 0.0
        assert page_result["wer"] == 0.0
        assert page_result["char_f1"] == 1.0
        assert page_result["word_f1"] == 1.0

    def test_cer_wer_consistency(self) -> None:
        """CER should be ≥ WER for the same data (chars more granular)."""
        ref_words = ["hello", "beautiful", "world"]
        hyp_words = ["hello", "world"]

        ref_text = " ".join(ref_words)
        hyp_text = " ".join(hyp_words)

        cer = compute_cer(ref_text, hyp_text)
        wer = compute_wer(ref_words, hyp_words)

        # WER = 1/3 ≈ 0.33, CER should be higher due to char deletions
        assert cer >= wer
        assert wer == pytest.approx(1.0 / 3.0)

    def test_hierarchical_page_data(self) -> None:
        """evaluate_run handles JSONB hierarchical page data."""
        run_data = {
            "pages": [
                {
                    "data": {
                        "blocks": [
                            {
                                "type": "text",
                                "lines": [
                                    {
                                        "words": [
                                            {"text": "hello", "bbox": [0, 0, 30, 15], "confidence": 0.9},
                                            {"text": "world", "bbox": [30, 0, 70, 15], "confidence": 0.85},
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                },
            ],
        }
        gt_data = {
            "pages": [
                {
                    "data": {
                        "blocks": [
                            {
                                "type": "text",
                                "lines": [
                                    {
                                        "words": [
                                            {"text": "hello", "bbox": [0, 0, 30, 15], "confidence": 1.0},
                                            {"text": "world", "bbox": [30, 0, 70, 15], "confidence": 1.0},
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                },
            ],
        }

        result = evaluate_run(run_data, gt_data)
        assert result["cer"] == 0.0
        assert result["wer"] == 0.0
        assert result["num_pages"] == 1
        assert result["char_f1"] == 1.0
        assert result["word_f1"] == 1.0
