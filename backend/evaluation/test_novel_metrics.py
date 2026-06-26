"""Tests for novel OCR evaluation metrics — Imagination Rate, Confidence
Calibration Error, Noise Sensitivity Index, and validation report."""

from __future__ import annotations

import pytest

from backend.evaluation.novel_metrics import (
    compute_all_novel_metrics,
    compute_confidence_calibration_error,
    compute_imagination_rate,
    compute_noise_sensitivity_index,
    validate_metrics_on_fixtures,
)


class TestImaginationRate:
    """Imagination Rate — hallucinated word detection."""

    def test_imagination_rate_no_hallucination(self) -> None:
        """OCR matches reference exactly → IR = 0.0."""
        ocr_words = [
            {"text": "hello", "confidence": 0.95},
            {"text": "world", "confidence": 0.92},
        ]
        ref_words = [
            {"text": "hello", "confidence": 1.0},
            {"text": "world", "confidence": 1.0},
        ]
        ir = compute_imagination_rate(ocr_words, ref_words)
        assert ir == 0.0

    def test_imagination_rate_with_hallucinations(self) -> None:
        """Extra words in OCR → IR > 0.0."""
        ocr_words = [
            {"text": "hello", "confidence": 0.95},
            {"text": "beautiful", "confidence": 0.85},
            {"text": "world", "confidence": 0.92},
        ]
        ref_words = [
            {"text": "hello", "confidence": 1.0},
            {"text": "world", "confidence": 1.0},
        ]
        ir = compute_imagination_rate(ocr_words, ref_words)
        # "beautiful" is an insertion (hallucination) → 1/3 ≈ 0.333
        assert ir == pytest.approx(1.0 / 3.0)

    def test_imagination_rate_all_hallucinated(self) -> None:
        """Empty reference → every OCR word is an insertion → IR = 1.0."""
        ocr_words = [
            {"text": "hello", "confidence": 0.95},
            {"text": "world", "confidence": 0.92},
        ]
        ref_words: list[dict] = []
        ir = compute_imagination_rate(ocr_words, ref_words)
        assert ir == 1.0

    def test_imagination_rate_empty_ocr(self) -> None:
        """Empty OCR list → IR = 0.0."""
        assert compute_imagination_rate([], [{"text": "hello"}]) == 0.0

    def test_imagination_rate_mixed(self) -> None:
        """Mix of matches and extra words."""
        ocr_words = [
            {"text": "the", "confidence": 0.95},
            {"text": "quick", "confidence": 0.93},
            {"text": "brown", "confidence": 0.97},
            {"text": "fox", "confidence": 0.94},
            {"text": "jumped", "confidence": 0.80},
            {"text": "extra", "confidence": 0.70},
        ]
        ref_words = [
            {"text": "the", "confidence": 1.0},
            {"text": "quick", "confidence": 1.0},
            {"text": "brown", "confidence": 1.0},
            {"text": "fox", "confidence": 1.0},
        ]
        ir = compute_imagination_rate(ocr_words, ref_words)
        # "jumped" and "extra" are insertions → 2/6 = 0.333...
        assert ir == pytest.approx(2.0 / 6.0)

    def test_imagination_rate_partial_matches(self) -> None:
        """Substitutions are NOT hallucinations — only insertions."""
        ocr_words = [
            {"text": "hello", "confidence": 0.80},
            {"text": "w0rld", "confidence": 0.85},  # substitution
        ]
        ref_words = [
            {"text": "hello", "confidence": 1.0},
            {"text": "world", "confidence": 1.0},
        ]
        ir = compute_imagination_rate(ocr_words, ref_words)
        # "w0rld" is a substitution (aligned, different text), not insertion.
        # IR should be 0.0 since every OCR word has an alignment partner.
        assert ir == 0.0


class TestConfidenceCalibrationError:
    """Confidence Calibration Error — Brier-score metric."""

    def test_confidence_calibration_perfect(self) -> None:
        """Confidence matches correctness perfectly → low CCE."""
        ocr_words = [
            {"text": "hello", "confidence": 0.95},
            {"text": "world", "confidence": 0.90},
        ]
        ref_words = [
            {"text": "hello", "confidence": 1.0},
            {"text": "world", "confidence": 1.0},
        ]
        cce = compute_confidence_calibration_error(ocr_words, ref_words)
        # Both words are correct → correctness = 1.0, 1.0.
        # CCE = ((0.95-1.0)^2 + (0.90-1.0)^2) / 2 = (0.0025 + 0.01) / 2
        expected = ((0.95 - 1.0) ** 2 + (0.90 - 1.0) ** 2) / 2.0
        assert cce == pytest.approx(expected)

    def test_confidence_calibration_poor(self) -> None:
        """High confidence on wrong words → high CCE.

        Uses words with fuzz.ratio < 60 (e.g. 'xxxxx' vs 'hello' = 0.0)
        so the aligner classifies them as substitutions, not matches.
        """
        ocr_words = [
            {"text": "xxxxx", "confidence": 0.92},  # substitution, high conf → bad
            {"text": "yyyyy", "confidence": 0.88},  # substitution, high conf → bad
        ]
        ref_words = [
            {"text": "hello", "confidence": 1.0},
            {"text": "world", "confidence": 1.0},
        ]
        cce_high = compute_confidence_calibration_error(ocr_words, ref_words)

        # Same errors but low confidence → lower CCE (better calibrated).
        ocr_words_low = [
            {"text": "xxxxx", "confidence": 0.30},  # substitution, knows unsure → good
            {"text": "yyyyy", "confidence": 0.25},  # substitution, knows unsure → good
        ]
        cce_low = compute_confidence_calibration_error(ocr_words_low, ref_words)

        assert cce_high > cce_low

    def test_cce_edge_cases_empty(self) -> None:
        """Empty OCR list → CCE = 0.0."""
        assert compute_confidence_calibration_error([], [{"text": "hello"}]) == 0.0

    def test_cce_edge_cases_all_correct(self) -> None:
        """All words correct → CCE is low but depends on confidence."""
        ocr_words = [
            {"text": "hello", "confidence": 0.95},
            {"text": "world", "confidence": 0.90},
        ]
        ref_words = [
            {"text": "hello", "confidence": 1.0},
            {"text": "world", "confidence": 1.0},
        ]
        cce = compute_confidence_calibration_error(ocr_words, ref_words)
        expected = ((0.95 - 1.0) ** 2 + (0.90 - 1.0) ** 2) / 2.0
        assert cce == pytest.approx(expected)

    def test_cce_edge_cases_all_wrong(self) -> None:
        """All words wrong, high confidence → high CCE."""
        ocr_words = [
            {"text": "xxxxx", "confidence": 0.95},
            {"text": "yyyyy", "confidence": 0.90},
        ]
        ref_words = [
            {"text": "hello", "confidence": 1.0},
            {"text": "world", "confidence": 1.0},
        ]
        cce = compute_confidence_calibration_error(ocr_words, ref_words)
        # Both wrong → correctness = 0.0, 0.0.
        # CCE = ((0.95-0.0)^2 + (0.90-0.0)^2) / 2
        expected = ((0.95 - 0.0) ** 2 + (0.90 - 0.0) ** 2) / 2.0
        assert cce == pytest.approx(expected)

    def test_cce_missing_confidence_default(self) -> None:
        """Missing confidence field defaults to 0.0."""
        ocr_words = [
            {"text": "hello"},  # no confidence key → defaults to 0.0
        ]
        ref_words = [
            {"text": "hello", "confidence": 1.0},
        ]
        cce = compute_confidence_calibration_error(ocr_words, ref_words)
        # correctness = 1.0 (match), confidence = 0.0 (default)
        # CCE = (0.0 - 1.0)^2 / 1 = 1.0
        assert cce == 1.0


class TestNoiseSensitivityIndex:
    """Noise Sensitivity Index — CER degradation across DPI."""

    def test_noise_sensitivity_basic(self) -> None:
        """NSI returns CER at each resolution and degradation slope."""
        ref_words = [
            {"text": "the", "confidence": 1.0},
            {"text": "quick", "confidence": 1.0},
            {"text": "brown", "confidence": 1.0},
            {"text": "fox", "confidence": 1.0},
        ]
        test_texts = {
            72: [
                {"text": "the", "confidence": 0.50},
                {"text": "qu1ck", "confidence": 0.40},
                {"text": "br0wn", "confidence": 0.35},
                {"text": "f0x", "confidence": 0.30},
            ],
            150: [
                {"text": "the", "confidence": 0.80},
                {"text": "quick", "confidence": 0.75},
                {"text": "br0wn", "confidence": 0.70},
                {"text": "fox", "confidence": 0.65},
            ],
            300: [
                {"text": "the", "confidence": 0.95},
                {"text": "quick", "confidence": 0.93},
                {"text": "brown", "confidence": 0.97},
                {"text": "fox", "confidence": 0.94},
            ],
        }

        result = compute_noise_sensitivity_index(
            _test_texts=test_texts,
            _test_reference=ref_words,
        )

        # Check keys.
        assert "cer_at_72" in result
        assert "cer_at_150" in result
        assert "cer_at_300" in result
        assert "degradation_slope" in result

        # Higher DPI should have lower CER.
        assert result["cer_at_72"] > result["cer_at_150"]
        assert result["cer_at_150"] >= result["cer_at_300"]

        # Degradation slope should be negative (CER decreases as DPI increases).
        assert result["degradation_slope"] < 0.0

    def test_noise_sensitivity_empty_texts(self) -> None:
        """NSI with empty data returns error dict."""
        result = compute_noise_sensitivity_index()
        assert "error" in result

    def test_noise_sensitivity_degradation_value(self) -> None:
        """CER values should be between 0 and 1."""
        ref_words = [
            {"text": "perfect", "confidence": 1.0},
        ]
        test_texts = {
            72: [{"text": "perfect", "confidence": 0.80}],
            300: [{"text": "perfect", "confidence": 0.95}],
        }

        result = compute_noise_sensitivity_index(
            _test_texts=test_texts,
            _test_reference=ref_words,
        )
        assert 0.0 <= result["cer_at_72"] <= 1.0
        assert 0.0 <= result["cer_at_300"] <= 1.0


class TestComputeAllNovelMetrics:
    """Combined computation of all novel metrics."""

    def test_all_novel_metrics(self) -> None:
        """All metrics computed in a single call."""
        ocr_data = {
            "words": [
                {"text": "the", "confidence": 0.95},
                {"text": "quick", "confidence": 0.93},
                {"text": "brown", "confidence": 0.97},
                {"text": "fox", "confidence": 0.94},
            ],
        }
        ref_data = {
            "words": [
                {"text": "the", "confidence": 1.0},
                {"text": "quick", "confidence": 1.0},
                {"text": "brown", "confidence": 1.0},
                {"text": "fox", "confidence": 1.0},
            ],
        }

        result = compute_all_novel_metrics(ocr_data, ref_data)

        # Check structure.
        assert "standard" in result
        assert "novel" in result
        assert "cer" in result["standard"]
        assert "wer" in result["standard"]
        assert "imagination_rate" in result["novel"]
        assert "confidence_calibration_error" in result["novel"]

        # Perfect OCR → CER=0, WER=0.
        assert result["standard"]["cer"] == 0.0
        assert result["standard"]["wer"] == 0.0
        assert result["novel"]["imagination_rate"] == 0.0

        # NSI should be None (no NSI args provided).
        assert result["novel"]["noise_sensitivity_index"] is None

    def test_all_novel_metrics_with_nsi(self) -> None:
        """All metrics including NSI."""
        ocr_data = {
            "words": [
                {"text": "the", "confidence": 0.95},
                {"text": "quick", "confidence": 0.93},
            ],
        }
        ref_data = {
            "words": [
                {"text": "the", "confidence": 1.0},
                {"text": "quick", "confidence": 1.0},
            ],
        }

        test_texts = {
            72: [{"text": "the", "confidence": 0.50}, {"text": "qu1ck", "confidence": 0.40}],
            300: [{"text": "the", "confidence": 0.95}, {"text": "quick", "confidence": 0.93}],
        }
        test_ref = [{"text": "the", "confidence": 1.0}, {"text": "quick", "confidence": 1.0}]

        result = compute_all_novel_metrics(
            ocr_data,
            ref_data,
            _test_texts=test_texts,
            _test_reference=test_ref,
        )

        assert result["novel"]["noise_sensitivity_index"] is not None
        nsi = result["novel"]["noise_sensitivity_index"]
        assert "cer_at_72" in nsi
        assert "cer_at_300" in nsi
        assert "degradation_slope" in nsi

    def test_all_novel_metrics_with_hallucinations(self) -> None:
        """Novel metrics correctly detect hallucinated words."""
        ocr_data = {
            "words": [
                {"text": "hello", "confidence": 0.95},
                {"text": "fake", "confidence": 0.70},
                {"text": "word", "confidence": 0.65},
                {"text": "world", "confidence": 0.92},
            ],
        }
        ref_data = {
            "words": [
                {"text": "hello", "confidence": 1.0},
                {"text": "world", "confidence": 1.0},
            ],
        }

        result = compute_all_novel_metrics(ocr_data, ref_data)
        # "fake" and "word" are insertions → IR = 2/4 = 0.5
        assert result["novel"]["imagination_rate"] == 0.5
        # CER should be 0.0 for the matched words, but "fake word"
        # are insertions at the end.
        # ref = "hello world", hyp = "hello fake word world"
        # Character alignment adds "fake word" as insertions.
        assert result["standard"]["cer"] > 0.0


class TestValidationReport:
    """Validation report comparing novel vs standard metrics."""

    def test_novel_metrics_report(self) -> None:
        """Validation report is generated successfully."""
        report = validate_metrics_on_fixtures()

        assert "scenarios" in report
        assert "verdicts" in report
        assert "all_pass" in report
        assert isinstance(report["all_pass"], bool)

        # Should have all 4 scenarios.
        assert "perfect" in report["scenarios"]
        assert "hallucinated" in report["scenarios"]
        assert "overconfident" in report["scenarios"]
        assert "well_calibrated" in report["scenarios"]

        # Each scenario has standard and novel metrics.
        for name in ("perfect", "hallucinated", "overconfident", "well_calibrated"):
            scenario = report["scenarios"][name]
            assert "standard" in scenario
            assert "novel" in scenario
            assert "cer" in scenario["standard"]
            assert "imagination_rate" in scenario["novel"]
            assert "confidence_calibration_error" in scenario["novel"]

        # Should have at least 5 verdicts.
        assert len(report["verdicts"]) >= 5

    def test_validation_verdicts_consistent(self) -> None:
        """Validation verdicts are consistent with known properties."""
        report = validate_metrics_on_fixtures()

        # Perfect scenario: IR should be 0.
        assert report["scenarios"]["perfect"]["novel"]["imagination_rate"] == 0.0

        # Hallucinated scenario: IR should be > 0.
        assert report["scenarios"]["hallucinated"]["novel"]["imagination_rate"] > 0.0

        # Overconfident should have higher CCE than well-calibrated.
        oc_cce = report["scenarios"]["overconfident"]["novel"]["confidence_calibration_error"]
        wc_cce = report["scenarios"]["well_calibrated"]["novel"]["confidence_calibration_error"]
        assert oc_cce > wc_cce

        # Hallucinated scenario has higher CER due to extra inserted words.
        assert (
            report["scenarios"]["hallucinated"]["standard"]["cer"]
            > report["scenarios"]["perfect"]["standard"]["cer"]
        )
        # IR is also higher — orthogonal signal.
        assert (
            report["scenarios"]["hallucinated"]["novel"]["imagination_rate"]
            > report["scenarios"]["perfect"]["novel"]["imagination_rate"]
        )


class TestEdgeCases:
    """Edge cases and integration tests."""

    def test_empty_inputs(self) -> None:
        """All metrics handle empty inputs gracefully."""
        ocr_words: list[dict] = []
        ref_words: list[dict] = [{"text": "hello"}]

        assert compute_imagination_rate(ocr_words, ref_words) == 0.0
        assert compute_confidence_calibration_error(ocr_words, ref_words) == 0.0

    def test_both_empty(self) -> None:
        """Both OCR and reference empty → all metrics return 0."""
        ocr: list[dict] = []
        ref: list[dict] = []

        assert compute_imagination_rate(ocr, ref) == 0.0
        assert compute_confidence_calibration_error(ocr, ref) == 0.0

    def test_imagination_vs_cer_orthogonal(self) -> None:
        """IR catches what CER misses: identical matched content with extras.

        Two OCR outputs with the same matched content but different numbers
        of hallucinated words should have the same CER but different IR.
        """
        ref = [{"text": w} for w in ["hello", "world"]]

        ocr_no_extra = [
            {"text": "hello", "confidence": 0.95},
            {"text": "world", "confidence": 0.92},
        ]
        ocr_with_extra = [
            {"text": "hello", "confidence": 0.95},
            {"text": "world", "confidence": 0.92},
            {"text": "extra", "confidence": 0.70},
            {"text": "words", "confidence": 0.65},
        ]

        ir_no_extra = compute_imagination_rate(ocr_no_extra, ref)
        ir_with_extra = compute_imagination_rate(ocr_with_extra, ref)

        # IR detects the extra words while CER may not capture them fully.
        assert ir_no_extra < ir_with_extra

    def test_cce_and_cer_orthogonal(self) -> None:
        """Two OCR outputs with same CER but different calibration.

        Demonstrates that CCE captures calibration quality that CER does not.
        Uses words with fuzz.ratio < 60 for definite substitution classification.
        """
        ref = [
            {"text": "hello", "confidence": 1.0},
            {"text": "world", "confidence": 1.0},
        ]

        # Both have the same substitution errors.
        well_cal = [
            {"text": "xxxxx", "confidence": 0.30},  # substitution, low conf → good
            {"text": "yyyyy", "confidence": 0.25},  # substitution, low conf → good
        ]
        badly_cal = [
            {"text": "xxxxx", "confidence": 0.95},  # substitution, high conf → bad
            {"text": "yyyyy", "confidence": 0.90},  # substitution, high conf → bad
        ]

        # Same CER because substitutions are identical.
        metrics_well = compute_all_novel_metrics({"words": well_cal}, {"words": ref})
        metrics_badly = compute_all_novel_metrics({"words": badly_cal}, {"words": ref})

        assert metrics_well["standard"]["cer"] == metrics_badly["standard"]["cer"]
        cce_well = metrics_well["novel"]["confidence_calibration_error"]
        cce_badly = metrics_badly["novel"]["confidence_calibration_error"]
        assert cce_well < cce_badly
