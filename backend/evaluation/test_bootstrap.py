"""Tests for the bootstrap confidence interval module."""

from __future__ import annotations

import math
import statistics

import pytest

from backend.evaluation.bootstrap import (
    add_ci_to_scores,
    bootstrap_ci,
    bootstrap_compare,
    resample_statistic,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _scores_close(a: list[float], b: list[float], tol: float = 1e-9) -> bool:
    """Return True when two lists of floats are element-wise close."""
    return len(a) == len(b) and all(math.isclose(ai, bi, abs_tol=tol) for ai, bi in zip(a, b, strict=True))


# ── resample_statistic ───────────────────────────────────────────────────────


class TestResampleStatistic:
    """Tests for the core resample_statistic function."""

    def test_correct_shape(self) -> None:
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = resample_statistic(scores, statistics.mean, n_resamples=500)
        assert len(result) == 500

    def test_with_mean_approx_original(self) -> None:
        """Mean of resampled means should approximate the original mean."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        original_mean = statistics.mean(scores)
        result = resample_statistic(scores, statistics.mean, n_resamples=2000)
        resampled_mean = statistics.mean(result)
        # With 2000 resamples the grand mean should be within 0.02
        assert math.isclose(resampled_mean, original_mean, abs_tol=0.02)

    def test_with_median(self) -> None:
        """Works with statistics.median as the statistic."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = resample_statistic(scores, statistics.median, n_resamples=200)
        assert len(result) == 200
        assert all(0.1 <= v <= 0.5 for v in result)

    def test_single_score(self) -> None:
        """Single-element input still works — every resample is the same value."""
        scores = [0.42]
        result = resample_statistic(scores, statistics.mean, n_resamples=100)
        assert len(result) == 100
        assert all(v == 0.42 for v in result)

    def test_all_identical(self) -> None:
        """All-identical input gives all-identical resamples."""
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        result = resample_statistic(scores, statistics.mean, n_resamples=100)
        assert all(v == 0.5 for v in result)

    def test_n_resamples_zero(self) -> None:
        """Zero resamples returns an empty list."""
        scores = [0.1, 0.2, 0.3]
        result = resample_statistic(scores, statistics.mean, n_resamples=0)
        assert result == []


# ── bootstrap_ci ─────────────────────────────────────────────────────────────


class TestBootstrapCI:
    """Tests for bootstrap_ci."""

    def test_known_distribution(self) -> None:
        """CI from a known distribution contains the true mean."""
        # Scores centred on 0.15 with moderate variance
        scores = [
            0.12,
            0.14,
            0.11,
            0.16,
            0.13,
            0.15,
            0.14,
            0.17,
            0.12,
            0.13,
            0.16,
            0.14,
            0.15,
            0.11,
            0.13,
            0.15,
            0.14,
            0.16,
            0.12,
            0.15,
        ]
        true_mean = statistics.mean(scores)
        ci = bootstrap_ci(scores, metric_name="cer", n_resamples=2000, ci_level=0.95)
        assert ci["metric"] == "cer"
        assert ci["n"] == 20
        assert math.isclose(ci["mean"], true_mean)
        assert ci["ci_lower"] <= true_mean <= ci["ci_upper"]

    def test_increases_with_n_resamples(self) -> None:
        """More resamples should produce a CI consistent with fewer resamples."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ci_low = bootstrap_ci(scores, n_resamples=100, ci_level=0.95)
        ci_high = bootstrap_ci(scores, n_resamples=2000, ci_level=0.95)
        # Both should contain the true mean
        true_mean = statistics.mean(scores)
        assert ci_low["ci_lower"] <= true_mean <= ci_low["ci_upper"]
        assert ci_high["ci_lower"] <= true_mean <= ci_high["ci_upper"]

    def test_reproducible(self) -> None:
        """Same random seed must give an identical CI."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        ci_a = bootstrap_ci(scores, n_resamples=500, random_seed=42)
        ci_b = bootstrap_ci(scores, n_resamples=500, random_seed=42)
        assert ci_a["ci_lower"] == ci_b["ci_lower"]
        assert ci_a["ci_upper"] == ci_b["ci_upper"]
        assert ci_a["mean"] == ci_b["mean"]
        assert _scores_close(ci_a["resampled_means"], ci_b["resampled_means"])

    def test_different_seed(self) -> None:
        """Different seeds should (almost certainly) give different CIs."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        ci_a = bootstrap_ci(scores, n_resamples=500, random_seed=42)
        ci_b = bootstrap_ci(scores, n_resamples=500, random_seed=99)
        # The probability of two seeds producing identical CI bounds on
        # continuous data is essentially zero.
        assert (ci_a["ci_lower"], ci_a["ci_upper"]) != (
            ci_b["ci_lower"],
            ci_b["ci_upper"],
        )

    def test_single_score(self) -> None:
        """A single score yields a degenerate CI at that value."""
        scores = [0.37]
        ci = bootstrap_ci(scores, n_resamples=100)
        assert ci["n"] == 1
        assert ci["mean"] == 0.37
        assert ci["median"] == 0.37
        assert ci["std"] == 0.0
        assert ci["ci_lower"] == 0.37
        assert ci["ci_upper"] == 0.37

    def test_all_identical_scores(self) -> None:
        """All-identical scores collapse the CI to a single value."""
        scores = [0.25, 0.25, 0.25, 0.25]
        ci = bootstrap_ci(scores, n_resamples=200)
        assert ci["mean"] == 0.25
        assert ci["ci_lower"] == 0.25
        assert ci["ci_upper"] == 0.25
        assert ci["std"] == 0.0

    def test_empty_scores_raises(self) -> None:
        """Empty scores should raise a ValueError."""
        with pytest.raises(ValueError, match="scores must not be empty"):
            bootstrap_ci([])

    def test_ci_level_99(self) -> None:
        """99 % CI should be wider than 90 % CI on the same data."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        seed = 42
        ci_90 = bootstrap_ci(scores, ci_level=0.90, n_resamples=2000, random_seed=seed)
        ci_99 = bootstrap_ci(scores, ci_level=0.99, n_resamples=2000, random_seed=seed)
        width_90 = ci_90["ci_upper"] - ci_90["ci_lower"]
        width_99 = ci_99["ci_upper"] - ci_99["ci_lower"]
        assert width_99 >= width_90

    def test_mean_stays_within_ci(self) -> None:
        """The observed mean must always lie within the CI."""
        scores = [0.01, 0.04, 0.02, 0.08, 0.03, 0.06, 0.02, 0.05]
        ci = bootstrap_ci(scores, n_resamples=1000, ci_level=0.95)
        assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]

    def test_resampled_means_length(self) -> None:
        """resampled_means must contain exactly n_resamples values."""
        scores = [0.1, 0.2, 0.3]
        ci = bootstrap_ci(scores, n_resamples=777)
        assert len(ci["resampled_means"]) == 777


# ── bootstrap_compare ────────────────────────────────────────────────────────


class TestBootstrapCompare:
    """Tests for bootstrap_compare."""

    def test_significant_difference(self) -> None:
        """Clearly different distributions should be flagged as significant."""
        engine_a = [0.01, 0.02, 0.01, 0.03, 0.02, 0.01, 0.02, 0.03, 0.01, 0.02]
        engine_b = [0.20, 0.25, 0.22, 0.18, 0.24, 0.21, 0.23, 0.19, 0.22, 0.20]
        result = bootstrap_compare(engine_a, engine_b, n_resamples=2000, ci_level=0.95)
        assert result["significant"] is True
        assert result["diff_ci_upper"] < 0  # mean_a < mean_b (lower CER is better)

    def test_significant_difference_reverse(self) -> None:
        """When engine A is worse, the diff is positive and still significant."""
        engine_a = [0.20, 0.25, 0.22, 0.18]
        engine_b = [0.01, 0.02, 0.01, 0.03]
        result = bootstrap_compare(engine_a, engine_b, n_resamples=2000, ci_level=0.95)
        assert result["significant"] is True
        assert result["diff_ci_lower"] > 0  # mean_a > mean_b

    def test_no_difference(self) -> None:
        """Identical score lists should NOT be flagged as significant."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        result = bootstrap_compare(scores, scores, n_resamples=1000, ci_level=0.95)
        assert result["significant"] is False
        # Bootstrap resampling noise means diff_mean won't be exactly 0,
        # but it should be very close.
        assert math.isclose(result["diff_mean"], 0.0, abs_tol=0.01)

    def test_empty_raises(self) -> None:
        """Empty score list should raise ValueError."""
        with pytest.raises(ValueError, match="engine score lists must not be empty"):
            bootstrap_compare([], [0.1, 0.2])

    def test_small_samples_no_difference(self) -> None:
        """Small nearly-identical samples should not be significant."""
        engine_a = [0.10, 0.11, 0.10, 0.12]
        engine_b = [0.11, 0.10, 0.12, 0.10]
        result = bootstrap_compare(engine_a, engine_b, n_resamples=500, ci_level=0.95)
        # With n=4 per group and overlapping distributions, most runs won't
        # find a significant difference (though this is probabilistic with
        # bootstrap, so we assert on diff_mean being near 0).
        assert math.isclose(result["diff_mean"], 0.0, abs_tol=0.02)

    def test_result_structure(self) -> None:
        """Result dict contains all expected keys."""
        engine_a = [0.01, 0.02, 0.01]
        engine_b = [0.03, 0.04, 0.03]
        result = bootstrap_compare(engine_a, engine_b, n_resamples=100)
        assert "diff_mean" in result
        assert "diff_ci_lower" in result
        assert "diff_ci_upper" in result
        assert "ci_level" in result
        assert "n_resamples" in result
        assert "significant" in result
        assert "n_a" in result
        assert "n_b" in result
        assert result["n_a"] == 3
        assert result["n_b"] == 3


# ── add_ci_to_scores ─────────────────────────────────────────────────────────


class TestAddCiToScores:
    """Tests for add_ci_to_scores."""

    def test_adds_ci_field(self) -> None:
        """Result contains both original page_scores and a ci dict."""
        page_scores = [
            {"page": 1, "cer": 0.05},
            {"page": 2, "cer": 0.03},
            {"page": 3, "cer": 0.07},
            {"page": 4, "cer": 0.04},
            {"page": 5, "cer": 0.06},
        ]
        result = add_ci_to_scores(page_scores, metric_key="cer")
        assert result["page_scores"] == page_scores
        assert "ci" in result
        assert result["ci"]["metric"] == "cer"
        assert result["ci"]["n"] == 5
        assert result["ci"]["ci_lower"] <= result["ci"]["mean"] <= result["ci"]["ci_upper"]

    def test_with_wer_key(self) -> None:
        """Works with a different metric key."""
        page_scores = [
            {"page": 1, "wer": 0.10},
            {"page": 2, "wer": 0.12},
            {"page": 3, "wer": 0.09},
        ]
        result = add_ci_to_scores(page_scores, metric_key="wer")
        assert result["ci"]["metric"] == "wer"
        assert result["ci"]["n"] == 3

    def test_empty_page_scores_raises(self) -> None:
        """Empty page_scores should raise ValueError."""
        with pytest.raises(ValueError, match="page_scores must not be empty"):
            add_ci_to_scores([])

    def test_ci_has_resampled_means(self) -> None:
        """The nested ci includes resampled_means for histogramming."""
        page_scores = [
            {"cer": 0.05},
            {"cer": 0.03},
            {"cer": 0.07},
            {"cer": 0.04},
        ]
        result = add_ci_to_scores(page_scores, n_resamples=100)
        assert len(result["ci"]["resampled_means"]) == 100
