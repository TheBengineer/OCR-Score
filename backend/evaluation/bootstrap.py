"""Bootstrap confidence interval computation for OCR evaluation metrics.

Provides pure-Python bootstrap resampling (no numpy/scipy dependency) to
compute confidence intervals on per-page OCR scores (CER, WER, accuracy, etc.)
and to compare score distributions between OCR engines.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable


def resample_statistic(
    scores: list[float],
    statistic: Callable[[list[float]], float] = statistics.mean,
    n_resamples: int = 1000,
) -> list[float]:
    """Core resampling function.

    Repeatedly draws *n* samples (where *n* = ``len(scores)``) **with
    replacement** from the observed scores and computes the *statistic* on
    each resample.

    Parameters
    ----------
    scores:
        Observed per-page scores.
    statistic:
        Function that reduces a list of floats to a single float
        (e.g. ``statistics.mean`` or ``statistics.median``).
    n_resamples:
        Number of bootstrap resamples to draw.

    Returns
    -------
    list[float]
        ``n_resamples`` values of the computed statistic, one per resample.
        This list is **not** sorted — use the caller's responsibility.
    """
    n = len(scores)
    return [statistic([scores[random.randrange(n)] for _ in range(n)]) for _ in range(n_resamples)]


def bootstrap_ci(
    scores: list[float],
    metric_name: str = "cer",
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    random_seed: int | None = None,
) -> dict:
    """Compute a bootstrap percentile confidence interval for *scores*.

    Algorithm
    ---------
    1. Compute observed statistics (mean, median, std) on the input.
    2. Resample with replacement ``n_resamples`` times, computing the mean
       of each resample.
    3. Sort the resampled means and take the
       ``(alpha/2, 1 - alpha/2)`` percentiles as the CI bounds.

    Parameters
    ----------
    scores:
        Per-page scores for one metric.
    metric_name:
        Label for the metric (e.g. ``"cer"``, ``"wer"``) — included in the
        result dict for downstream display.
    n_resamples:
        Number of bootstrap resamples.
    ci_level:
        Confidence level (e.g. ``0.95`` for 95 % CI).
    random_seed:
        If provided, the global ``random`` module is seeded before resampling
        so results are reproducible.  ``None`` (default) leaves the state
        alone.

    Returns
    -------
    dict
        ``{
            "metric": metric_name,
            "n": len(scores),
            "mean": ...,
            "median": ...,
            "std": ...,
            "ci_level": ci_level,
            "ci_lower": ...,
            "ci_upper": ...,
            "n_resamples": n_resamples,
            "resampled_means": [...],
        }``
    """
    if not scores:
        raise ValueError("scores must not be empty")

    if random_seed is not None:
        random.seed(random_seed)

    n = len(scores)
    mean = statistics.mean(scores)
    median = statistics.median(scores)
    std = statistics.stdev(scores) if n >= 2 else 0.0

    resampled = resample_statistic(scores, statistics.mean, n_resamples)
    resampled.sort()

    alpha = 1.0 - ci_level
    lower_idx: int = max(0, int(n_resamples * alpha / 2))
    upper_idx: int = min(n_resamples - 1, int(n_resamples * (1.0 - alpha / 2)))

    return {
        "metric": metric_name,
        "n": n,
        "mean": mean,
        "median": median,
        "std": std,
        "ci_level": ci_level,
        "ci_lower": resampled[lower_idx],
        "ci_upper": resampled[upper_idx],
        "n_resamples": n_resamples,
        "resampled_means": resampled,
    }


def bootstrap_compare(
    engine_a_scores: list[float],
    engine_b_scores: list[float],
    n_resamples: int = 1000,
    ci_level: float = 0.95,
) -> dict:
    """Compare two engines' score distributions via bootstrap difference of means.

    Repeatedly resamples both score vectors, computes
    ``mean_a - mean_b`` each time, and builds a CI around the difference.
    If the CI does **not** contain 0 the difference is considered
    statistically significant at the chosen confidence level.

    Parameters
    ----------
    engine_a_scores:
        Per-page scores for engine A.
    engine_b_scores:
        Per-page scores for engine B.
    n_resamples:
        Number of bootstrap resamples.
    ci_level:
        Confidence level (e.g. ``0.95`` for 95 % CI).

    Returns
    -------
    dict
        ``{
            "diff_mean": ...,
            "diff_ci_lower": ...,
            "diff_ci_upper": ...,
            "ci_level": ci_level,
            "n_resamples": n_resamples,
            "significant": bool,
            "n_a": len(engine_a_scores),
            "n_b": len(engine_b_scores),
        }``
    """
    if not engine_a_scores or not engine_b_scores:
        raise ValueError("engine score lists must not be empty")

    n_a = len(engine_a_scores)
    n_b = len(engine_b_scores)

    diffs: list[float] = []
    for _ in range(n_resamples):
        mean_a = statistics.mean([engine_a_scores[random.randrange(n_a)] for _ in range(n_a)])
        mean_b = statistics.mean([engine_b_scores[random.randrange(n_b)] for _ in range(n_b)])
        diffs.append(mean_a - mean_b)

    diffs.sort()
    alpha = 1.0 - ci_level
    lower_idx: int = max(0, int(n_resamples * alpha / 2))
    upper_idx: int = min(n_resamples - 1, int(n_resamples * (1.0 - alpha / 2)))

    diff_mean = statistics.mean(diffs)
    ci_lower = diffs[lower_idx]
    ci_upper = diffs[upper_idx]

    return {
        "diff_mean": diff_mean,
        "diff_ci_lower": ci_lower,
        "diff_ci_upper": ci_upper,
        "ci_level": ci_level,
        "n_resamples": n_resamples,
        "significant": ci_lower > 0 or ci_upper < 0,
        "n_a": n_a,
        "n_b": n_b,
    }


def add_ci_to_scores(
    page_scores: list[dict],
    metric_key: str = "cer",
    n_resamples: int = 1000,
    ci_level: float = 0.95,
) -> dict:
    """Convenience wrapper that adds a CI field to an existing score dict list.

    Each element of *page_scores* is expected to contain *metric_key* as a
    float value (e.g. ``{"page": 1, "cer": 0.05}``).

    Parameters
    ----------
    page_scores:
        List of per-page score dicts.
    metric_key:
        Key whose values are extracted for bootstrapping.
    n_resamples:
        Number of bootstrap resamples (forwarded to ``bootstrap_ci``).
    ci_level:
        Confidence level (forwarded to ``bootstrap_ci``).

    Returns
    -------
    dict
        ``{"page_scores": [...], "ci": {...}}`` where ``ci`` is the full
        result of ``bootstrap_ci``.
    """
    if not page_scores:
        raise ValueError("page_scores must not be empty")

    scores = [ps[metric_key] for ps in page_scores]
    ci = bootstrap_ci(
        scores=scores,
        metric_name=metric_key,
        n_resamples=n_resamples,
        ci_level=ci_level,
    )
    return {
        "page_scores": page_scores,
        "ci": ci,
    }
