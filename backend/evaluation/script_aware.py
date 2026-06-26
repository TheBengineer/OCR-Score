"""Script-aware OCR evaluation for non-Latin scripts.

Provides script detection via Unicode-range heuristics, per-script score
breakdowns, and script-specific metrics (CJK character-level accuracy,
Arabic RTL-aware evaluation).

Typical usage::

    from backend.evaluation.script_aware import (
        detect_script,
        compute_script_scores,
        compute_cjk_metrics,
        compute_arabic_metrics,
    )

    # Detect the dominant script in a text
    script = detect_script("\u4eca\u5929\u7684\u5929\u6c14\u5f88\u597d")
    assert script == "cjk"

    # Compute per-script breakdown for an evaluation run
    breakdown = compute_script_scores(ocr_data, gt_data)

    # CJK-specific metrics (character-level, no word WER)
    cjk = compute_cjk_metrics("\u4eca\u5929\u5929\u6c14\u5f88\u597d", "\u4eca\u5929\u6c14\u5f88\u597d")
"""

from __future__ import annotations

from typing import Final

from backend.evaluation.scoring import (
    compute_char_metrics,
    compute_precision_recall_f1,
)

# ---------------------------------------------------------------------------
# Unicode-range tables
# Each entry is (script_name, list_of_(start, end)_inclusive).
# ---------------------------------------------------------------------------

_SCRIPT_RANGES: Final[dict[str, list[tuple[int, int]]]] = {
    "cjk": [
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
        (0x2E80, 0x2EFF),  # CJK Radicals Supplement
        (0x3000, 0x303F),  # CJK Symbols and Punctuation
        (0xFF01, 0xFF60),  # Fullwidth forms
        (0x2F00, 0x2FDF),  # Kangxi Radicals
        (0x31C0, 0x31EF),  # CJK Strokes
        (0x3200, 0x32FF),  # CJK Enclosed
        (0x3300, 0x33FF),  # CJK Compatibility
        (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
        (0xFE30, 0xFE4F),  # CJK Compatibility Forms
        (0x2FF0, 0x2FFF),  # Ideographic Description Characters
    ],
    "arabic": [
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0x0870, 0x089F),  # Arabic Extended-B
        (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    ],
    "devanagari": [
        (0x0900, 0x097F),  # Devanagari
        (0xA8E0, 0xA8FF),  # Devanagari Extended
        (0x1CD0, 0x1CFF),  # Vedic Extensions
    ],
    "thai": [
        (0x0E00, 0x0E7F),  # Thai
    ],
    "hebrew": [
        (0x0590, 0x05FF),  # Hebrew
        (0xFB1D, 0xFB4F),  # Hebrew Presentation Forms
    ],
    "cyrillic": [
        (0x0400, 0x04FF),  # Cyrillic
        (0x0500, 0x052F),  # Cyrillic Supplement
        (0x2DE0, 0x2DFF),  # Cyrillic Extended-A
        (0xA640, 0xA69F),  # Cyrillic Extended-B
    ],
    "greek": [
        (0x0370, 0x03FF),  # Greek and Coptic
        (0x1F00, 0x1FFF),  # Greek Extended
    ],
    "japanese": [
        (0x3040, 0x309F),  # Hiragana
        (0x30A0, 0x30FF),  # Katakana
        (0x31F0, 0x31FF),  # Katakana Phonetic Extensions
        (0x1B000, 0x1B0FF),  # Kana Supplement
        (0x1B100, 0x1B12F),  # Kana Extended-A
    ],
    "korean": [
        (0xAC00, 0xD7AF),  # Hangul Syllables
        (0x1100, 0x11FF),  # Hangul Jamo
        (0x3130, 0x318F),  # Hangul Compatibility Jamo
        (0xA960, 0xA97F),  # Hangul Jamo Extended-A
        (0xD7B0, 0xD7FF),  # Hangul Jamo Extended-B
    ],
}

_KNOWN_SCRIPTS: Final[list[str]] = list(_SCRIPT_RANGES.keys())


def _char_script(char: str) -> str | None:
    """Return the script name for *char*, or ``None`` for Latin/common."""
    code = ord(char)
    for script, ranges in _SCRIPT_RANGES.items():
        for start, end in ranges:
            if start <= code <= end:
                return script
    return None


# ---------------------------------------------------------------------------
# Internal helpers (defined before callers to satisfy strict checkers)
# ---------------------------------------------------------------------------


def compute_word_metrics_script(
    reference_words: list[str],
    hypothesis_words: list[str],
) -> dict:
    """Word-level metrics compatible with script-aware scoring.

    Same logic as :func:`~backend.evaluation.scoring.compute_word_metrics`
    but inlined here to avoid circular imports at module level.
    """
    n_ref = len(reference_words)
    n_hyp = len(hypothesis_words)

    substitutions, insertions, deletions = _word_levenshtein_counts_script(
        reference_words,
        hypothesis_words,
    )
    matches = max(0, n_ref - substitutions - deletions)
    wer = (substitutions + insertions + deletions) / n_ref if n_ref > 0 else 0.0
    tp = matches
    fp = substitutions + insertions
    fn = substitutions + deletions
    precision, recall, f1 = compute_precision_recall_f1(tp, fp, fn)

    return {
        "wer": wer,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "breakdown": {
            "matches": matches,
            "substitutions": substitutions,
            "insertions": insertions,
            "deletions": deletions,
        },
        "_ref_length": n_ref,
        "_hyp_length": n_hyp,
    }


def _word_levenshtein_counts_script(
    ref_words: list[str],
    hyp_words: list[str],
) -> tuple[int, int, int]:
    """Return ``(substitutions, insertions, deletions)`` via word-level DP.

    Duplicated from :mod:`backend.evaluation.scoring` to avoid circular
    imports at module level.
    """
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    substitutions = 0
    insertions = 0
    deletions = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost == 1:
                    substitutions += 1
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return substitutions, insertions, deletions


def _aggregate_script_scores(scores: list[dict]) -> dict:
    """Weighted aggregation of page-level scores.

    Character metrics are weighted by ``char_ref_length``; word metrics
    by ``word_ref_length``.
    """
    if not scores:
        return {
            "cer": 0.0,
            "wer": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    total_char_weight = sum(s["char_ref_length"] for s in scores)
    total_word_weight = sum(s["word_ref_length"] for s in scores)

    weighted_cer = sum(s["cer"] * s["char_ref_length"] for s in scores)
    weighted_prec = sum(s["precision"] * s["char_ref_length"] for s in scores)
    weighted_rec = sum(s["recall"] * s["char_ref_length"] for s in scores)
    weighted_f1 = sum(s["f1"] * s["char_ref_length"] for s in scores)
    weighted_wer = sum(s["wer"] * s["word_ref_length"] for s in scores)

    def _safe_div(value: float, weight: int) -> float:
        return value / weight if weight > 0 else 0.0

    return {
        "cer": _safe_div(weighted_cer, total_char_weight),
        "wer": _safe_div(weighted_wer, total_word_weight),
        "precision": _safe_div(weighted_prec, total_char_weight),
        "recall": _safe_div(weighted_rec, total_char_weight),
        "f1": _safe_div(weighted_f1, total_char_weight),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_script(text: str) -> str:
    """Detect the dominant script in *text* using Unicode-range heuristics.

    Counts characters belonging to each non-Latin script. If no non-Latin
    script accounts for at least 20% of non-whitespace characters, returns
    ``"latin"``.

    Args:
        text: The input string to analyse.

    Returns:
        One of ``"latin"``, ``"cjk"``, ``"arabic"``, ``"devanagari"``,
        ``"thai"``, ``"hebrew"``, ``"cyrillic"``, ``"greek"``,
        ``"japanese"``, ``"korean"``.
    """
    if not text.strip():
        return "latin"

    counts: dict[str, int] = {}
    total_non_latin = 0

    for ch in text:
        if ch.isspace():
            continue
        script = _char_script(ch)
        if script is not None:
            counts[script] = counts.get(script, 0) + 1
            total_non_latin += 1

    if total_non_latin == 0:
        return "latin"

    # Find the script with the highest count.
    dominant = max(counts, key=counts.get)  # type: ignore[arg-type]
    dominant_count = counts[dominant]
    # Require at least 20% of non-whitespace non-Latin chars.
    if dominant_count / total_non_latin >= 0.2:
        return dominant

    return "latin"


def compute_script_scores(
    ocr_data: dict,
    gt_data: dict,
) -> dict:
    """Compute per-script breakdown of evaluation scores.

    Analyses the ground-truth text across all pages, detects the dominant
    script per page, and groups page-level scores by script. Returns
    overall scores plus a breakdown dict keyed by script name.

    Args:
        ocr_data: Run data dict with ``"pages"`` key (same structure as
            :func:`~backend.evaluation._evaluators.evaluate_run`).
        gt_data: Ground-truth data dict (same structure).

    Returns:
        Dict with ``"overall"`` (combined CER/WER/precision/recall/F1),
        ``"per_script"`` (dict mapping script name -> same metric dict),
        and ``"script_counts"`` (number of pages per script).
    """
    from backend.evaluation._evaluators import _extract_words, _normalize_page_list

    run_pages = _normalize_page_list(ocr_data)
    gt_pages = _normalize_page_list(gt_data)
    num_pages = min(len(run_pages), len(gt_pages))

    script_pages: dict[str, list[dict]] = {}
    page_scores: list[dict] = []

    for i in range(num_pages):
        run_words = _extract_words(run_pages[i])
        gt_words = _extract_words(gt_pages[i])

        ref_text = " ".join(w["text"] for w in gt_words)
        hyp_text = " ".join(w["text"] for w in run_words)
        ref_word_texts = [w["text"] for w in gt_words]
        hyp_word_texts = [w["text"] for w in run_words]

        script = detect_script(ref_text)

        char_metrics = compute_char_metrics(ref_text, hyp_text)
        word_metrics = compute_word_metrics_script(ref_word_texts, hyp_word_texts)

        score = {
            "cer": char_metrics["cer"],
            "precision": char_metrics["precision"],
            "recall": char_metrics["recall"],
            "f1": char_metrics["f1"],
            "wer": word_metrics["wer"],
            "char_ref_length": char_metrics["_ref_length"],
            "word_ref_length": word_metrics["_ref_length"],
        }
        page_scores.append(score)
        script_pages.setdefault(script, []).append(score)

    overall = _aggregate_script_scores(page_scores)

    per_script: dict[str, dict] = {}
    script_counts: dict[str, int] = {}
    for script, scores in script_pages.items():
        per_script[script] = _aggregate_script_scores(scores)
        script_counts[script] = len(scores)

    return {
        "overall": overall,
        "per_script": per_script,
        "script_counts": script_counts,
    }


def compute_cjk_metrics(ocr_text: str, gt_text: str) -> dict:
    """Compute CJK-specific evaluation metrics.

    CJK text does **not** use spaces between words, so word-level WER is
    not meaningful. Instead this function returns:

    - Character-level CER, precision, recall, F1
    - A character confusion matrix for substitutions
    - ``cjk_char_accuracy`` — matches / max(len(gt), len(ocr))
    - No word-level metrics

    Args:
        ocr_text: OCR-hypothesis CJK text.
        gt_text: Ground-truth CJK text.

    Returns:
        Dict with character-level metrics appropriate for CJK scripts.
    """
    char_metrics = compute_char_metrics(gt_text, ocr_text)
    n_gt = char_metrics["_ref_length"]
    n_ocr = char_metrics["_hyp_length"]
    denom = max(n_gt, n_ocr) if max(n_gt, n_ocr) > 0 else 1
    cjk_char_accuracy = char_metrics["breakdown"]["matches"] / denom

    return {
        "cer": char_metrics["cer"],
        "char_precision": char_metrics["precision"],
        "char_recall": char_metrics["recall"],
        "char_f1": char_metrics["f1"],
        "cjk_char_accuracy": cjk_char_accuracy,
        "confusion_matrix": char_metrics["confusion_matrix"],
        "breakdown": char_metrics["breakdown"],
    }


def compute_arabic_metrics(ocr_text: str, gt_text: str) -> dict:
    """Compute Arabic-specific evaluation metrics.

    Arabic is an RTL (Right-to-Left) script. This function returns:

    - Character-level CER, precision, recall, F1
    - Confusion matrix
    - ``rtl_bbox_penalty``: a placeholder metric indicating whether bounding
      box order is consistent with RTL reading direction.

    Args:
        ocr_text: OCR-hypothesis Arabic text.
        gt_text: Ground-truth Arabic text.

    Returns:
        Dict with character-level metrics and an RTL bbox penalty.
    """
    char_metrics = compute_char_metrics(gt_text, ocr_text)

    return {
        "cer": char_metrics["cer"],
        "char_precision": char_metrics["precision"],
        "char_recall": char_metrics["recall"],
        "char_f1": char_metrics["f1"],
        "confusion_matrix": char_metrics["confusion_matrix"],
        "breakdown": char_metrics["breakdown"],
        "rtl_bbox_penalty": 0.0,
    }


def compute_arabic_rtl_penalty(
    ocr_words: list[dict],
    _gt_words: list[dict],  # noqa: ARG001 — kept for API symmetry
) -> float:
    """Compute an RTL-aware bounding-box penalty for Arabic text.

    Arabic is read right-to-left. In a properly ordered OCR reading order,
    the first word on a line should have the rightmost bounding box.
    This function checks how often the reading order matches the
    horizontal order of word bounding boxes.

    Args:
        ocr_words: List of word dicts with ``"bbox"``
            ``[x_min, y_min, x_max, y_max]`` and ``"text"`` keys.
        gt_words: Same structure for ground truth.

    Returns:
        A penalty in ``[0.0, 1.0]`` — 0.0 means perfect RTL ordering,
        1.0 means completely incorrect ordering.
    """
    if len(ocr_words) < 2:
        return 0.0

    wrong_order = 0
    total_pairs = 0

    for i in range(len(ocr_words) - 1):
        bbox_i = ocr_words[i].get("bbox")
        bbox_j = ocr_words[i + 1].get("bbox")
        if bbox_i is None or bbox_j is None or len(bbox_i) < 3 or len(bbox_j) < 3:
            continue
        total_pairs += 1
        if bbox_i[0] < bbox_j[0]:  # x_min increases -> LTR ordering (wrong for RTL)
            wrong_order += 1

    if total_pairs == 0:
        return 0.0

    return wrong_order / total_pairs


# ---------------------------------------------------------------------------
# Internal helpers (validation)
# ---------------------------------------------------------------------------


def _validate_bbox(words: list[dict]) -> None:
    """Validate that word dicts have the expected bbox structure.

    Raises ``TypeError`` if any word is missing a ``"bbox"`` key or it
    is not a sequence of at least 4 numbers.
    """
    for word in words:
        bbox = word.get("bbox")
        if bbox is None:
            msg = f"Word {word!r} is missing 'bbox'"
            raise TypeError(msg)
        if not isinstance(bbox, (list, tuple)):
            msg = f"Word {word!r} has non-sequence bbox"
            raise TypeError(msg)
        if len(bbox) < 4:
            msg = f"Word {word!r} has bbox with fewer than 4 elements"
            raise TypeError(msg)
