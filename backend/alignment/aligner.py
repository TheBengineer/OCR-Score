"""Sequence alignment algorithms for OCR evaluation.

Provides Needleman-Wunsch (global) and Smith-Waterman (local) alignment,
character-level alignment, and the full ``align_ocr_texts`` pipeline that
compares OCR outputs against ground truth using Levenshtein similarity
and optional bounding-box tiebreaking.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from rapidfuzz import fuzz

# ── Core NW Implementation ──────────────────────────────────────────────────────


def needleman_wunsch(
    seq_a: list[str],
    seq_b: list[str],
    match_score: float = 1.0,
    mismatch_penalty: float = -1.0,
    gap_penalty: float = -1.0,
    scoring_func: Callable[[str, str], float] | None = None,
) -> tuple[list[tuple[int | None, int | None]], float]:
    """Global (Needleman-Wunsch) alignment of two string sequences.

    Args:
        seq_a: First sequence (e.g. reference words).
        seq_b: Second sequence (e.g. OCR words).
        match_score: Score for an exact match (used when ``scoring_func``
            is ``None``).
        mismatch_penalty: Penalty for a mismatch (used when ``scoring_func``
            is ``None``).
        gap_penalty: Penalty for inserting or deleting an element.
        scoring_func: Optional custom scoring function ``f(a, b) -> float``
            that returns the substitution score for a pair of elements.
            When provided, ``match_score`` and ``mismatch_penalty`` are
            ignored.

    Returns:
        ``(alignment_path, score)`` where *alignment_path* is a list of
        ``(index_a, index_b)`` tuples (``None`` = gap) and *score* is the
        total alignment score.
    """
    n, m = len(seq_a), len(seq_b)

    # Use the banded Hirschberg path for large sequences (O(n) memory).
    if n > 500 or m > 500:
        return _hirschberg_banded(seq_a, seq_b, match_score, mismatch_penalty, gap_penalty, scoring_func)

    # ── Full DP matrix ────────────────────────────────────────────────────
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_penalty

    for i in range(1, n + 1):
        a_i = seq_a[i - 1]
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + _sub_score(
                a_i, seq_b[j - 1], match_score, mismatch_penalty, scoring_func
            )
            ins = dp[i][j - 1] + gap_penalty
            dele = dp[i - 1][j] + gap_penalty
            dp[i][j] = max(sub, ins, dele)

    # ── Traceback ─────────────────────────────────────────────────────────
    path: list[tuple[int | None, int | None]] = []
    i, j = n, m
    score = dp[n][m]

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub = dp[i - 1][j - 1] + _sub_score(
                seq_a[i - 1], seq_b[j - 1], match_score, mismatch_penalty, scoring_func
            )
            if math.isclose(dp[i][j], sub, rel_tol=1e-9):
                path.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and math.isclose(dp[i][j], dp[i - 1][j] + gap_penalty, rel_tol=1e-9):
            path.append((i - 1, None))
            i -= 1
        else:
            path.append((None, j - 1))
            j -= 1

    path.reverse()
    return path, score


# ── Smith-Waterman (local) ──────────────────────────────────────────────────────


def smith_waterman(
    seq_a: list[str],
    seq_b: list[str],
    match_score: float = 2.0,
    mismatch_penalty: float = -1.0,
    gap_penalty: float = -1.0,
    scoring_func: Callable[[str, str], float] | None = None,
) -> tuple[list[tuple[int | None, int | None]], float]:
    """Local (Smith-Waterman) alignment of two string sequences.

    Finds the highest-scoring local segment between the two sequences.
    Useful for finding partial matches (e.g. a phrase from the reference
    appearing within a larger OCR block).

    Args:
        seq_a: First sequence.
        seq_b: Second sequence.
        match_score: Score for an exact match.
        mismatch_penalty: Penalty for a mismatch.
        gap_penalty: Penalty for inserting or deleting.
        scoring_func: Optional custom scoring function ``f(a, b) -> float``
            that returns the substitution score for a pair of elements.
            When provided, ``match_score`` and ``mismatch_penalty`` are
            ignored.

    Returns:
        ``(alignment_path, score)`` where *alignment_path* is the local
        alignment (may be a subset of the full sequences) and *score* is
        the maximum local alignment score.
    """
    n, m = len(seq_a), len(seq_b)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]

    max_score = 0.0
    max_i, max_j = 0, 0

    for i in range(1, n + 1):
        a_i = seq_a[i - 1]
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + _sub_score(
                a_i, seq_b[j - 1], match_score, mismatch_penalty, scoring_func
            )
            ins = dp[i][j - 1] + gap_penalty
            dele = dp[i - 1][j] + gap_penalty
            dp[i][j] = max(0.0, sub, ins, dele)

            if dp[i][j] > max_score:
                max_score = dp[i][j]
                max_i, max_j = i, j

    # ── Traceback from the maximum cell ────────────────────────────────
    path: list[tuple[int | None, int | None]] = []
    i, j = max_i, max_j

    while i > 0 and j > 0 and dp[i][j] > 0:
        if dp[i][j] == 0:
            break
        if i > 0 and j > 0:
            sub = dp[i - 1][j - 1] + _sub_score(
                seq_a[i - 1], seq_b[j - 1], match_score, mismatch_penalty, scoring_func
            )
            if math.isclose(dp[i][j], sub, rel_tol=1e-9):
                path.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and math.isclose(dp[i][j], dp[i - 1][j] + gap_penalty, rel_tol=1e-9):
            path.append((i - 1, None))
            i -= 1
        elif j > 0:
            path.append((None, j - 1))
            j -= 1
        else:
            break

    path.reverse()
    return path, max_score


# ── Banded Hirschberg (space-efficient NW for large sequences) ──────────────────


def _hirschberg_banded(
    seq_a: Sequence[str],
    seq_b: Sequence[str],
    match_score: float,
    mismatch_penalty: float,
    gap_penalty: float,
    scoring_func: Callable[[str, str], float] | None = None,
) -> tuple[list[tuple[int | None, int | None]], float]:
    """Banded Hirschberg divide-and-conquer alignment.

    Uses O(min(n,m)) space by recursively splitting the longer sequence and
    only computing a diagonal band of the DP matrix at each step.

    Falls back to full Needleman-Wunsch if the sequences are short or the
    band is too narrow.
    """
    n, m = len(seq_a), len(seq_b)

    # Base case: small enough for full DP.
    if n <= 500 and m <= 500:
        return _nw_full(seq_a, seq_b, match_score, mismatch_penalty, gap_penalty, scoring_func)

    band_width = max(n, m) * 0.1

    # Fallback: if band is too narrow relative to sequence length, use full NW.
    if band_width < 50:
        return _nw_full(seq_a, seq_b, match_score, mismatch_penalty, gap_penalty, scoring_func)

    # Ensure seq_a is the longer one for the divide step.
    if n < m:
        seq_a, seq_b = seq_b, seq_a
        n, m = m, n
        swapped = True
    else:
        swapped = False

    mid = n // 2

    # Forward pass: scores from start to mid.
    fwd = _nw_banded_row(seq_a[:mid], seq_b, match_score, mismatch_penalty, gap_penalty, scoring_func, band_width)

    # Backward pass: scores from end to mid.
    rev_a = list(reversed(seq_a[mid:]))
    rev_b = list(reversed(seq_b))
    rev = _nw_banded_row(rev_a, rev_b, match_score, mismatch_penalty, gap_penalty, scoring_func, band_width)

    # Combine forward and backward to find the split point.
    best_split = 0
    best_total = -float("inf")
    for k in range(m + 1):
        left_val = fwd[k] if k < len(fwd) else -float("inf")
        right_val = rev[m - k] if 0 <= m - k < len(rev) else -float("inf")
        total = left_val + right_val
        if total > best_total:
            best_total = total
            best_split = k

    # Recurse on left and right halves.
    left_path, left_score = _hirschberg_banded(
        seq_a[:mid], seq_b[:best_split], match_score, mismatch_penalty, gap_penalty, scoring_func
    )
    right_path, right_score = _hirschberg_banded(
        seq_a[mid:], seq_b[best_split:], match_score, mismatch_penalty, gap_penalty, scoring_func
    )

    path = left_path + right_path

    if swapped:
        # Swap indices back.
        path = [(b, a) if a is not None and b is not None else (b, a) for a, b in path]

    return path, left_score + right_score


def _nw_banded_row(
    seq_a: Sequence[str],
    seq_b: Sequence[str],
    match_score: float,
    mismatch_penalty: float,
    gap_penalty: float,
    scoring_func: Callable[[str, str], float] | None = None,
    band_width: float = 100.0,
) -> list[float]:
    """Compute the last row of a banded NW matrix.

    Only cells within ``band_width`` of the main diagonal (|i-j| <= band_width)
    are computed.
    """
    n, m = len(seq_a), len(seq_b)
    band = int(band_width)
    # Previous row of the banded matrix.
    prev: dict[int, float] = {0: 0.0}
    for j in range(1, m + 1):
        if j <= band:
            prev[-j] = -float("inf")
            prev[j] = prev.get(j - 1, -float("inf")) + gap_penalty

    for i in range(1, n + 1):
        curr: dict[int, float] = {}
        a_i = seq_a[i - 1]
        # Only compute cells within band_width.
        j_start = max(1, i - band)
        j_end = min(m, i + band)

        # Gap from top.
        if i <= band:
            curr[-i] = prev.get(-(i - 1), -float("inf")) + gap_penalty

        for j in range(j_start, j_end + 1):
            sub = prev.get(j - i, -float("inf")) + _sub_score(
                a_i, seq_b[j - 1], match_score, mismatch_penalty, scoring_func
            )
            ins = curr.get(j - i - 1, -float("inf")) + gap_penalty if j > j_start else -float("inf")
            dele = prev.get(j - i + 1, -float("inf")) + gap_penalty
            curr[j - i] = max(sub, ins, dele)

        prev = curr

    # Build the last row as a list indexed by j.
    last_row: list[float] = []
    for j in range(m + 1):
        last_row.append(prev.get(j - n, -float("inf")))
    return last_row


def _nw_full(
    seq_a: Sequence[str],
    seq_b: Sequence[str],
    match_score: float,
    mismatch_penalty: float,
    gap_penalty: float,
    scoring_func: Callable[[str, str], float] | None = None,
) -> tuple[list[tuple[int | None, int | None]], float]:
    """Standard Needleman-Wunsch (non-banded) returned as path + score."""
    n, m = len(seq_a), len(seq_b)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_penalty

    for i in range(1, n + 1):
        a_i = seq_a[i - 1]
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + _sub_score(a_i, seq_b[j - 1], match_score, mismatch_penalty, scoring_func)
            ins = dp[i][j - 1] + gap_penalty
            dele = dp[i - 1][j] + gap_penalty
            dp[i][j] = max(sub, ins, dele)

    path: list[tuple[int | None, int | None]] = []
    i, j = n, m
    score = dp[n][m]

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub = dp[i - 1][j - 1] + _sub_score(
                seq_a[i - 1], seq_b[j - 1], match_score, mismatch_penalty, scoring_func
            )
            if math.isclose(dp[i][j], sub, rel_tol=1e-9):
                path.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and math.isclose(dp[i][j], dp[i - 1][j] + gap_penalty, rel_tol=1e-9):
            path.append((i - 1, None))
            i -= 1
        else:
            path.append((None, j - 1))
            j -= 1

    path.reverse()
    return path, score


# ── Helpers ──────────────────────────────────────────────────────────────────────


def _sub_score(
    a: str,
    b: str,
    match_score: float,
    mismatch_penalty: float,
    scoring_func: Callable[[str, str], float] | None = None,
) -> float:
    """Compute substitution score between two elements."""
    if scoring_func is not None:
        return scoring_func(a, b)
    return match_score if a == b else mismatch_penalty


# ── Character-level alignment ────────────────────────────────────────────────────


def character_level_align(
    ref_text: str,
    ocr_text: str,
) -> list[tuple[str | None, str | None, str]]:
    """Character-level alignment between reference and OCR text strings.

    Uses Needleman-Wunsch on individual characters and annotates each
    aligned pair with an operation type.

    Args:
        ref_text: Ground-truth reference string.
        ocr_text: OCR output string.

    Returns:
        List of ``(ref_char, ocr_char, operation)`` tuples where:
        - ``ref_char`` / ``ocr_char`` is ``None`` for insertions / deletions
        - ``operation`` is one of ``"match"``, ``"substitution"``,
          ``"insertion"``, or ``"deletion"``
    """
    ref_chars = list(ref_text)
    ocr_chars = list(ocr_text)

    path, _ = needleman_wunsch(ref_chars, ocr_chars)

    result: list[tuple[str | None, str | None, str]] = []
    for ref_idx, ocr_idx in path:
        if ref_idx is not None and ocr_idx is not None:
            rc = ref_chars[ref_idx]
            oc = ocr_chars[ocr_idx]
            op = "match" if rc == oc else "substitution"
            result.append((rc, oc, op))
        elif ref_idx is not None:
            result.append((ref_chars[ref_idx], None, "deletion"))
        else:
            result.append((None, ocr_chars[ocr_idx], "insertion"))

    return result


# ── OCR text alignment pipeline ──────────────────────────────────────────────────


def _bbox_iou(bbox_a: list[float], bbox_b: list[float]) -> float:
    """Compute Intersection-over-Union between two bounding boxes.

    Bounding boxes are ``[x0, y0, x1, y1]`` in page-space coordinates.
    Returns a value in ``[0.0, 1.0]`` where 1.0 means identical boxes.
    """
    x_left = max(bbox_a[0], bbox_b[0])
    y_top = max(bbox_a[1], bbox_b[1])
    x_right = min(bbox_a[2], bbox_b[2])
    y_bottom = min(bbox_a[3], bbox_b[3])

    intersection = max(0.0, x_right - x_left) * max(0.0, y_bottom - y_top)
    if intersection == 0.0:
        return 0.0

    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _default_config() -> dict:
    """Return the default configuration dict for ``align_ocr_texts``."""
    return {
        "match_score": 1.0,
        "mismatch_penalty": -1.0,
        "gap_penalty": -1.0,
        "bbox_low_confidence_threshold": 0.5,
        "similarity_threshold": 0.6,
    }


def _word_scoring_func(ref_text: str, ocr_text: str) -> float:
    """Levenshtein-similarity-based scoring function for word alignment.

    Uses ``rapidfuzz.fuzz.ratio`` which returns 0-100; we normalise to 0-1
    and then scale to the ``[-1.0, 1.0]`` range expected by NW.
    """
    similarity = fuzz.ratio(ref_text, ocr_text) / 100.0
    # Map [0, 1] similarity to [-1, 1] score range.
    return 2.0 * similarity - 1.0


def align_ocr_texts(
    ocr_words: list[dict],
    reference_words: list[dict],
    config: dict | None = None,
) -> dict:
    """Align OCR word results against reference (ground-truth) words.

    Uses Needleman-Wunsch with Levenshtein similarity as the substitution
    scoring function.  When a word's confidence is below the threshold and
    bounding boxes are available, falls back to spatial (bbox IOU) matching
    for that region.

    Args:
        ocr_words: List of word dicts from OCR output. Each dict must
            contain ``"text"`` and may contain ``"bbox"`` (``[x0, y0, x1, y1]``)
            and ``"confidence"`` (``float`` in ``[0, 1]``).
        reference_words: List of word dicts from ground truth (same format).
        config: Optional configuration dict.  Default keys:
            - ``match_score`` (``float``, default ``1.0``)
            - ``mismatch_penalty`` (``float``, default ``-1.0``)
            - ``gap_penalty`` (``float``, default ``-1.0``)
            - ``bbox_low_confidence_threshold`` (``float``, default ``0.5``)
            - ``similarity_threshold`` (``float``, default ``0.6``)

    Returns:
        Alignment result dict with:
        ``word_pairs`` — per-word alignment entries
        ``score`` — overall alignment score
        ``stats`` — counts of match / substitution / insertion / deletion
    """
    cfg = {**_default_config(), **(config or {})}
    match_score: float = cfg["match_score"]
    gap_penalty: float = cfg["gap_penalty"]

    ocr_texts = [w["text"] for w in ocr_words]
    ref_texts = [w["text"] for w in reference_words]

    # ── Run Needleman-Wunsch with similarity scoring ────────────────────
    path, score = needleman_wunsch(
        ref_texts,
        ocr_texts,
        match_score=match_score,
        mismatch_penalty=-1.0,
        gap_penalty=gap_penalty,
        scoring_func=_word_scoring_func,
    )

    # ── Build output pairs ──────────────────────────────────────────────
    word_pairs: list[dict] = []
    stats: dict[str, int] = {"matches": 0, "substitutions": 0, "insertions": 0, "deletions": 0}

    for ref_idx, ocr_idx in path:
        pair: dict = {
            "ref_idx": ref_idx,
            "ocr_idx": ocr_idx,
            "ref_text": None,
            "ocr_text": None,
            "operation": "match",
            "similarity": 1.0,
            "bbox_iou": None,
        }

        if ref_idx is not None and ocr_idx is not None:
            ref_text = reference_words[ref_idx]["text"]
            ocr_text = ocr_words[ocr_idx]["text"]
            similarity = fuzz.ratio(ref_text, ocr_text) / 100.0

            pair["ref_text"] = ref_text
            pair["ocr_text"] = ocr_text
            pair["similarity"] = similarity

            # Determine operation from similarity threshold.
            if similarity >= cfg["similarity_threshold"]:
                pair["operation"] = "match"
                stats["matches"] += 1
            else:
                pair["operation"] = "substitution"
                stats["substitutions"] += 1

            # Bbox-based tiebreaking for low-confidence words.
            ref_word = reference_words[ref_idx]
            ocr_word = ocr_words[ocr_idx]
            ref_conf = ref_word.get("confidence", 1.0)
            ocr_conf = ocr_word.get("confidence", 1.0)

            threshold = cfg["bbox_low_confidence_threshold"]
            low_conf = ref_conf < threshold or ocr_conf < threshold
            if low_conf and "bbox" in ref_word and "bbox" in ocr_word:
                pair["bbox_iou"] = _bbox_iou(ref_word["bbox"], ocr_word["bbox"])
                # Override operation if bbox IOU confirms a match.
                if pair["bbox_iou"] is not None and pair["bbox_iou"] > 0.7 and pair["operation"] == "substitution":
                    pair["operation"] = "match"

        elif ref_idx is not None:
            pair["ref_text"] = reference_words[ref_idx]["text"]
            pair["operation"] = "deletion"
            stats["deletions"] += 1
        else:
            pair["ocr_text"] = ocr_words[ocr_idx]["text"]
            pair["operation"] = "insertion"
            stats["insertions"] += 1

        word_pairs.append(pair)

    return {
        "word_pairs": word_pairs,
        "score": score / max(len(ref_texts), len(ocr_texts), 1),
        "stats": stats,
    }
