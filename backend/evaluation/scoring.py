"""CER / WER scoring pipeline for OCR evaluation.

Computes character-level and word-level evaluation metrics by comparing OCR
output against ground truth using the alignment algorithms from
:mod:`backend.alignment.aligner`.

Typical usage::

    from backend.evaluation import compute_cer, compute_wer, evaluate_page

    # Simple metric
    cer = compute_cer("hello world", "hello wor1d")

    # Per-page evaluation
    page_results = [{"text": "hello", "bbox": [...], "confidence": 0.9}, ...]
    ground_truth = [{"text": "hello", "bbox": [...], "confidence": 1.0}, ...]
    scores = evaluate_page(page_results, ground_truth)
"""

from __future__ import annotations

from backend.alignment.aligner import character_level_align

# ── Core metric functions ─────────────────────────────────────────────────────


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate between a reference and hypothesis string.

    Defined as ``(substitutions + insertions + deletions) / len(reference)``
    using a Needleman-Wunsch character-level alignment.

    Args:
        reference: Ground-truth string.
        hypothesis: OCR-hypothesis string.

    Returns:
        CER in ``[0.0, 1.0]``.  Returns ``0.0`` when both strings are empty.
        Returns ``1.0`` when the reference is empty but hypothesis is not.
    """
    n_ref = len(reference)

    if n_ref == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    aligned = character_level_align(reference, hypothesis)

    substitutions = sum(1 for _, _, op in aligned if op == "substitution")
    insertions = sum(1 for _, _, op in aligned if op == "insertion")
    deletions = sum(1 for _, _, op in aligned if op == "deletion")

    return (substitutions + insertions + deletions) / n_ref


def compute_wer(
    reference_words: list[str],
    hypothesis_words: list[str],
) -> float:
    """Word Error Rate between a reference and hypothesis word list.

    Defined as ``(substitutions + insertions + deletions) / len(reference_words)``.

    Uses Needleman-Wunsch alignment on word sequences with exact-match scoring.

    Args:
        reference_words: Ground-truth word sequence.
        hypothesis_words: OCR-hypothesis word sequence.

    Returns:
        WER in ``[0.0, 1.0]``.  Returns ``0.0`` when both word lists are empty.
        Returns ``1.0`` when the reference is empty but hypothesis is not.
    """
    n_ref = len(reference_words)

    if n_ref == 0:
        return 0.0 if len(hypothesis_words) == 0 else 1.0

    substitutions, insertions, deletions = _word_levenshtein_counts(
        reference_words,
        hypothesis_words,
    )

    return (substitutions + insertions + deletions) / n_ref


def compute_precision_recall_f1(
    tp: int,
    fp: int,
    fn: int,
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 score from classification counts.

    Args:
        tp: True positives (correctly recognised elements).
        fp: False positives (spurious / extra elements).
        fn: False negatives (elements missed from reference).

    Returns:
        ``(precision, recall, f1)`` — each in ``[0.0, 1.0]``.
        All three are ``1.0`` when ``tp > 0`` and ``fp == fn == 0``.
        All three are ``0.0`` when ``tp == 0``.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ── Aggregate metric builders ────────────────────────────────────────────────


def compute_char_metrics(
    reference: str,
    hypothesis: str,
) -> dict:
    """Compute full character-level evaluation metrics.

    Returns a dict with:
        - ``cer`` — Character Error Rate
        - ``precision`` — correct / (correct + substituted + inserted)
        - ``recall`` — correct / (correct + substituted + deleted)
        - ``f1`` — harmonic mean of precision and recall
        - ``breakdown`` — ``{matches, substitutions, insertions, deletions}``
        - ``confusion_matrix`` — nested dict ``{ref_char: {hyp_char: count}}``
          for substitution pairs only (matches and gaps are excluded)

    Args:
        reference: Ground-truth string.
        hypothesis: OCR-hypothesis string.
    """
    n_ref = len(reference)
    n_hyp = len(hypothesis)

    aligned = character_level_align(reference, hypothesis)

    matches = 0
    substitutions = 0
    insertions = 0
    deletions = 0
    confusion: dict[str, dict[str, int]] = {}

    for ref_char, hyp_char, op in aligned:
        if op == "match":
            matches += 1
        elif op == "substitution":
            substitutions += 1
            # Build confusion matrix: only substitutions.
            if ref_char is not None and hyp_char is not None:
                confusion.setdefault(ref_char, {})
                confusion[ref_char][hyp_char] = confusion[ref_char].get(hyp_char, 0) + 1
        elif op == "insertion":
            insertions += 1
        elif op == "deletion":
            deletions += 1

    cer = (substitutions + insertions + deletions) / n_ref if n_ref > 0 else 0.0

    # Character-level precision/recall/F1.
    tp = matches
    fp = substitutions + insertions
    fn = substitutions + deletions
    precision, recall, f1 = compute_precision_recall_f1(tp, fp, fn)

    return {
        "cer": cer,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "breakdown": {
            "matches": matches,
            "substitutions": substitutions,
            "insertions": insertions,
            "deletions": deletions,
        },
        "confusion_matrix": confusion,
        "_ref_length": n_ref,
        "_hyp_length": n_hyp,
    }


def compute_word_metrics(
    reference_words: list[str],
    hypothesis_words: list[str],
) -> dict:
    """Compute full word-level evaluation metrics.

    Returns a dict with:
        - ``wer`` — Word Error Rate
        - ``precision`` — correct / (correct + substituted + inserted)
        - ``recall`` — correct / (correct + substituted + deleted)
        - ``f1`` — harmonic mean of precision and recall
        - ``breakdown`` — ``{matches, substitutions, insertions, deletions}``
          with per-type counts

    Args:
        reference_words: Ground-truth word sequence.
        hypothesis_words: OCR-hypothesis word sequence.
    """
    n_ref = len(reference_words)
    n_hyp = len(hypothesis_words)

    substitutions, insertions, deletions = _word_levenshtein_counts(
        reference_words,
        hypothesis_words,
    )
    # Matches are reference words that survived — they are neither deleted
    # nor substituted.  This is equivalent to counting aligned word pairs
    # whose text is identical.
    matches = max(0, n_ref - substitutions - deletions)

    wer = (substitutions + insertions + deletions) / n_ref if n_ref > 0 else 0.0

    # Word-level precision/recall/F1.
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


# ── Internal helpers ──────────────────────────────────────────────────────────


def _word_levenshtein_counts(
    ref_words: list[str],
    hyp_words: list[str],
) -> tuple[int, int, int]:
    """Return ``(substitutions, insertions, deletions)`` via word-level DP.

    Uses Levenshtein distance (edit distance) on word sequences with exact
    string equality as the matching criterion.
    """
    n, m = len(ref_words), len(hyp_words)

    # DP table: dp[i][j] = edit distance between ref_words[:i] and hyp_words[:j].
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i  # i deletions
    for j in range(1, m + 1):
        dp[0][j] = j  # j insertions

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution or match
            )

    # Backtrace to count operations.
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


def _normalize_page_list(data: dict) -> list:
    """Extract a list of page entries from a run/gt data dict.

    Handles both ``{"pages": [...]}`` and raw list inputs stored as dict values.
    """
    pages = data.get("pages", data) if isinstance(data, dict) else data
    if isinstance(pages, dict):
        # Fallback: if the dict itself looks like a page result, wrap it.
        if "results" in pages or "data" in pages:
            return [pages]
        return []
    if isinstance(pages, list):
        return pages
    return []


def _extract_words(page_entry: dict) -> list[dict]:
    """Extract a flat list of word dicts from a page entry.

    Handles:
    - ``{"results": [word_dict, ...]}`` — flat word list.
    - ``{"data": {"blocks": [{"lines": [{"words": [...]}]}]}}`` — hierarchical.
    - A list of word dicts directly (if passed as page entry).
    """
    # Direct result list.
    results = page_entry.get("results")
    if results is not None and isinstance(results, list):
        return results

    # Hierarchical JSONB data.
    data = page_entry.get("data", page_entry)
    if isinstance(data, dict):
        words: list[dict] = []
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                words.extend(line.get("words", []))
        # If the data dict itself contains word-level entries with "text".
        if not words and "text" in data:
            return [data]
        return words

    # If the page_entry itself is a list of word dicts.
    if isinstance(page_entry, list):
        return page_entry

    return []
