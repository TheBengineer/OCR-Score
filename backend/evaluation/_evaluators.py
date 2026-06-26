"""Page-level and run-level OCR evaluators.

Separated from :mod:`backend.evaluation.scoring` to keep the core metric
module under 250 pure LOC.
"""

from __future__ import annotations

from backend.alignment.aligner import align_ocr_texts
from backend.evaluation.scoring import compute_char_metrics, compute_word_metrics


def evaluate_page(
    page_results: list[dict],
    ground_truth: list[dict],
    config: dict | None = None,
) -> dict:
    """Full evaluation of a single page's OCR results against ground truth.

    The function computes:
    - Character-level metrics (CER, precision, recall, F1, confusion matrix)
    - Word-level metrics (WER, precision, recall, F1, I/D/S breakdown)
    - Block-level alignment statistics

    Args:
        page_results: List of word dicts from OCR output. Each dict must
            contain ``"text"`` and may contain ``"bbox"`` and ``"confidence"``.
        ground_truth: List of word dicts from ground truth (same format).
        config: Optional configuration dict forwarded to
            :func:`~backend.alignment.aligner.align_ocr_texts`.

    Returns:
        Dict with ``cer``, ``wer``, ``char_precision``, ``char_recall``,
        ``char_f1``, ``word_precision``, ``word_recall``, ``word_f1``,
        ``char_confusion``, ``word_breakdown``, and ``alignment_stats``.
    """
    ref_text = " ".join(entry["text"] for entry in ground_truth)
    hyp_text = " ".join(entry["text"] for entry in page_results)

    char_metrics = compute_char_metrics(ref_text, hyp_text)

    ref_word_texts = [entry["text"] for entry in ground_truth]
    hyp_word_texts = [entry["text"] for entry in page_results]
    word_metrics = compute_word_metrics(ref_word_texts, hyp_word_texts)

    alignment = align_ocr_texts(page_results, ground_truth, config)

    return {
        "cer": char_metrics["cer"],
        "wer": word_metrics["wer"],
        "char_precision": char_metrics["precision"],
        "char_recall": char_metrics["recall"],
        "char_f1": char_metrics["f1"],
        "word_precision": word_metrics["precision"],
        "word_recall": word_metrics["recall"],
        "word_f1": word_metrics["f1"],
        "char_confusion": char_metrics["confusion_matrix"],
        "word_breakdown": word_metrics["breakdown"],
        "alignment_stats": alignment.get("stats", {}),
    }


def evaluate_run(run_data: dict, gt_data: dict) -> dict:
    """Aggregate evaluation scores across all pages in a run.

    Each page is evaluated individually then aggregated using weighted
    averages — character-level metrics are weighted by the number of
    reference characters on each page; word-level metrics are weighted
    by the number of reference words.

    Args:
        run_data: Dict with a ``"pages"`` key mapping to a list of page
            entries.  Each page entry is a dict that contains either a
            ``"results"`` key (list of word dicts) or a ``"data"`` key
            (JSONB hierarchy from which words are extracted via
            :func:`_extract_words`).
        gt_data: Same structure as *run_data*, representing ground truth.

    Returns:
        Aggregated dict with ``cer``, ``wer``, ``char_*``, ``word_*``
        scores, plus ``per_page`` and ``num_pages``.
    """
    run_pages = _normalize_page_list(run_data)
    gt_pages = _normalize_page_list(gt_data)

    per_page: list[dict] = []
    total_char_weight = 0
    total_word_weight = 0
    weighted_cer = 0.0
    weighted_char_prec = 0.0
    weighted_char_rec = 0.0
    weighted_char_f1 = 0.0
    weighted_wer = 0.0
    weighted_word_prec = 0.0
    weighted_word_rec = 0.0
    weighted_word_f1 = 0.0

    num_pages = min(len(run_pages), len(gt_pages))
    for i in range(num_pages):
        run_page_words = _extract_words(run_pages[i])
        gt_page_words = _extract_words(gt_pages[i])

        page_result = evaluate_page(run_page_words, gt_page_words)
        per_page.append(page_result)

        ref_text = " ".join(w["text"] for w in gt_page_words)
        ref_words = [w["text"] for w in gt_page_words]
        cw = len(ref_text)
        ww = len(ref_words)

        total_char_weight += cw
        total_word_weight += ww
        weighted_cer += page_result["cer"] * cw
        weighted_char_prec += page_result["char_precision"] * cw
        weighted_char_rec += page_result["char_recall"] * cw
        weighted_char_f1 += page_result["char_f1"] * cw
        weighted_wer += page_result["wer"] * ww
        weighted_word_prec += page_result["word_precision"] * ww
        weighted_word_rec += page_result["word_recall"] * ww
        weighted_word_f1 += page_result["word_f1"] * ww

    def _safe_div(value: float, weight: int) -> float:
        return value / weight if weight > 0 else 0.0

    return {
        "cer": _safe_div(weighted_cer, total_char_weight),
        "wer": _safe_div(weighted_wer, total_word_weight),
        "char_precision": _safe_div(weighted_char_prec, total_char_weight),
        "char_recall": _safe_div(weighted_char_rec, total_char_weight),
        "char_f1": _safe_div(weighted_char_f1, total_char_weight),
        "word_precision": _safe_div(weighted_word_prec, total_word_weight),
        "word_recall": _safe_div(weighted_word_rec, total_word_weight),
        "word_f1": _safe_div(weighted_word_f1, total_word_weight),
        "per_page": per_page,
        "num_pages": num_pages,
    }


def _normalize_page_list(data: dict) -> list:
    """Extract a list of page entries from a run/gt data dict.

    Handles both ``{"pages": [...]}`` and raw list inputs stored as dict values.
    """
    pages = data.get("pages", data) if isinstance(data, dict) else data
    if isinstance(pages, dict):
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
    results = page_entry.get("results")
    if results is not None and isinstance(results, list):
        return results

    data = page_entry.get("data", page_entry)
    if isinstance(data, dict):
        words: list[dict] = []
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                words.extend(line.get("words", []))
        if not words and "text" in data:
            return [data]
        return words

    if isinstance(page_entry, list):
        return page_entry

    return []
