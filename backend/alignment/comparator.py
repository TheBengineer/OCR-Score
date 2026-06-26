"""Multi-engine comparison module for OCRScore.

Provides functions to align OCR outputs from multiple engines and build
a comparison grid response suitable for the frontend overlay system.

The alignment strategy:
1. Pairwise Needleman-Wunsch against the first engine as reference.
2. For each aligned position, collect all engines' interpretations.
3. Compute consensus text via majority vote (confidence-weighted for ties).
4. Compute per-position consensus entropy and overall entropy.
5. Determine ``status`` per engine per position:
   - ``"match"`` — engine text matches the consensus
   - ``"wrong"`` — engine text differs from consensus
   - ``"missing"`` — engine has a gap at this position
   - ``"extra"`` — engine has an insertion not in the reference
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from backend.alignment.aligner import needleman_wunsch


def _extract_flat_words(page_result: dict) -> list[dict]:
    """Flatten word list from page result dict (supports flat words, JSONB, and partial JSONB formats)."""
    words: list[dict] | None = page_result.get("words")
    if words and isinstance(words, list) and words:
        return words

    data: dict = page_result.get("data", page_result)
    if isinstance(data, dict):
        result_words: list[dict] = []
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                result_words.extend(line.get("words", []))
        if result_words:
            return result_words

    inner_data: dict | None = data.get("data")
    if isinstance(inner_data, dict):
        result_words = []
        for block in inner_data.get("blocks", []):
            for line in block.get("lines", []):
                result_words.extend(line.get("words", []))
        if result_words:
            return result_words

    return []


def _compute_consensus_for_position(
    position_words: dict[str, dict | None],
) -> tuple[str | None, float]:
    """Majority vote (equal-weight) with confidence-weighted tiebreaker."""
    text_counts: Counter[str] = Counter()
    text_conf_sum: dict[str, float] = {}
    text_conf_n: dict[str, int] = {}

    for _eng_name, word in position_words.items():
        if word and word.get("text"):
            text = word["text"]
            text_counts[text] += 1
            conf = word.get("confidence")
            if conf is not None:
                text_conf_sum[text] = text_conf_sum.get(text, 0.0) + conf
                text_conf_n[text] = text_conf_n.get(text, 0) + 1

    if not text_counts:
        return None, 0.0

    max_count = max(text_counts.values())
    tied = [t for t, c in text_counts.items() if c == max_count]

    if len(tied) == 1:
        consensus_text = tied[0]
    else:
        best_text = tied[0]
        best_avg = 0.0
        for text in tied:
            n_c = text_conf_n.get(text, 0)
            avg = text_conf_sum.get(text, 0.0) / n_c if n_c > 0 else 0.0
            if avg > best_avg:
                best_avg, best_text = avg, text
        consensus_text = best_text

    n = text_conf_n.get(consensus_text, 0)
    consensus_confidence = (text_conf_sum.get(consensus_text, 0.0) / n) if n > 0 else 1.0
    return consensus_text, consensus_confidence


def _compute_position_entropy(position_texts: list[str | None]) -> float:
    """Normalised entropy for a single aligned position in [0.0, 1.0]."""
    non_null = [t for t in position_texts if t is not None]
    n = len(non_null)
    if n < 2:
        return 0.0

    counter: Counter[str] = Counter(non_null)
    entropy = 0.0
    for count in counter.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)

    max_entropy = math.log2(n)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _build_aligned_positions(
    ref_words: list[dict],
    word_lists: list[list[dict]],
    engine_names: list[str],
    alignments: list[list[tuple[int | None, int | None]]],
) -> list[dict[str, Any]]:
    """Project all engines onto reference word positions using pairwise NW alignments."""
    aligned_words: list[dict[str, Any]] = []

    for ref_idx, ref_word in enumerate(ref_words):
        position_word_map: dict[str, dict | None] = {engine_names[0]: ref_word}

        for eng_idx, path in enumerate(alignments):
            eng_word: dict | None = None
            target_list = word_lists[eng_idx + 1]
            for r_idx, e_idx in path:
                if r_idx == ref_idx:
                    if e_idx is not None:
                        eng_word = target_list[e_idx]
                    break
            position_word_map[engine_names[eng_idx + 1]] = eng_word

        consensus_text, consensus_conf = _compute_consensus_for_position(position_word_map)

        engine_entries: dict[str, dict] = {}
        for eng_name, word in position_word_map.items():
            if word is not None:
                status = "match" if word["text"] == consensus_text else "wrong"
                engine_entries[eng_name] = {
                    "text": word.get("text"),
                    "confidence": word.get("confidence"),
                    "bbox": word.get("bbox"),
                    "status": status,
                }
            else:
                engine_entries[eng_name] = {
                    "text": None, "confidence": None, "bbox": None,
                    "status": "missing",
                }

        aligned_words.append({
            "position": ref_idx,
            "consensus": consensus_text,
            "consensus_confidence": consensus_conf,
            "engines": engine_entries,
        })

    return aligned_words


def _compute_overall_entropy(aligned_words: list[dict], engine_names: list[str]) -> float:
    """Average normalised consensus entropy across all positions."""
    total = 0.0
    count = 0
    for pos in aligned_words:
        texts: list[str | None] = []
        for eng_name in engine_names:
            eng_info = pos["engines"].get(eng_name, {})
            texts.append(eng_info.get("text"))
        total += _compute_position_entropy(texts)
        count += 1
    return total / count if count > 0 else 0.0


def _compute_engine_stats(
    aligned_words: list[dict],
    engine_names: list[str],
    extras_by_engine: dict[str, list[dict]],
) -> dict[str, dict[str, int]]:
    """Compute per-engine match/wrong/missing/extra counts."""
    stats: dict[str, dict[str, int]] = {}
    for eng_name in engine_names:
        counts: dict[str, int] = {"match": 0, "wrong": 0, "missing": 0, "extra": 0}
        for pos in aligned_words:
            status = pos["engines"].get(eng_name, {}).get("status", "missing")
            if status in counts:
                counts[status] += 1
        counts["extra"] = len(extras_by_engine.get(eng_name, []))
        stats[eng_name] = counts
    return stats


def _empty_result(engine_names: list[str]) -> dict:
    """Return an empty alignment result."""
    extras: dict[str, list[dict]] = {name: [] for name in engine_names}
    zero_stats: dict[str, int] = {"match": 0, "wrong": 0, "missing": 0, "extra": 0}
    return {
        "aligned_words": [],
        "engines": engine_names,
        "consensus_entropy": 0.0,
        "num_positions": 0,
        "extras": extras,
        "stats": {"total_words": 0, "engine_stats": {name: dict(zero_stats) for name in engine_names}},
    }


def align_multiple_engine_pages(
    engine_results: list[dict],
    config: dict | None = None,  # noqa: ARG001 — reserved for future alignment thresholds
) -> dict:
    """Align OCR outputs from multiple engines into a common word grid.

    Uses pairwise Needleman-Wunsch against the **first** engine as reference,
    then projects all engines onto the reference word positions.

    Args:
        engine_results: List of page result dicts. Each should have an
            ``"engine"`` key (slug/name) and contain word data in either
            flat ``{"words": [...]}`` or canonical JSONB format.
        config: Reserved for future alignment thresholds.

    Returns:
        Aligned data dict with ``aligned_words``, ``engines``,
        ``consensus_entropy``, ``num_positions``, ``extras``, and ``stats``.
    """
    if not engine_results:
        return _empty_result([])

    word_lists: list[list[dict]] = [_extract_flat_words(er) for er in engine_results]
    engine_names: list[str] = [
        er.get("engine", er.get("engine_id", f"engine_{i}"))
        for i, er in enumerate(engine_results)
    ]

    if not word_lists[0]:
        return _empty_result(engine_names)

    ref_words = word_lists[0]
    ref_texts = [w["text"] for w in ref_words]
    alignments = [
        needleman_wunsch(ref_texts, [w["text"] for w in word_lists[i]])[0]
        for i in range(1, len(engine_results))
    ]
    extras_by_engine: dict[str, list[dict]] = {name: [] for name in engine_names}

    aligned_words = _build_aligned_positions(ref_words, word_lists, engine_names, alignments)

    for ei, path in enumerate(alignments):
        ename = engine_names[ei + 1]
        for r_idx, e_idx in path:
            if r_idx is None and e_idx is not None:
                w = word_lists[ei + 1][e_idx]
                extras_by_engine[ename].append({k: w.get(k) for k in ("text", "confidence", "bbox")})

    overall_entropy = _compute_overall_entropy(aligned_words, engine_names)
    engine_stats = _compute_engine_stats(aligned_words, engine_names, extras_by_engine)

    return {
        "aligned_words": aligned_words,
        "engines": engine_names,
        "consensus_entropy": overall_entropy,
        "num_positions": len(aligned_words),
        "extras": extras_by_engine,
        "stats": {"total_words": len(ref_words), "engine_stats": engine_stats},
    }


def build_comparison_grid(
    aligned_data: dict,
    page_number: int = 1,
    dimensions: dict | None = None,
) -> dict:
    """Build the comparison grid response from aligned data.

    Args:
        aligned_data: Output from :func:`align_multiple_engine_pages`.
        page_number: 1-based page number.
        dimensions: Optional ``{"width": float, "height": float}``.

    Returns:
        Full comparison grid response dict.
    """
    return {
        "page_number": page_number,
        "dimensions": dimensions or {"width": 0.0, "height": 0.0},
        "engines": aligned_data.get("engines", []),
        "alignment": {"aligned_words": aligned_data.get("aligned_words", [])},
        "consensus_entropy": aligned_data.get("consensus_entropy", 0.0),
        "stats": aligned_data.get("stats", {"total_words": 0, "engine_stats": {}}),
    }
