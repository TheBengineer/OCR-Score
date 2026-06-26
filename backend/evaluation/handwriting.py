"""Handwriting detection and tagging for OCR evaluation.

Detects handwriting regions in OCR engine output using heuristic rules
(low confidence + irregular character spacing, or engine metadata) and
tags them so they can be excluded from scoring or scored separately.

Typical usage::

    from backend.evaluation.handwriting import (
        tag_handwriting_regions,
        filter_handwriting_from_scoring,
    )

    # Tag handwriting regions in an engine's output
    tagged = tag_handwriting_regions(engine_output)

    # Separate handwriting and non-handwriting content for scoring
    text_only, handwriting_only = filter_handwriting_from_scoring(page_data)
"""

from __future__ import annotations

# Default confidence threshold below which a word may be considered
# handwriting if other heuristics also apply.
_HANDWRITING_CONFIDENCE_THRESHOLD: float = 0.5

# The ratio of (max_char_width / min_char_width) within a block above
# which spacing is considered "irregular" — a handwriting indicator.
_IRREGULAR_SPACING_RATIO: float = 3.0

# Keys in an engine output dict that may indicate handwriting.
_HANDWRITING_METADATA_KEYS: set[str] = {
    "handwriting",
    "handwritten",
    "is_handwriting",
    "script_type",
    "writing_style",
}


def tag_handwriting_regions(engine_output: dict) -> dict:
    """Detect and tag handwriting regions in an OCR engine output dict.

    Heuristics applied (in order):

    1. **Engine metadata** — if the engine output already contains a field
       named ``"handwriting"``, ``"handwritten"``, ``"is_handwriting"``,
       ``"script_type"``, or ``"writing_style"`` with a value indicating
       handwriting, the entire document is tagged.
    2. **Low confidence + irregular spacing** — words with confidence < 0.5
       whose block has highly irregular inter-character spacing.
    3. **Low confidence on its own** — words with confidence < 0.3 are also
       tagged, as very low confidence often indicates handwriting that the
       engine cannot parse properly.

    The function modifies *engine_output* in place, adding
    ``"handwriting": True`` to each tagged word dict, and also sets
    ``"has_handwriting": True`` at the top level.

    Args:
        engine_output: Engine output dict. Expected to contain either a
            ``"pages"`` key (list of page dicts) or a top-level
            ``"blocks"`` key (list of block dicts).

    Returns:
        The same dict (mutated) with handwriting tags added.
    """
    # Check top-level engine metadata first.
    if _check_engine_metadata(engine_output):
        # If the engine says it's all handwriting, tag everything.
        _tag_all(engine_output)
        engine_output["has_handwriting"] = True
        return engine_output

    pages = engine_output.get("pages", [engine_output] if "blocks" in engine_output else [])

    has_handwriting = False
    for page in pages:
        page_tagged = _tag_page(page)
        if page_tagged:
            has_handwriting = True

    engine_output["has_handwriting"] = has_handwriting
    return engine_output


def filter_handwriting_from_scoring(page_data: dict) -> tuple[dict, dict]:
    """Separate handwriting-tagged content from text for scoring.

    Splits a page-level data dict into two dicts:

    * ``text_data`` — contains only entries **without** the
      ``"handwriting": True`` flag.
    * ``handwriting_data`` — contains only entries **with** the
      ``"handwriting": True`` flag.

    Both output dicts preserve the original structure
    (``"results"`` list or ``"data"`` with nested blocks/lines/words).

    Args:
        page_data: A page-level dict in either flat
            ``{"results": [word_dict, ...]}`` format or hierarchical
            ``{"data": {"blocks": [...]}}`` format.

    Returns:
        ``(text_data, handwriting_data)`` — two dicts with the same
        structural shape as *page_data*.
    """
    results = page_data.get("results")
    if results is not None and isinstance(results, list):
        text_words = [w for w in results if not w.get("handwriting")]
        hw_words = [w for w in results if w.get("handwriting")]
        return {"results": text_words}, {"results": hw_words}

    data = page_data.get("data", page_data) if isinstance(page_data, dict) else page_data
    if isinstance(data, dict) and "blocks" in data:
        text_blocks: list[dict] = []
        hw_blocks: list[dict] = []
        for block in data["blocks"]:
            text_lines: list[dict] = []
            hw_lines: list[dict] = []
            for line in block.get("lines", []):
                text_words_line: list[dict] = []
                hw_words_line: list[dict] = []
                for word in line.get("words", []):
                    if word.get("handwriting"):
                        hw_words_line.append(word)
                    else:
                        text_words_line.append(word)
                if text_words_line:
                    text_lines.append({**line, "words": text_words_line})
                if hw_words_line:
                    hw_lines.append({**line, "words": hw_words_line})
            if text_lines:
                text_blocks.append({**block, "lines": text_lines})
            if hw_lines:
                hw_blocks.append({**block, "lines": hw_lines})
        return {"data": {"blocks": text_blocks}}, {"data": {"blocks": hw_blocks}}

    # Unknown structure — return everything as text.
    return page_data, {"data": {"blocks": []}}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_engine_metadata(output: dict) -> bool:
    """Check if engine metadata indicates handwriting."""
    for key in _HANDWRITING_METADATA_KEYS:
        value = output.get(key)
        if value is not None:
            if isinstance(value, str) and value.lower() in ("handwriting", "handwritten", "cursive"):
                return True
            if isinstance(value, bool) and value:
                return True
    return False


def _tag_all(output: dict) -> None:
    """Recursively tag every word in the output as handwriting."""
    pages = output.get("pages", [output] if "blocks" in output else [])
    for page in pages:
        _tag_words_in_page(page)


def _tag_page(page: dict) -> bool:
    """Tag handwriting words on a single page. Returns True if any tagged."""
    words = _collect_words(page)
    if not words:
        return False

    # Group words by block index for spacing analysis.
    block_word_map: dict[int, list[dict]] = {}
    for w in words:
        block_idx = w.get("block_index", 0) or 0
        block_word_map.setdefault(block_idx, []).append(w)

    # Precompute irregular blocks.
    irregular_blocks: set[int] = set()
    for block_idx, block_words in block_word_map.items():
        widths = [_word_width(w) for w in block_words if _word_width(w) is not None]
        if len(widths) >= 3:
            min_w = min(widths)
            max_w = max(widths)
            if min_w > 0 and (max_w / min_w) > _IRREGULAR_SPACING_RATIO:
                irregular_blocks.add(block_idx)

    tagged_any = False
    for w in words:
        conf = w.get("confidence")
        if conf is None or not isinstance(conf, (int, float)):
            continue

        block_idx = w.get("block_index", 0) or 0
        is_irregular = block_idx in irregular_blocks

        if conf < 0.3:
            # Very low confidence -> handwriting.
            w["handwriting"] = True
            tagged_any = True
        elif conf < _HANDWRITING_CONFIDENCE_THRESHOLD and is_irregular:
            # Low confidence + irregular spacing -> handwriting.
            w["handwriting"] = True
            tagged_any = True

    return tagged_any


def _collect_words(page: dict) -> list[dict]:
    """Flatten all word dicts from a page entry."""
    results = page.get("results")
    if results is not None and isinstance(results, list):
        return results

    words: list[dict] = []
    data = page.get("data", page)
    if isinstance(data, dict):
        for block_idx, block in enumerate(data.get("blocks", [])):
            for line in block.get("lines", []):
                for word in line.get("words", []):
                    word["block_index"] = block_idx
                    words.append(word)
    return words


def _word_width(word: dict) -> float | None:
    """Compute the width of a word from its bbox, or None."""
    bbox = word.get("bbox")
    if bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    return float(bbox[2] - bbox[0])


def _tag_words_in_page(page: dict) -> None:
    """Tag all words in a page as handwriting."""
    words = _collect_words(page)
    for w in words:
        w["handwriting"] = True
