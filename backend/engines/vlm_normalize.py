"""VLM output normalisation — converts free-form VLM text into ``NormalizedPage``.

VLM-based OCR engines (olmOCR, DeepSeek-OCR) return unstructured or
semi-structured text without positional data.  This module provides
functions to parse markdown or JSON output into the canonical
``NormalizedDocument`` hierarchy using heuristic bounding boxes.

Since VLMs do not provide character-level bounding boxes, the output is
marked as **lossy** via ``VLM_LOSSY_METADATA``.
"""

from __future__ import annotations

import json
from typing import Any

from backend.engine.normalized_schema import (
    Character as NormalizedCharacter,
)
from backend.engine.normalized_schema import (
    NormalizedPage,
)
from backend.engine.normalized_schema import (
    TextBlock as NormalizedTextBlock,
)
from backend.engine.normalized_schema import (
    TextLine as NormalizedTextLine,
)
from backend.engine.normalized_schema import (
    Word as NormalizedWord,
)
from backend.engines.vlm_layout import (
    heuristic_block_bbox,
    heuristic_line_bboxes,
    split_markdown_blocks,
)


def normalize_vlm_output(
    raw_text: str,
    page_dims: tuple[float, float],
    _dpi: int = 300,
    output_format: str = "markdown",
) -> NormalizedPage:
    """Convert VLM text to ``NormalizedPage`` with heuristic bboxes."""
    if output_format == "json":
        return normalize_json_output(raw_text, page_dims)
    return normalize_markdown_output(raw_text, page_dims)


def normalize_markdown_output(
    raw_text: str,
    page_dims: tuple[float, float],
) -> NormalizedPage:
    """Convert markdown VLM output — blocks distributed heuristically on page."""
    page_width, page_height = page_dims
    raw_blocks = split_markdown_blocks(raw_text)

    blocks: list[NormalizedTextBlock] = []
    block_order = 0

    for b_idx, raw_block in enumerate(raw_blocks):
        if raw_block["type"] == "heading":
            text = raw_block["text"]
            block_bbox = heuristic_block_bbox(
                b_idx, len(raw_blocks), page_width, page_height, len(text)
            )
            lines = text_to_lines(text, block_bbox)
            blocks.append(
                NormalizedTextBlock(
                    type="text",
                    bbox=block_bbox,
                    confidence=1.0,
                    order=block_order,
                    lines=lines,
                )
            )
            block_order += 1

        elif raw_block["type"] in {"text", "code"}:
            text_lines = raw_block["lines"]
            block_bbox = heuristic_block_bbox(
                b_idx,
                len(raw_blocks),
                page_width,
                page_height,
                sum(len(line_) for line_ in text_lines),
            )
            line_bbox_list = heuristic_line_bboxes(len(text_lines), block_bbox)

            lines: list[NormalizedTextLine] = []
            for l_idx, line_text in enumerate(text_lines):
                line_stripped = line_text.strip()
                if not line_stripped:
                    continue

                line_bbox = (
                    line_bbox_list[l_idx]
                    if l_idx < len(line_bbox_list)
                    else block_bbox
                )
                words = text_to_word_objects(line_stripped, line_bbox)
                lines.append(
                    NormalizedTextLine(
                        text=line_stripped,
                        bbox=line_bbox,
                        confidence=1.0,
                        order=l_idx,
                        words=words,
                    )
                )

            if lines:
                block_bbox = [
                    min(ln.bbox[0] for ln in lines),
                    min(ln.bbox[1] for ln in lines),
                    max(ln.bbox[2] for ln in lines),
                    max(ln.bbox[3] for ln in lines),
                ]
                blocks.append(
                    NormalizedTextBlock(
                        type="text",
                        bbox=block_bbox,
                        confidence=1.0,
                        order=block_order,
                        lines=lines,
                    )
                )
                block_order += 1

    return NormalizedPage(
        page_number=1,
        width=page_width,
        height=page_height,
        blocks=blocks,
        tables=[],
    )


def normalize_json_output(
    raw_text: str,
    page_dims: tuple[float, float],
) -> NormalizedPage:
    """Convert JSON VLM output — falls back to markdown if parsing fails."""
    page_width, page_height = page_dims
    blocks: list[NormalizedTextBlock] = []

    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, ValueError, TypeError):
        return normalize_markdown_output(raw_text, page_dims)

    items: list[dict[str, Any]] = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get(
            "blocks",
            data.get("pages", data.get("text_blocks", [data])),
        )
    else:
        return normalize_markdown_output(raw_text, page_dims)

    for b_idx, item in enumerate(items):
        if isinstance(item, str):
            text = item
            item_conf = 1.0
        elif isinstance(item, dict):
            text = str(item.get("text", item.get("content", "")))
            item_conf = float(item.get("confidence", 1.0) or 1.0)
        else:
            continue

        if not text or not text.strip():
            continue

        block_bbox = heuristic_block_bbox(
            b_idx, len(items), page_width, page_height, len(text)
        )
        lines = text_to_lines(text, block_bbox)
        blocks.append(
            NormalizedTextBlock(
                type="text",
                bbox=block_bbox,
                confidence=min(item_conf, 1.0),
                order=b_idx,
                lines=lines,
            )
        )

    return NormalizedPage(
        page_number=1,
        width=page_width,
        height=page_height,
        blocks=blocks,
        tables=[],
    )


def text_to_lines(
    text: str,
    block_bbox: list[float],
    _block_order: int = 0,
) -> list[NormalizedTextLine]:
    """Split text into ``TextLine`` objects with heuristic bboxes."""
    lines_raw = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines_raw:
        lines_raw = [text.strip()]

    line_bbox_list = heuristic_line_bboxes(len(lines_raw), block_bbox)
    result_lines: list[NormalizedTextLine] = []
    for l_idx, line_text in enumerate(lines_raw):
        line_bbox = (
            line_bbox_list[l_idx] if l_idx < len(line_bbox_list) else block_bbox
        )
        words = text_to_word_objects(line_text, line_bbox)
        result_lines.append(
            NormalizedTextLine(
                text=line_text,
                bbox=line_bbox,
                confidence=1.0,
                order=l_idx,
                words=words,
            )
        )
    return result_lines


def text_to_word_objects(
    text: str,
    bbox: list[float],
) -> list[NormalizedWord]:
    """Split a line of text into ``Word`` objects with synthesised chars.

    Since VLM output lacks character-level bounding boxes, characters
    are evenly distributed across each word's extent.
    """
    words_raw = text.split()
    if not words_raw:
        return []

    word_width = (bbox[2] - bbox[0]) / max(len(words_raw), 1)
    result_words: list[NormalizedWord] = []
    for w_idx, word_text in enumerate(words_raw):
        w_x0 = bbox[0] + w_idx * word_width
        w_x1 = w_x0 + word_width
        word_bbox = [w_x0, bbox[1], w_x1, bbox[3]]

        char_width = (word_bbox[2] - word_bbox[0]) / max(len(word_text), 1)
        chars: list[NormalizedCharacter] = []
        for c_idx, ch in enumerate(word_text):
            chars.append(
                NormalizedCharacter(
                    char=ch,
                    bbox=[
                        word_bbox[0] + c_idx * char_width,
                        word_bbox[1],
                        word_bbox[0] + (c_idx + 1) * char_width,
                        word_bbox[3],
                    ],
                    confidence=1.0,
                    order=c_idx,
                )
            )

        result_words.append(
            NormalizedWord(
                text=word_text,
                bbox=word_bbox,
                confidence=1.0,
                order=w_idx,
                chars=chars,
            )
        )
    return result_words
