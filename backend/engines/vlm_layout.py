"""Heuristic layout helpers for VLM-based OCR engines.

Since VLM output typically lacks positional information (no character-level
or word-level bounding boxes), these helpers distribute blocks, lines, and
words heuristically across the page based on reading order.
"""

from __future__ import annotations

import re
from typing import Any


def pixel_to_point(pixel: float, dpi: int) -> float:
    """Convert a pixel coordinate to points at 72 DPI.

    Args:
        pixel: Value in pixels.
        dpi: The DPI the pixel measurement was taken at.

    Returns:
        Equivalent value in points (1/72 inch).
    """
    return pixel * 72.0 / dpi


def split_markdown_blocks(markdown_text: str) -> list[dict[str, Any]]:
    """Split markdown text into logical blocks.

    Recognises headings (``#`` … ``######``), blank-line paragraph breaks,
    code fences (`` ``` ``), and consecutive non-blank lines as text blocks.

    Args:
        markdown_text: Raw markdown text from a VLM.

    Returns:
        List of block dicts with ``type``, optionally ``level`` (for
        headings), and ``lines`` (list of text lines) or ``text``.
    """
    blocks: list[dict[str, Any]] = []
    lines = markdown_text.split("\n")
    current_lines: list[str] = []
    current_type = "text"
    in_code_block = False

    for line in lines:
        stripped = line.strip()

        # ── Code fence toggle ────────────────────────────────────────
        if stripped.startswith("```"):
            if current_lines:
                blocks.append({"type": current_type, "lines": current_lines})
                current_lines = []
            in_code_block = not in_code_block
            current_type = "code" if in_code_block else "text"
            continue

        if in_code_block:
            current_lines.append(line)
            continue

        # ── Markdown heading ─────────────────────────────────────────
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            if current_lines:
                blocks.append({"type": current_type, "lines": current_lines})
                current_lines = []
            blocks.append({
                "type": "heading",
                "level": len(heading_match.group(1)),
                "text": heading_match.group(2),
            })
            continue

        # ── Blank line = paragraph break ─────────────────────────────
        if not stripped:
            if current_lines:
                blocks.append({"type": current_type, "lines": current_lines})
                current_lines = []
            continue

        current_lines.append(line)

    if current_lines:
        blocks.append({"type": current_type, "lines": current_lines})

    return blocks


def heuristic_block_bbox(
    block_idx: int,
    num_blocks: int,
    page_width_pts: float,
    page_height_pts: float,
    _text_length: int | None = None,
) -> list[float]:
    """Compute a heuristic bounding box for a block.

    Since VLM output does not contain positional information, blocks are
    distributed vertically on the page with 5% horizontal margins.

    Args:
        block_idx: Zero-based index of this block.
        num_blocks: Total number of blocks on the page.
        page_width_pts: Page width in points.
        page_height_pts: Page height in points.
        _text_length: Unused; retained for future length-aware layout.

    Returns:
        ``[x0, y0, x1, y1]`` in page-space points, top-left origin.
    """
    margin = page_width_pts * 0.05
    block_count = max(num_blocks, 1)
    slice_height = page_height_pts / block_count
    y0 = slice_height * block_idx + slice_height * 0.1
    y1 = y0 + slice_height * 0.8
    return [margin, y0, page_width_pts - margin, y1]


def heuristic_line_bboxes(
    num_lines: int,
    block_bbox: list[float],
) -> list[list[float]]:
    """Compute heuristic bounding boxes for lines within a block.

    Lines are distributed evenly within the block's vertical extent.

    Args:
        num_lines: Number of lines in the block.
        block_bbox: ``[x0, y0, x1, y1]`` of the parent block.

    Returns:
        List of ``[x0, y0, x1, y1]`` per line, in reading order.
    """
    if num_lines == 0:
        return []

    line_height = (block_bbox[3] - block_bbox[1]) / max(num_lines, 1)
    return [
        [
            block_bbox[0],
            block_bbox[1] + i * line_height,
            block_bbox[2],
            block_bbox[1] + (i + 1) * line_height,
        ]
        for i in range(num_lines)
    ]
