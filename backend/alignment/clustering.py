"""Spatial clustering for block-level OCR alignment.  # noqa: SIZE_OK

Groups individual word bounding boxes into logical text blocks (paragraphs,
columns, headings) based on their spatial proximity on the page. The output
feeds into the Needleman-Wunsch word-level alignment (Task 12).

Coordinate system
-----------------
All bounding boxes use **page-space coordinates**:
    - Unit: points (1/72 inch) at **72 DPI**
    - Origin: **top-left** corner of the page
    - Format: ``[x0, y0, x1, y1]``
"""

from __future__ import annotations

import math
import statistics
from typing import TypedDict

# ── Types ─────────────────────────────────────────────────────────────────────


class WordDict(TypedDict):
    """A single word matching the NormalizedSchema.Word subset used here."""

    text: str
    bbox: list[float]  # [x0, y0, x1, y1]
    confidence: float
    order: int


class BlockDict(TypedDict):
    """A grouped block of spatially contiguous words with layout metadata."""

    bbox: list[float]
    words: list[WordDict]
    text: str
    confidence: float
    order: int
    column: int


# ── Internal helpers ──────────────────────────────────────────────────────────


def _y_overlap_iou(bbox_a: list[float], bbox_b: list[float]) -> float:
    """Compute the IoU of two bounding boxes projected onto the y-axis.

    This measures how much two words vertically overlap regardless of
    their horizontal positions.  Used to decide whether two words are on
    the same text line.
    """
    y0_a, y1_a = bbox_a[1], bbox_a[3]
    y0_b, y1_b = bbox_b[1], bbox_b[3]
    overlap = max(0.0, min(y1_a, y1_b) - max(y0_a, y0_b))
    height_a = y1_a - y0_a
    height_b = y1_b - y0_b
    union = height_a + height_b - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def _words_to_lines(
    words: list[WordDict],
    iou_threshold: float = 0.3,
) -> list[list[WordDict]]:
    """Group word dicts into text lines by y-axis overlap IoU.

    Words whose y-projection IoU exceeds *iou_threshold* are considered
    to be on the same text line.  Words are sorted top-to-bottom then
    left-to-right before grouping.
    """
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: (w["bbox"][1], w["bbox"][0]))

    lines: list[list[WordDict]] = []
    for word in sorted_words:
        placed = False
        for line in lines:
            if any(_y_overlap_iou(word["bbox"], lw["bbox"]) > iou_threshold for lw in line):
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])

    # Sort words left-to-right within each line (reading direction).
    for line in lines:
        line.sort(key=lambda w: w["bbox"][0])

    return lines


def _compute_vertical_gaps(lines: list[list[WordDict]]) -> list[float]:
    """Compute the positive vertical gaps between consecutive lines."""
    if len(lines) < 2:
        return []
    gaps: list[float] = []
    for i in range(len(lines) - 1):
        bottom = max(w["bbox"][3] for w in lines[i])
        top = min(w["bbox"][1] for w in lines[i + 1])
        gap = top - bottom
        if gap > 0:
            gaps.append(gap)
    return gaps


def _make_block(lines: list[list[WordDict]]) -> BlockDict:
    """Merge a group of text lines into a single block dict.

    Computes the union bounding box, average confidence, and concatenated
    text of all words in the lines.
    """
    all_words = [w for line in lines for w in line]

    x0 = min(w["bbox"][0] for w in all_words)
    y0 = min(w["bbox"][1] for w in all_words)
    x1 = max(w["bbox"][2] for w in all_words)
    y1 = max(w["bbox"][3] for w in all_words)
    avg_conf = sum(w["confidence"] for w in all_words) / len(all_words)
    text = " ".join(w["text"] for w in all_words)

    return BlockDict(
        bbox=[x0, y0, x1, y1],
        words=all_words,
        text=text,
        confidence=avg_conf,
        order=0,
        column=0,
    )


def _group_lines_to_blocks(
    lines: list[list[WordDict]],
    gap_threshold: float,
) -> list[BlockDict]:
    """Group text lines into blocks separated by gaps above *gap_threshold*.

    Lines whose vertical gap is at or below the threshold are merged into
    the same block.  A gap larger than the threshold starts a new block.
    """
    if not lines:
        return []

    blocks: list[BlockDict] = []
    current: list[list[WordDict]] = [lines[0]]

    for i in range(1, len(lines)):
        prev_bottom = max(w["bbox"][3] for w in lines[i - 1])
        curr_top = min(w["bbox"][1] for w in lines[i])
        gap = curr_top - prev_bottom

        if gap <= gap_threshold:
            current.append(lines[i])
        else:
            blocks.append(_make_block(current))
            current = [lines[i]]

    if current:
        blocks.append(_make_block(current))

    return blocks


def _whitespace_gaps(
    words: list[WordDict],
    page_width: float,
    page_height: float | None = None,
) -> list[float]:
    """Find x-positions of whitespace gaps that are column separators.

    Builds a 1-D occupancy grid across the page width and identifies
    contiguous runs of empty bins wider than ``page_width × 0.05``.
    Gaps that fall outside the content area (page-edge margins) are
    always excluded.

    When *page_height* is provided, an extra filter is applied: the
    gap must have words on both sides whose vertical spans each exceed
    50 % of *page_height*.  This prevents word-level spacing within a
    single line (e.g. a large gap between left-aligned labels and
    tab-stop values) from being treated as a column separator.
    """
    if not words:
        return []

    num_bins = max(1, int(page_width))
    occupied = [False] * num_bins

    for w in words:
        x0 = max(0, int(w["bbox"][0]))
        x1 = min(num_bins - 1, int(w["bbox"][2]))
        for i in range(x0, x1 + 1):
            occupied[i] = True

    first_occupied = next((i for i in range(num_bins) if occupied[i]), 0)
    last_occupied = next((i for i in range(num_bins - 1, -1, -1) if occupied[i]), num_bins - 1)

    min_gap = max(1, int(page_width * 0.05))
    raw_gaps: list[tuple[float, float]] = []
    in_gap = False
    start = 0

    for i in range(num_bins):
        if not occupied[i]:
            if not in_gap:
                start = i
                in_gap = True
        elif in_gap:
            width = i - start
            if width >= min_gap:
                raw_gaps.append(((start + i - 1) / 2.0, float(width)))
            in_gap = False

    if in_gap:
        width = num_bins - start
        if width >= min_gap:
            raw_gaps.append(((start + num_bins - 1) / 2.0, float(width)))

    if page_height is None:
        return [g[0] for g in raw_gaps if first_occupied < g[0] < last_occupied]

    # When page_height is available, also verify the vertical-span criterion.
    min_vertical_span = page_height * 0.5

    result: list[float] = []
    for gap_center, _ in raw_gaps:
        if not (first_occupied < gap_center < last_occupied):
            continue

        left_words = [w for w in words if w["bbox"][2] < gap_center]
        right_words = [w for w in words if w["bbox"][0] > gap_center]

        if not left_words or not right_words:
            continue

        left_span = max(w["bbox"][3] for w in left_words) - min(w["bbox"][1] for w in left_words)
        right_span = max(w["bbox"][3] for w in right_words) - min(w["bbox"][1] for w in right_words)

        if left_span >= min_vertical_span and right_span >= min_vertical_span:
            result.append(gap_center)

    return result


def _prim_order(centers: list[tuple[float, float]]) -> list[int]:
    """Return block indices in MST-based reading order via Prim's algorithm.

    Prim's algorithm starting from the top-left block center produces a
    minimum spanning tree that naturally follows reading order: within a
    column, vertical neighbours are closest and get processed first; the
    jump between columns happens only when one column is exhausted.

    The tiebreaker sequence ``(distance, y, x, index)`` ensures
    deterministic top-to-bottom ordering when distances are equal.
    """
    n = len(centers)
    if n <= 1:
        return list(range(n))

    start = min(range(n), key=lambda i: (centers[i][1], centers[i][0]))

    visited = [False] * n
    min_dist = [float("inf")] * n
    min_dist[start] = 0
    order: list[int] = []

    for _ in range(n):
        u = min(
            (i for i in range(n) if not visited[i]),
            key=lambda i: (min_dist[i], centers[i][1], centers[i][0], i),
        )
        visited[u] = True
        order.append(u)

        for v in range(n):
            if not visited[v]:
                dx = centers[u][0] - centers[v][0]
                dy = centers[u][1] - centers[v][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < min_dist[v]:
                    min_dist[v] = dist

    return order


def _split_blocks_at_gaps(
    blocks: list[BlockDict],
    page_width: float,
    page_height: float | None = None,
) -> list[BlockDict]:
    """Split any block that spans a detected column-separator gap.

    After line-to-block grouping, a single block may contain words from two
    different columns if they happen to share a y-range.  This function
    detects those crossing blocks and splits them at the column boundary.
    """
    if not blocks:
        return []

    all_words = [w for b in blocks for w in b["words"]]
    gaps = _whitespace_gaps(all_words, page_width, page_height)
    if not gaps:
        return blocks

    result: list[BlockDict] = []
    for block in blocks:
        bx0 = block["bbox"][0]
        bx1 = block["bbox"][2]

        relevant_gaps = [g for g in gaps if bx0 < g < bx1]
        if not relevant_gaps:
            result.append(block)
            continue

        gap = relevant_gaps[0]
        left_words = [w for w in block["words"] if w["bbox"][2] < gap]
        right_words = [w for w in block["words"] if w["bbox"][0] > gap]

        if left_words:
            left_lines = _words_to_lines(left_words)
            if left_lines:
                result.append(_make_block(left_lines))
        if right_words:
            right_lines = _words_to_lines(right_words)
            if right_lines:
                result.append(_make_block(right_lines))

    return result


# ── Public API ────────────────────────────────────────────────────────────────


def cluster_words_to_blocks(
    words: list[WordDict],
    page_width: float,
    page_height: float,
    config: dict | None = None,
) -> list[BlockDict]:
    """Group word dicts into spatially coherent text blocks.

    The grouping uses a three-step process:
    1. Group words into lines by y-axis overlap IoU (>0.3 by default).
    2. Group lines into blocks by an adaptive vertical gap threshold
       (median gap × 1.5 by default).
    3. Detect columns and assign column indices to each block.

    Args:
        words: List of word dicts conforming to ``NormalizedSchema.Word``.
        page_width: Page width in points at 72 DPI.
        page_height: Page height in points at 72 DPI.
        config: Optional config override dict:
            - ``y_overlap_iou_threshold`` (float, default ``0.3``)
            - ``gap_multiplier`` (float, default ``1.5``)

    Returns:
        List of block dicts, each containing the grouped words plus a
        union bounding box and metadata.
    """
    if not words:
        return []

    cfg = config or {}
    iou_threshold = cfg.get("y_overlap_iou_threshold", 0.3)
    gap_multiplier = cfg.get("gap_multiplier", 1.5)

    # Step 1: Group words into lines by y-overlap.
    lines = _words_to_lines(words, iou_threshold=iou_threshold)
    if not lines:
        return []

    # Step 2: Compute adaptive vertical gap threshold.
    gaps = _compute_vertical_gaps(lines)
    gap_threshold = statistics.median(gaps) * gap_multiplier if gaps else page_height * 0.01

    # Step 3: Group lines into blocks.
    blocks = _group_lines_to_blocks(lines, gap_threshold)

    # Step 4: Split blocks that span column boundaries.
    blocks = _split_blocks_at_gaps(blocks, page_width, page_height)

    # Step 5: Detect columns and assign column indices.
    column_indices = detect_columns(blocks, page_width)
    for block, col_idx in zip(blocks, column_indices, strict=True):
        block["column"] = col_idx

    return blocks


def detect_columns(
    blocks: list[BlockDict],
    page_width: float,
) -> list[int]:
    """Detect multi-column layout and assign column indices to blocks.

    Uses a whitespace-occupancy histogram across the page width.
    Vertical strips of the page that contain no word content are
    identified as column separators.  Each block is assigned a
    column index based on its x-center relative to the gaps.

    Args:
        blocks: List of block dicts.
        page_width: Page width in points at 72 DPI.

    Returns:
        List of column indices (0-based), one per block.
    """
    if not blocks:
        return []

    all_words = [w for block in blocks for w in block["words"]]
    if not all_words:
        return [0] * len(blocks)

    gaps = _whitespace_gaps(all_words, page_width)

    result: list[int] = []
    for block in blocks:
        cx = (block["bbox"][0] + block["bbox"][2]) / 2.0
        col = sum(1 for g in gaps if cx > g)
        result.append(col)

    return result


def estimate_reading_order(
    blocks: list[BlockDict],
    page_width: float,
    page_height: float,  # noqa: ARG001 — kept for API consistency
) -> list[BlockDict]:
    """Sort blocks in reading order.

    For single-column layouts the sort is a simple top-to-bottom /
    left-to-right sort.  For multi-column layouts, blocks are grouped
    by column (detected via whitespace gaps), sorted left-to-right by
    column, then top-to-bottom within each column.

    Args:
        blocks: List of block dicts (typically from
            :func:`cluster_words_to_blocks`).
        page_width: Page width in points at 72 DPI.
        page_height: Page height in points (unused, for API consistency).

    Returns:
        Blocks sorted in reading order with their ``order`` field updated.
    """
    if not blocks:
        return []

    column_indices = detect_columns(blocks, page_width)
    num_columns = max(column_indices) + 1 if column_indices else 1

    if num_columns <= 1:
        # Single column: simple top-to-bottom, left-to-right.
        ordered = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    else:
        # Multi-column: group by column, sort columns left-to-right,
        # sort blocks top-to-bottom within each column.
        col_groups: dict[int, list[BlockDict]] = {}
        for block, col in zip(blocks, column_indices, strict=True):
            col_groups.setdefault(col, []).append(block)

        ordered: list[BlockDict] = []
        for col in sorted(col_groups):
            col_groups[col].sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
            ordered.extend(col_groups[col])

    for i, block in enumerate(ordered):
        block["order"] = i

    return ordered
