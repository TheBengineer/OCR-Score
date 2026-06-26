"""Tests for the spatial clustering module.

Covers block grouping, column detection, reading-order estimation,
and edge cases (empty input, headers/footers, tab-stop alignment).
"""

from __future__ import annotations

from backend.alignment.clustering import (
    BlockDict,
    WordDict,
    _prim_order,
    _words_to_lines,
    cluster_words_to_blocks,
    detect_columns,
    estimate_reading_order,
)

# ── Shared test dimensions ────────────────────────────────────────────────────

PAGE_W: float = 612.0  # US Letter width in points
PAGE_H: float = 792.0  # US Letter height in points


# ── Internal helper tests ─────────────────────────────────────────────────────


class TestWordsToLines:
    """Tests for the y-overlap line-grouping helper."""

    def test_same_line(self) -> None:
        """Words with y-overlap -> single line."""
        words: list[WordDict] = [
            {"text": "Hello", "bbox": [0.0, 0.0, 50.0, 20.0], "confidence": 0.9, "order": 0},
            {"text": "world", "bbox": [55.0, 2.0, 110.0, 22.0], "confidence": 0.9, "order": 1},
        ]
        lines = _words_to_lines(words)
        assert len(lines) == 1
        assert len(lines[0]) == 2

    def test_no_y_overlap(self) -> None:
        """Words with no y-overlap -> separate lines."""
        words: list[WordDict] = [
            {"text": "Top", "bbox": [0.0, 0.0, 50.0, 20.0], "confidence": 0.9, "order": 0},
            {"text": "Bottom", "bbox": [0.0, 100.0, 60.0, 120.0], "confidence": 0.9, "order": 1},
        ]
        lines = _words_to_lines(words)
        assert len(lines) == 2


class TestPrimOrder:
    """MST-based reading order via Prim's algorithm."""

    def test_single_center(self) -> None:
        """Single point -> [0]."""
        assert _prim_order([(100.0, 100.0)]) == [0]

    def test_two_points(self) -> None:
        """Two points -> top-left first."""
        centers = [(200.0, 50.0), (50.0, 100.0)]
        order = _prim_order(centers)
        # (50, 100) has smaller (y, x) than (200, 50) because y=50 < y=100
        # Wait: min by (y, x): (200,50) has y=50, (50,100) has y=100
        # So start at index 0 (200, 50).
        # Next closest: (50, 100) at dist ~158.
        assert order == [0, 1]

    def test_column_order(self) -> None:
        """Two-column layout: left column before right column."""
        centers = [
            (100.0, 100.0),   # 0: left col, top
            (100.0, 200.0),   # 1: left col, bottom
            (400.0, 100.0),   # 2: right col, top
            (400.0, 200.0),   # 3: right col, bottom
        ]
        order = _prim_order(centers)
        # Start at (100, 100) [index 0]
        # Nearest: (100, 200) [idx 1] at dist=100
        # Nearest from {0,1}: (400, 100) [idx 2] at dist=300 or (400, 200) [idx 3] at dist~316
        # idx 2 is closer (dist 300 < 316) and has smaller y (100 < 200)
        # So order is [0, 1, 2, 3]
        assert order == [0, 1, 2, 3]


# ── Block grouping tests ──────────────────────────────────────────────────────


class TestBasicBlockGrouping:
    """Words on same line and close vertically -> one block."""

    def test_single_line_block(self) -> None:
        """Three words on the same line -> single block."""
        words: list[WordDict] = [
            {"text": "The", "bbox": [0.0, 0.0, 30.0, 20.0], "confidence": 0.95, "order": 0},
            {"text": "quick", "bbox": [35.0, 0.0, 80.0, 20.0], "confidence": 0.90, "order": 1},
            {"text": "fox", "bbox": [85.0, 0.0, 110.0, 20.0], "confidence": 0.85, "order": 2},
        ]
        blocks = cluster_words_to_blocks(words, PAGE_W, PAGE_H)
        assert len(blocks) == 1
        assert blocks[0]["text"] == "The quick fox"

    def test_multi_line_block(self) -> None:
        """Two closely spaced lines -> single block."""
        words: list[WordDict] = [
            {"text": "First", "bbox": [0.0, 0.0, 50.0, 20.0], "confidence": 0.9, "order": 0},
            {"text": "line", "bbox": [55.0, 0.0, 90.0, 20.0], "confidence": 0.9, "order": 1},
            {"text": "Second", "bbox": [0.0, 25.0, 60.0, 45.0], "confidence": 0.9, "order": 2},
            {"text": "line", "bbox": [65.0, 25.0, 100.0, 45.0], "confidence": 0.9, "order": 3},
        ]
        blocks = cluster_words_to_blocks(words, PAGE_W, PAGE_H)
        assert len(blocks) == 1


class TestSeparateBlocks:
    """Words with large vertical gap -> separate blocks."""

    def test_two_separate_paragraphs(self) -> None:
        """Two multi-line groups with a large gap -> two blocks."""
        words: list[WordDict] = [
            # Top paragraph: 3 lines, intra gap ~10
            {"text": "Top1", "bbox": [0.0, 0.0, 50.0, 20.0], "confidence": 0.9, "order": 0},
            {"text": "Top2", "bbox": [0.0, 30.0, 60.0, 50.0], "confidence": 0.9, "order": 1},
            {"text": "Top3", "bbox": [0.0, 60.0, 70.0, 80.0], "confidence": 0.9, "order": 2},
            # Bottom paragraph: 3 lines
            {"text": "Bot1", "bbox": [0.0, 300.0, 60.0, 320.0], "confidence": 0.9, "order": 3},
            {"text": "Bot2", "bbox": [0.0, 330.0, 60.0, 350.0], "confidence": 0.9, "order": 4},
            {"text": "Bot3", "bbox": [0.0, 360.0, 70.0, 380.0], "confidence": 0.9, "order": 5},
        ]
        blocks = cluster_words_to_blocks(words, PAGE_W, PAGE_H)
        assert len(blocks) == 2


# ── Column detection tests ────────────────────────────────────────────────────


class TestMultiColumnDetection:
    """Two columns of text -> columns correctly assigned."""

    def _make_column_words(self, x0: float, x1: float, prefix: str, start_order: int = 0) -> list[WordDict]:
        """Build a tall column of words spanning y=50 to y=550 (>50% page height)."""
        lines: list[WordDict] = []
        for i, y_pos in enumerate([50, 100, 200, 300, 400, 500]):
            lines.append({
                "text": f"{prefix}{i}",
                "bbox": [x0, float(y_pos), x1, float(y_pos + 20)],
                "confidence": 0.9,
                "order": start_order + i,
            })
        return lines

    def test_two_columns(self) -> None:
        """Blocks in left and right columns get column indices 0 and 1."""
        words = self._make_column_words(50, 150, "L") + self._make_column_words(400, 500, "R", 6)
        blocks = cluster_words_to_blocks(words, PAGE_W, PAGE_H)
        columns = [b["column"] for b in blocks]
        assert columns == [0, 1]

    def test_single_column(self) -> None:
        """All blocks in a single column layout -> all column 0."""
        words: list[WordDict] = [
            {"text": "A", "bbox": [50.0, 100.0, 100.0, 120.0], "confidence": 0.9, "order": 0},
            {"text": "B", "bbox": [50.0, 130.0, 100.0, 150.0], "confidence": 0.9, "order": 1},
        ]
        blocks = cluster_words_to_blocks(words, PAGE_W, PAGE_H)
        assert len(blocks) >= 1
        for b in blocks:
            assert b["column"] == 0

    def test_detect_columns_only(self) -> None:
        """detect_columns standalone returns correct indices with pre-built blocks."""
        left_words: list[WordDict] = [
            {"text": "L1", "bbox": [50.0, 50.0, 100.0, 70.0], "confidence": 0.9, "order": 0},
            {"text": "L2", "bbox": [50.0, 150.0, 100.0, 170.0], "confidence": 0.9, "order": 1},
            {"text": "L3", "bbox": [50.0, 250.0, 100.0, 270.0], "confidence": 0.9, "order": 2},
            {"text": "L4", "bbox": [50.0, 350.0, 100.0, 370.0], "confidence": 0.9, "order": 3},
            {"text": "L5", "bbox": [50.0, 450.0, 100.0, 470.0], "confidence": 0.9, "order": 4},
            {"text": "L6", "bbox": [50.0, 550.0, 100.0, 570.0], "confidence": 0.9, "order": 5},
        ]
        right_words: list[WordDict] = [
            {"text": "R1", "bbox": [400.0, 50.0, 450.0, 70.0], "confidence": 0.9, "order": 6},
            {"text": "R2", "bbox": [400.0, 150.0, 450.0, 170.0], "confidence": 0.9, "order": 7},
            {"text": "R3", "bbox": [400.0, 250.0, 450.0, 270.0], "confidence": 0.9, "order": 8},
            {"text": "R4", "bbox": [400.0, 350.0, 450.0, 370.0], "confidence": 0.9, "order": 9},
            {"text": "R5", "bbox": [400.0, 450.0, 450.0, 470.0], "confidence": 0.9, "order": 10},
            {"text": "R6", "bbox": [400.0, 550.0, 450.0, 570.0], "confidence": 0.9, "order": 11},
        ]
        blocks = cluster_words_to_blocks(left_words + right_words, PAGE_W, PAGE_H)
        cols = detect_columns(blocks, PAGE_W)
        assert cols == [0, 1], f"Expected [0, 1], got {cols}"


# ── Reading order tests ───────────────────────────────────────────────────────


class TestReadingOrderTwoColumns:
    """Two columns -> left column before right."""

    def test_left_column_first(self) -> None:
        """In a two-column layout, the left column comes first."""
        left_words: list[WordDict] = [
            {"text": "L1", "bbox": [50.0, 50.0, 150.0, 70.0], "confidence": 0.9, "order": 0},
            {"text": "L2", "bbox": [50.0, 150.0, 150.0, 170.0], "confidence": 0.9, "order": 1},
            {"text": "L3", "bbox": [50.0, 250.0, 150.0, 270.0], "confidence": 0.9, "order": 2},
            {"text": "L4", "bbox": [50.0, 350.0, 150.0, 370.0], "confidence": 0.9, "order": 3},
            {"text": "L5", "bbox": [50.0, 450.0, 150.0, 470.0], "confidence": 0.9, "order": 4},
            {"text": "L6", "bbox": [50.0, 550.0, 150.0, 570.0], "confidence": 0.9, "order": 5},
        ]
        right_words: list[WordDict] = [
            {"text": "R1", "bbox": [400.0, 50.0, 500.0, 70.0], "confidence": 0.9, "order": 6},
            {"text": "R2", "bbox": [400.0, 150.0, 500.0, 170.0], "confidence": 0.9, "order": 7},
            {"text": "R3", "bbox": [400.0, 250.0, 500.0, 270.0], "confidence": 0.9, "order": 8},
            {"text": "R4", "bbox": [400.0, 350.0, 500.0, 370.0], "confidence": 0.9, "order": 9},
            {"text": "R5", "bbox": [400.0, 450.0, 500.0, 470.0], "confidence": 0.9, "order": 10},
            {"text": "R6", "bbox": [400.0, 550.0, 500.0, 570.0], "confidence": 0.9, "order": 11},
        ]
        blocks = cluster_words_to_blocks(left_words + right_words, PAGE_W, PAGE_H)
        assert len(blocks) == 2, f"Expected 2 blocks, got {len(blocks)}"
        ordered = estimate_reading_order(blocks, PAGE_W, PAGE_H)
        assert "L" in ordered[0]["text"]
        assert "R" in ordered[1]["text"]
        assert ordered[0]["order"] < ordered[1]["order"]


class TestReadingOrderTopToBottom:
    """Single column -> top-to-bottom order."""

    def test_top_to_bottom_single_column(self) -> None:
        """Single column blocks sorted top-to-bottom."""
        # Build blocks directly with distinct y positions to test ordering.
        blocks: list[BlockDict] = [
            {
                "bbox": [0.0, 300.0, 80.0, 320.0],
                "words": [{"text": "Bottom", "bbox": [0.0, 300.0, 80.0, 320.0], "confidence": 0.9, "order": 2}],
                "text": "Bottom",
                "confidence": 0.9,
                "order": 0,
                "column": 0,
            },
            {
                "bbox": [0.0, 150.0, 80.0, 170.0],
                "words": [{"text": "Middle", "bbox": [0.0, 150.0, 80.0, 170.0], "confidence": 0.9, "order": 1}],
                "text": "Middle",
                "confidence": 0.9,
                "order": 0,
                "column": 0,
            },
            {
                "bbox": [0.0, 0.0, 50.0, 20.0],
                "words": [{"text": "Top", "bbox": [0.0, 0.0, 50.0, 20.0], "confidence": 0.9, "order": 0}],
                "text": "Top",
                "confidence": 0.9,
                "order": 0,
                "column": 0,
            },
        ]
        ordered = estimate_reading_order(blocks, PAGE_W, PAGE_H)
        texts = [b["text"] for b in ordered]
        assert texts == ["Top", "Middle", "Bottom"], f"Got {texts}"


# ── Tab-stop alignment test ────────────────────────────────────────────────────


class TestTabStopAlignment:
    """Words aligned to tab stops -> same block."""

    def test_tab_aligned_lines(self) -> None:
        """Words at consistent x-positions across adjacent lines -> one block."""
        words: list[WordDict] = [
            # Line 1: "Name:" tab-stop "John"
            {"text": "Name:", "bbox": [0.0, 0.0, 40.0, 20.0], "confidence": 0.9, "order": 0},
            {"text": "John", "bbox": [100.0, 0.0, 140.0, 20.0], "confidence": 0.9, "order": 1},
            # Line 2: "Age:" tab-stop "30"
            {"text": "Age:", "bbox": [0.0, 25.0, 35.0, 45.0], "confidence": 0.9, "order": 2},
            {"text": "30", "bbox": [100.0, 25.0, 120.0, 45.0], "confidence": 0.9, "order": 3},
        ]
        blocks = cluster_words_to_blocks(words, PAGE_W, PAGE_H)
        assert len(blocks) == 1
        assert "Name:" in blocks[0]["text"]


# ── Edge case tests ────────────────────────────────────────────────────────────


class TestEmptyInput:
    """Empty list -> empty result."""

    def test_empty_words(self) -> None:
        """Empty word list returns empty block list."""
        assert cluster_words_to_blocks([], PAGE_W, PAGE_H) == []

    def test_empty_detect_columns(self) -> None:
        """Empty blocks returns empty column list."""
        assert detect_columns([], PAGE_W) == []

    def test_empty_reading_order(self) -> None:
        """Empty blocks returns empty list."""
        assert estimate_reading_order([], PAGE_W, PAGE_H) == []


class TestHeaderFooterDetection:
    """Text at page margins -> separate block."""

    def test_header_body_footer_separate(self) -> None:
        """Header, body, and footer form three distinct blocks."""
        words: list[WordDict] = [
            # Header at top margin
            {"text": "Header", "bbox": [200.0, 10.0, 300.0, 30.0], "confidence": 0.9, "order": 0},
            # Body text: multiple lines
            {"text": "Body1", "bbox": [50.0, 100.0, 150.0, 120.0], "confidence": 0.9, "order": 1},
            {"text": "Body2", "bbox": [50.0, 130.0, 160.0, 150.0], "confidence": 0.9, "order": 2},
            {"text": "Body3", "bbox": [50.0, 160.0, 170.0, 180.0], "confidence": 0.9, "order": 3},
            # Footer at bottom margin
            {"text": "Footer", "bbox": [200.0, 700.0, 300.0, 720.0], "confidence": 0.9, "order": 4},
        ]
        blocks = cluster_words_to_blocks(words, PAGE_W, PAGE_H)
        # Header, body block, footer -> 3 separate blocks
        assert len(blocks) == 3
        texts = {b["text"] for b in blocks}
        assert "Header" in texts
        assert "Footer" in texts
