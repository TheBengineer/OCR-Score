"""Tests for the GriTS table scoring module — GriTS_Top, GriTS_Con, GriTS_Loc,
cell matrix building, text similarity, IoU, and full table-structure metrics."""

from __future__ import annotations

import pytest

from backend.evaluation.table_scoring import (
    _build_cell_matrix,
    _cell_text_similarity,
    _compute_aligned_cells,
    _compute_structure_precision_recall,
    _iou,
    _row_similarity,
    _transpose,
    compute_table_structure_metrics,
    grits_con,
    grits_loc,
    grits_top,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Test fixtures — 2×2 tables with and without merged cells
# ═══════════════════════════════════════════════════════════════════════════════

A = {"row": 0, "col": 0, "row_span": 1, "col_span": 1, "text": "A", "bbox": [0, 0, 50, 20], "confidence": 1.0}
B = {"row": 0, "col": 1, "row_span": 1, "col_span": 1, "text": "B", "bbox": [50, 0, 100, 20], "confidence": 1.0}
C = {"row": 1, "col": 0, "row_span": 1, "col_span": 1, "text": "C", "bbox": [0, 20, 50, 40], "confidence": 1.0}
D = {"row": 1, "col": 1, "row_span": 1, "col_span": 1, "text": "D", "bbox": [50, 20, 100, 40], "confidence": 1.0}

CELLS_2X2 = [A, B, C, D]

# A single merged cell covering the whole 2×2 area
MERGE_ALL = {
    "row": 0, "col": 0, "row_span": 2, "col_span": 2,
    "text": "ABCD", "bbox": [0, 0, 100, 40], "confidence": 1.0,
}
CELLS_MERGED = [MERGE_ALL]

# A 2×2 table with first row merged across both columns
MERGE_ROW0 = {
    "row": 0, "col": 0, "row_span": 1, "col_span": 2,
    "text": "AB", "bbox": [0, 0, 100, 20], "confidence": 1.0,
}
C2 = {
    "row": 1, "col": 0, "row_span": 1, "col_span": 1,
    "text": "C", "bbox": [0, 20, 50, 40], "confidence": 1.0,
}
D2 = {
    "row": 1, "col": 1, "row_span": 1, "col_span": 1,
    "text": "D", "bbox": [50, 20, 100, 40], "confidence": 1.0,
}
CELLS_2X2_MERGED_ROW = [MERGE_ROW0, C2, D2]

# ── Alternative text (partial content mismatch) ───────────────────────────────

X = {"row": 0, "col": 0, "row_span": 1, "col_span": 1, "text": "X", "bbox": [0, 0, 50, 20], "confidence": 1.0}
Y = {"row": 1, "col": 1, "row_span": 1, "col_span": 1, "text": "Y", "bbox": [50, 20, 100, 40], "confidence": 1.0}
CELLS_2X2_PARTIAL_TEXT = [X, B, C, Y]

# ── Alternative bboxes (partial location mismatch) ────────────────────────────

A_SHIFTED = {
    "row": 0, "col": 0, "row_span": 1, "col_span": 1,
    "text": "A", "bbox": [20, 0, 50, 20], "confidence": 1.0,
}
D_SHIFTED = {
    "row": 1, "col": 1, "row_span": 1, "col_span": 1,
    "text": "D", "bbox": [100, 100, 110, 110], "confidence": 1.0,
}
CELLS_2X2_PARTIAL_BBOX = [A_SHIFTED, B, C, D_SHIFTED]


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestCellMatrixBuilding:
    """``_build_cell_matrix`` — flat cell list → 2D grid."""

    def test_basic_2x2(self) -> None:
        """2×2 table with no spans produces a 2×2 matrix."""
        mat = _build_cell_matrix(CELLS_2X2)
        assert len(mat) == 2
        assert len(mat[0]) == 2
        assert mat[0][0] is A
        assert mat[0][1] is B
        assert mat[1][0] is C
        assert mat[1][1] is D

    def test_merged_cell(self) -> None:
        """A cell spanning 2×2 produces a 2×2 matrix with None for covered positions."""
        mat = _build_cell_matrix(CELLS_MERGED)
        assert len(mat) == 2
        assert len(mat[0]) == 2
        assert mat[0][0] is MERGE_ALL
        assert mat[0][1] is None
        assert mat[1][0] is None
        assert mat[1][1] is None

    def test_merged_row(self) -> None:
        """First row merged across both columns."""
        mat = _build_cell_matrix(CELLS_2X2_MERGED_ROW)
        assert len(mat) == 2
        assert len(mat[0]) == 2
        assert mat[0][0] is MERGE_ROW0
        assert mat[0][1] is None
        assert mat[1][0] is C2
        assert mat[1][1] is D2

    def test_empty_cells(self) -> None:
        """Empty cell list → empty matrix."""
        assert _build_cell_matrix([]) == []

    def test_inferred_dims(self) -> None:
        """Matrix dimensions inferred from max row/col + span."""
        cells = [
            {"row": 2, "col": 3, "row_span": 1, "col_span": 1, "text": "Z", "bbox": [0, 0, 10, 10], "confidence": 1.0},
        ]
        mat = _build_cell_matrix(cells)
        assert len(mat) == 3  # max row = 2, so rows = 0,1,2 → 3
        assert len(mat[0]) == 4  # max col = 3, so cols = 0,1,2,3 → 4
        assert mat[2][3] == cells[0]


class TestTranspose:
    """``_transpose`` — 2D list transposition."""

    def test_square(self) -> None:
        """Square matrix transposed correctly."""
        m = [[1, 2], [3, 4]]
        t = _transpose(m)
        assert t == [[1, 3], [2, 4]]

    def test_rectangular(self) -> None:
        """Rectangular (2×3 → 3×2) matrix."""
        m = [[1, 2, 3], [4, 5, 6]]
        t = _transpose(m)
        assert t == [[1, 4], [2, 5], [3, 6]]

    def test_empty(self) -> None:
        """Empty inputs → empty output."""
        assert _transpose([]) == []
        assert _transpose([[]]) == []


class TestRowSimilarity:
    """``_row_similarity`` — 1D-LCS between row vectors."""

    def test_identical_rows(self) -> None:
        """Two identical all-cell rows → 1.0."""
        row_a = [A, B]
        row_b = [A, B]
        assert _row_similarity(row_a, row_b) == 1.0

    def test_completely_different(self) -> None:
        """Rows with different lengths — cell vs None mismatch."""
        row_a = [A, B]  # two cells
        row_b = [MERGE_ALL, None]  # one cell + None
        sim = _row_similarity(row_a, row_b)
        # LCS = 1 (position 0: both cells, position 1: cell vs None → no match)
        # sim = 2*1/(2+2) = 0.5
        assert sim == pytest.approx(0.5)

    def test_both_empty(self) -> None:
        """Both rows empty → 1.0."""
        assert _row_similarity([], []) == 1.0

    def test_one_empty(self) -> None:
        """One empty, one non-empty → 0.0."""
        assert _row_similarity([A], []) == 0.0
        assert _row_similarity([], [A]) == 0.0


class TestCellTextSimilarity:
    """``_cell_text_similarity`` — Levenshtein-based similarity."""

    def test_identical(self) -> None:
        """Identical strings → 1.0."""
        assert _cell_text_similarity("hello", "hello") == 1.0

    def test_completely_different(self) -> None:
        """Totally different strings → 0.0 when no overlap."""
        # "abc" vs "xyz": distance = 3, max_len = 3, sim = 0.0
        sim = _cell_text_similarity("abc", "xyz")
        assert sim == pytest.approx(0.0)

    def test_partial_match(self) -> None:
        """Partial edit distance."""
        # "kitten" vs "sitting": 3 ops (k→s, e→i, +g), max_len=7
        sim = _cell_text_similarity("kitten", "sitting")
        assert sim == pytest.approx(1.0 - 3.0 / 7.0)

    def test_both_empty(self) -> None:
        """Both empty → 1.0."""
        assert _cell_text_similarity("", "") == 1.0

    def test_one_empty(self) -> None:
        """One empty → 0.0."""
        assert _cell_text_similarity("a", "") == 0.0
        assert _cell_text_similarity("", "a") == 0.0

    def test_case_sensitive(self) -> None:
        """Levenshtein is case-sensitive."""
        sim = _cell_text_similarity("Hello", "hello")
        assert sim < 1.0


class TestIoU:
    """``_iou`` — Intersection over Union."""

    def test_identical(self) -> None:
        """Same bbox → 1.0."""
        assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0

    def test_no_overlap(self) -> None:
        """Non-overlapping bboxes → 0.0."""
        assert _iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_partial_overlap(self) -> None:
        """Partial overlap → correct IoU."""
        # a: [0,0,10,10] area=100; b: [5,0,15,10] area=100
        # intersection: x_left=5, y_top=0, x_right=10, y_bottom=10 → 5*10=50
        # union = 100+100-50 = 150; IoU = 50/150 = 1/3
        assert _iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1.0 / 3.0)

    def test_contained(self) -> None:
        """One bbox fully contains the other → area ratio."""
        # a: [0,0,10,10] area=100; b: [2,2,8,8] area=36
        # intersection = 36; union = 100; IoU = 0.36
        assert _iou([0, 0, 10, 10], [2, 2, 8, 8]) == pytest.approx(0.36)

    def test_zero_area(self) -> None:
        """Zero-area bbox (degenerate) → 0.0."""
        assert _iou([0, 0, 0, 10], [0, 0, 10, 10]) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# GriTS_Top
# ═══════════════════════════════════════════════════════════════════════════════


class TestGritsTop:
    """GriTS_Top — structural topology similarity."""

    def test_exact_match(self) -> None:
        """Identical 2×2 structure → 1.0."""
        assert grits_top(CELLS_2X2, CELLS_2X2) == 1.0

    def test_exact_match_merged(self) -> None:
        """Identical merged structure → 1.0."""
        assert grits_top(CELLS_MERGED, CELLS_MERGED) == 1.0

    def test_completely_wrong(self) -> None:
        """2×2 vs single merged cell → no structural overlap → 0.0."""
        score = grits_top(CELLS_2X2, CELLS_MERGED)
        # RowSim: all row-pair similarities ≤ 0.5 → no row matches → 0.0
        # ColSim: all col-pair similarities ≤ 0.5 → no col matches → 0.0
        assert score == pytest.approx(0.0)

    def test_partial(self) -> None:
        """2×2 vs merged-first-row → partial overlap → ~0.5."""
        score = grits_top(CELLS_2X2, CELLS_2X2_MERGED_ROW)
        # RowSim = 0.5 (1 of 2 rows matched), ColSim = 0.5 (1 of 2 cols matched)
        assert score == pytest.approx(0.5)

    def test_merged_cells(self) -> None:
        """Both tables have merged cells in the same way → 1.0."""
        assert grits_top(CELLS_2X2_MERGED_ROW, CELLS_2X2_MERGED_ROW) == 1.0

    def test_different_structure_same_content(self) -> None:
        """Different structural split but text content happens to match → penalized."""
        # pred: 2×2 all separate cells
        # gt: same 2×2 but first row merged
        # Even though text is "A","B","C","D" vs "AB","C","D", the structure differs
        score = grits_top(CELLS_2X2, CELLS_2X2_MERGED_ROW)
        assert 0.0 < score < 1.0

    def test_both_empty(self) -> None:
        """No cells on either side → 1.0."""
        assert grits_top([], []) == 1.0

    def test_pred_empty(self) -> None:
        """No pred cells but gt has cells → 0.0."""
        assert grits_top([], CELLS_2X2) == 0.0

    def test_gt_empty(self) -> None:
        """No gt cells but pred has cells → 0.0."""
        assert grits_top(CELLS_2X2, []) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# GriTS_Con
# ═══════════════════════════════════════════════════════════════════════════════


class TestGritsCon:
    """GriTS_Con — cell text content similarity."""

    def test_exact_match(self) -> None:
        """Identical cells → 1.0."""
        assert grits_con(CELLS_2X2, CELLS_2X2) == 1.0

    def test_partial_match(self) -> None:
        """2 correct cells, 2 wrong → 0.5."""
        score = grits_con(CELLS_2X2, CELLS_2X2_PARTIAL_TEXT)
        # (A→X: 0.0, B→B: 1.0, C→C: 1.0, D→Y: 0.0) / 4 = 0.5
        assert score == pytest.approx(0.5)

    def test_empty_cells(self) -> None:
        """Empty-text cells handled correctly."""
        empty = {
            "row": 0, "col": 0, "row_span": 1, "col_span": 1,
            "text": "", "bbox": [0, 0, 10, 10], "confidence": 1.0,
        }
        non_empty = {
            "row": 0, "col": 0, "row_span": 1, "col_span": 1,
            "text": "A", "bbox": [0, 0, 10, 10], "confidence": 1.0,
        }
        score = grits_con([empty], [non_empty])
        assert score == pytest.approx(0.0)

    def test_both_empty_cells(self) -> None:
        """Both cells have empty text → 1.0."""
        empty_a = {
            "row": 0, "col": 0, "row_span": 1, "col_span": 1,
            "text": "", "bbox": [0, 0, 10, 10], "confidence": 1.0,
        }
        empty_b = {
            "row": 0, "col": 0, "row_span": 1, "col_span": 1,
            "text": "", "bbox": [0, 0, 10, 10], "confidence": 1.0,
        }
        assert grits_con([empty_a], [empty_b]) == 1.0

    def test_all_mismatched(self) -> None:
        """All cells have different text → 0.0."""
        # Structure matches (2×2), but all text differs
        def _cell(row, col):
            return {
                "row": row, "col": col, "row_span": 1, "col_span": 1,
                "text": "X", "bbox": [col * 10, row * 10, col * 10 + 10, row * 10 + 10],
                "confidence": 1.0,
            }
        all_x = [_cell(0, 0), _cell(0, 1), _cell(1, 0), _cell(1, 1)]
        score = grits_con(CELLS_2X2, all_x)
        assert score == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# GriTS_Loc
# ═══════════════════════════════════════════════════════════════════════════════


class TestGritsLoc:
    """GriTS_Loc — cell bounding box similarity."""

    def test_exact_match(self) -> None:
        """Same bboxes → 1.0."""
        assert grits_loc(CELLS_2X2, CELLS_2X2) == 1.0

    def test_partial_overlap(self) -> None:
        """Two cells with different bboxes, two the same → ~0.5."""
        # cell (0,0): bbox [0,0,50,20] vs [20,0,50,20] → IoU = 0.6
        # cell (0,1): [50,0,100,20] vs [50,0,100,20] → IoU = 1.0
        # cell (1,0): [0,20,50,40] vs [0,20,50,40] → IoU = 1.0
        # cell (1,1): [50,20,100,40] vs [100,100,110,110] → IoU = 0.0
        # Mean = (0.6 + 1.0 + 1.0 + 0.0) / 4 = 0.65
        score = grits_loc(CELLS_2X2, CELLS_2X2_PARTIAL_BBOX)
        # Compute expected: pred (0,0)=[0,0,50,20], gt=[20,0,50,20]
        # intersection: [20,0,50,20] → 30*20=600
        # area_a=1000, area_b=600, union=1000, IoU=600/1000=0.6
        # (0,1)=1.0, (1,0)=1.0, (1,1)=0.0
        expected = (0.6 + 1.0 + 1.0 + 0.0) / 4.0
        assert score == pytest.approx(expected)

    def test_no_overlap(self) -> None:
        """All bboxes completely different → 0.0."""
        def _cell(row, col, offset):
            return {
                "row": row, "col": col, "row_span": 1, "col_span": 1,
                "text": chr(65 + row * 2 + col),
                "bbox": [offset, offset, offset + 10, offset + 10],
                "confidence": 1.0,
            }
        far_away = [_cell(0, 0, 1000), _cell(0, 1, 2000), _cell(1, 0, 3000), _cell(1, 1, 4000)]
        score = grits_loc(CELLS_2X2, far_away)
        assert score == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Alignment and structure precision/recall
# ═══════════════════════════════════════════════════════════════════════════════


class TestAlignedCells:
    """``_compute_aligned_cells`` — separable 2D-LCS alignment."""

    def test_exact_match(self) -> None:
        """Identical tables → all rows and cols matched."""
        _, _, row_pairs, col_pairs = _compute_aligned_cells(CELLS_2X2, CELLS_2X2)
        assert row_pairs == [(0, 0), (1, 1)]
        assert col_pairs == [(0, 0), (1, 1)]

    def test_merged_vs_plain(self) -> None:
        """Merged vs 2×2 → no alignment."""
        _, _, row_pairs, col_pairs = _compute_aligned_cells(CELLS_2X2, CELLS_MERGED)
        assert len(row_pairs) == 0
        assert len(col_pairs) == 0


class TestStructurePrecisionRecall:
    """``_compute_structure_precision_recall``."""

    def test_exact_match(self) -> None:
        """Identical tables → precision and recall = 1.0."""
        p, r = _compute_structure_precision_recall(CELLS_2X2, CELLS_2X2)
        assert (p, r) == (1.0, 1.0)

    def test_merged_vs_plain(self) -> None:
        """2×2 vs merged cell → 0 precision and recall."""
        p, r = _compute_structure_precision_recall(CELLS_2X2, CELLS_MERGED)
        assert p == 0.0
        assert r == 0.0

    def test_partial(self) -> None:
        """2×2 vs merged-first-row → partial precision/recall."""
        p, r = _compute_structure_precision_recall(CELLS_2X2, CELLS_2X2_MERGED_ROW)
        # row_pairs = [(1, 1)], col_pairs = [(0, 0)] or [(1, 1)] depending on LCS
        # Structure: row 0 differs (merged vs separate), row 1 matches
        # For precision: pred has 4 cells, 2 in the aligned region (row 1 only) → 2/4 = 0.5
        # For recall: gt has 3 cells, cells in aligned region...
        # The exact values depend on which columns are matched
        assert 0.0 < p < 1.0
        assert 0.0 < r < 1.0

    def test_empty(self) -> None:
        """Both empty → 1.0, 1.0."""
        p, r = _compute_structure_precision_recall([], [])
        assert (p, r) == (1.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# compute_table_structure_metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeTableStructureMetrics:
    """``compute_table_structure_metrics`` — full table evaluation."""

    def test_all_metrics(self) -> None:
        """All metrics computed for a matched table pair."""
        pred_tables = [
            {"bbox": [0, 0, 100, 50], "cells": CELLS_2X2},
        ]
        gt_tables = [
            {"bbox": [0, 0, 100, 50], "cells": CELLS_2X2},
        ]

        result = compute_table_structure_metrics(pred_tables, gt_tables)

        assert result["grits_top"] == 1.0
        assert result["grits_con"] == 1.0
        assert result["grits_loc"] == 1.0
        assert result["structure_precision"] == 1.0
        assert result["structure_recall"] == 1.0
        assert result["cell_accuracy"] == 1.0
        assert result["table_detection_precision"] == 1.0
        assert result["table_detection_recall"] == 1.0

    def test_no_tables(self) -> None:
        """No tables → all 1.0 (nothing to miss)."""
        result = compute_table_structure_metrics([], [])
        for key in ("grits_top", "grits_con", "grits_loc", "structure_precision",
                     "structure_recall", "cell_accuracy", "table_detection_precision",
                     "table_detection_recall"):
            assert result[key] == 1.0, f"{key} should be 1.0"

    def test_missing_table(self) -> None:
        """One table found, one gt table missed → recall penalized."""
        table1 = {"bbox": [0, 0, 100, 50], "cells": CELLS_2X2}
        table2 = {"bbox": [200, 0, 300, 50], "cells": CELLS_2X2}

        pred_tables = [table1]
        gt_tables = [table1, table2]  # table2 has no matching pred

        result = compute_table_structure_metrics(pred_tables, gt_tables)

        assert result["table_detection_precision"] == 1.0  # no false positives
        assert result["table_detection_recall"] == 0.5     # 1 of 2 found
        assert result["grits_top"] == 1.0  # matched table is identical
        assert result["grits_con"] == 1.0
        assert result["grits_loc"] == 1.0

    def test_extra_pred_table(self) -> None:
        """Extra predicted table not in GT → precision penalized."""
        table1 = {"bbox": [0, 0, 100, 50], "cells": CELLS_2X2}
        table2 = {"bbox": [200, 0, 300, 50], "cells": CELLS_2X2}

        pred_tables = [table1, table2]  # table2 is extra
        gt_tables = [table1]

        result = compute_table_structure_metrics(pred_tables, gt_tables)

        assert result["table_detection_precision"] == 0.5  # 1 of 2 correct
        assert result["table_detection_recall"] == 1.0     # found the only GT
        assert result["grits_top"] == 1.0

    def test_no_cells_in_tables(self) -> None:
        """Tables with no cells → scores default to 1.0."""
        pred_tables = [
            {"bbox": [0, 0, 100, 50], "cells": []},
        ]
        gt_tables = [
            {"bbox": [0, 0, 100, 50], "cells": []},
        ]

        result = compute_table_structure_metrics(pred_tables, gt_tables)

        assert result["grits_top"] == 1.0
        assert result["grits_con"] == 1.0
        assert result["grits_loc"] == 1.0

    def test_partial_table_match(self) -> None:
        """Partially matching tables → intermediate scores."""
        pred_tables = [
            {"bbox": [0, 0, 100, 50], "cells": CELLS_2X2},
            {"bbox": [200, 0, 300, 50], "cells": CELLS_2X2_MERGED_ROW},
        ]
        # table2's GT is same, table1's GT differs
        gt_tables = [
            {"bbox": [0, 0, 100, 50], "cells": CELLS_2X2_PARTIAL_TEXT},
            {"bbox": [200, 0, 300, 50], "cells": CELLS_2X2_MERGED_ROW},
        ]

        result = compute_table_structure_metrics(pred_tables, gt_tables)

        assert result["table_detection_precision"] == 1.0
        assert result["table_detection_recall"] == 1.0
        # Both table pairs match → GriTS averages across them
        # Pair 0: same structure, different text → griTS_top=1.0, grits_con=0.5, grits_loc=1.0
        # Pair 1: same structure and text → all 1.0
        # Average: grits_top=1.0, grits_con=0.75, grits_loc=1.0
        assert result["grits_top"] == pytest.approx(1.0)
        assert result["grits_con"] == pytest.approx(0.75)
        assert result["grits_loc"] == pytest.approx(1.0)

    def test_with_merged_cells_in_metrics(self) -> None:
        """Tables with merged cells produce correct metrics."""
        pred_tables = [
            {"bbox": [0, 0, 100, 50], "cells": CELLS_2X2_MERGED_ROW},
        ]
        gt_tables = [
            {"bbox": [0, 0, 100, 50], "cells": CELLS_2X2_MERGED_ROW},
        ]

        result = compute_table_structure_metrics(pred_tables, gt_tables)

        assert result["grits_top"] == 1.0
        assert result["grits_con"] == 1.0
        assert result["grits_loc"] == 1.0
        assert result["structure_precision"] == 1.0
        assert result["structure_recall"] == 1.0

    def test_greedy_iou_matching(self) -> None:
        """Tables matched by IoU, not by index."""
        # pred[0] matches gt[1] better than gt[0]
        pred_tables = [
            {"bbox": [0, 0, 100, 50], "cells": CELLS_2X2},
        ]
        gt_tables = [
            {"bbox": [200, 200, 300, 250], "cells": CELLS_2X2_PARTIAL_TEXT},
            {"bbox": [0, 0, 100, 50], "cells": CELLS_2X2},
        ]

        result = compute_table_structure_metrics(pred_tables, gt_tables)

        assert result["table_detection_recall"] == 0.5  # only one GT matched
        assert result["grits_top"] == 1.0  # matched to the structurally identical one
