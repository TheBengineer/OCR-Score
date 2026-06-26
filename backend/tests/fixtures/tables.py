"""Synthetic table test fixtures with known ground truth.  # noqa: SIZE_OK

8 table definitions × 5 pred-variant sets = 40 data fixtures.
This is a data module — every block is a realistic table with typed
coordinates and content.  Splitting it would reduce consumer readability.

Each fixture provides:
- ``gt_cells`` — ground-truth cells
- ``pred_cells_exact`` — identical cells for exact-match tests
- ``pred_cells_wrong`` — same structure, wrong text (content errors)
- ``pred_cells_wrong_structure`` — different topology (structure errors)
- ``pred_cells_partial`` — mix of correct/incorrect (partial-score tests)
- ``table_metadata`` — table dimensions, bbox, and caption

Coordinate system: US Letter (72 DPI), 612 × 792 pt, top-left origin.
All cell dicts follow the ``TableCell`` schema from
``backend/engine/normalized_schema.py``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# ── Helper utilities ───────────────────────────────────────────────────────────


def _cell_bbox(
    row: int,
    col: int,
    row_span: int,
    col_span: int,
    x0: float,
    y0: float,
    col_widths: list[float],
    row_heights: list[float],
) -> list[float]:
    """Compute cell bounding box from grid position."""
    cx = x0 + sum(col_widths[:col])
    cy = y0 + sum(row_heights[:row])
    cx1 = x0 + sum(col_widths[: col + col_span])
    cy1 = y0 + sum(row_heights[: row + row_span])
    return [cx, cy, cx1, cy1]


def _make_cell(
    row: int,
    col: int,
    text: str,
    x0: float,
    y0: float,
    col_widths: list[float],
    row_heights: list[float],
    row_span: int = 1,
    col_span: int = 1,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Build a single cell dict for a table fixture."""
    return {
        "row": row,
        "col": col,
        "row_span": row_span,
        "col_span": col_span,
        "text": text,
        "bbox": _cell_bbox(row, col, row_span, col_span, x0, y0, col_widths, row_heights),
        "confidence": confidence,
    }


def _make_metadata(
    x0: float,
    y0: float,
    col_widths: list[float],
    row_heights: list[float],
    caption: str,
) -> dict[str, Any]:
    """Build the table-metadata dict for a fixture."""
    return {
        "num_rows": len(row_heights),
        "num_cols": len(col_widths),
        "bbox": [x0, y0, x0 + sum(col_widths), y0 + sum(row_heights)],
        "caption": caption,
    }


def _cells_equal(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    """Return True when two cell lists are structurally and textually identical."""
    return a == b


# ── 1. SIMPLE_TABLE — 3 rows × 3 cols, header + 2 data rows ───────────────────


def _build_simple_table() -> dict[str, Any]:
    x0, y0 = 80.0, 280.0
    col_widths = [160.0, 140.0, 160.0]
    row_heights = [40.0, 35.0, 35.0]

    gt_cells = [
        _make_cell(0, 0, "Name", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Department", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Email", x0, y0, col_widths, row_heights),
        _make_cell(1, 0, "Alice Johnson", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "Engineering", x0, y0, col_widths, row_heights),
        _make_cell(1, 2, "alice@example.com", x0, y0, col_widths, row_heights),
        _make_cell(2, 0, "Bob Smith", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "Marketing", x0, y0, col_widths, row_heights),
        _make_cell(2, 2, "bob@example.com", x0, y0, col_widths, row_heights),
    ]
    metadata = _make_metadata(x0, y0, col_widths, row_heights, "Table 1: Employee Directory")

    # pred_cells_exact — structure + content identical to GT
    pred_cells_exact = deepcopy(gt_cells)

    # pred_cells_wrong — same structure, different text in some cells
    pred_cells_wrong = [
        _make_cell(0, 0, "Name", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Dept", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Email Address", x0, y0, col_widths, row_heights),
        _make_cell(1, 0, "Alice Johnsøn", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "Engineering", x0, y0, col_widths, row_heights),
        _make_cell(1, 2, "alice@example.com", x0, y0, col_widths, row_heights),
        _make_cell(2, 0, "Bob Smith", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "Marketng", x0, y0, col_widths, row_heights),
        _make_cell(2, 2, "bob@example.org", x0, y0, col_widths, row_heights),
    ]

    # pred_cells_wrong_structure — merged header cells (different topology)
    # 8 cells instead of 9: "Name" and "Department" merged in header row
    pred_cells_wrong_structure = [
        # Row 0: Name merged across cols 0-1, Email stands alone
        _make_cell(0, 0, "Name & Department", x0, y0, col_widths, row_heights, col_span=2),
        _make_cell(0, 2, "Email", x0, y0, col_widths, row_heights),
        # Row 1
        _make_cell(1, 0, "Alice Johnson", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "Engineering", x0, y0, col_widths, row_heights),
        _make_cell(1, 2, "alice@example.com", x0, y0, col_widths, row_heights),
        # Row 2
        _make_cell(2, 0, "Bob Smith", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "Marketing", x0, y0, col_widths, row_heights),
        _make_cell(2, 2, "bob@example.com", x0, y0, col_widths, row_heights),
    ]

    # pred_cells_partial — mix of correct and wrong text, same structure
    pred_cells_partial = [
        _make_cell(0, 0, "Full Name", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Dept", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Email", x0, y0, col_widths, row_heights),
        _make_cell(1, 0, "Alice Johnson", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "Engineering", x0, y0, col_widths, row_heights),
        _make_cell(1, 2, "alice@co.com", x0, y0, col_widths, row_heights),
        _make_cell(2, 0, "Robert Smith", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "Marketing", x0, y0, col_widths, row_heights),
        _make_cell(2, 2, "bob@example.com", x0, y0, col_widths, row_heights),
    ]

    return {
        "gt_cells": gt_cells,
        "pred_cells_exact": pred_cells_exact,
        "pred_cells_wrong": pred_cells_wrong,
        "pred_cells_wrong_structure": pred_cells_wrong_structure,
        "pred_cells_partial": pred_cells_partial,
        "table_metadata": metadata,
    }


SIMPLE_TABLE = _build_simple_table()


# ── 2. TABLE_WITH_MERGED_CELLS — 4 rows × 3 cols, rowspan + colspan ────────────


def _build_merged_cells_table() -> dict[str, Any]:
    x0, y0 = 100.0, 250.0
    col_widths = [120.0, 160.0, 160.0]
    row_heights = [40.0, 35.0, 35.0, 35.0]

    gt_cells = [
        # Header
        _make_cell(0, 0, "Feature", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Basic", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Premium", x0, y0, col_widths, row_heights),
        # Row 1: Price — Premium spans 2 rows
        _make_cell(1, 0, "Price", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "$99", x0, y0, col_widths, row_heights),
        _make_cell(1, 2, "$199", x0, y0, col_widths, row_heights, row_span=2),
        # Row 2: Storage — (2,2) occupied by rowspan from (1,2)
        _make_cell(2, 0, "Storage", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "256GB", x0, y0, col_widths, row_heights),
        # Row 3
        _make_cell(3, 0, "Support", x0, y0, col_widths, row_heights),
        _make_cell(3, 1, "Email", x0, y0, col_widths, row_heights),
        _make_cell(3, 2, "24/7 Phone", x0, y0, col_widths, row_heights),
    ]
    metadata = _make_metadata(x0, y0, col_widths, row_heights, "Table 2: Pricing Plans")

    pred_cells_exact = deepcopy(gt_cells)

    # wrong — same structure, wrong prices
    pred_cells_wrong = [
        _make_cell(0, 0, "Feature", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Basic", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Pro", x0, y0, col_widths, row_heights),
        _make_cell(1, 0, "Price", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "$89", x0, y0, col_widths, row_heights),
        _make_cell(1, 2, "$179", x0, y0, col_widths, row_heights, row_span=2),
        _make_cell(2, 0, "Storage", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "128GB", x0, y0, col_widths, row_heights),
        _make_cell(3, 0, "Support", x0, y0, col_widths, row_heights),
        _make_cell(3, 1, "Email", x0, y0, col_widths, row_heights),
        _make_cell(3, 2, "Phone Only", x0, y0, col_widths, row_heights),
    ]

    # wrong_structure — flatten all merged cells (12 cells, 1×1 each)
    pred_cells_wrong_structure = [
        _make_cell(0, 0, "Feature", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Basic", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Premium", x0, y0, col_widths, row_heights),
        _make_cell(1, 0, "Price", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "$99", x0, y0, col_widths, row_heights),
        _make_cell(1, 2, "$199", x0, y0, col_widths, row_heights),
        _make_cell(2, 0, "Storage", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "256GB", x0, y0, col_widths, row_heights),
        _make_cell(2, 2, "", x0, y0, col_widths, row_heights),
        _make_cell(3, 0, "Support", x0, y0, col_widths, row_heights),
        _make_cell(3, 1, "Email", x0, y0, col_widths, row_heights),
        _make_cell(3, 2, "24/7 Phone", x0, y0, col_widths, row_heights),
    ]

    # partial — mixed: some merges correct, one text wrong, one span wrong
    pred_cells_partial = [
        _make_cell(0, 0, "Feature", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Basic", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Premium", x0, y0, col_widths, row_heights),
        _make_cell(1, 0, "Price", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "$99", x0, y0, col_widths, row_heights),
        _make_cell(1, 2, "$199", x0, y0, col_widths, row_heights),  # missing rowspan
        _make_cell(2, 0, "Storage", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "256GB", x0, y0, col_widths, row_heights),
        _make_cell(2, 2, "Included", x0, y0, col_widths, row_heights),  # extra cell
        _make_cell(3, 0, "Support", x0, y0, col_widths, row_heights),
        _make_cell(3, 1, "Email", x0, y0, col_widths, row_heights),
        _make_cell(3, 2, "24/7 Phone", x0, y0, col_widths, row_heights),
    ]

    return {
        "gt_cells": gt_cells,
        "pred_cells_exact": pred_cells_exact,
        "pred_cells_wrong": pred_cells_wrong,
        "pred_cells_wrong_structure": pred_cells_wrong_structure,
        "pred_cells_partial": pred_cells_partial,
        "table_metadata": metadata,
    }


TABLE_WITH_MERGED_CELLS = _build_merged_cells_table()


# ── 3. TABLE_NUMERIC — 5 rows × 4 cols, quarterly financial data ──────────────


def _build_numeric_table() -> dict[str, Any]:
    x0, y0 = 60.0, 200.0
    col_widths = [100.0, 120.0, 110.0, 110.0]
    row_heights = [40.0, 30.0, 30.0, 30.0, 30.0]

    gt_cells = [
        _make_cell(0, 0, "Quarter", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Revenue", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Costs", x0, y0, col_widths, row_heights),
        _make_cell(0, 3, "Profit", x0, y0, col_widths, row_heights),
        _make_cell(1, 0, "Q1 2024", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "150000", x0, y0, col_widths, row_heights),
        _make_cell(1, 2, "95000", x0, y0, col_widths, row_heights),
        _make_cell(1, 3, "55000", x0, y0, col_widths, row_heights),
        _make_cell(2, 0, "Q2 2024", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "165000", x0, y0, col_widths, row_heights),
        _make_cell(2, 2, "102000", x0, y0, col_widths, row_heights),
        _make_cell(2, 3, "63000", x0, y0, col_widths, row_heights),
        _make_cell(3, 0, "Q3 2024", x0, y0, col_widths, row_heights),
        _make_cell(3, 1, "180000", x0, y0, col_widths, row_heights),
        _make_cell(3, 2, "110000", x0, y0, col_widths, row_heights),
        _make_cell(3, 3, "70000", x0, y0, col_widths, row_heights),
        _make_cell(4, 0, "Q4 2024", x0, y0, col_widths, row_heights),
        _make_cell(4, 1, "175000", x0, y0, col_widths, row_heights),
        _make_cell(4, 2, "108000", x0, y0, col_widths, row_heights),
        _make_cell(4, 3, "67000", x0, y0, col_widths, row_heights),
    ]
    metadata = _make_metadata(x0, y0, col_widths, row_heights, "Table 3: 2024 Quarterly Financials")

    pred_cells_exact = deepcopy(gt_cells)

    # wrong — same structure, shifted numbers
    pred_cells_wrong = [
        _make_cell(0, 0, "Quarter", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Revenue", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Costs", x0, y0, col_widths, row_heights),
        _make_cell(0, 3, "Profit", x0, y0, col_widths, row_heights),
        _make_cell(1, 0, "Q1 2024", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "148000", x0, y0, col_widths, row_heights),
        _make_cell(1, 2, "95000", x0, y0, col_widths, row_heights),
        _make_cell(1, 3, "53000", x0, y0, col_widths, row_heights),
        _make_cell(2, 0, "Q2 2024", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "165000", x0, y0, col_widths, row_heights),
        _make_cell(2, 2, "100000", x0, y0, col_widths, row_heights),
        _make_cell(2, 3, "65000", x0, y0, col_widths, row_heights),
        _make_cell(3, 0, "Q3 2024", x0, y0, col_widths, row_heights),
        _make_cell(3, 1, "180000", x0, y0, col_widths, row_heights),
        _make_cell(3, 2, "110000", x0, y0, col_widths, row_heights),
        _make_cell(3, 3, "70000", x0, y0, col_widths, row_heights),
        _make_cell(4, 0, "Q4 2024", x0, y0, col_widths, row_heights),
        _make_cell(4, 1, "172000", x0, y0, col_widths, row_heights),
        _make_cell(4, 2, "108000", x0, y0, col_widths, row_heights),
        _make_cell(4, 3, "64000", x0, y0, col_widths, row_heights),
    ]

    # wrong_structure — missing Q3 row entirely (4 rows instead of 5)
    row_heights_short = [40.0, 30.0, 30.0, 30.0]
    pred_cells_wrong_structure = [
        _make_cell(0, 0, "Quarter", x0, y0, col_widths, row_heights_short),
        _make_cell(0, 1, "Revenue", x0, y0, col_widths, row_heights_short),
        _make_cell(0, 2, "Costs", x0, y0, col_widths, row_heights_short),
        _make_cell(0, 3, "Profit", x0, y0, col_widths, row_heights_short),
        _make_cell(1, 0, "Q1 2024", x0, y0, col_widths, row_heights_short),
        _make_cell(1, 1, "150000", x0, y0, col_widths, row_heights_short),
        _make_cell(1, 2, "95000", x0, y0, col_widths, row_heights_short),
        _make_cell(1, 3, "55000", x0, y0, col_widths, row_heights_short),
        _make_cell(2, 0, "Q2 2024", x0, y0, col_widths, row_heights_short),
        _make_cell(2, 1, "165000", x0, y0, col_widths, row_heights_short),
        _make_cell(2, 2, "102000", x0, y0, col_widths, row_heights_short),
        _make_cell(2, 3, "63000", x0, y0, col_widths, row_heights_short),
        _make_cell(3, 0, "Q4 2024", x0, y0, col_widths, row_heights_short),
        _make_cell(3, 1, "175000", x0, y0, col_widths, row_heights_short),
        _make_cell(3, 2, "108000", x0, y0, col_widths, row_heights_short),
        _make_cell(3, 3, "67000", x0, y0, col_widths, row_heights_short),
    ]

    # partial — Q1 correct, Q2 numbers off, Q3-Q4 correct
    pred_cells_partial = [
        _make_cell(0, 0, "Quarter", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Revenue", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Costs", x0, y0, col_widths, row_heights),
        _make_cell(0, 3, "Profit", x0, y0, col_widths, row_heights),
        _make_cell(1, 0, "Q1 2024", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "150000", x0, y0, col_widths, row_heights),
        _make_cell(1, 2, "95000", x0, y0, col_widths, row_heights),
        _make_cell(1, 3, "55000", x0, y0, col_widths, row_heights),
        _make_cell(2, 0, "Q2 2024", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "165500", x0, y0, col_widths, row_heights),
        _make_cell(2, 2, "101000", x0, y0, col_widths, row_heights),
        _make_cell(2, 3, "64500", x0, y0, col_widths, row_heights),
        _make_cell(3, 0, "Q3 2024", x0, y0, col_widths, row_heights),
        _make_cell(3, 1, "180000", x0, y0, col_widths, row_heights),
        _make_cell(3, 2, "110000", x0, y0, col_widths, row_heights),
        _make_cell(3, 3, "70000", x0, y0, col_widths, row_heights),
        _make_cell(4, 0, "Q4 2024", x0, y0, col_widths, row_heights),
        _make_cell(4, 1, "175000", x0, y0, col_widths, row_heights),
        _make_cell(4, 2, "108000", x0, y0, col_widths, row_heights),
        _make_cell(4, 3, "67000", x0, y0, col_widths, row_heights),
    ]

    return {
        "gt_cells": gt_cells,
        "pred_cells_exact": pred_cells_exact,
        "pred_cells_wrong": pred_cells_wrong,
        "pred_cells_wrong_structure": pred_cells_wrong_structure,
        "pred_cells_partial": pred_cells_partial,
        "table_metadata": metadata,
    }


TABLE_NUMERIC = _build_numeric_table()


# ── 4. TABLE_WITH_EMPTY_CELLS — 3 rows × 3 cols, some empty cells ────────────


def _build_empty_cells_table() -> dict[str, Any]:
    x0, y0 = 100.0, 350.0
    col_widths = [100.0, 180.0, 160.0]
    row_heights = [40.0, 35.0, 35.0]

    gt_cells = [
        _make_cell(0, 0, "Name", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Email", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Phone", x0, y0, col_widths, row_heights),
        _make_cell(1, 0, "Alice Johnson", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "alice@example.com", x0, y0, col_widths, row_heights),
        _make_cell(1, 2, "", x0, y0, col_widths, row_heights),
        _make_cell(2, 0, "Bob Smith", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "", x0, y0, col_widths, row_heights),
        _make_cell(2, 2, "555-0199", x0, y0, col_widths, row_heights),
    ]
    metadata = _make_metadata(x0, y0, col_widths, row_heights, "Table 4: Contact Directory")

    pred_cells_exact = deepcopy(gt_cells)

    # wrong — same structure, some empty cells filled with wrong data
    pred_cells_wrong = [
        _make_cell(0, 0, "Name", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Email", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Phone", x0, y0, col_widths, row_heights),
        _make_cell(1, 0, "Alice Johnson", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "alice@example.com", x0, y0, col_widths, row_heights),
        _make_cell(1, 2, "555-0100", x0, y0, col_widths, row_heights),
        _make_cell(2, 0, "Robert Smith", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "bob@example.com", x0, y0, col_widths, row_heights),
        _make_cell(2, 2, "555-0199", x0, y0, col_widths, row_heights),
    ]

    # wrong_structure — extra column added (3 rows × 4 cols)
    col_widths_ext = [100.0, 180.0, 120.0, 100.0]
    row_heights_ext = [40.0, 35.0, 35.0]
    pred_cells_wrong_structure = [
        _make_cell(0, 0, "Name", x0, y0, col_widths_ext, row_heights_ext),
        _make_cell(0, 1, "Email", x0, y0, col_widths_ext, row_heights_ext),
        _make_cell(0, 2, "Phone", x0, y0, col_widths_ext, row_heights_ext),
        _make_cell(0, 3, "Extension", x0, y0, col_widths_ext, row_heights_ext),
        _make_cell(1, 0, "Alice Johnson", x0, y0, col_widths_ext, row_heights_ext),
        _make_cell(1, 1, "alice@example.com", x0, y0, col_widths_ext, row_heights_ext),
        _make_cell(1, 2, "", x0, y0, col_widths_ext, row_heights_ext),
        _make_cell(1, 3, "101", x0, y0, col_widths_ext, row_heights_ext),
        _make_cell(2, 0, "Bob Smith", x0, y0, col_widths_ext, row_heights_ext),
        _make_cell(2, 1, "", x0, y0, col_widths_ext, row_heights_ext),
        _make_cell(2, 2, "555-0199", x0, y0, col_widths_ext, row_heights_ext),
        _make_cell(2, 3, "102", x0, y0, col_widths_ext, row_heights_ext),
    ]

    # partial — half the empty cells filled, one text wrong
    pred_cells_partial = [
        _make_cell(0, 0, "Name", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Email", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Phone", x0, y0, col_widths, row_heights),
        _make_cell(1, 0, "Alice Johnson", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "alice@example.com", x0, y0, col_widths, row_heights),
        _make_cell(1, 2, "N/A", x0, y0, col_widths, row_heights),
        _make_cell(2, 0, "Bob Smith", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "bob@example.com", x0, y0, col_widths, row_heights),
        _make_cell(2, 2, "555-0199", x0, y0, col_widths, row_heights),
    ]

    return {
        "gt_cells": gt_cells,
        "pred_cells_exact": pred_cells_exact,
        "pred_cells_wrong": pred_cells_wrong,
        "pred_cells_wrong_structure": pred_cells_wrong_structure,
        "pred_cells_partial": pred_cells_partial,
        "table_metadata": metadata,
    }


TABLE_WITH_EMPTY_CELLS = _build_empty_cells_table()


# ── 5. TABLE_SINGLE_ROW — 1 row × 5 cols, header only ────────────────────────


def _build_single_row_table() -> dict[str, Any]:
    x0, y0 = 50.0, 100.0
    col_widths = [90.0, 160.0, 90.0, 90.0, 100.0]
    row_heights = [40.0]

    gt_cells = [
        _make_cell(0, 0, "Item", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Description", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Quantity", x0, y0, col_widths, row_heights),
        _make_cell(0, 3, "Unit Price", x0, y0, col_widths, row_heights),
        _make_cell(0, 4, "Total", x0, y0, col_widths, row_heights),
    ]
    metadata = _make_metadata(x0, y0, col_widths, row_heights, "Table 5: Order Headers")

    pred_cells_exact = deepcopy(gt_cells)

    # wrong — same structure, wrong text
    pred_cells_wrong = [
        _make_cell(0, 0, "SKU", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Product Name", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Qty", x0, y0, col_widths, row_heights),
        _make_cell(0, 3, "Price", x0, y0, col_widths, row_heights),
        _make_cell(0, 4, "Amount", x0, y0, col_widths, row_heights),
    ]

    # wrong_structure — split into 2 rows via a colspan
    row_heights_2 = [40.0, 35.0]
    pred_cells_wrong_structure = [
        _make_cell(0, 0, "Item", x0, y0, col_widths, row_heights_2),
        _make_cell(0, 1, "Description & Quantity", x0, y0, col_widths, row_heights_2, col_span=2),
        _make_cell(0, 3, "Unit Price", x0, y0, col_widths, row_heights_2),
        _make_cell(0, 4, "Total", x0, y0, col_widths, row_heights_2),
        _make_cell(1, 0, "", x0, y0, col_widths, row_heights_2),
        _make_cell(1, 1, "", x0, y0, col_widths, row_heights_2),
        _make_cell(1, 2, "", x0, y0, col_widths, row_heights_2),
        _make_cell(1, 3, "", x0, y0, col_widths, row_heights_2),
        _make_cell(1, 4, "", x0, y0, col_widths, row_heights_2),
    ]

    # partial — two headers wrong
    pred_cells_partial = [
        _make_cell(0, 0, "Item", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Description", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Count", x0, y0, col_widths, row_heights),
        _make_cell(0, 3, "Unit Price", x0, y0, col_widths, row_heights),
        _make_cell(0, 4, "Sum", x0, y0, col_widths, row_heights),
    ]

    return {
        "gt_cells": gt_cells,
        "pred_cells_exact": pred_cells_exact,
        "pred_cells_wrong": pred_cells_wrong,
        "pred_cells_wrong_structure": pred_cells_wrong_structure,
        "pred_cells_partial": pred_cells_partial,
        "table_metadata": metadata,
    }


TABLE_SINGLE_ROW = _build_single_row_table()


# ── 6. TABLE_SINGLE_CELL — 1 row × 1 col, trivial table ───────────────────────


def _build_single_cell_table() -> dict[str, Any]:
    x0, y0 = 200.0, 400.0
    col_widths = [200.0]
    row_heights = [40.0]

    gt_cells = [
        _make_cell(0, 0, "Total Amount Due: $1,234.56", x0, y0, col_widths, row_heights),
    ]
    metadata = _make_metadata(x0, y0, col_widths, row_heights, "Table 6: Invoice Total")

    pred_cells_exact = deepcopy(gt_cells)

    # wrong — wrong amount
    pred_cells_wrong = [
        _make_cell(0, 0, "Total Amount Due: $1,199.99", x0, y0, col_widths, row_heights),
    ]

    # wrong_structure — split into 2 rows (label + value)
    row_heights_2 = [25.0, 25.0]
    pred_cells_wrong_structure = [
        _make_cell(0, 0, "Total Amount Due:", x0, y0, col_widths, row_heights_2),
        _make_cell(1, 0, "$1,234.56", x0, y0, col_widths, row_heights_2),
    ]

    # partial — wrong text
    pred_cells_partial = [
        _make_cell(0, 0, "Total: $1,234.56", x0, y0, col_widths, row_heights),
    ]

    return {
        "gt_cells": gt_cells,
        "pred_cells_exact": pred_cells_exact,
        "pred_cells_wrong": pred_cells_wrong,
        "pred_cells_wrong_structure": pred_cells_wrong_structure,
        "pred_cells_partial": pred_cells_partial,
        "table_metadata": metadata,
    }


TABLE_SINGLE_CELL = _build_single_cell_table()


# ── 7. TABLE_IRREGULAR — 4 rows × 4 cols, various row/col spans ───────────────


def _build_irregular_table() -> dict[str, Any]:
    x0, y0 = 80.0, 300.0
    col_widths = [130.0, 130.0, 130.0, 130.0]
    row_heights = [40.0, 35.0, 35.0, 35.0]

    gt_cells = [
        # Header
        _make_cell(0, 0, "Department", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Q1", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Q2", x0, y0, col_widths, row_heights),
        _make_cell(0, 3, "Q3", x0, y0, col_widths, row_heights),
        # Row 1: Engineering, colspan=3 across Q1-Q3
        _make_cell(1, 0, "Engineering", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "Product Launch", x0, y0, col_widths, row_heights, col_span=3),
        # Row 2: Marketing with rowspan on col 1, Campaign B and C on cols 2-3
        _make_cell(2, 0, "Marketing", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "Campaign A", x0, y0, col_widths, row_heights, row_span=2),
        _make_cell(2, 2, "Campaign B", x0, y0, col_widths, row_heights),
        _make_cell(2, 3, "Campaign C", x0, y0, col_widths, row_heights),
        # Row 3: Sales, cols 2-3 filled (col 1 occupied by rowspan)
        _make_cell(3, 0, "Sales Support", x0, y0, col_widths, row_heights),
        _make_cell(3, 2, "Campaign D", x0, y0, col_widths, row_heights),
        _make_cell(3, 3, "Campaign E", x0, y0, col_widths, row_heights),
    ]
    metadata = _make_metadata(x0, y0, col_widths, row_heights, "Table 7: Campaign Tracker")

    pred_cells_exact = deepcopy(gt_cells)

    # wrong — same structure, different campaign names
    pred_cells_wrong = [
        _make_cell(0, 0, "Department", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Q1", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Q2", x0, y0, col_widths, row_heights),
        _make_cell(0, 3, "Q3", x0, y0, col_widths, row_heights),
        _make_cell(1, 0, "Engineering", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "Launch Event", x0, y0, col_widths, row_heights, col_span=3),
        _make_cell(2, 0, "Marketing", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "Ad Campaign", x0, y0, col_widths, row_heights, row_span=2),
        _make_cell(2, 2, "Campaign B", x0, y0, col_widths, row_heights),
        _make_cell(2, 3, "Campaign C", x0, y0, col_widths, row_heights),
        _make_cell(3, 0, "Sales", x0, y0, col_widths, row_heights),
        _make_cell(3, 2, "Campaign D", x0, y0, col_widths, row_heights),
        _make_cell(3, 3, "Campaign F", x0, y0, col_widths, row_heights),
    ]

    # wrong_structure — flatten all spans to 1×1, resulting in extra cells
    pred_cells_wrong_structure = [
        _make_cell(0, 0, "Department", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Q1", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Q2", x0, y0, col_widths, row_heights),
        _make_cell(0, 3, "Q3", x0, y0, col_widths, row_heights),
        _make_cell(1, 0, "Engineering", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "Product Launch", x0, y0, col_widths, row_heights),
        _make_cell(1, 2, "", x0, y0, col_widths, row_heights),
        _make_cell(1, 3, "", x0, y0, col_widths, row_heights),
        _make_cell(2, 0, "Marketing", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "Campaign A", x0, y0, col_widths, row_heights),
        _make_cell(2, 2, "Campaign B", x0, y0, col_widths, row_heights),
        _make_cell(2, 3, "Campaign C", x0, y0, col_widths, row_heights),
        _make_cell(3, 0, "Sales Support", x0, y0, col_widths, row_heights),
        _make_cell(3, 1, "", x0, y0, col_widths, row_heights),
        _make_cell(3, 2, "Campaign D", x0, y0, col_widths, row_heights),
        _make_cell(3, 3, "Campaign E", x0, y0, col_widths, row_heights),
    ]

    # partial — one span wrong, one text wrong
    pred_cells_partial = [
        _make_cell(0, 0, "Department", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Q1", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Q2", x0, y0, col_widths, row_heights),
        _make_cell(0, 3, "Q3", x0, y0, col_widths, row_heights),
        _make_cell(1, 0, "Engineering", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "Product Launch", x0, y0, col_widths, row_heights),  # missing colspan
        _make_cell(1, 2, "", x0, y0, col_widths, row_heights),
        _make_cell(1, 3, "", x0, y0, col_widths, row_heights),
        _make_cell(2, 0, "Marketing", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "Campaign A", x0, y0, col_widths, row_heights, row_span=2),
        _make_cell(2, 2, "Campaign B", x0, y0, col_widths, row_heights),
        _make_cell(2, 3, "Campaign C", x0, y0, col_widths, row_heights),
        _make_cell(3, 0, "Sales Support", x0, y0, col_widths, row_heights),
        _make_cell(3, 2, "Campaign D", x0, y0, col_widths, row_heights),
        _make_cell(3, 3, "Campaign F", x0, y0, col_widths, row_heights),
    ]

    return {
        "gt_cells": gt_cells,
        "pred_cells_exact": pred_cells_exact,
        "pred_cells_wrong": pred_cells_wrong,
        "pred_cells_wrong_structure": pred_cells_wrong_structure,
        "pred_cells_partial": pred_cells_partial,
        "table_metadata": metadata,
    }


TABLE_IRREGULAR = _build_irregular_table()


# ── 8. TABLE_LARGE — 10 rows × 8 cols, mixed content ─────────────────────────


def _build_large_table() -> dict[str, Any]:
    x0, y0 = 40.0, 150.0
    col_widths = [50.0, 110.0, 80.0, 60.0, 60.0, 60.0, 50.0, 80.0]
    row_heights = [35.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0]

    gt_cells = [
        # Header
        _make_cell(0, 0, "ID", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Product Name", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Category", x0, y0, col_widths, row_heights),
        _make_cell(0, 3, "Price", x0, y0, col_widths, row_heights),
        _make_cell(0, 4, "Quantity", x0, y0, col_widths, row_heights),
        _make_cell(0, 5, "In Stock", x0, y0, col_widths, row_heights),
        _make_cell(0, 6, "Rating", x0, y0, col_widths, row_heights),
        _make_cell(0, 7, "Updated", x0, y0, col_widths, row_heights),
        # Row 1
        _make_cell(1, 0, "PRD-001", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "Wireless Mouse", x0, y0, col_widths, row_heights),
        _make_cell(1, 2, "Electronics", x0, y0, col_widths, row_heights),
        _make_cell(1, 3, "29.99", x0, y0, col_widths, row_heights),
        _make_cell(1, 4, "150", x0, y0, col_widths, row_heights),
        _make_cell(1, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(1, 6, "4.5", x0, y0, col_widths, row_heights),
        _make_cell(1, 7, "2024-01-15", x0, y0, col_widths, row_heights),
        # Row 2
        _make_cell(2, 0, "PRD-002", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "Mechanical Keyboard", x0, y0, col_widths, row_heights),
        _make_cell(2, 2, "Electronics", x0, y0, col_widths, row_heights),
        _make_cell(2, 3, "89.99", x0, y0, col_widths, row_heights),
        _make_cell(2, 4, "75", x0, y0, col_widths, row_heights),
        _make_cell(2, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(2, 6, "4.8", x0, y0, col_widths, row_heights),
        _make_cell(2, 7, "2024-01-20", x0, y0, col_widths, row_heights),
        # Row 3
        _make_cell(3, 0, "PRD-003", x0, y0, col_widths, row_heights),
        _make_cell(3, 1, "USB-C Hub", x0, y0, col_widths, row_heights),
        _make_cell(3, 2, "Accessories", x0, y0, col_widths, row_heights),
        _make_cell(3, 3, "45.00", x0, y0, col_widths, row_heights),
        _make_cell(3, 4, "200", x0, y0, col_widths, row_heights),
        _make_cell(3, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(3, 6, "4.3", x0, y0, col_widths, row_heights),
        _make_cell(3, 7, "2024-02-01", x0, y0, col_widths, row_heights),
        # Row 4
        _make_cell(4, 0, "PRD-004", x0, y0, col_widths, row_heights),
        _make_cell(4, 1, "Monitor Stand", x0, y0, col_widths, row_heights),
        _make_cell(4, 2, "Furniture", x0, y0, col_widths, row_heights),
        _make_cell(4, 3, "59.99", x0, y0, col_widths, row_heights),
        _make_cell(4, 4, "45", x0, y0, col_widths, row_heights),
        _make_cell(4, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(4, 6, "4.6", x0, y0, col_widths, row_heights),
        _make_cell(4, 7, "2024-02-10", x0, y0, col_widths, row_heights),
        # Row 5
        _make_cell(5, 0, "PRD-005", x0, y0, col_widths, row_heights),
        _make_cell(5, 1, "Webcam HD", x0, y0, col_widths, row_heights),
        _make_cell(5, 2, "Electronics", x0, y0, col_widths, row_heights),
        _make_cell(5, 3, "79.99", x0, y0, col_widths, row_heights),
        _make_cell(5, 4, "0", x0, y0, col_widths, row_heights),
        _make_cell(5, 5, "No", x0, y0, col_widths, row_heights),
        _make_cell(5, 6, "4.1", x0, y0, col_widths, row_heights),
        _make_cell(5, 7, "2024-03-05", x0, y0, col_widths, row_heights),
        # Row 6
        _make_cell(6, 0, "PRD-006", x0, y0, col_widths, row_heights),
        _make_cell(6, 1, "Desk Lamp", x0, y0, col_widths, row_heights),
        _make_cell(6, 2, "Furniture", x0, y0, col_widths, row_heights),
        _make_cell(6, 3, "34.99", x0, y0, col_widths, row_heights),
        _make_cell(6, 4, "120", x0, y0, col_widths, row_heights),
        _make_cell(6, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(6, 6, "4.7", x0, y0, col_widths, row_heights),
        _make_cell(6, 7, "2024-03-15", x0, y0, col_widths, row_heights),
        # Row 7
        _make_cell(7, 0, "PRD-007", x0, y0, col_widths, row_heights),
        _make_cell(7, 1, "Noise Cancelling Headphones", x0, y0, col_widths, row_heights),
        _make_cell(7, 2, "Electronics", x0, y0, col_widths, row_heights),
        _make_cell(7, 3, "249.99", x0, y0, col_widths, row_heights),
        _make_cell(7, 4, "30", x0, y0, col_widths, row_heights),
        _make_cell(7, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(7, 6, "4.9", x0, y0, col_widths, row_heights),
        _make_cell(7, 7, "2024-04-01", x0, y0, col_widths, row_heights),
        # Row 8
        _make_cell(8, 0, "PRD-008", x0, y0, col_widths, row_heights),
        _make_cell(8, 1, "Mouse Pad", x0, y0, col_widths, row_heights),
        _make_cell(8, 2, "Accessories", x0, y0, col_widths, row_heights),
        _make_cell(8, 3, "12.99", x0, y0, col_widths, row_heights),
        _make_cell(8, 4, "500", x0, y0, col_widths, row_heights),
        _make_cell(8, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(8, 6, "4.2", x0, y0, col_widths, row_heights),
        _make_cell(8, 7, "2024-04-10", x0, y0, col_widths, row_heights),
        # Row 9
        _make_cell(9, 0, "PRD-009", x0, y0, col_widths, row_heights),
        _make_cell(9, 1, "Ergonomic Chair", x0, y0, col_widths, row_heights),
        _make_cell(9, 2, "Furniture", x0, y0, col_widths, row_heights),
        _make_cell(9, 3, "499.99", x0, y0, col_widths, row_heights),
        _make_cell(9, 4, "15", x0, y0, col_widths, row_heights),
        _make_cell(9, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(9, 6, "4.8", x0, y0, col_widths, row_heights),
        _make_cell(9, 7, "2024-05-01", x0, y0, col_widths, row_heights),
    ]
    metadata = _make_metadata(x0, y0, col_widths, row_heights, "Table 8: Product Inventory")

    pred_cells_exact = deepcopy(gt_cells)

    # wrong — same structure, some prices/quantities shifted
    pred_cells_wrong = [
        _make_cell(0, 0, "ID", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Product Name", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Category", x0, y0, col_widths, row_heights),
        _make_cell(0, 3, "Price", x0, y0, col_widths, row_heights),
        _make_cell(0, 4, "Quantity", x0, y0, col_widths, row_heights),
        _make_cell(0, 5, "In Stock", x0, y0, col_widths, row_heights),
        _make_cell(0, 6, "Rating", x0, y0, col_widths, row_heights),
        _make_cell(0, 7, "Updated", x0, y0, col_widths, row_heights),
        _make_cell(1, 0, "PRD-001", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "Wireless Mouse", x0, y0, col_widths, row_heights),
        _make_cell(1, 2, "Electronics", x0, y0, col_widths, row_heights),
        _make_cell(1, 3, "27.99", x0, y0, col_widths, row_heights),
        _make_cell(1, 4, "150", x0, y0, col_widths, row_heights),
        _make_cell(1, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(1, 6, "4.5", x0, y0, col_widths, row_heights),
        _make_cell(1, 7, "2024-01-15", x0, y0, col_widths, row_heights),
        _make_cell(2, 0, "PRD-002", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "Mechanical Keyboard", x0, y0, col_widths, row_heights),
        _make_cell(2, 2, "Electronics", x0, y0, col_widths, row_heights),
        _make_cell(2, 3, "89.99", x0, y0, col_widths, row_heights),
        _make_cell(2, 4, "75", x0, y0, col_widths, row_heights),
        _make_cell(2, 5, "No", x0, y0, col_widths, row_heights),
        _make_cell(2, 6, "4.8", x0, y0, col_widths, row_heights),
        _make_cell(2, 7, "2024-01-20", x0, y0, col_widths, row_heights),
        _make_cell(3, 0, "PRD-003", x0, y0, col_widths, row_heights),
        _make_cell(3, 1, "USB-C Hub", x0, y0, col_widths, row_heights),
        _make_cell(3, 2, "Accessories", x0, y0, col_widths, row_heights),
        _make_cell(3, 3, "45.00", x0, y0, col_widths, row_heights),
        _make_cell(3, 4, "200", x0, y0, col_widths, row_heights),
        _make_cell(3, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(3, 6, "4.3", x0, y0, col_widths, row_heights),
        _make_cell(3, 7, "2024-02-01", x0, y0, col_widths, row_heights),
        _make_cell(4, 0, "PRD-004", x0, y0, col_widths, row_heights),
        _make_cell(4, 1, "Monitor Stand", x0, y0, col_widths, row_heights),
        _make_cell(4, 2, "Furniture", x0, y0, col_widths, row_heights),
        _make_cell(4, 3, "59.99", x0, y0, col_widths, row_heights),
        _make_cell(4, 4, "45", x0, y0, col_widths, row_heights),
        _make_cell(4, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(4, 6, "4.6", x0, y0, col_widths, row_heights),
        _make_cell(4, 7, "2024-02-10", x0, y0, col_widths, row_heights),
        _make_cell(5, 0, "PRD-005", x0, y0, col_widths, row_heights),
        _make_cell(5, 1, "Webcam HD", x0, y0, col_widths, row_heights),
        _make_cell(5, 2, "Electronics", x0, y0, col_widths, row_heights),
        _make_cell(5, 3, "74.99", x0, y0, col_widths, row_heights),
        _make_cell(5, 4, "5", x0, y0, col_widths, row_heights),
        _make_cell(5, 5, "No", x0, y0, col_widths, row_heights),
        _make_cell(5, 6, "4.1", x0, y0, col_widths, row_heights),
        _make_cell(5, 7, "2024-03-05", x0, y0, col_widths, row_heights),
        _make_cell(6, 0, "PRD-006", x0, y0, col_widths, row_heights),
        _make_cell(6, 1, "Desk Lamp", x0, y0, col_widths, row_heights),
        _make_cell(6, 2, "Furniture", x0, y0, col_widths, row_heights),
        _make_cell(6, 3, "34.99", x0, y0, col_widths, row_heights),
        _make_cell(6, 4, "120", x0, y0, col_widths, row_heights),
        _make_cell(6, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(6, 6, "4.7", x0, y0, col_widths, row_heights),
        _make_cell(6, 7, "2024-03-15", x0, y0, col_widths, row_heights),
        _make_cell(7, 0, "PRD-007", x0, y0, col_widths, row_heights),
        _make_cell(7, 1, "Noise Cancelling Headphones", x0, y0, col_widths, row_heights),
        _make_cell(7, 2, "Electronics", x0, y0, col_widths, row_heights),
        _make_cell(7, 3, "249.99", x0, y0, col_widths, row_heights),
        _make_cell(7, 4, "30", x0, y0, col_widths, row_heights),
        _make_cell(7, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(7, 6, "4.9", x0, y0, col_widths, row_heights),
        _make_cell(7, 7, "2024-04-01", x0, y0, col_widths, row_heights),
        _make_cell(8, 0, "PRD-008", x0, y0, col_widths, row_heights),
        _make_cell(8, 1, "Mouse Pad", x0, y0, col_widths, row_heights),
        _make_cell(8, 2, "Accessories", x0, y0, col_widths, row_heights),
        _make_cell(8, 3, "12.99", x0, y0, col_widths, row_heights),
        _make_cell(8, 4, "500", x0, y0, col_widths, row_heights),
        _make_cell(8, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(8, 6, "4.2", x0, y0, col_widths, row_heights),
        _make_cell(8, 7, "2024-04-10", x0, y0, col_widths, row_heights),
        _make_cell(9, 0, "PRD-009", x0, y0, col_widths, row_heights),
        _make_cell(9, 1, "Ergonomic Chair", x0, y0, col_widths, row_heights),
        _make_cell(9, 2, "Furniture", x0, y0, col_widths, row_heights),
        _make_cell(9, 3, "499.99", x0, y0, col_widths, row_heights),
        _make_cell(9, 4, "15", x0, y0, col_widths, row_heights),
        _make_cell(9, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(9, 6, "4.8", x0, y0, col_widths, row_heights),
        _make_cell(9, 7, "2024-05-01", x0, y0, col_widths, row_heights),
    ]

    # wrong_structure — missing a column (no "Rating" column, 10×7)
    col_widths_7 = [50.0, 110.0, 80.0, 60.0, 60.0, 60.0, 80.0]
    pred_cells_wrong_structure = [
        _make_cell(0, 0, "ID", x0, y0, col_widths_7, row_heights),
        _make_cell(0, 1, "Product Name", x0, y0, col_widths_7, row_heights),
        _make_cell(0, 2, "Category", x0, y0, col_widths_7, row_heights),
        _make_cell(0, 3, "Price", x0, y0, col_widths_7, row_heights),
        _make_cell(0, 4, "Quantity", x0, y0, col_widths_7, row_heights),
        _make_cell(0, 5, "In Stock", x0, y0, col_widths_7, row_heights),
        _make_cell(0, 6, "Updated", x0, y0, col_widths_7, row_heights),
        # Row 1
        _make_cell(1, 0, "PRD-001", x0, y0, col_widths_7, row_heights),
        _make_cell(1, 1, "Wireless Mouse", x0, y0, col_widths_7, row_heights),
        _make_cell(1, 2, "Electronics", x0, y0, col_widths_7, row_heights),
        _make_cell(1, 3, "29.99", x0, y0, col_widths_7, row_heights),
        _make_cell(1, 4, "150", x0, y0, col_widths_7, row_heights),
        _make_cell(1, 5, "Yes", x0, y0, col_widths_7, row_heights),
        _make_cell(1, 6, "2024-01-15", x0, y0, col_widths_7, row_heights),
        # Row 2
        _make_cell(2, 0, "PRD-002", x0, y0, col_widths_7, row_heights),
        _make_cell(2, 1, "Mechanical Keyboard", x0, y0, col_widths_7, row_heights),
        _make_cell(2, 2, "Electronics", x0, y0, col_widths_7, row_heights),
        _make_cell(2, 3, "89.99", x0, y0, col_widths_7, row_heights),
        _make_cell(2, 4, "75", x0, y0, col_widths_7, row_heights),
        _make_cell(2, 5, "Yes", x0, y0, col_widths_7, row_heights),
        _make_cell(2, 6, "2024-01-20", x0, y0, col_widths_7, row_heights),
        # Row 3
        _make_cell(3, 0, "PRD-003", x0, y0, col_widths_7, row_heights),
        _make_cell(3, 1, "USB-C Hub", x0, y0, col_widths_7, row_heights),
        _make_cell(3, 2, "Accessories", x0, y0, col_widths_7, row_heights),
        _make_cell(3, 3, "45.00", x0, y0, col_widths_7, row_heights),
        _make_cell(3, 4, "200", x0, y0, col_widths_7, row_heights),
        _make_cell(3, 5, "Yes", x0, y0, col_widths_7, row_heights),
        _make_cell(3, 6, "2024-02-01", x0, y0, col_widths_7, row_heights),
        # Row 4
        _make_cell(4, 0, "PRD-004", x0, y0, col_widths_7, row_heights),
        _make_cell(4, 1, "Monitor Stand", x0, y0, col_widths_7, row_heights),
        _make_cell(4, 2, "Furniture", x0, y0, col_widths_7, row_heights),
        _make_cell(4, 3, "59.99", x0, y0, col_widths_7, row_heights),
        _make_cell(4, 4, "45", x0, y0, col_widths_7, row_heights),
        _make_cell(4, 5, "Yes", x0, y0, col_widths_7, row_heights),
        _make_cell(4, 6, "2024-02-10", x0, y0, col_widths_7, row_heights),
        # Row 5
        _make_cell(5, 0, "PRD-005", x0, y0, col_widths_7, row_heights),
        _make_cell(5, 1, "Webcam HD", x0, y0, col_widths_7, row_heights),
        _make_cell(5, 2, "Electronics", x0, y0, col_widths_7, row_heights),
        _make_cell(5, 3, "79.99", x0, y0, col_widths_7, row_heights),
        _make_cell(5, 4, "0", x0, y0, col_widths_7, row_heights),
        _make_cell(5, 5, "No", x0, y0, col_widths_7, row_heights),
        _make_cell(5, 6, "2024-03-05", x0, y0, col_widths_7, row_heights),
        # Row 6
        _make_cell(6, 0, "PRD-006", x0, y0, col_widths_7, row_heights),
        _make_cell(6, 1, "Desk Lamp", x0, y0, col_widths_7, row_heights),
        _make_cell(6, 2, "Furniture", x0, y0, col_widths_7, row_heights),
        _make_cell(6, 3, "34.99", x0, y0, col_widths_7, row_heights),
        _make_cell(6, 4, "120", x0, y0, col_widths_7, row_heights),
        _make_cell(6, 5, "Yes", x0, y0, col_widths_7, row_heights),
        _make_cell(6, 6, "2024-03-15", x0, y0, col_widths_7, row_heights),
        # Row 7
        _make_cell(7, 0, "PRD-007", x0, y0, col_widths_7, row_heights),
        _make_cell(7, 1, "Noise Cancelling Headphones", x0, y0, col_widths_7, row_heights),
        _make_cell(7, 2, "Electronics", x0, y0, col_widths_7, row_heights),
        _make_cell(7, 3, "249.99", x0, y0, col_widths_7, row_heights),
        _make_cell(7, 4, "30", x0, y0, col_widths_7, row_heights),
        _make_cell(7, 5, "Yes", x0, y0, col_widths_7, row_heights),
        _make_cell(7, 6, "2024-04-01", x0, y0, col_widths_7, row_heights),
        # Row 8
        _make_cell(8, 0, "PRD-008", x0, y0, col_widths_7, row_heights),
        _make_cell(8, 1, "Mouse Pad", x0, y0, col_widths_7, row_heights),
        _make_cell(8, 2, "Accessories", x0, y0, col_widths_7, row_heights),
        _make_cell(8, 3, "12.99", x0, y0, col_widths_7, row_heights),
        _make_cell(8, 4, "500", x0, y0, col_widths_7, row_heights),
        _make_cell(8, 5, "Yes", x0, y0, col_widths_7, row_heights),
        _make_cell(8, 6, "2024-04-10", x0, y0, col_widths_7, row_heights),
        # Row 9
        _make_cell(9, 0, "PRD-009", x0, y0, col_widths_7, row_heights),
        _make_cell(9, 1, "Ergonomic Chair", x0, y0, col_widths_7, row_heights),
        _make_cell(9, 2, "Furniture", x0, y0, col_widths_7, row_heights),
        _make_cell(9, 3, "499.99", x0, y0, col_widths_7, row_heights),
        _make_cell(9, 4, "15", x0, y0, col_widths_7, row_heights),
        _make_cell(9, 5, "Yes", x0, y0, col_widths_7, row_heights),
        _make_cell(9, 6, "2024-05-01", x0, y0, col_widths_7, row_heights),
    ]

    # partial — first 2 rows correct, row 2 wrong price, row 3-9 correct
    pred_cells_partial = [
        _make_cell(0, 0, "ID", x0, y0, col_widths, row_heights),
        _make_cell(0, 1, "Product Name", x0, y0, col_widths, row_heights),
        _make_cell(0, 2, "Category", x0, y0, col_widths, row_heights),
        _make_cell(0, 3, "Price", x0, y0, col_widths, row_heights),
        _make_cell(0, 4, "Quantity", x0, y0, col_widths, row_heights),
        _make_cell(0, 5, "In Stock", x0, y0, col_widths, row_heights),
        _make_cell(0, 6, "Rating", x0, y0, col_widths, row_heights),
        _make_cell(0, 7, "Updated", x0, y0, col_widths, row_heights),
        _make_cell(1, 0, "PRD-001", x0, y0, col_widths, row_heights),
        _make_cell(1, 1, "Wireless Mouse", x0, y0, col_widths, row_heights),
        _make_cell(1, 2, "Electronics", x0, y0, col_widths, row_heights),
        _make_cell(1, 3, "29.99", x0, y0, col_widths, row_heights),
        _make_cell(1, 4, "150", x0, y0, col_widths, row_heights),
        _make_cell(1, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(1, 6, "4.5", x0, y0, col_widths, row_heights),
        _make_cell(1, 7, "2024-01-15", x0, y0, col_widths, row_heights),
        # Row 2: Mechanical Keyboard — wrong "In Stock", wrong "Quantity"
        _make_cell(2, 0, "PRD-002", x0, y0, col_widths, row_heights),
        _make_cell(2, 1, "Mechanical Keyboard", x0, y0, col_widths, row_heights),
        _make_cell(2, 2, "Electronics", x0, y0, col_widths, row_heights),
        _make_cell(2, 3, "89.99", x0, y0, col_widths, row_heights),
        _make_cell(2, 4, "75", x0, y0, col_widths, row_heights),
        _make_cell(2, 5, "No", x0, y0, col_widths, row_heights),
        _make_cell(2, 6, "4.8", x0, y0, col_widths, row_heights),
        _make_cell(2, 7, "2024-01-20", x0, y0, col_widths, row_heights),
        # Rest identical to GT (copied below)
        _make_cell(3, 0, "PRD-003", x0, y0, col_widths, row_heights),
        _make_cell(3, 1, "USB-C Hub", x0, y0, col_widths, row_heights),
        _make_cell(3, 2, "Accessories", x0, y0, col_widths, row_heights),
        _make_cell(3, 3, "45.00", x0, y0, col_widths, row_heights),
        _make_cell(3, 4, "200", x0, y0, col_widths, row_heights),
        _make_cell(3, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(3, 6, "4.3", x0, y0, col_widths, row_heights),
        _make_cell(3, 7, "2024-02-01", x0, y0, col_widths, row_heights),
        _make_cell(4, 0, "PRD-004", x0, y0, col_widths, row_heights),
        _make_cell(4, 1, "Monitor Stand", x0, y0, col_widths, row_heights),
        _make_cell(4, 2, "Furniture", x0, y0, col_widths, row_heights),
        _make_cell(4, 3, "59.99", x0, y0, col_widths, row_heights),
        _make_cell(4, 4, "45", x0, y0, col_widths, row_heights),
        _make_cell(4, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(4, 6, "4.6", x0, y0, col_widths, row_heights),
        _make_cell(4, 7, "2024-02-10", x0, y0, col_widths, row_heights),
        _make_cell(5, 0, "PRD-005", x0, y0, col_widths, row_heights),
        _make_cell(5, 1, "Webcam HD", x0, y0, col_widths, row_heights),
        _make_cell(5, 2, "Electronics", x0, y0, col_widths, row_heights),
        _make_cell(5, 3, "79.99", x0, y0, col_widths, row_heights),
        _make_cell(5, 4, "0", x0, y0, col_widths, row_heights),
        _make_cell(5, 5, "No", x0, y0, col_widths, row_heights),
        _make_cell(5, 6, "4.1", x0, y0, col_widths, row_heights),
        _make_cell(5, 7, "2024-03-05", x0, y0, col_widths, row_heights),
        _make_cell(6, 0, "PRD-006", x0, y0, col_widths, row_heights),
        _make_cell(6, 1, "Desk Lamp", x0, y0, col_widths, row_heights),
        _make_cell(6, 2, "Furniture", x0, y0, col_widths, row_heights),
        _make_cell(6, 3, "34.99", x0, y0, col_widths, row_heights),
        _make_cell(6, 4, "120", x0, y0, col_widths, row_heights),
        _make_cell(6, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(6, 6, "4.7", x0, y0, col_widths, row_heights),
        _make_cell(6, 7, "2024-03-15", x0, y0, col_widths, row_heights),
        _make_cell(7, 0, "PRD-007", x0, y0, col_widths, row_heights),
        _make_cell(7, 1, "Noise Cancelling Headphones", x0, y0, col_widths, row_heights),
        _make_cell(7, 2, "Electronics", x0, y0, col_widths, row_heights),
        _make_cell(7, 3, "249.99", x0, y0, col_widths, row_heights),
        _make_cell(7, 4, "30", x0, y0, col_widths, row_heights),
        _make_cell(7, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(7, 6, "4.9", x0, y0, col_widths, row_heights),
        _make_cell(7, 7, "2024-04-01", x0, y0, col_widths, row_heights),
        _make_cell(8, 0, "PRD-008", x0, y0, col_widths, row_heights),
        _make_cell(8, 1, "Mouse Pad", x0, y0, col_widths, row_heights),
        _make_cell(8, 2, "Accessories", x0, y0, col_widths, row_heights),
        _make_cell(8, 3, "12.99", x0, y0, col_widths, row_heights),
        _make_cell(8, 4, "500", x0, y0, col_widths, row_heights),
        _make_cell(8, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(8, 6, "4.2", x0, y0, col_widths, row_heights),
        _make_cell(8, 7, "2024-04-10", x0, y0, col_widths, row_heights),
        _make_cell(9, 0, "PRD-009", x0, y0, col_widths, row_heights),
        _make_cell(9, 1, "Ergonomic Chair", x0, y0, col_widths, row_heights),
        _make_cell(9, 2, "Furniture", x0, y0, col_widths, row_heights),
        _make_cell(9, 3, "499.99", x0, y0, col_widths, row_heights),
        _make_cell(9, 4, "15", x0, y0, col_widths, row_heights),
        _make_cell(9, 5, "Yes", x0, y0, col_widths, row_heights),
        _make_cell(9, 6, "4.8", x0, y0, col_widths, row_heights),
        _make_cell(9, 7, "2024-05-01", x0, y0, col_widths, row_heights),
    ]

    return {
        "gt_cells": gt_cells,
        "pred_cells_exact": pred_cells_exact,
        "pred_cells_wrong": pred_cells_wrong,
        "pred_cells_wrong_structure": pred_cells_wrong_structure,
        "pred_cells_partial": pred_cells_partial,
        "table_metadata": metadata,
    }


TABLE_LARGE = _build_large_table()
