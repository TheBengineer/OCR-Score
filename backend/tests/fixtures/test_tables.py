"""Tests for table alignment test fixtures.

Verifies that each fixture:
- Has the correct structure (required fields, valid types)
- Has a consistent grid (cells occupy non-overlapping positions)
- Provides internally consistent pred variants relative to gt_cells
- Produces the expected match characteristics (exact, partial, structural diff)
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.tests.fixtures.tables import (
    SIMPLE_TABLE,
    TABLE_IRREGULAR,
    TABLE_LARGE,
    TABLE_NUMERIC,
    TABLE_SINGLE_CELL,
    TABLE_SINGLE_ROW,
    TABLE_WITH_EMPTY_CELLS,
    TABLE_WITH_MERGED_CELLS,
    _cells_equal,
)

# ── Helpers ─────────────────────────────────────────────────────────────────────


def _cell_key(cell: dict[str, Any]) -> str:
    """Unique key for a cell's grid position."""
    return f"{cell['row']},{cell['col']}"


def _validate_fixture_structure(fixture: dict[str, Any], name: str) -> None:
    """Validate that a fixture has all required keys and valid types."""
    expected_keys = {
        "gt_cells",
        "pred_cells_exact",
        "pred_cells_wrong",
        "pred_cells_wrong_structure",
        "pred_cells_partial",
        "table_metadata",
    }
    assert fixture.keys() == expected_keys, f"{name}: unexpected keys"

    meta = fixture["table_metadata"]
    assert isinstance(meta["num_rows"], int), f"{name}: num_rows not int"
    assert isinstance(meta["num_cols"], int), f"{name}: num_cols not int"
    assert meta["num_rows"] > 0, f"{name}: num_rows must be positive"
    assert meta["num_cols"] > 0, f"{name}: num_cols must be positive"
    assert len(meta["bbox"]) == 4, f"{name}: bbox must have 4 elements"
    assert isinstance(meta["caption"], str), f"{name}: caption not str"

    for key in ("gt_cells", "pred_cells_exact", "pred_cells_wrong", "pred_cells_wrong_structure", "pred_cells_partial"):
        cells = fixture[key]
        assert isinstance(cells, list), f"{name}.{key}: not a list"
        for cell in cells:
            _validate_cell(cell, name, key)


def _validate_cell(cell: dict[str, Any], fixture_name: str, variant: str) -> None:
    """Validate a single cell dict."""
    for field in ("row", "col", "row_span", "col_span"):
        assert field in cell, f"{fixture_name}.{variant}: missing {field}"
        assert isinstance(cell[field], int), f"{fixture_name}.{variant}: {field} not int"
        assert cell[field] >= 0, f"{fixture_name}.{variant}: {field} negative"
    assert cell["row_span"] >= 1, f"{fixture_name}.{variant}: row_span < 1"
    assert cell["col_span"] >= 1, f"{fixture_name}.{variant}: col_span < 1"
    assert "text" in cell, f"{fixture_name}.{variant}: missing text"
    assert isinstance(cell["text"], str), f"{fixture_name}.{variant}: text not str"
    assert len(cell["bbox"]) == 4, f"{fixture_name}.{variant}: bbox wrong length"
    assert 0.0 <= cell["confidence"] <= 1.0, f"{fixture_name}.{variant}: confidence out of range"


def _grid_positions(cells: list[dict[str, Any]]) -> set[tuple[int, int]]:
    """Compute the set of (row, col) positions occupied by cells."""
    occupied: set[tuple[int, int]] = set()
    for cell in cells:
        r, c = cell["row"], cell["col"]
        rs, cs = cell["row_span"], cell["col_span"]
        for dr in range(rs):
            for dc in range(cs):
                pos = (r + dr, c + dc)
                assert pos not in occupied, f"Overlapping cell at row={pos[0]}, col={pos[1]}"
                occupied.add(pos)
    return occupied


def _max_grid(cells: list[dict[str, Any]]) -> tuple[int, int]:
    """Compute (max_row+1, max_col+1) of the occupied grid."""
    positions = _grid_positions(cells)
    if not positions:
        return (0, 0)
    max_r = max(p[0] for p in positions)
    max_c = max(p[1] for p in positions)
    return (max_r + 1, max_c + 1)


# ── Structural validation (all fixtures) ────────────────────────────────────────


@pytest.mark.parametrize(
    "fixture,name",
    [
        (SIMPLE_TABLE, "SIMPLE_TABLE"),
        (TABLE_WITH_MERGED_CELLS, "TABLE_WITH_MERGED_CELLS"),
        (TABLE_NUMERIC, "TABLE_NUMERIC"),
        (TABLE_WITH_EMPTY_CELLS, "TABLE_WITH_EMPTY_CELLS"),
        (TABLE_SINGLE_ROW, "TABLE_SINGLE_ROW"),
        (TABLE_SINGLE_CELL, "TABLE_SINGLE_CELL"),
        (TABLE_IRREGULAR, "TABLE_IRREGULAR"),
        (TABLE_LARGE, "TABLE_LARGE"),
    ],
)
def test_fixture_structure(fixture: dict[str, Any], name: str) -> None:
    """All fixtures have the correct structure and valid data types."""
    _validate_fixture_structure(fixture, name)


@pytest.mark.parametrize(
    "fixture,name",
    [
        (SIMPLE_TABLE, "SIMPLE_TABLE"),
        (TABLE_WITH_MERGED_CELLS, "TABLE_WITH_MERGED_CELLS"),
        (TABLE_NUMERIC, "TABLE_NUMERIC"),
        (TABLE_WITH_EMPTY_CELLS, "TABLE_WITH_EMPTY_CELLS"),
        (TABLE_SINGLE_ROW, "TABLE_SINGLE_ROW"),
        (TABLE_SINGLE_CELL, "TABLE_SINGLE_CELL"),
        (TABLE_IRREGULAR, "TABLE_IRREGULAR"),
        (TABLE_LARGE, "TABLE_LARGE"),
    ],
)
def test_no_overlapping_cells(fixture: dict[str, Any], name: str) -> None:  # noqa: ARG001
    """GT cells occupy non-overlapping grid positions."""
    _grid_positions(fixture["gt_cells"])


# ── Metadata consistency ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fixture,name",
    [
        (SIMPLE_TABLE, "SIMPLE_TABLE"),
        (TABLE_WITH_MERGED_CELLS, "TABLE_WITH_MERGED_CELLS"),
        (TABLE_NUMERIC, "TABLE_NUMERIC"),
        (TABLE_WITH_EMPTY_CELLS, "TABLE_WITH_EMPTY_CELLS"),
        (TABLE_SINGLE_ROW, "TABLE_SINGLE_ROW"),
        (TABLE_SINGLE_CELL, "TABLE_SINGLE_CELL"),
        (TABLE_IRREGULAR, "TABLE_IRREGULAR"),
        (TABLE_LARGE, "TABLE_LARGE"),
    ],
)
def test_metadata_matches_grid(fixture: dict[str, Any], name: str) -> None:
    """table_metadata num_rows/num_cols matches the occupied grid."""
    gt_cells = fixture["gt_cells"]
    meta = fixture["table_metadata"]
    max_r, max_c = _max_grid(gt_cells)
    assert meta["num_rows"] == max_r, f"{name}: num_rows {meta['num_rows']} != grid {max_r}"
    assert meta["num_cols"] == max_c, f"{name}: num_cols {meta['num_cols']} != grid {max_c}"


# ── Exact match tests ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fixture,name",
    [
        (SIMPLE_TABLE, "SIMPLE_TABLE"),
        (TABLE_WITH_MERGED_CELLS, "TABLE_WITH_MERGED_CELLS"),
        (TABLE_NUMERIC, "TABLE_NUMERIC"),
        (TABLE_WITH_EMPTY_CELLS, "TABLE_WITH_EMPTY_CELLS"),
        (TABLE_SINGLE_ROW, "TABLE_SINGLE_ROW"),
        (TABLE_SINGLE_CELL, "TABLE_SINGLE_CELL"),
        (TABLE_IRREGULAR, "TABLE_IRREGULAR"),
        (TABLE_LARGE, "TABLE_LARGE"),
    ],
)
def test_pred_cells_exact_identical(fixture: dict[str, Any], name: str) -> None:
    """pred_cells_exact must be identical to gt_cells (exact match → GriTS=1.0)."""
    assert _cells_equal(fixture["gt_cells"], fixture["pred_cells_exact"]), (
        f"{name}: pred_cells_exact differs from gt_cells"
    )


# ── Content error tests (wrong text, same structure) ──────────────────────────


@pytest.mark.parametrize(
    "fixture,name",
    [
        (SIMPLE_TABLE, "SIMPLE_TABLE"),
        (TABLE_WITH_MERGED_CELLS, "TABLE_WITH_MERGED_CELLS"),
        (TABLE_NUMERIC, "TABLE_NUMERIC"),
        (TABLE_WITH_EMPTY_CELLS, "TABLE_WITH_EMPTY_CELLS"),
        (TABLE_SINGLE_ROW, "TABLE_SINGLE_ROW"),
        (TABLE_SINGLE_CELL, "TABLE_SINGLE_CELL"),
        (TABLE_IRREGULAR, "TABLE_IRREGULAR"),
        (TABLE_LARGE, "TABLE_LARGE"),
    ],
)
def test_pred_cells_wrong_same_structure(fixture: dict[str, Any], name: str) -> None:
    """pred_cells_wrong has the same structure (rows, cols, spans) as GT."""
    gt = fixture["gt_cells"]
    wrong = fixture["pred_cells_wrong"]
    assert len(gt) == len(wrong), f"{name}: wrong has different cell count"
    for g, w in zip(gt, wrong, strict=True):
        assert g["row"] == w["row"], f"{name}: row mismatch"
        assert g["col"] == w["col"], f"{name}: col mismatch"
        assert g["row_span"] == w["row_span"], f"{name}: row_span mismatch"
        assert g["col_span"] == w["col_span"], f"{name}: col_span mismatch"


@pytest.mark.parametrize(
    "fixture,name",
    [
        (SIMPLE_TABLE, "SIMPLE_TABLE"),
        (TABLE_WITH_MERGED_CELLS, "TABLE_WITH_MERGED_CELLS"),
        (TABLE_NUMERIC, "TABLE_NUMERIC"),
        (TABLE_WITH_EMPTY_CELLS, "TABLE_WITH_EMPTY_CELLS"),
        (TABLE_SINGLE_ROW, "TABLE_SINGLE_ROW"),
        (TABLE_SINGLE_CELL, "TABLE_SINGLE_CELL"),
        (TABLE_IRREGULAR, "TABLE_IRREGULAR"),
        (TABLE_LARGE, "TABLE_LARGE"),
    ],
)
def test_pred_cells_wrong_has_diff_text(fixture: dict[str, Any], name: str) -> None:
    """pred_cells_wrong has at least one cell with different text than GT."""
    gt = fixture["gt_cells"]
    wrong = fixture["pred_cells_wrong"]
    assert not _cells_equal(gt, wrong), f"{name}: pred_cells_wrong identical to gt_cells"
    # Verify difference is textual, not structural
    texts_differ = any(g["text"] != w["text"] for g, w in zip(gt, wrong, strict=True))
    assert texts_differ, f"{name}: pred_cells_wrong text identical to gt_cells"


# ── Structure error tests ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fixture,name",
    [
        (SIMPLE_TABLE, "SIMPLE_TABLE"),
        (TABLE_WITH_MERGED_CELLS, "TABLE_WITH_MERGED_CELLS"),
        (TABLE_NUMERIC, "TABLE_NUMERIC"),
        (TABLE_WITH_EMPTY_CELLS, "TABLE_WITH_EMPTY_CELLS"),
        (TABLE_SINGLE_ROW, "TABLE_SINGLE_ROW"),
        (TABLE_SINGLE_CELL, "TABLE_SINGLE_CELL"),
        (TABLE_IRREGULAR, "TABLE_IRREGULAR"),
        (TABLE_LARGE, "TABLE_LARGE"),
    ],
)
def test_pred_cells_wrong_structure_differs(fixture: dict[str, Any], name: str) -> None:
    """pred_cells_wrong_structure has genuinely different topology from GT."""
    gt = fixture["gt_cells"]
    ws = fixture["pred_cells_wrong_structure"]
    # Must differ in count or at least one structural field
    if len(gt) == len(ws):
        differs = any(
            g["row"] != w["row"]
            or g["col"] != w["col"]
            or g["row_span"] != w["row_span"]
            or g["col_span"] != w["col_span"]
            for g, w in zip(gt, ws, strict=True)
        )
        assert differs, f"{name}: pred_cells_wrong_structure structurally same as gt_cells"
    else:
        # Different cell count is sufficient
        pass  # already different


# ── Partial match tests ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fixture,name",
    [
        (SIMPLE_TABLE, "SIMPLE_TABLE"),
        (TABLE_WITH_MERGED_CELLS, "TABLE_WITH_MERGED_CELLS"),
        (TABLE_NUMERIC, "TABLE_NUMERIC"),
        (TABLE_WITH_EMPTY_CELLS, "TABLE_WITH_EMPTY_CELLS"),
        (TABLE_SINGLE_ROW, "TABLE_SINGLE_ROW"),
        (TABLE_SINGLE_CELL, "TABLE_SINGLE_CELL"),
        (TABLE_IRREGULAR, "TABLE_IRREGULAR"),
        (TABLE_LARGE, "TABLE_LARGE"),
    ],
)
def test_pred_cells_partial_not_exact(fixture: dict[str, Any], name: str) -> None:
    """pred_cells_partial is not identical to gt_cells."""
    assert not _cells_equal(fixture["gt_cells"], fixture["pred_cells_partial"]), (
        f"{name}: pred_cells_partial identical to gt_cells"
    )


    @pytest.mark.parametrize(
        "fixture,name",
        [
            (SIMPLE_TABLE, "SIMPLE_TABLE"),
            (TABLE_NUMERIC, "TABLE_NUMERIC"),
            (TABLE_WITH_EMPTY_CELLS, "TABLE_WITH_EMPTY_CELLS"),
            (TABLE_SINGLE_ROW, "TABLE_SINGLE_ROW"),
            (TABLE_LARGE, "TABLE_LARGE"),
        ],
    )
    def test_pred_cells_partial_some_correct(fixture: dict[str, Any], name: str) -> None:
        """pred_cells_partial has at least one matching cell and at least one differing cell."""
        gt = fixture["gt_cells"]
        partial = fixture["pred_cells_partial"]
        # Use the shorter list for zipping if lengths differ
        min_len = min(len(gt), len(partial))
        matches = sum(
            1 for i in range(min_len) if gt[i] == partial[i]
        )
        assert 0 < matches < min_len or (len(gt) != len(partial) and matches > 0), (
            f"{name}: pred_cells_partial has {matches} matches out of {min_len}"
        )


# ── Fixture-specific tests ─────────────────────────────────────────────────────


def test_simple_table_exact_match() -> None:
    """GriTS_Top=1.0, GriTS_Con=1.0 for pred_cells_exact."""
    fixture = SIMPLE_TABLE
    assert _cells_equal(fixture["gt_cells"], fixture["pred_cells_exact"])


def test_simple_table_content_error() -> None:
    """GriTS_Top=1.0, GriTS_Con<1.0 for pred_cells_wrong."""
    fixture = SIMPLE_TABLE
    gt = fixture["gt_cells"]
    wrong = fixture["pred_cells_wrong"]
    # Same structure
    assert len(gt) == len(wrong)
    for g, w in zip(gt, wrong, strict=True):
        assert g["row"] == w["row"]
        assert g["col"] == w["col"]
        assert g["row_span"] == w["row_span"]
        assert g["col_span"] == w["col_span"]
    # Different text
    assert not _cells_equal(gt, wrong)


def test_simple_table_structure_error() -> None:
    """GriTS_Top<1.0 for pred_cells_wrong_structure."""
    fixture = SIMPLE_TABLE
    gt = fixture["gt_cells"]
    ws = fixture["pred_cells_wrong_structure"]
    # Different cell count (8 vs 9) due to merged header
    assert len(gt) != len(ws)
    # Grid dimensions should still match metadata
    meta = fixture["table_metadata"]
    max_r, max_c = _max_grid(ws)
    assert max_r <= meta["num_rows"]
    assert max_c <= meta["num_cols"]


def test_merged_cells_exact() -> None:
    """Merged cell fixture has correct spans."""
    fixture = TABLE_WITH_MERGED_CELLS
    gt = fixture["gt_cells"]
    # Find the merged cell
    merged = [c for c in gt if c["row_span"] > 1 or c["col_span"] > 1]
    assert len(merged) >= 1, "No merged cells found in TABLE_WITH_MERGED_CELLS"
    # Verify the specific merged cell
    premium_cell = next(c for c in gt if c["row"] == 1 and c["col"] == 2)
    assert premium_cell["row_span"] == 2
    assert premium_cell["text"] == "$199"
    # No overlapping positions
    _grid_positions(gt)
    # All grid positions should be covered by metadata
    meta = fixture["table_metadata"]
    assert meta["num_rows"] == 4
    assert meta["num_cols"] == 3


def test_numeric_table_content() -> None:
    """Numeric table cell content is numeric strings."""
    fixture = TABLE_NUMERIC
    gt = fixture["gt_cells"]
    # Header row validates as is, data rows should have numeric values in columns 1-3
    for cell in gt:
        if cell["row"] > 0 and cell["col"] > 0:
            # Should be parseable as integer
            int(cell["text"])  # raises if not numeric
    # Verify pred_cells_exact matches
    assert _cells_equal(gt, fixture["pred_cells_exact"])


def test_empty_cells() -> None:
    """Empty cells fixture contains empty strings."""
    fixture = TABLE_WITH_EMPTY_CELLS
    gt = fixture["gt_cells"]
    empty_cells = [c for c in gt if c["text"] == ""]
    assert len(empty_cells) >= 2, "Expected at least 2 empty cells"
    # Verify the empty cell positions
    assert any(c["row"] == 1 and c["col"] == 2 for c in empty_cells), "Missing (1,2) empty cell"
    assert any(c["row"] == 2 and c["col"] == 1 for c in empty_cells), "Missing (2,1) empty cell"
    # pred_cells_wrong replaces empty cells with data
    wrong = fixture["pred_cells_wrong"]
    wrong_empty = [c for c in wrong if c["text"] == ""]
    assert len(wrong_empty) == 0, "pred_cells_wrong should have no empty cells"


def test_single_row_table() -> None:
    """Header-only table has 1 row with metadata matching."""
    fixture = TABLE_SINGLE_ROW
    meta = fixture["table_metadata"]
    assert meta["num_rows"] == 1
    assert meta["num_cols"] == 5
    for cell in fixture["gt_cells"]:
        assert cell["row"] == 0
        assert cell["col_span"] == 1
        assert cell["row_span"] == 1


def test_single_cell_table() -> None:
    """Single cell table is trivial 1×1."""
    fixture = TABLE_SINGLE_CELL
    meta = fixture["table_metadata"]
    assert meta["num_rows"] == 1
    assert meta["num_cols"] == 1
    assert len(fixture["gt_cells"]) == 1
    cell = fixture["gt_cells"][0]
    assert cell["row"] == 0
    assert cell["col"] == 0
    assert cell["row_span"] == 1
    assert cell["col_span"] == 1
    assert len(cell["text"]) > 0


def test_large_table() -> None:
    """Large table fixture loads correctly with full dimensions."""
    fixture = TABLE_LARGE
    meta = fixture["table_metadata"]
    assert meta["num_rows"] == 10
    assert meta["num_cols"] == 8
    assert len(fixture["gt_cells"]) == 10 * 8  # 80 cells
    # All cells have the expected grid positions
    positions = _grid_positions(fixture["gt_cells"])
    assert len(positions) == 80
    # Every GT cell has non-empty text
    for cell in fixture["gt_cells"]:
        assert len(cell["text"]) > 0, f"Empty cell at ({cell['row']},{cell['col']})"


def test_large_table_metadata_bbox() -> None:
    """Large table bbox fits within US Letter page bounds (612×792)."""
    fixture = TABLE_LARGE
    bbox = fixture["table_metadata"]["bbox"]
    assert bbox[0] >= 0  # x0
    assert bbox[1] >= 0  # y0
    assert bbox[2] <= 612  # x1 fits within page width
    assert bbox[3] <= 792  # y1 fits within page height


def test_all_bboxes_within_page() -> None:
    """All cell bboxes across all fixtures fit within US Letter page bounds."""
    all_fixtures = [
        SIMPLE_TABLE,
        TABLE_WITH_MERGED_CELLS,
        TABLE_NUMERIC,
        TABLE_WITH_EMPTY_CELLS,
        TABLE_SINGLE_ROW,
        TABLE_SINGLE_CELL,
        TABLE_IRREGULAR,
        TABLE_LARGE,
    ]
    variants = ("gt_cells", "pred_cells_exact", "pred_cells_wrong",
                "pred_cells_wrong_structure", "pred_cells_partial")
    for fixture in all_fixtures:
        for variant in variants:
            for cell in fixture[variant]:
                bbox = cell["bbox"]
                assert bbox[0] >= 0 and bbox[1] >= 0, "Negative bbox origin"
                assert bbox[2] > bbox[0], "x1 not > x0"
                assert bbox[3] > bbox[1], "y1 not > y0"
                assert bbox[2] <= 612, f"x1 exceeds page width: {bbox[2]}"
                assert bbox[3] <= 792, f"y1 exceeds page height: {bbox[3]}"


def test_fixtures_importable() -> None:
    """Fixtures can be imported by other test modules via the package."""
    from backend.tests.fixtures import SIMPLE_TABLE as _st  # noqa: F811, N811
    from backend.tests.fixtures import TABLE_LARGE as _tl  # noqa: F811, N811
    assert _st is SIMPLE_TABLE
    assert _tl is TABLE_LARGE
