"""Table alignment test fixtures for OCRScore.

These are synthetic tables with known ground truth that can be used to test
the GriTS scoring pipeline and the multi-engine compare endpoint.
"""

from backend.tests.fixtures.tables import (
    SIMPLE_TABLE,
    TABLE_IRREGULAR,
    TABLE_LARGE,
    TABLE_NUMERIC,
    TABLE_SINGLE_CELL,
    TABLE_SINGLE_ROW,
    TABLE_WITH_EMPTY_CELLS,
    TABLE_WITH_MERGED_CELLS,
)

__all__ = [
    "SIMPLE_TABLE",
    "TABLE_WITH_MERGED_CELLS",
    "TABLE_NUMERIC",
    "TABLE_WITH_EMPTY_CELLS",
    "TABLE_SINGLE_ROW",
    "TABLE_SINGLE_CELL",
    "TABLE_IRREGULAR",
    "TABLE_LARGE",
]
