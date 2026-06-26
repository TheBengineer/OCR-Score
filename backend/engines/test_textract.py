"""Tests for the AWS Textract OCR engine module.

The test suite covers:
- ABC conformance (``OCREngine`` subclass, attributes, methods).
- Config schema validation.
- Raw output normalisation (``normalize()`` with mock data).
- Hierarchy construction (PAGE → LINE → WORD, TABLE → CELL).
- Confidence value normalisation (0‑100 → 0.0‑1.0).
- Bounding box coordinate transformation (normalised → page-space points).
- Reading order preservation via CHILD relationship ordering.
- Edge cases (empty pages, missing blocks, KEY_VALUE_SET, SELECTION_ELEMENT).
- Polygon → bounding-box merge helper.

All tests are self-contained — they do **not** require AWS credentials, a
Textract client, or any external service.  Mock fixtures use plain dicts
that mirror the Textract API response structure.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.engine.base import OCREngine
from backend.engine.normalized_schema import (
    NormalizedDocument,
    TextBlock,
)

# Module to test
from backend.engines.textract import (
    TextractEngine,
    _bbox_to_points,
    _build_block_index,
    _get_child_ids,
    _normalize_confidence,
    _polygon_to_bbox,
)

# ── Mock data factory ───────────────────────────────────────────────────────


def _make_block(
    block_id: str = "id1",
    block_type: str = "WORD",
    text: str = "",
    confidence: float = 99.0,
    left: float = 0.0,
    top: float = 0.0,
    width: float = 0.1,
    height: float = 0.1,
    page: int = 1,
    child_ids: list[str] | None = None,
    entity_types: list[str] | None = None,
    row_index: int | None = None,
    column_index: int | None = None,
    row_span: int | None = None,
    column_span: int | None = None,
    selection_status: str | None = None,
    text_types: list[str] | None = None,
) -> dict[str, Any]:
    """Build a Textract-like block dict.

    Args:
        block_id: Unique block identifier.
        block_type: Textract ``BlockType`` (``PAGE``, ``LINE``, ``WORD``,
            ``TABLE``, ``CELL``, ``KEY_VALUE_SET``, ``SELECTION_ELEMENT``).
        text: Block text content (for ``WORD``, ``LINE``, ``CELL``).
        confidence: Confidence value (0‑100).
        left / top / width / height: Normalised bounding box (0‑1).
        page: Page number.
        child_ids: IDs for ``CHILD`` relationship entries.
        entity_types: Entity type labels (e.g. ``["KEY"]``, ``["VALUE"]``).
        row_index / column_index: 1‑based cell position (for ``CELL``).
        row_span / column_span: Cell span (for ``CELL``).
        selection_status: ``"SELECTED"`` or ``"NOT_SELECTED"``.
        text_types: E.g. ``["HANDWRITING"]``.
    """
    block: dict[str, Any] = {
        "Id": block_id,
        "BlockType": block_type,
        "Confidence": confidence,
        "Page": page,
        "Geometry": {
            "BoundingBox": {
                "Width": width,
                "Height": height,
                "Left": left,
                "Top": top,
            },
            "Polygon": [
                {"X": left, "Y": top},
                {"X": left + width, "Y": top},
                {"X": left + width, "Y": top + height},
                {"X": left, "Y": top + height},
            ],
        },
    }
    if text:
        block["Text"] = text
    if child_ids is not None:
        block["Relationships"] = [{"Type": "CHILD", "Ids": child_ids}]
    if entity_types is not None:
        block["EntityTypes"] = entity_types
    if row_index is not None:
        block["RowIndex"] = row_index
    if column_index is not None:
        block["ColumnIndex"] = column_index
    if row_span is not None:
        block["RowSpan"] = row_span
    if column_span is not None:
        block["ColumnSpan"] = column_span
    if selection_status is not None:
        block["SelectionStatus"] = selection_status
    if text_types is not None:
        block["TextTypes"] = text_types
    return block


def _make_mock_raw(
    blocks: list[dict[str, Any]] | None = None,
    engine_id: str = "aws-textract",
    engine_version: str = "1.34.0",
    page_count: int = 1,
) -> dict[str, Any]:
    """Build a minimal raw output dict as returned by ``process_pdf()``.

    Args:
        blocks: List of Textract-like block dicts.
        engine_id: Engine identifier.
        engine_version: Engine version string.
        page_count: Number of pages in the document.

    Returns:
        A dict matching the structure returned by ``TextractEngine.process_pdf()``.
    """
    return {
        "blocks": blocks or [],
        "document_metadata": {"Pages": page_count},
        "engine_id": engine_id,
        "engine_version": engine_version,
        "config_snapshot": {
            "region": "us-east-1",
            "timeout_seconds": 300,
        },
        "page_count": page_count,
    }


# ── Mock blocks for basic "Hello World" test ────────────────────────────────

HELLO_WORLD_PAGE = _make_block(
    "page1", "PAGE", page=1, child_ids=["line1"],
    width=8.5, height=11.0,
)

HELLO_WORLD_LINE = _make_block(
    "line1", "LINE", text="Hello World", confidence=99.0,
    left=0.1, top=0.1, width=0.8, height=0.1, page=1,
    child_ids=["word1", "word2"],
)

HELLO_WORLD_WORDS = [
    _make_block(
        "word1", "WORD", text="Hello", confidence=98.0,
        left=0.1, top=0.1, width=0.2, height=0.1, page=1,
    ),
    _make_block(
        "word2", "WORD", text="World", confidence=97.0,
        left=0.35, top=0.1, width=0.55, height=0.1, page=1,
    ),
]

HELLO_WORLD_BLOCKS = [HELLO_WORLD_PAGE, HELLO_WORLD_LINE, *HELLO_WORLD_WORDS]


# ── Mock blocks for table tests ─────────────────────────────────────────────

TABLE_PAGE = _make_block(
    "page1", "PAGE", page=1, child_ids=["table1"],
    width=8.5, height=11.0,
)

TABLE_BLOCK = _make_block(
    "table1", "TABLE", confidence=98.0,
    left=0.1, top=0.3, width=0.8, height=0.4, page=1,
    child_ids=["cell1", "cell2", "cell3", "cell4"],
)

TABLE_CELLS = [
    _make_block(
        "cell1", "CELL", text="A", confidence=95.0,
        left=0.1, top=0.3, width=0.35, height=0.1, page=1,
        row_index=1, column_index=1, row_span=1, column_span=1,
    ),
    _make_block(
        "cell2", "CELL", text="B", confidence=94.0,
        left=0.55, top=0.3, width=0.35, height=0.1, page=1,
        row_index=1, column_index=2, row_span=1, column_span=1,
    ),
    _make_block(
        "cell3", "CELL", text="C", confidence=93.0,
        left=0.1, top=0.5, width=0.35, height=0.1, page=1,
        row_index=2, column_index=1, row_span=1, column_span=1,
    ),
    _make_block(
        "cell4", "CELL", text="D", confidence=92.0,
        left=0.55, top=0.5, width=0.35, height=0.1, page=1,
        row_index=2, column_index=2, row_span=1, column_span=1,
    ),
]

TABLE_BLOCKS = [TABLE_PAGE, TABLE_BLOCK, *TABLE_CELLS]


# ── Mock blocks for form tests ──────────────────────────────────────────────

FORM_PAGE = _make_block(
    "page1", "PAGE", page=1, child_ids=["kv1", "kv2"],
    width=8.5, height=11.0,
)

# Key block with a VALUE relationship.
FORM_KEY = _make_block(
    "kv1", "KEY_VALUE_SET", confidence=99.0,
    left=0.1, top=0.1, width=0.2, height=0.05, page=1,
    entity_types=["KEY"],
    child_ids=["key_word1", "key_word2"],
)

KEY_WORDS = [
    _make_block(
        "key_word1", "WORD", text="Name:", confidence=98.0,
        left=0.1, top=0.1, width=0.1, height=0.05, page=1,
    ),
]

# Value block (referenced by KEY_VALUE_SET but also a top-level child).
FORM_VALUE = _make_block(
    "kv2", "KEY_VALUE_SET", confidence=99.0,
    left=0.35, top=0.1, width=0.3, height=0.05, page=1,
    entity_types=["VALUE"],
    child_ids=["val_word1"],
)

VALUE_WORDS = [
    _make_block(
        "val_word1", "WORD", text="John", confidence=97.0,
        left=0.35, top=0.1, width=0.3, height=0.05, page=1,
    ),
]

FORM_BLOCKS = [FORM_PAGE, FORM_KEY, *KEY_WORDS, FORM_VALUE, *VALUE_WORDS]


# ── Mock blocks for reading-order test ──────────────────────────────────────

ORDER_PAGE = _make_block(
    "page1", "PAGE", page=1, child_ids=["line2", "line1"],
    width=8.5, height=11.0,
)

ORDER_LINE_2 = _make_block(
    "line2", "LINE", text="Second line in PAGE children",
    confidence=98.0,
    left=0.1, top=0.5, width=0.8, height=0.1, page=1,
    child_ids=["word3"],
)

ORDER_LINE_1 = _make_block(
    "line1", "LINE", text="First line in PAGE children",
    confidence=99.0,
    left=0.1, top=0.1, width=0.8, height=0.1, page=1,
    child_ids=["word1"],
)

ORDER_WORDS = [
    _make_block(
        "word1", "WORD", text="First", confidence=99.0,
        left=0.1, top=0.1, width=0.3, height=0.1, page=1,
    ),
    _make_block(
        "word3", "WORD", text="Second", confidence=98.0,
        left=0.1, top=0.5, width=0.3, height=0.1, page=1,
    ),
]

ORDER_BLOCKS = [ORDER_PAGE, ORDER_LINE_1, ORDER_LINE_2, *ORDER_WORDS]


# ── Mock blocks for selection-element test ──────────────────────────────────

SELECTION_PAGE = _make_block(
    "page1", "PAGE", page=1, child_ids=["sel1", "sel2"],
    width=8.5, height=11.0,
)

SELECTION_ELEMENTS = [
    _make_block(
        "sel1", "SELECTION_ELEMENT", confidence=99.0,
        left=0.1, top=0.1, width=0.03, height=0.03, page=1,
        selection_status="SELECTED",
    ),
    _make_block(
        "sel2", "SELECTION_ELEMENT", confidence=98.0,
        left=0.1, top=0.2, width=0.03, height=0.03, page=1,
        selection_status="NOT_SELECTED",
    ),
]

SELECTION_BLOCKS = [SELECTION_PAGE, *SELECTION_ELEMENTS]


# ── Mock blocks for handwriting test ────────────────────────────────────────

HANDWRITING_PAGE = _make_block(
    "page1", "PAGE", page=1, child_ids=["hline1"],
    width=8.5, height=11.0,
)

HANDWRITING_LINE = _make_block(
    "hline1", "LINE", text="Handwritten note", confidence=85.0,
    left=0.1, top=0.1, width=0.8, height=0.1, page=1,
    child_ids=["hword1"],
)

HANDWRITING_WORD = _make_block(
    "hword1", "WORD", text="Handwritten", confidence=85.0,
    left=0.1, top=0.1, width=0.5, height=0.1, page=1,
    text_types=["HANDWRITING"],
)

HANDWRITING_BLOCKS = [HANDWRITING_PAGE, HANDWRITING_LINE, HANDWRITING_WORD]


# ── Helper unit tests ───────────────────────────────────────────────────────


class TestNormalizeConfidence:
    """Verify ``_normalize_confidence()`` handles all Textract confidence cases."""

    def test_positive_value(self) -> None:
        """Given a confidence 0‑100, When normalising, Then it maps to 0‑1."""
        assert _normalize_confidence(100.0) == 1.0
        assert _normalize_confidence(99.5) == 0.995
        assert _normalize_confidence(50.0) == 0.5
        assert _normalize_confidence(0.0) == 0.0

    def test_none(self) -> None:
        """Given None, When normalising, Then 0.0 is returned."""
        assert _normalize_confidence(None) == 0.0

    def test_negative(self) -> None:
        """Given a negative value, When normalising, Then 0.0 is returned."""
        assert _normalize_confidence(-1.0) == 0.0
        assert _normalize_confidence(-100.0) == 0.0

    def test_unparseable(self) -> None:
        """Given an unparseable value, When normalising, Then 0.0 is returned."""
        assert _normalize_confidence("invalid") == 0.0
        assert _normalize_confidence({}) == 0.0

    def test_clamps_above_100(self) -> None:
        """Given confidence > 100, When normalising, Then it is clamped to 1.0."""
        assert _normalize_confidence(150.0) == 1.0
        assert _normalize_confidence(200.0) == 1.0


class TestBboxToPoints:
    """Verify ``_bbox_to_points()`` coordinate conversion."""

    def test_full_page(self) -> None:
        """Given a full-page bbox on US Letter, When converting, Then it spans the page."""
        bbox = {"Left": 0.0, "Top": 0.0, "Width": 1.0, "Height": 1.0}
        result = _bbox_to_points(bbox, 612.0, 792.0)
        assert result == pytest.approx([0.0, 0.0, 612.0, 792.0])

    def test_partial_region(self) -> None:
        """Given normalised coords (0.1, 0.1, 0.3, 0.2), When converting, Then correct points."""
        bbox = {"Left": 0.1, "Top": 0.1, "Width": 0.3, "Height": 0.2}
        result = _bbox_to_points(bbox, 612.0, 792.0)
        assert result == pytest.approx([61.2, 79.2, 244.8, 237.6])

    def test_empty_bbox(self) -> None:
        """Given None or empty bbox, When converting, Then zeros are returned."""
        assert _bbox_to_points(None, 612.0, 792.0) == [0.0, 0.0, 0.0, 0.0]
        assert _bbox_to_points({}, 612.0, 792.0) == [0.0, 0.0, 0.0, 0.0]


class TestPolygonToBbox:
    """Verify ``_polygon_to_bbox()`` polygon → bounding-box conversion."""

    def test_basic_polygon(self) -> None:
        """Given a rectangular polygon, When converting, Then tight bbox is returned."""
        polygon = [
            {"X": 0.1, "Y": 0.2},
            {"X": 0.8, "Y": 0.2},
            {"X": 0.8, "Y": 0.6},
            {"X": 0.1, "Y": 0.6},
        ]
        result = _polygon_to_bbox(polygon, 612.0, 792.0)
        assert result == pytest.approx([61.2, 158.4, 489.6, 475.2])

    def test_rotated_polygon(self) -> None:
        """Given a rotated (non-axis-aligned) polygon, When converting, Then tight bbox is returned."""
        polygon = [
            {"X": 0.2, "Y": 0.1},
            {"X": 0.8, "Y": 0.3},
            {"X": 0.7, "Y": 0.7},
            {"X": 0.1, "Y": 0.5},
        ]
        result = _polygon_to_bbox(polygon, 612.0, 792.0)
        assert result == pytest.approx([61.2, 79.2, 489.6, 554.4])

    def test_empty_polygon(self) -> None:
        """Given None or empty polygon, When converting, Then zeros are returned."""
        assert _polygon_to_bbox(None, 612.0, 792.0) == [0.0, 0.0, 0.0, 0.0]
        assert _polygon_to_bbox([], 612.0, 792.0) == [0.0, 0.0, 0.0, 0.0]


class TestBuildBlockIndex:
    """Verify ``_build_block_index()`` index creation."""

    def test_basic_index(self) -> None:
        """Given a list of blocks, When indexing, Then a correct ID→block map is returned."""
        blocks = [
            {"Id": "a", "BlockType": "WORD"},
            {"Id": "b", "BlockType": "LINE"},
        ]
        index = _build_block_index(blocks)
        assert index["a"]["BlockType"] == "WORD"
        assert index["b"]["BlockType"] == "LINE"
        assert len(index) == 2

    def test_skips_blocks_without_id(self) -> None:
        """Given a block without Id, When indexing, Then it is skipped."""
        blocks = [
            {"BlockType": "WORD"},  # no Id
            {"Id": "b", "BlockType": "LINE"},
        ]
        index = _build_block_index(blocks)
        assert len(index) == 1
        assert "b" in index


class TestGetChildIds:
    """Verify ``_get_child_ids()`` relationship extraction."""

    def test_basic_child_relationship(self) -> None:
        """Given a block with CHILD relationship, When extracting, Then child IDs are returned."""
        block = {"Relationships": [{"Type": "CHILD", "Ids": ["a", "b"]}]}
        assert _get_child_ids(block) == ["a", "b"]

    def test_no_relationships(self) -> None:
        """Given a block with no Relationships key, When extracting, Then empty list is returned."""
        assert _get_child_ids({}) == []

    def test_other_rel_type(self) -> None:
        """Given a block with a non-CHILD relationship, When extracting CHILD, Then empty list."""
        block = {"Relationships": [{"Type": "VALUE", "Ids": ["v1"]}]}
        assert _get_child_ids(block) == []

    def test_custom_rel_type(self) -> None:
        """Given a block with VALUE relationship, When extracting VALUE, Then IDs are returned."""
        block = {"Relationships": [{"Type": "VALUE", "Ids": ["v1"]}]}
        assert _get_child_ids(block, rel_type="VALUE") == ["v1"]


# ── ABC conformance ─────────────────────────────────────────────────────────


class TestTextractABCConformance:
    """Verify that ``TextractEngine`` satisfies the ``OCREngine`` ABC."""

    def test_engine_is_ocrenigne_subclass(self) -> None:
        """Given TextractEngine, When checking MRO, Then it is an OCREngine subclass."""
        assert issubclass(TextractEngine, OCREngine)

    def test_engine_conforms_to_abc(self) -> None:
        """Given a TextractEngine instance, When inspecting interface, Then all ABC members exist."""
        engine = TextractEngine()

        # -- class-level attributes
        assert engine.engine_id == "aws-textract"
        assert engine.display_name == "AWS Textract"
        assert isinstance(engine.version, str)
        assert len(engine.version) > 0

        # -- required methods
        assert callable(engine.get_config_schema)
        assert callable(engine.normalize)

        # process_pdf must be a coroutine
        import asyncio

        assert asyncio.iscoroutinefunction(engine.process_pdf)


# ── Config schema ───────────────────────────────────────────────────────────


class TestTextractConfigSchema:
    """Verify ``get_config_schema()`` returns valid JSON Schema."""

    def test_schema_structure(self) -> None:
        """Given a TextractEngine, When calling get_config_schema(), Then it returns valid schema."""
        engine = TextractEngine()
        schema = engine.get_config_schema()

        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert schema["required"] == []

    def test_schema_has_expected_keys(self) -> None:
        """Given a TextractEngine, When inspecting schema properties, Then all keys exist."""
        engine = TextractEngine()
        props = engine.get_config_schema()["properties"]

        assert "access_key_id" in props
        assert props["access_key_id"]["type"] == "string"

        assert "secret_access_key" in props
        assert props["secret_access_key"]["type"] == "string"

        assert "region" in props
        assert props["region"]["type"] == "string"
        assert props["region"]["default"] == "us-east-1"

        assert "timeout_seconds" in props
        assert props["timeout_seconds"]["type"] == "integer"
        assert props["timeout_seconds"]["default"] == 300
        assert props["timeout_seconds"]["minimum"] == 30
        assert props["timeout_seconds"]["maximum"] == 3600


# ── Confidence parsing ──────────────────────────────────────────────────────


class TestTextractConfidenceParsing:
    """Verify confidence values normalise correctly through ``normalize()``."""

    def test_confidence_parsing(self) -> None:
        """Given blocks with various confidence values, When normalizing, Then 0‑1 range is enforced."""
        word_with_100 = _make_block(
            "w100", "WORD", text="Full", confidence=100.0,
            left=0.1, top=0.1, width=0.2, height=0.1, page=1,
        )
        word_with_50 = _make_block(
            "w050", "WORD", text="Half", confidence=50.0,
            left=0.1, top=0.2, width=0.2, height=0.1, page=1,
        )
        word_with_0 = _make_block(
            "w000", "WORD", text="Zero", confidence=0.0,
            left=0.1, top=0.3, width=0.2, height=0.1, page=1,
        )

        line_full = _make_block(
            "l100", "LINE", text="Full", confidence=99.0,
            left=0.1, top=0.1, width=0.8, height=0.1, page=1,
            child_ids=["w100"],
        )
        line_half = _make_block(
            "l050", "LINE", text="Half", confidence=99.0,
            left=0.1, top=0.2, width=0.8, height=0.1, page=1,
            child_ids=["w050"],
        )
        line_zero = _make_block(
            "l000", "LINE", text="Zero", confidence=99.0,
            left=0.1, top=0.3, width=0.8, height=0.1, page=1,
            child_ids=["w000"],
        )

        page = _make_block(
            "page1", "PAGE", page=1, child_ids=["l100", "l050", "l000"],
            width=8.5, height=11.0,
        )

        blocks = [page, line_full, line_half, line_zero,
                  word_with_100, word_with_50, word_with_0]
        raw = _make_mock_raw(blocks=blocks)
        result = TextractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        page_out = validated.pages[0]

        assert len(page_out.blocks) == 3

        # Word confidences: 100.0 → 1.0, 50.0 → 0.5, 0.0 → 0.0
        assert page_out.blocks[0].lines[0].words[0].confidence == pytest.approx(1.0)
        assert page_out.blocks[1].lines[0].words[0].confidence == pytest.approx(0.5)
        assert page_out.blocks[2].lines[0].words[0].confidence == pytest.approx(0.0)

        # Characters inherit word confidence
        for word in page_out.blocks[0].lines[0].words:
            for ch in word.chars:
                assert ch.confidence == pytest.approx(1.0)


# ── Coordinate normalisation ────────────────────────────────────────────────


class TestTextractCoordinateNormalization:
    """Verify coordinate normalisation from normalised coords to page-space points."""

    def test_coordinate_normalization(self) -> None:
        """Given normalised coords at US Letter (612×792 pt), When normalizing, Then bboxes are in points.

        Textract normalised coords are scaled by page dimensions:
            point_x = normalised_x * page_width_inches * 72
            point_y = normalised_y * page_height_inches * 72
        """
        word = _make_block(
            "w1", "WORD", text="Test", confidence=98.0,
            left=0.1, top=0.1, width=0.2, height=0.1, page=1,
        )
        line = _make_block(
            "l1", "LINE", text="Test", confidence=99.0,
            left=0.1, top=0.1, width=0.2, height=0.1, page=1,
            child_ids=["w1"],
        )
        page = _make_block(
            "p1", "PAGE", page=1, child_ids=["l1"],
            width=8.5, height=11.0,
        )

        raw = _make_mock_raw(blocks=[page, line, word])
        result = TextractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        page_out = validated.pages[0]

        # Page dimensions: 8.5 in × 72 = 612 pts, 11 in × 72 = 792 pts
        assert page_out.width == pytest.approx(612.0)
        assert page_out.height == pytest.approx(792.0)

        # Word bbox: normalised (0.1, 0.1, 0.2+0.1=0.3, 0.1+0.1=0.2)
        # → points: [0.1*612, 0.1*792, 0.3*612, 0.2*792]
        # = [61.2, 79.2, 183.6, 158.4]
        word_out = page_out.blocks[0].lines[0].words[0]
        assert word_out.bbox == pytest.approx([61.2, 79.2, 183.6, 158.4])

        # Character bboxes should be synthesised within the word bbox
        assert len(word_out.chars) == 4  # "Test"
        char_width = (183.6 - 61.2) / 4  # = 30.6
        assert word_out.chars[0].bbox == pytest.approx(
            [61.2, 79.2, 61.2 + char_width, 158.4]
        )
        assert word_out.chars[3].bbox == pytest.approx(
            [61.2 + 3 * char_width, 79.2, 61.2 + 4 * char_width, 158.4]
        )


# ── Normalize output ────────────────────────────────────────────────────────


class TestTextractNormalizeBasic:
    """Verify ``normalize()`` produces correct ``NormalizedDocument`` structure."""

    def test_empty_response(self) -> None:
        """Given an empty blocks list, When normalizing, Then a valid empty document is produced."""
        raw = _make_mock_raw(blocks=[])
        result = TextractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert len(validated.pages) == 1
        assert validated.pages[0].page_number == 1
        assert validated.pages[0].blocks == []
        assert validated.pages[0].tables == []

    def test_basic_hierarchy(self) -> None:
        """Given a page with a LINE containing two WORDS, When normalizing, Then hierarchy is correct."""
        raw = _make_mock_raw(blocks=HELLO_WORLD_BLOCKS)
        result = TextractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert len(validated.pages) == 1
        page_out = validated.pages[0]

        assert page_out.page_number == 1
        assert page_out.width == pytest.approx(612.0)
        assert page_out.height == pytest.approx(792.0)

        # Should have one block with one line containing two words
        assert len(page_out.blocks) == 1
        block = page_out.blocks[0]
        assert isinstance(block, TextBlock)
        assert block.type == "text"
        assert 0.0 <= block.confidence <= 1.0
        assert block.order == 0

        assert len(block.lines) == 1
        line = block.lines[0]
        assert line.text == "Hello World"
        assert 0.0 <= line.confidence <= 1.0
        assert line.order == 0

        assert len(line.words) == 2
        assert line.words[0].text == "Hello"
        assert line.words[1].text == "World"
        assert line.words[0].order == 0
        assert line.words[1].order == 1

        # Words should have synthesised characters
        for word in line.words:
            assert len(word.chars) == len(word.text), (
                f"Word '{word.text}' has {len(word.chars)} chars, "
                f"expected {len(word.text)}"
            )
            for char_entry in word.chars:
                assert len(char_entry.char) == 1
                assert 0.0 <= char_entry.confidence <= 1.0
                assert char_entry.order >= 0

    def test_metadata(self) -> None:
        """Given mock raw data, When normalizing, Then engine metadata is preserved."""
        raw = _make_mock_raw(
            blocks=HELLO_WORLD_BLOCKS,
            engine_id="aws-textract",
            engine_version="1.34.0",
        )
        result = TextractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert validated.engine_id == "aws-textract"
        assert validated.engine_version == "1.34.0"
        assert validated.config_snapshot["region"] == "us-east-1"

    def test_empty_blocks_per_page(self) -> None:
        """Given a PAGE block with no CHILD relationships, When normalizing, Then empty page is returned."""
        page = _make_block(
            "p1", "PAGE", page=1, child_ids=[],
            width=8.5, height=11.0,
        )
        raw = _make_mock_raw(blocks=[page])
        result = TextractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert len(validated.pages) == 1
        assert validated.pages[0].blocks == []
        assert validated.pages[0].tables == []


class TestTextractNormalizeTables:
    """Verify table extraction from Textract TABLE blocks."""

    def test_normalize_tables(self) -> None:
        """Given a page with a TABLE, When normalizing, Then table structure is correct."""
        raw = _make_mock_raw(blocks=TABLE_BLOCKS)
        result = TextractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert len(validated.pages[0].tables) == 1

        table_out = validated.pages[0].tables[0]
        assert table_out.num_rows == 2
        assert table_out.num_cols == 2
        assert len(table_out.cells) == 4

        # Cell coordinates: Textract 1-based → schema 0-based
        # row=0, col=0 → Textract RowIndex=1, ColumnIndex=1
        assert table_out.cells[0].row == 0
        assert table_out.cells[0].col == 0
        assert table_out.cells[0].text == "A"
        assert table_out.cells[0].row_span == 1
        assert table_out.cells[0].col_span == 1

        # row=1, col=1 → Textract RowIndex=2, ColumnIndex=2
        assert table_out.cells[3].row == 1
        assert table_out.cells[3].col == 1
        assert table_out.cells[3].text == "D"

    def test_normalize_table_block_marker(self) -> None:
        """Given a TABLE block, When normalizing, Then a TableBlock marker is in the blocks list."""
        raw = _make_mock_raw(blocks=TABLE_BLOCKS)
        result = TextractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        blocks = validated.pages[0].blocks
        assert len(blocks) == 1
        assert blocks[0].type == "table"

    def test_no_tables(self) -> None:
        """Given a page with no TABLE blocks, When normalizing, Then tables list is empty."""
        raw = _make_mock_raw(blocks=HELLO_WORLD_BLOCKS)
        result = TextractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert validated.pages[0].tables == []


class TestTextractNormalizeCells:
    """Verify CELL block conversion to TableCells with row/col indices."""

    def test_cell_indices(self) -> None:
        """Given cells with row/col spans, When normalizing, Then 0-based indices and spans are correct."""
        # A 2×3 table with a spanned cell.
        big_cell = _make_block(
            "cell_big", "CELL", text="Header", confidence=95.0,
            left=0.1, top=0.3, width=0.8, height=0.1, page=1,
            row_index=1, column_index=1, row_span=1, column_span=3,
        )
        cells_2row = [
            _make_block(
                f"cell_body_{i}", "CELL", text=f"D{i}", confidence=94.0,
                left=0.1 + (i - 1) * 0.3, top=0.5, width=0.25, height=0.1, page=1,
                row_index=2, column_index=i, row_span=1, column_span=1,
            )
            for i in range(1, 4)
        ]
        table_block = _make_block(
            "t1", "TABLE", confidence=98.0,
            left=0.1, top=0.3, width=0.8, height=0.3, page=1,
            child_ids=["cell_big", "cell_body_1", "cell_body_2", "cell_body_3"],
        )
        page = _make_block(
            "p1", "PAGE", page=1, child_ids=["t1"],
            width=8.5, height=11.0,
        )

        blocks = [page, table_block, big_cell, *cells_2row]
        raw = _make_mock_raw(blocks=blocks)
        result = TextractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        table_out = validated.pages[0].tables[0]

        # num_rows should be 2 (max row=1 + span 1)
        assert table_out.num_rows == 2
        # num_cols should be 3 (max col=2 + span 1)
        assert table_out.num_cols == 3

        # First cell: row=0, col=0, col_span=3
        assert table_out.cells[0].row == 0
        assert table_out.cells[0].col == 0
        assert table_out.cells[0].col_span == 3
        assert table_out.cells[0].text == "Header"

        # Body cells: row=1, col=0..2
        for i in range(3):
            assert table_out.cells[1 + i].row == 1
            assert table_out.cells[1 + i].col == i


class TestTextractNormalizeForms:
    """Verify KEY_VALUE_SET handling (forms, future use).

    Currently the engine handles KEY_VALUE_SET blocks gracefully without
    crashing.  Full form extraction is a future enhancement.
    """

    def test_forms_handled_without_error(self) -> None:
        """Given KEY_VALUE_SET blocks, When normalizing, Then no error is raised."""
        raw = _make_mock_raw(blocks=FORM_BLOCKS)
        result = TextractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        # Forms should not produce any blocks or tables yet
        assert len(validated.pages[0].blocks) == 0
        assert len(validated.pages[0].tables) == 0


class TestTextractReadingOrder:
    """Verify reading order preservation from CHILD relationship ordering."""

    def test_reading_order(self) -> None:
        """Given PAGE children in non‑natural order, When normalizing, Then block order follows CHILD list."""
        raw = _make_mock_raw(blocks=ORDER_BLOCKS)
        result = TextractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        page_out = validated.pages[0]

        assert len(page_out.blocks) == 2
        # PAGE CHILD ids are ["line2", "line1"], so block 0 = line2, block 1 = line1
        assert page_out.blocks[0].order == 0
        assert page_out.blocks[1].order == 1

        # First block should have the "Second" word (from line2)
        assert page_out.blocks[0].lines[0].words[0].text == "Second"

        # Second block should have the "First" word (from line1)
        assert page_out.blocks[1].lines[0].words[0].text == "First"


class TestTextractSelectionElements:
    """Verify SELECTION_ELEMENT handling.

    Selection elements are not yet mapped to the output schema but must
    be handled without errors.
    """

    def test_selection_elements_handled(self) -> None:
        """Given SELECTION_ELEMENT blocks, When normalizing, Then no error is raised."""
        raw = _make_mock_raw(blocks=SELECTION_BLOCKS)
        result = TextractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        # Selection elements should not produce blocks yet
        assert len(validated.pages[0].blocks) == 0


class TestTextractHandwriting:
    """Verify HANDWRITING detection in Textract WORD blocks.

    Textract WORD blocks may carry ``TextTypes=["HANDWRITING"]`` which
    the engine records in metadata for future use.
    """

    def test_handwriting_detected(self) -> None:
        """Given a WORD with HANDWRITING TextType, When normalizing, Then the word is processed normally."""
        raw = _make_mock_raw(blocks=HANDWRITING_BLOCKS)
        result = TextractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        page_out = validated.pages[0]

        assert len(page_out.blocks) == 1
        word = page_out.blocks[0].lines[0].words[0]
        assert word.text == "Handwritten"

        # Confidence normalised: 85.0 → 0.85
        assert word.confidence == pytest.approx(0.85)

        # Characters should be synthesised
        assert len(word.chars) == len("Handwritten")


class TestTextractMergeGeometry:
    """Verify polygon → bbox conversion utility."""

    def test_polygon_to_bbox_merged(self) -> None:
        """Given a WORD block with Polygon, When normalizing, Then the bbox is derived from polygon vertices.

        The engine uses BoundingBox for the canonical bbox, but the
        ``_polygon_to_bbox`` helper is available for tighter bounds when
        polygon data is preferred.
        """
        # The _polygon_to_bbox function is tested separately in TestPolygonToBbox
        # This integration test verifies that the overall pipeline works.
        word = _make_block(
            "w1", "WORD", text="Test", confidence=98.0,
            left=0.1, top=0.1, width=0.2, height=0.1, page=1,
        )
        line = _make_block(
            "l1", "LINE", text="Test", confidence=99.0,
            left=0.1, top=0.1, width=0.2, height=0.1, page=1,
            child_ids=["w1"],
        )
        page = _make_block(
            "p1", "PAGE", page=1, child_ids=["l1"],
            width=8.5, height=11.0,
        )

        blocks = [page, line, word]
        raw = _make_mock_raw(blocks=blocks)
        result = TextractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        word_out = validated.pages[0].blocks[0].lines[0].words[0]

        # The bbox should be consistent whether derived from BoundingBox or Polygon
        polygon_bbox = _polygon_to_bbox(
            word.get("Geometry", {}).get("Polygon"),
            612.0, 792.0,
        )
        assert word_out.bbox == pytest.approx(polygon_bbox)


# ── Multiple pages ──────────────────────────────────────────────────────────


class TestTextractNormalizeMultiplePages:
    """Verify ``normalize()`` handles multiple pages."""

    def test_two_pages(self) -> None:
        """Given raw data with two pages, When normalizing, Then two NormalizedPages are produced."""
        page1 = _make_block(
            "p1", "PAGE", page=1, child_ids=["l1a"],
            width=8.5, height=11.0,
        )
        line1a = _make_block(
            "l1a", "LINE", text="Page one", confidence=99.0,
            left=0.1, top=0.1, width=0.5, height=0.1, page=1,
            child_ids=["w1a"],
        )
        word1a = _make_block(
            "w1a", "WORD", text="Page", confidence=99.0,
            left=0.1, top=0.1, width=0.3, height=0.1, page=1,
        )

        page2 = _make_block(
            "p2", "PAGE", page=2, child_ids=["l2a"],
            width=8.5, height=11.0,
        )
        line2a = _make_block(
            "l2a", "LINE", text="Page two", confidence=98.0,
            left=0.1, top=0.1, width=0.5, height=0.1, page=2,
            child_ids=["w2a"],
        )
        word2a = _make_block(
            "w2a", "WORD", text="Page", confidence=98.0,
            left=0.1, top=0.1, width=0.3, height=0.1, page=2,
        )

        blocks = [page1, line1a, word1a, page2, line2a, word2a]
        raw = _make_mock_raw(blocks=blocks, page_count=2)
        result = TextractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert len(validated.pages) == 2
        assert validated.pages[0].page_number == 1
        assert validated.pages[1].page_number == 2
        assert validated.pages[0].blocks[0].lines[0].words[0].text == "Page"
        assert validated.pages[1].blocks[0].lines[0].words[0].text == "Page"


# ── Process PDF (mocked) ────────────────────────────────────────────────────


class TestTextractProcessPDF:
    """Verify ``process_pdf()`` error handling and graceful degradation.

    These tests do NOT require AWS credentials.  The boto3 client is
    fully mocked.
    """

    @pytest.mark.asyncio
    async def test_handles_missing_file(self) -> None:
        """Given a non-existent PDF path, When processing, Then FileNotFoundError is raised."""
        engine = TextractEngine()

        with pytest.raises(FileNotFoundError, match="not found"):
            await engine.process_pdf("/nonexistent/path.pdf")

    @pytest.mark.asyncio
    async def test_graceful_no_boto3(self) -> None:
        """Given missing boto3 SDK, When processing, Then RuntimeError is raised gracefully.

        We simulate this by patching HAS_BOTO3 to False within the engine module.
        """
        engine = TextractEngine()

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("backend.engines.textract.HAS_BOTO3", False),
            pytest.raises(RuntimeError, match="boto3"),
        ):
            await engine.process_pdf("/fake/path.pdf")

    @pytest.mark.asyncio
    async def test_process_pdf_calls_progress(self) -> None:
        """Given mock dependencies, When processing, Then progress callback is invoked."""
        engine = TextractEngine()
        captured: list[int] = []

        def track(v: int) -> None:
            captured.append(v)

        mock_client = MagicMock()
        mock_client.analyze_document.return_value = {
            "Blocks": HELLO_WORLD_BLOCKS,
            "DocumentMetadata": {"Pages": 1},
        }

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_bytes", return_value=b"%PDF-1.4 fake pdf content"),
            patch("backend.engines.textract.boto3.client", return_value=mock_client),
        ):
            await engine.process_pdf(
                "/fake/path.pdf",
                config={"region": "us-east-1", "timeout_seconds": 300},
                progress=track,
            )

        assert len(captured) >= 3
        assert captured[0] == 0
        assert captured[-1] == 100

    @pytest.mark.asyncio
    async def test_process_pdf_returns_expected_structure(self) -> None:
        """Given mock dependencies, When processing, Then raw output has expected keys."""
        engine = TextractEngine()

        mock_client = MagicMock()
        mock_client.analyze_document.return_value = {
            "Blocks": HELLO_WORLD_BLOCKS,
            "DocumentMetadata": {"Pages": 1},
        }

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_bytes", return_value=b"%PDF-1.4 fake pdf content"),
            patch("backend.engines.textract.boto3.client", return_value=mock_client),
        ):
            result = await engine.process_pdf(
                "/fake/path.pdf",
                config={"region": "us-east-1"},
            )

        assert "blocks" in result
        assert result["engine_id"] == "aws-textract"
        assert "engine_version" in result
        assert "config_snapshot" in result
        assert result["page_count"] == 1
        assert len(result["blocks"]) == len(HELLO_WORLD_BLOCKS)


# ── Registry integration ────────────────────────────────────────────────────


class TestTextractRegistry:
    """Verify that TextractEngine is registered with the global registry."""

    def test_registered_in_registry(self) -> None:
        """Given the global registry, When retrieving by ID, Then TextractEngine is returned."""
        from backend.engine.registry import registry as global_registry

        engine = global_registry.get("aws-textract")
        assert isinstance(engine, TextractEngine)
        assert engine.engine_id == "aws-textract"

    def test_in_registry_list(self) -> None:
        """Given the global registry, When listing engines, Then aws-textract is present."""
        from backend.engine.registry import registry as global_registry

        engine_ids = [e.engine_id for e in global_registry.list()]
        assert "aws-textract" in engine_ids
