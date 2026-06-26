"""Tests for the GCP Document AI OCR engine module.

The test suite covers:
- ABC conformance (``OCREngine`` subclass, attributes, methods).
- Config schema validation.
- Raw output normalisation (``normalize()`` with mock data).
- Hierarchy construction (paragraph → line → token → symbol).
- Table extraction and cell mapping.
- Confidence value propagation.
- Bounding box coordinate transformation (normalised → page-space points).
- Edge cases (empty pages, no text, missing fields).

All tests are self-contained — they do **not** require GCP credentials, a
Document AI processor, or any external service.  Mock fixtures use plain
dicts that mirror the proto JSON structure with snake_case field names.
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.engine.base import OCREngine
from backend.engine.normalized_schema import (
    NormalizedDocument,
    TextBlock,
)

# Module to test
from backend.engines.gcp_document_ai import (
    GcpDocumentAiEngine,
    _extract_text,
    _find_contained,
    _get_text_range,
    _layout_to_bbox,
)

# ── Mock data factories ─────────────────────────────────────────────────────


def _make_poly(
    nx0: float = 0.0,
    ny0: float = 0.0,
    nx1: float = 1.0,
    ny1: float = 1.0,
) -> dict[str, Any]:
    """Create a bounding polygon with normalised vertices.

    Returns a quadrilateral in clockwise order: top-left → top-right →
    bottom-right → bottom-left.
    """
    return {
        "normalized_vertices": [
            {"x": nx0, "y": ny0},
            {"x": nx1, "y": ny0},
            {"x": nx1, "y": ny1},
            {"x": nx0, "y": ny1},
        ]
    }


def _make_layout(
    nx0: float = 0.0,
    ny0: float = 0.0,
    nx1: float = 1.0,
    ny1: float = 1.0,
    *,
    confidence: float = 1.0,
    start_index: int = 0,
    end_index: int = 0,
) -> dict[str, Any]:
    """Create a Document AI layout dict with normalised coordinates."""
    result: dict[str, Any] = {
        "bounding_poly": _make_poly(nx0, ny0, nx1, ny1),
        "confidence": confidence,
    }
    if end_index > start_index:
        result["text_anchor"] = {
            "text_segments": [{"start_index": start_index, "end_index": end_index}]
        }
    return result


def _make_symbol(
    _char: str,
    nx0: float,
    ny0: float,
    nx1: float,
    ny1: float,
    *,
    confidence: float = 0.99,
    start_index: int = 0,
    end_index: int = 0,
) -> dict[str, Any]:
    """Create a Document AI symbol dict."""
    return {
        "layout": _make_layout(nx0, ny0, nx1, ny1, confidence=confidence, start_index=start_index, end_index=end_index),
    }


def _make_token(
    _text: str,
    nx0: float,
    ny0: float,
    nx1: float,
    ny1: float,
    *,
    confidence: float = 0.99,
    start_index: int = 0,
    end_index: int = 0,
    symbols: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a Document AI token dict."""
    token: dict[str, Any] = {
        "layout": _make_layout(nx0, ny0, nx1, ny1, confidence=confidence, start_index=start_index, end_index=end_index),
    }
    if symbols:
        token["symbols"] = symbols
    return token


def _make_line(
    nx0: float,
    ny0: float,
    nx1: float,
    ny1: float,
    *,
    confidence: float = 0.98,
    start_index: int = 0,
    end_index: int = 0,
) -> dict[str, Any]:
    """Create a Document AI line dict."""
    return {
        "layout": _make_layout(nx0, ny0, nx1, ny1, confidence=confidence, start_index=start_index, end_index=end_index),
    }


def _make_paragraph(
    nx0: float,
    ny0: float,
    nx1: float,
    ny1: float,
    *,
    confidence: float = 0.97,
    start_index: int = 0,
    end_index: int = 0,
) -> dict[str, Any]:
    """Create a Document AI paragraph dict."""
    return {
        "layout": _make_layout(nx0, ny0, nx1, ny1, confidence=confidence, start_index=start_index, end_index=end_index),
    }


def _make_mock_page(
    *,
    page_number: int = 1,
    width: float = 612.0,
    height: float = 792.0,
    paragraphs: list[dict[str, Any]] | None = None,
    lines: list[dict[str, Any]] | None = None,
    tokens: list[dict[str, Any]] | None = None,
    symbols: list[dict[str, Any]] | None = None,
    tables: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a mock Document AI page dict."""
    page: dict[str, Any] = {
        "page_number": page_number,
        "dimensions": {"width": width, "height": height, "unit": "pt"},
        "layout": _make_layout(0.0, 0.0, 1.0, 1.0, confidence=1.0),
    }
    if paragraphs is not None:
        page["paragraphs"] = paragraphs
    if lines is not None:
        page["lines"] = lines
    if tokens is not None:
        page["tokens"] = tokens
    if symbols is not None:
        page["symbols"] = symbols
    if tables is not None:
        page["tables"] = tables
    return page


def _make_mock_document(
    text: str = "",
    pages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a mock Document AI Document dict."""
    return {
        "text": text,
        "pages": pages or [],
    }


def _make_mock_raw(
    *,
    text: str = "",
    pages: list[dict[str, Any]] | None = None,
    engine_id: str = "gcp-document-ai",
    engine_version: str = "2.36.0",
) -> dict[str, Any]:
    """Build a minimal raw output dict as returned by ``process_pdf()``."""
    return {
        "document": _make_mock_document(text=text, pages=pages),
        "engine_id": engine_id,
        "engine_version": engine_version,
        "config_snapshot": {
            "processor_id": "test-processor",
            "location": "us",
            "project_id": "test-project",
            "timeout_seconds": 300,
            "mime_type": "application/pdf",
        },
        "page_count": len(pages) if pages else 0,
    }


# ── Test data ───────────────────────────────────────────────────────────────

# A simple "Hello World" document at 612×792 pt (US Letter).
# Text offsets: "Hello World" = indices 0-10 (with space at index 5).
HELLO_WORLD_TEXT = "Hello World"
HELLO_WORLD_PARAGRAPHS = [
    _make_paragraph(0.1, 0.1, 0.9, 0.2, confidence=0.97, start_index=0, end_index=11),
]
HELLO_WORLD_LINES = [
    _make_line(0.1, 0.1, 0.9, 0.2, confidence=0.98, start_index=0, end_index=11),
]
HELLO_WORLD_TOKENS = [
    _make_token("Hello", 0.1, 0.1, 0.3, 0.2, confidence=0.99, start_index=0, end_index=5),
    _make_token("World", 0.35, 0.1, 0.9, 0.2, confidence=0.98, start_index=6, end_index=11),
]
HELLO_WORLD_SYMBOLS = [
    # "Hello" symbols
    _make_symbol("H", 0.10, 0.10, 0.13, 0.20, confidence=0.99, start_index=0, end_index=1),
    _make_symbol("e", 0.14, 0.10, 0.17, 0.20, confidence=0.99, start_index=1, end_index=2),
    _make_symbol("l", 0.18, 0.10, 0.21, 0.20, confidence=0.99, start_index=2, end_index=3),
    _make_symbol("l", 0.22, 0.10, 0.25, 0.20, confidence=0.99, start_index=3, end_index=4),
    _make_symbol("o", 0.26, 0.10, 0.30, 0.20, confidence=0.99, start_index=4, end_index=5),
    # "World" symbols
    _make_symbol("W", 0.35, 0.10, 0.42, 0.20, confidence=0.98, start_index=6, end_index=7),
    _make_symbol("o", 0.43, 0.10, 0.55, 0.20, confidence=0.98, start_index=7, end_index=8),
    _make_symbol("r", 0.56, 0.10, 0.67, 0.20, confidence=0.98, start_index=8, end_index=9),
    _make_symbol("l", 0.68, 0.10, 0.78, 0.20, confidence=0.98, start_index=9, end_index=10),
    _make_symbol("d", 0.79, 0.10, 0.90, 0.20, confidence=0.98, start_index=10, end_index=11),
]


# ── Helper unit tests ───────────────────────────────────────────────────────


class TestLayoutToBBox:
    """Verify ``_layout_to_bbox()`` coordinate conversion."""

    def test_normalised_coordinates(self) -> None:
        """Given normalised coords (0-1) at 612×792 pt, When converting, Then bbox is in points."""
        layout = _make_layout(0.1, 0.2, 0.5, 0.8)
        bbox = _layout_to_bbox(layout, 612.0, 792.0)
        assert bbox == pytest.approx([61.2, 158.4, 306.0, 633.6])

    def test_empty_layout(self) -> None:
        """Given an empty layout, When converting, Then zeros are returned."""
        assert _layout_to_bbox(None, 612.0, 792.0) == [0.0, 0.0, 0.0, 0.0]
        assert _layout_to_bbox({}, 612.0, 792.0) == [0.0, 0.0, 0.0, 0.0]

    def test_full_page(self) -> None:
        """Given a full-page bbox, When converting, Then it spans the page dimensions."""
        layout = _make_layout(0.0, 0.0, 1.0, 1.0)
        bbox = _layout_to_bbox(layout, 612.0, 792.0)
        assert bbox == pytest.approx([0.0, 0.0, 612.0, 792.0])


class TestExtractText:
    """Verify ``_extract_text()`` text extraction from text_anchor offsets."""

    def test_single_segment(self) -> None:
        """Given a single text segment, When extracting, Then correct substring is returned."""
        element = {"layout": {"text_anchor": {"text_segments": [{"start_index": 0, "end_index": 5}]}}}
        result = _extract_text("Hello World", element)
        assert result == "Hello"

    def test_multiple_segments(self) -> None:
        """Given multiple text segments, When extracting, Then they are concatenated."""
        element = {
            "layout": {
                "text_anchor": {
                    "text_segments": [
                        {"start_index": 0, "end_index": 5},
                        {"start_index": 6, "end_index": 11},
                    ]
                }
            }
        }
        result = _extract_text("Hello World", element)
        assert result == "HelloWorld"

    def test_empty_text(self) -> None:
        """Given empty full text, When extracting, Then empty string is returned."""
        element = {"layout": {"text_anchor": {"text_segments": [{"start_index": 0, "end_index": 5}]}}}
        result = _extract_text("", element)
        assert result == ""

    def test_no_text_anchor(self) -> None:
        """Given element with no text_anchor, When extracting, Then empty string is returned."""
        result = _extract_text("Hello World", {"layout": {}})
        assert result == ""

    def test_out_of_bounds_indices(self) -> None:
        """Given out-of-bounds indices, When extracting, Then they are skipped."""
        element = {"layout": {"text_anchor": {"text_segments": [{"start_index": -1, "end_index": 100}]}}}
        result = _extract_text("Hello", element)
        assert result == ""


class TestGetTextRange:
    """Verify ``_get_text_range()`` range extraction."""

    def test_basic_range(self) -> None:
        """Given an element with text segments, When getting range, Then correct start/end is returned."""
        element = {"layout": {"text_anchor": {"text_segments": [{"start_index": 3, "end_index": 10}]}}}
        assert _get_text_range(element) == (3, 10)

    def test_no_segments(self) -> None:
        """Given element with no text segments, When getting range, Then (0, 0) is returned."""
        assert _get_text_range({"layout": {}}) == (0, 0)

    def test_multiple_segments_uses_first_and_last(self) -> None:
        """Given multiple segments, When getting range, Then start of first and end of last are used."""
        element = {
            "layout": {
                "text_anchor": {
                    "text_segments": [
                        {"start_index": 2, "end_index": 5},
                        {"start_index": 6, "end_index": 10},
                    ]
                }
            }
        }
        assert _get_text_range(element) == (2, 10)


class TestFindContained:
    """Verify ``_find_contained()`` containment detection."""

    def test_simple_containment(self) -> None:
        """Given children with nested ranges, When filtering, Then only contained children remain."""
        parent = {"layout": {"text_anchor": {"text_segments": [{"start_index": 0, "end_index": 10}]}}}
        children = [
            {"layout": {"text_anchor": {"text_segments": [{"start_index": 2, "end_index": 5}]}}},
            {"layout": {"text_anchor": {"text_segments": [{"start_index": 8, "end_index": 15}]}}},  # outside
            {"layout": {"text_anchor": {"text_segments": [{"start_index": 0, "end_index": 10}]}}},
        ]
        result = _find_contained(children, parent)
        assert len(result) == 2
        assert result[0]["layout"]["text_anchor"]["text_segments"][0]["start_index"] == 2
        assert result[1]["layout"]["text_anchor"]["text_segments"][0]["start_index"] == 0

    def test_no_segments_on_parent(self) -> None:
        """Given parent with no text segments, When filtering, Then all children are returned."""
        parent = {"layout": {}}
        children = [{"layout": {"text_anchor": {"text_segments": [{"start_index": 0, "end_index": 5}]}}}]
        result = _find_contained(children, parent)
        assert len(result) == 1

    def test_no_children(self) -> None:
        """Given an empty children list, When filtering, Then empty list is returned."""
        parent = {"layout": {"text_anchor": {"text_segments": [{"start_index": 0, "end_index": 10}]}}}
        assert _find_contained([], parent) == []


# ── ABC conformance ─────────────────────────────────────────────────────────


class TestGcpABCConformance:
    """Verify that ``GcpDocumentAiEngine`` satisfies the ``OCREngine`` ABC."""

    def test_gcp_engine_is_ocrenigne_subclass(self) -> None:
        """Given GcpDocumentAiEngine, When checking MRO, Then it is an OCREngine subclass."""
        assert issubclass(GcpDocumentAiEngine, OCREngine)

    def test_gcp_engine_conforms_to_abc(self) -> None:
        """Given a GcpDocumentAiEngine instance, When inspecting interface, Then all ABC members exist."""
        engine = GcpDocumentAiEngine()

        # -- class-level attributes
        assert engine.engine_id == "gcp-document-ai"
        assert engine.display_name == "GCP Document AI"
        assert isinstance(engine.version, str)
        assert len(engine.version) > 0

        # -- required methods
        assert callable(engine.get_config_schema)
        assert callable(engine.normalize)

        # process_pdf must be a coroutine
        import asyncio

        assert asyncio.iscoroutinefunction(engine.process_pdf)


# ── Config schema ───────────────────────────────────────────────────────────


class TestGcpConfigSchema:
    """Verify ``get_config_schema()`` returns valid JSON Schema."""

    def test_gcp_config_schema_structure(self) -> None:
        """Given a GcpDocumentAiEngine, When calling get_config_schema(), Then it returns valid schema."""
        engine = GcpDocumentAiEngine()
        schema = engine.get_config_schema()

        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert "processor_id" in schema["required"]
        assert "project_id" in schema["required"]

    def test_gcp_config_schema_has_expected_keys(self) -> None:
        """Given a GcpDocumentAiEngine, When inspecting schema properties, Then all keys exist."""
        engine = GcpDocumentAiEngine()
        props = engine.get_config_schema()["properties"]

        assert "processor_id" in props
        assert props["processor_id"]["type"] == "string"

        assert "location" in props
        assert props["location"]["type"] == "string"
        assert props["location"]["default"] == "us"

        assert "project_id" in props
        assert props["project_id"]["type"] == "string"

        assert "credentials_path" in props
        assert props["credentials_path"]["type"] == "string"

        assert "timeout_seconds" in props
        assert props["timeout_seconds"]["type"] == "integer"
        assert props["timeout_seconds"]["default"] == 300
        assert props["timeout_seconds"]["minimum"] == 30

        assert "mime_type" in props
        assert props["mime_type"]["type"] == "string"
        assert props["mime_type"]["default"] == "application/pdf"


# ── Normalize output ────────────────────────────────────────────────────────


class TestGcpNormalizeBasic:
    """Verify ``normalize()`` produces correct ``NormalizedDocument`` structure."""

    def test_gcp_normalize_empty_response(self) -> None:
        """Given an empty raw document, When normalizing, Then a valid empty document is produced."""
        raw = _make_mock_raw()
        result = GcpDocumentAiEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert len(validated.pages) == 1
        assert validated.pages[0].page_number == 1
        assert validated.pages[0].blocks == []
        assert validated.pages[0].tables == []

    def test_gcp_normalize_no_text(self) -> None:
        """Given a page with no text content, When normalizing, Then empty blocks are produced."""
        page = _make_mock_page(
            paragraphs=[],
            lines=[],
            tokens=[],
            symbols=[],
        )
        raw = _make_mock_raw(pages=[page])
        result = GcpDocumentAiEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert validated.pages[0].blocks == []

    def test_gcp_normalize_basic_document(self) -> None:
        """Given a basic document with one paragraph, When normalizing, Then hierarchy is correct."""
        page = _make_mock_page(
            paragraphs=HELLO_WORLD_PARAGRAPHS,
            lines=HELLO_WORLD_LINES,
            tokens=HELLO_WORLD_TOKENS,
            symbols=HELLO_WORLD_SYMBOLS,
        )
        raw = _make_mock_raw(text=HELLO_WORLD_TEXT, pages=[page])
        result = GcpDocumentAiEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert len(validated.pages) == 1
        page_out = validated.pages[0]

        assert page_out.page_number == 1
        assert page_out.width == 612.0
        assert page_out.height == 792.0

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

        # Words should have matched characters from symbols
        for word in line.words:
            assert len(word.chars) == len(word.text), (
                f"Word '{word.text}' has {len(word.chars)} chars, expected {len(word.text)}"
            )
            for char_entry in word.chars:
                assert len(char_entry.char) == 1
                assert 0.0 <= char_entry.confidence <= 1.0
                assert char_entry.order >= 0

    def test_gcp_normalize_metadata(self) -> None:
        """Given mock raw data, When normalizing, Then engine metadata is preserved."""
        page = _make_mock_page(
            paragraphs=HELLO_WORLD_PARAGRAPHS,
            lines=HELLO_WORLD_LINES,
            tokens=HELLO_WORLD_TOKENS,
            symbols=HELLO_WORLD_SYMBOLS,
        )
        raw = _make_mock_raw(text=HELLO_WORLD_TEXT, pages=[page])
        result = GcpDocumentAiEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert validated.engine_id == "gcp-document-ai"
        assert validated.engine_version == "2.36.0"
        assert validated.config_snapshot["processor_id"] == "test-processor"
        assert validated.config_snapshot["location"] == "us"


class TestGcpNormalizeParagraphsLinesWords:
    """Verify hierarchy preservation with multiple paragraphs."""

    def test_gcp_normalize_multiple_paragraphs(self) -> None:
        """Given a page with two paragraphs, When normalizing, Then both are preserved as blocks."""
        text = "First paragraph.\nSecond paragraph."
        paragraphs = [
            _make_paragraph(0.1, 0.1, 0.9, 0.15, confidence=0.97, start_index=0, end_index=16),
            _make_paragraph(0.1, 0.2, 0.9, 0.25, confidence=0.96, start_index=17, end_index=34),
        ]
        lines = [
            _make_line(0.1, 0.1, 0.9, 0.15, confidence=0.98, start_index=0, end_index=16),
            _make_line(0.1, 0.2, 0.9, 0.25, confidence=0.97, start_index=17, end_index=34),
        ]
        tokens = [
            _make_token("First", 0.10, 0.10, 0.25, 0.15, confidence=0.99, start_index=0, end_index=5),
            _make_token("paragraph.", 0.26, 0.10, 0.50, 0.15, confidence=0.98, start_index=6, end_index=16),
            _make_token("Second", 0.10, 0.20, 0.28, 0.25, confidence=0.99, start_index=17, end_index=23),
            _make_token("paragraph.", 0.29, 0.20, 0.58, 0.25, confidence=0.98, start_index=24, end_index=34),
        ]

        page = _make_mock_page(paragraphs=paragraphs, lines=lines, tokens=tokens)
        raw = _make_mock_raw(text=text, pages=[page])
        result = GcpDocumentAiEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert len(validated.pages[0].blocks) == 2

        block0 = validated.pages[0].blocks[0]
        block1 = validated.pages[0].blocks[1]

        assert block0.order == 0
        assert block1.order == 1

        # Each block should have one line
        assert len(block0.lines) == 1
        assert len(block1.lines) == 1

        assert block0.lines[0].words[0].text == "First"
        assert block1.lines[0].words[0].text == "Second"


class TestGcpNormalizeTables:
    """Verify table extraction from Document AI table data."""

    def test_gcp_normalize_tables(self) -> None:
        """Given a page with a table, When normalizing, Then table structure is correct."""
        text = "Header1Header2Data1Data2"
        table_layout = _make_layout(0.1, 0.3, 0.9, 0.7, confidence=0.95)

        # Build table cells with text_anchor offsets into the full text.
        # Each cell is a dict with layout and content offsets.
        header_cell1_text = "Header1"
        header_cell2_text = "Header2"
        body_cell1_text = "Data1"
        body_cell2_text = "Data2"

        # Offsets: Header1=0-7, Header2=7-14, Data1=14-19, Data2=19-24
        table = {
            "layout": table_layout,
            "header_rows": [
                {
                    "cells": [
                        {
                            "layout": _make_layout(
                                0.1, 0.3, 0.45, 0.4,
                                confidence=0.95,
                                start_index=0, end_index=len(header_cell1_text),
                            ),
                            "col_index": 0,
                            "col_span": 1,
                            "row_span": 1,
                        },
                        {
                            "layout": _make_layout(
                                0.55, 0.3, 0.9, 0.4,
                                confidence=0.94,
                                start_index=len(header_cell1_text),
                                end_index=len(header_cell1_text) + len(header_cell2_text),
                            ),
                            "col_index": 1,
                            "col_span": 1,
                            "row_span": 1,
                        },
                    ]
                }
            ],
            "body_rows": [
                {
                    "cells": [
                        {
                            "layout": _make_layout(
                                0.1, 0.5, 0.45, 0.6,
                                confidence=0.93,
                                start_index=len(header_cell1_text) + len(header_cell2_text),
                                end_index=len(header_cell1_text) + len(header_cell2_text)
                                + len(body_cell1_text),
                            ),
                            "col_index": 0,
                            "col_span": 1,
                            "row_span": 1,
                        },
                        {
                            "layout": _make_layout(
                                0.55, 0.5, 0.9, 0.6,
                                confidence=0.92,
                                start_index=len(header_cell1_text) + len(header_cell2_text)
                                + len(body_cell1_text),
                                end_index=len(header_cell1_text) + len(header_cell2_text)
                                + len(body_cell1_text) + len(body_cell2_text),
                            ),
                            "col_index": 1,
                            "col_span": 1,
                            "row_span": 1,
                        },
                    ]
                }
            ],
        }

        # The table text is NOT part of the main paragraph text in DAI,
        # so we just need the full text string for the text_anchor lookups.
        page = _make_mock_page(
            paragraphs=[],
            lines=[],
            tokens=[],
            tables=[table],
        )
        raw = _make_mock_raw(text=text, pages=[page])
        result = GcpDocumentAiEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert len(validated.pages[0].tables) == 1

        table_out = validated.pages[0].tables[0]
        assert table_out.num_rows == 2  # 1 header + 1 body
        assert table_out.num_cols == 2
        assert len(table_out.cells) == 4

        # Check header cells
        header_cells = [c for c in table_out.cells if c.row == 0]
        assert len(header_cells) == 2
        assert header_cells[0].text == "Header1"
        assert header_cells[1].text == "Header2"

        # Check body cells
        body_cells = [c for c in table_out.cells if c.row == 1]
        assert len(body_cells) == 2
        assert body_cells[0].text == "Data1"
        assert body_cells[1].text == "Data2"

    def test_gcp_normalize_no_tables(self) -> None:
        """Given a page with no tables, When normalizing, Then tables list is empty."""
        page = _make_mock_page(
            paragraphs=HELLO_WORLD_PARAGRAPHS,
            lines=HELLO_WORLD_LINES,
            tokens=HELLO_WORLD_TOKENS,
            symbols=HELLO_WORLD_SYMBOLS,
        )
        raw = _make_mock_raw(text=HELLO_WORLD_TEXT, pages=[page])
        result = GcpDocumentAiEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert validated.pages[0].tables == []


class TestGcpConfidenceParsing:
    """Verify confidence values propagate correctly."""

    def test_gcp_confidence_parsing(self) -> None:
        """Given confidence values (0.0-1.0) in layouts, When normalizing, Then they propagate correctly."""
        # Confidence values are 0.0-1.0 per layout element
        paragraphs = [
            _make_paragraph(0.1, 0.1, 0.9, 0.2, confidence=0.95, start_index=0, end_index=11),
        ]
        lines = [
            _make_line(0.1, 0.1, 0.9, 0.2, confidence=0.94, start_index=0, end_index=11),
        ]
        tokens = [
            _make_token("Hello", 0.10, 0.10, 0.30, 0.20, confidence=0.93, start_index=0, end_index=5),
            _make_token("World", 0.35, 0.10, 0.90, 0.20, confidence=0.92, start_index=6, end_index=11),
        ]
        symbols = [
            _make_symbol("H", 0.10, 0.10, 0.13, 0.20, confidence=0.91, start_index=0, end_index=1),
            _make_symbol("e", 0.14, 0.10, 0.17, 0.20, confidence=0.90, start_index=1, end_index=2),
            _make_symbol("l", 0.18, 0.10, 0.21, 0.20, confidence=0.89, start_index=2, end_index=3),
            _make_symbol("l", 0.22, 0.10, 0.25, 0.20, confidence=0.88, start_index=3, end_index=4),
            _make_symbol("o", 0.26, 0.10, 0.30, 0.20, confidence=0.87, start_index=4, end_index=5),
            _make_symbol("W", 0.35, 0.10, 0.42, 0.20, confidence=0.86, start_index=6, end_index=7),
            _make_symbol("o", 0.43, 0.10, 0.55, 0.20, confidence=0.85, start_index=7, end_index=8),
            _make_symbol("r", 0.56, 0.10, 0.67, 0.20, confidence=0.84, start_index=8, end_index=9),
            _make_symbol("l", 0.68, 0.10, 0.78, 0.20, confidence=0.83, start_index=9, end_index=10),
            _make_symbol("d", 0.79, 0.10, 0.90, 0.20, confidence=0.82, start_index=10, end_index=11),
        ]

        page = _make_mock_page(
            paragraphs=paragraphs,
            lines=lines,
            tokens=tokens,
            symbols=symbols,
        )
        raw = _make_mock_raw(text=HELLO_WORLD_TEXT, pages=[page])
        result = GcpDocumentAiEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        block = validated.pages[0].blocks[0]
        line = block.lines[0]

        # Word confidences should match the token confidences
        assert line.words[0].confidence == pytest.approx(0.93)
        assert line.words[1].confidence == pytest.approx(0.92)

        # Character confidences should match the symbol confidences
        assert line.words[0].chars[0].confidence == pytest.approx(0.91)  # H
        assert line.words[0].chars[4].confidence == pytest.approx(0.87)  # o
        assert line.words[1].chars[0].confidence == pytest.approx(0.86)  # W
        assert line.words[1].chars[4].confidence == pytest.approx(0.82)  # d


class TestGcpCoordinateNormalization:
    """Verify coordinate normalization from Document AI coords to page-space."""

    def test_gcp_coordinate_normalization(self) -> None:
        """Given normalized coords (0-1) at 612×792 pt, When normalizing, Then bboxes are in points.

        Document AI normalized coords are scaled by page dimensions:
            point_x = normalized_x * page_width
            point_y = normalized_y * page_height
        """
        paragraphs = [
            _make_paragraph(0.1, 0.1, 0.5, 0.2, confidence=0.97, start_index=0, end_index=11),
        ]
        lines = [
            _make_line(0.1, 0.1, 0.5, 0.2, confidence=0.98, start_index=0, end_index=11),
        ]
        tokens = [
            _make_token("Hello", 0.10, 0.10, 0.25, 0.20, confidence=0.99, start_index=0, end_index=5),
        ]
        symbols = [
            _make_symbol("H", 0.10, 0.10, 0.13, 0.20, confidence=0.99, start_index=0, end_index=1),
            _make_symbol("e", 0.14, 0.10, 0.17, 0.20, confidence=0.99, start_index=1, end_index=2),
        ]

        page = _make_mock_page(
            width=612.0,
            height=792.0,
            paragraphs=paragraphs,
            lines=lines,
            tokens=tokens,
            symbols=symbols,
        )
        raw = _make_mock_raw(text="Hello World", pages=[page])
        result = GcpDocumentAiEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        page_out = validated.pages[0]

        # Page dimensions preserved
        assert page_out.width == 612.0
        assert page_out.height == 792.0

        # Word bbox: normalized [0.10, 0.10, 0.25, 0.20]
        # → points: [61.2, 79.2, 153.0, 158.4]
        word = page_out.blocks[0].lines[0].words[0]
        expected_bbox = [61.2, 79.2, 153.0, 158.4]
        assert word.bbox == pytest.approx(expected_bbox)

        # Character bbox for 'H': normalized [0.10, 0.10, 0.13, 0.20]
        # → points: [61.2, 79.2, 79.56, 158.4]
        char_h = word.chars[0]
        assert char_h.bbox == pytest.approx([61.2, 79.2, 79.56, 158.4])

        # Character bbox for 'e': normalized [0.14, 0.10, 0.17, 0.20]
        # → points: [85.68, 79.2, 104.04, 158.4]
        char_e = word.chars[1]
        assert char_e.bbox == pytest.approx([85.68, 79.2, 104.04, 158.4])


class TestGcpNormalizeNoSymbols:
    """Verify fallback when symbols are missing."""

    def test_gcp_normalize_synthesises_chars_when_no_symbols(self) -> None:
        """Given tokens but no symbols, When normalizing, Then chars are synthesised from token text."""
        page = _make_mock_page(
            paragraphs=HELLO_WORLD_PARAGRAPHS,
            lines=HELLO_WORLD_LINES,
            tokens=HELLO_WORLD_TOKENS,
            # No symbols
        )
        raw = _make_mock_raw(text=HELLO_WORLD_TEXT, pages=[page])
        result = GcpDocumentAiEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        word = validated.pages[0].blocks[0].lines[0].words[0]

        assert word.text == "Hello"
        assert len(word.chars) == 5
        assert word.chars[0].char == "H"
        assert word.chars[4].char == "o"
        assert 0.0 <= word.chars[0].confidence <= 1.0

        # Synthesised chars should span the word bbox
        word_bbox = word.bbox
        assert word.chars[0].bbox[0] == word_bbox[0]  # first char at left edge
        assert word.chars[4].bbox[2] == word_bbox[2]  # last char at right edge


class TestGcpNormalizeMultiplePages:
    """Verify ``normalize()`` handles multiple pages."""

    def test_gcp_normalize_two_pages(self) -> None:
        """Given raw data with two pages, When normalizing, Then two pages are produced."""
        page1_text = "Page one"
        page2_text = "Page two"

        page1 = _make_mock_page(
            page_number=1,
            paragraphs=[
                _make_paragraph(0.1, 0.1, 0.5, 0.2, confidence=0.97, start_index=0, end_index=8),
            ],
            lines=[
                _make_line(0.1, 0.1, 0.5, 0.2, confidence=0.98, start_index=0, end_index=8),
            ],
            tokens=[
                _make_token("Page", 0.10, 0.10, 0.20, 0.20, confidence=0.99, start_index=0, end_index=4),
                _make_token("one", 0.25, 0.10, 0.50, 0.20, confidence=0.99, start_index=5, end_index=8),
            ],
        )
        page2 = _make_mock_page(
            page_number=2,
            paragraphs=[
                _make_paragraph(0.1, 0.1, 0.5, 0.2, confidence=0.96, start_index=0, end_index=8),
            ],
            lines=[
                _make_line(0.1, 0.1, 0.5, 0.2, confidence=0.97, start_index=0, end_index=8),
            ],
            tokens=[
                _make_token("Page", 0.10, 0.10, 0.20, 0.20, confidence=0.99, start_index=0, end_index=4),
                _make_token("two", 0.25, 0.10, 0.50, 0.20, confidence=0.98, start_index=5, end_index=8),
            ],
        )

        # Each page has its own text in the overall Document.text
        # For this test, each page consumes a contiguous region.
        full_text = page1_text + page2_text
        raw = _make_mock_raw(
            text=full_text,
            pages=[page1, page2],
        )
        result = GcpDocumentAiEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert len(validated.pages) == 2
        assert validated.pages[0].page_number == 1
        assert validated.pages[1].page_number == 2
        assert validated.pages[0].blocks[0].lines[0].words[0].text == "Page"
        assert validated.pages[1].blocks[0].lines[0].words[0].text == "Page"

    def test_gcp_normalize_page_number_fallback(self) -> None:
        """Given a page with page_number=0, When normalizing, Then index+1 is used."""
        page_data = _make_mock_page(
            page_number=0,  # unset/zero in proto
            paragraphs=[],
            lines=[],
        )
        raw = _make_mock_raw(pages=[page_data])
        result = GcpDocumentAiEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert validated.pages[0].page_number == 1


class TestGcpNormalizeLinesOnly:
    """Verify fallback when paragraphs are missing but lines exist."""

    def test_gcp_normalize_lines_without_paragraphs(self) -> None:
        """Given lines but no paragraphs, When normalizing, Then each line becomes a block."""
        text = "Line one"
        lines = [
            _make_line(0.1, 0.1, 0.9, 0.2, confidence=0.98, start_index=0, end_index=8),
        ]
        tokens = [
            _make_token("Line", 0.10, 0.10, 0.25, 0.20, confidence=0.99, start_index=0, end_index=4),
            _make_token("one", 0.30, 0.10, 0.50, 0.20, confidence=0.98, start_index=5, end_index=8),
        ]

        page = _make_mock_page(
            paragraphs=[],  # explicit empty — no paragraphs
            lines=lines,
            tokens=tokens,
        )
        raw = _make_mock_raw(text=text, pages=[page])
        result = GcpDocumentAiEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert len(validated.pages[0].blocks) == 1

        block = validated.pages[0].blocks[0]
        assert len(block.lines) == 1
        assert block.lines[0].text == "Line one"
        assert block.lines[0].words[0].text == "Line"
        assert block.lines[0].words[1].text == "one"


# ── Process PDF (mocked) — the engine's process_pdf calls the GCP SDK.
# These tests verify the orchestration layer without real credentials.
# ─────────────────────────────────────────────────────────────────────────────


class TestGcpProcessPDF:
    """Verify ``process_pdf()`` error handling and graceful degradation.

    Full SDK mocking is not done here (the SDK's gRPC internals are complex).
    Instead we verify:
    - Missing files raise FileNotFoundError.
    - Missing SDK raises RuntimeError (when HAS_DOCUMENT_AI is False).
    """

    @pytest.mark.asyncio
    async def test_gcp_process_pdf_handles_missing_file(self) -> None:
        """Given a non-existent PDF path, When processing, Then FileNotFoundError is raised."""
        engine = GcpDocumentAiEngine()

        with pytest.raises(FileNotFoundError, match="not found"):
            await engine.process_pdf("/nonexistent/path.pdf")

    @pytest.mark.asyncio
    async def test_gcp_process_pdf_graceful_no_sdk(self) -> None:
        """Given missing google-cloud-documentai SDK, When processing, Then RuntimeError is raised gracefully.

        We simulate this by patching HAS_DOCUMENT_AI to False within the engine module.
        """
        from unittest.mock import patch

        engine = GcpDocumentAiEngine()

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("backend.engines.gcp_document_ai.HAS_DOCUMENT_AI", False),
            pytest.raises(RuntimeError, match="google-cloud-documentai"),
        ):
            await engine.process_pdf("/fake/path.pdf")


class TestGcpRegistry:
    """Verify that GcpDocumentAiEngine is registered with the global registry."""

    def test_gcp_registered_in_registry(self) -> None:
        """Given the global registry, When retrieving by ID, Then GcpDocumentAiEngine is returned."""
        from backend.engine.registry import registry as global_registry

        engine = global_registry.get("gcp-document-ai")
        assert isinstance(engine, GcpDocumentAiEngine)
        assert engine.engine_id == "gcp-document-ai"

    def test_gcp_in_registry_list(self) -> None:
        """Given the global registry, When listing engines, Then gcp-document-ai is present."""
        from backend.engine.registry import registry as global_registry

        engine_ids = [e.engine_id for e in global_registry.list()]
        assert "gcp-document-ai" in engine_ids
