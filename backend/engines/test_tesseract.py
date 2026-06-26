"""Tests for the Tesseract OCR engine module.

The test suite covers:
- ABC conformance (``OCREngine`` subclass, attributes, methods).
- Config schema validation.
- Raw output normalisation (``normalize()`` with mock data).
- Confidence value parsing.
- Bounding box coordinate transformation (pixel → page-space points).

All tests are self-contained — they do **not** require a Tesseract binary,
poppler-utils, or any external service.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.engine.base import OCREngine
from backend.engine.normalized_schema import Character, NormalizedDocument, TextBlock

# Module to test
from backend.engines.tesseract import (
    TesseractEngine,
    _parse_boxes,
    _parse_confidence,
    _pixel_to_point,
)

# ── Pillow image factory (used for mocking pdf2image) ───────────────────────


def _make_mock_image(width_px: int = 2550, height_px: int = 3300) -> MagicMock:
    """Create a mock PIL Image with the given pixel dimensions."""
    img = MagicMock()
    img.size = (width_px, height_px)
    return img


# ── Mock pytesseract data factories ─────────────────────────────────────────


def _make_mock_image_to_data(
    words: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a synthetic ``image_to_data(…, output_type=Output.DICT)`` result.

    Each entry in *words* may contain:
        - ``text`` (str, required)
        - ``left``, ``top``, ``width``, ``height`` (int, default 0)
        - ``conf`` (int/str, default ``'95'``)
        - ``block_num``, ``par_num``, ``line_num``, ``word_num`` (int, default 0)

    The returned dict includes the full hierarchy (page, block, para, line,
    word levels) so it mirrors a real Tesseract response.
    """
    words = words or []

    # Determine the number of distinct lines and blocks.
    seen_block_par_line: set[tuple[int, int, int]] = set()
    for w in words:
        bn = w.get("block_num", 0)
        pn = w.get("par_num", 0)
        ln = w.get("line_num", 0)
        seen_block_par_line.add((bn, pn, ln))
    has_content = len(words) > 0

    base: dict[str, list[Any]] = {
        "level": [],
        "page_num": [],
        "block_num": [],
        "par_num": [],
        "line_num": [],
        "word_num": [],
        "left": [],
        "top": [],
        "width": [],
        "height": [],
        "conf": [],
        "text": [],
    }

    # Page-level entry (level 1)
    base["level"].append(1)
    base["page_num"].append(1)
    for key in ("block_num", "par_num", "line_num", "word_num"):
        base[key].append(0)
    base["left"].append(0)
    base["top"].append(0)
    base["width"].append(0)
    base["height"].append(0)
    base["conf"].append("-1")
    base["text"].append("")

    if has_content:
        for (bn, pn, ln) in sorted(seen_block_par_line):
            # Block-level entry (level 2)
            base["level"].append(2)
            base["page_num"].append(1)
            base["block_num"].append(bn)
            base["par_num"].append(0)
            base["line_num"].append(0)
            base["word_num"].append(0)
            base["left"].append(0)
            base["top"].append(0)
            base["width"].append(0)
            base["height"].append(0)
            base["conf"].append("-1")
            base["text"].append("")

            # Paragraph-level entry (level 3)
            base["level"].append(3)
            base["page_num"].append(1)
            base["block_num"].append(bn)
            base["par_num"].append(pn)
            base["line_num"].append(0)
            base["word_num"].append(0)
            base["left"].append(0)
            base["top"].append(0)
            base["width"].append(0)
            base["height"].append(0)
            base["conf"].append("-1")
            base["text"].append("")

            # Line-level entry (level 4)
            base["level"].append(4)
            base["page_num"].append(1)
            base["block_num"].append(bn)
            base["par_num"].append(pn)
            base["line_num"].append(ln)
            base["word_num"].append(0)
            base["left"].append(0)
            base["top"].append(0)
            base["width"].append(0)
            base["height"].append(0)
            base["conf"].append("-1")
            base["text"].append("")

            # Word-level entries (level 5)
            for w in words:
                if (
                    w.get("block_num", 0) == bn
                    and w.get("par_num", 0) == pn
                    and w.get("line_num", 0) == ln
                ):
                    base["level"].append(5)
                    base["page_num"].append(1)
                    base["block_num"].append(bn)
                    base["par_num"].append(pn)
                    base["line_num"].append(ln)
                    base["word_num"].append(w.get("word_num", 0))
                    base["left"].append(w.get("left", 0))
                    base["top"].append(w.get("top", 0))
                    base["width"].append(w.get("width", 0))
                    base["height"].append(w.get("height", 0))
                    base["conf"].append(str(w.get("conf", 95)))
                    base["text"].append(w.get("text", ""))

    return base


def _make_mock_boxes(chars: list[dict[str, Any]]) -> str:
    """Build a synthetic ``image_to_boxes()`` result string.

    Each entry in *chars* may contain:
        - ``char`` (str, single character)
        - ``left``, ``bottom``, ``right``, ``top`` (int)
        - ``page`` (int, default 1)

    The returned string mirrors Tesseract's box format::

        <char> <left> <bottom> <right> <top> <page>
    """
    lines: list[str] = []
    for ch in chars:
        left = ch.get("left", 0)
        bottom = ch.get("bottom", 0)
        right = ch.get("right", 0)
        top = ch.get("top", 0)
        page = ch.get("page", 1)
        lines.append(f"{ch['char']} {left} {bottom} {right} {top} {page}")
    return "\n".join(lines)


def _make_mock_raw_page(
    *,
    page_number: int = 1,
    width: float = 612.0,
    height: float = 792.0,
    dpi: int = 300,
    word_data: dict[str, Any] | None = None,
    boxes_str: str = "",
) -> dict[str, Any]:
    """Build a single raw page entry as produced by ``process_pdf()``."""
    return {
        "page_number": page_number,
        "width": width,
        "height": height,
        "dpi": dpi,
        "image_to_data": word_data or _make_mock_image_to_data(),
        "image_to_boxes": boxes_str,
    }


def _make_minimal_raw(
    pages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a minimal raw output dict as returned by ``process_pdf()``."""
    return {
        "raw_pages": pages or [_make_mock_raw_page()],
        "engine_id": "tesseract",
        "engine_version": "0.3.13",
        "config_snapshot": {"lang": "eng", "psm": 3, "oem": 3, "dpi": 300},
        "page_count": len(pages) if pages else 1,
    }


# ── Test data ───────────────────────────────────────────────────────────────

SAMPLE_WORDS: list[dict[str, Any]] = [
    {
        "text": "Hello",
        "left": 100,
        "top": 50,
        "width": 150,
        "height": 40,
        "conf": 92,
        "block_num": 0,
        "par_num": 0,
        "line_num": 0,
        "word_num": 0,
    },
    {
        "text": "World",
        "left": 270,
        "top": 50,
        "width": 140,
        "height": 40,
        "conf": 85,
        "block_num": 0,
        "par_num": 0,
        "line_num": 0,
        "word_num": 1,
    },
]

# Character boxes for "Hello" and "World" at DPI 300 for a 612×792 pt page.
# Image dimensions: 2550 × 3300 px.
# Char boxes use bottom-left origin pixel coords.
# y_page = image_height - box_y → top-left origin.
SAMPLE_CHARACTERS: list[dict[str, Any]] = [
    # "Hello" characters — left=100 to left=250 approx
    {"char": "H", "left": 100, "bottom": 3230, "right": 125, "top": 3290},
    {"char": "e", "left": 128, "bottom": 3230, "right": 146, "top": 3290},
    {"char": "l", "left": 150, "bottom": 3230, "right": 168, "top": 3290},
    {"char": "l", "left": 172, "bottom": 3230, "right": 190, "top": 3290},
    {"char": "o", "left": 194, "bottom": 3230, "right": 220, "top": 3290},
    # "World" characters — left=270 to left=410 approx
    {"char": "W", "left": 270, "bottom": 3230, "right": 298, "top": 3290},
    {"char": "o", "left": 302, "bottom": 3230, "right": 320, "top": 3290},
    {"char": "r", "left": 324, "bottom": 3230, "right": 340, "top": 3290},
    {"char": "l", "left": 344, "bottom": 3230, "right": 362, "top": 3290},
    {"char": "d", "left": 366, "bottom": 3230, "right": 390, "top": 3290},
]


# ── ABC conformance ─────────────────────────────────────────────────────────


class TestTesseractABCConformance:
    """Verify that ``TesseractEngine`` satisfies the ``OCREngine`` ABC."""

    def test_tesseract_engine_is_ocrenigne_subclass(self) -> None:
        """Given TesseractEngine, When checking MRO, Then it is an OCREngine subclass."""
        assert issubclass(TesseractEngine, OCREngine)

    def test_tesseract_engine_conforms_to_abc(self) -> None:
        """Given a TesseractEngine instance, When inspecting interface, Then all ABC members exist."""
        engine = TesseractEngine()

        # -- class-level attributes
        assert engine.engine_id == "tesseract"
        assert engine.display_name == "Tesseract OCR"
        assert isinstance(engine.version, str)
        assert len(engine.version) > 0

        # -- required methods
        assert callable(engine.get_config_schema)
        assert callable(engine.normalize)

        # process_pdf must be a coroutine
        import asyncio

        assert asyncio.iscoroutinefunction(engine.process_pdf)


# ── Config schema ───────────────────────────────────────────────────────────


class TestTesseractConfigSchema:
    """Verify ``get_config_schema()`` returns valid JSON Schema."""

    def test_tesseract_config_schema_structure(self) -> None:
        """Given a TesseractEngine, When calling get_config_schema(), Then it returns valid schema."""
        engine = TesseractEngine()
        schema = engine.get_config_schema()

        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert schema["required"] == []

    def test_tesseract_config_schema_has_expected_keys(self) -> None:
        """Given a TesseractEngine, When inspecting schema properties, Then all keys exist."""
        engine = TesseractEngine()
        props = engine.get_config_schema()["properties"]

        assert "lang" in props
        assert props["lang"]["type"] == "string"
        assert props["lang"]["default"] == "eng"

        assert "psm" in props
        assert props["psm"]["type"] == "integer"
        assert props["psm"]["default"] == 3

        assert "oem" in props
        assert props["oem"]["type"] == "integer"
        assert props["oem"]["default"] == 3

        assert "dpi" in props
        assert props["dpi"]["type"] == "integer"
        assert props["dpi"]["default"] == 300


# ── Confidence parsing ──────────────────────────────────────────────────────


class TestConfidenceParsing:
    """Verify ``_parse_confidence()`` handles all Tesseract confidence cases."""

    def test_confidence_positive_value(self) -> None:
        """Given a positive Tesseract confidence, When parsing, Then it is normalised to 0‑1."""
        assert _parse_confidence("95") == 0.95
        assert _parse_confidence("100") == 1.0
        assert _parse_confidence("0") == 0.0
        assert _parse_confidence(92) == 0.92

    def test_confidence_negative_value(self) -> None:
        """Given a negative confidence (-1), When parsing, Then it returns 0."""
        assert _parse_confidence("-1") == 0.0
        assert _parse_confidence(-1) == 0.0

    def test_confidence_none(self) -> None:
        """Given None, When parsing, Then it returns 0."""
        assert _parse_confidence(None) == 0.0

    def test_confidence_empty_string(self) -> None:
        """Given an empty string, When parsing, Then it returns 0."""
        assert _parse_confidence("") == 0.0

    def test_confidence_clamps_above_100(self) -> None:
        """Given a confidence above 100, When parsing, Then it is clamped to 1.0."""
        assert _parse_confidence("150") == 1.0


# ── Coordinate helpers ──────────────────────────────────────────────────────


class TestPixelToPoint:
    """Verify ``_pixel_to_point()`` coordinate conversion."""

    def test_pixel_to_point_at_300_dpi(self) -> None:
        """Given a pixel value at 300 DPI, When converting to points, Then result is correct."""
        # point = pixel * 72 / dpi
        assert _pixel_to_point(300, 300) == 72.0  # 1 inch at 300 DPI
        assert _pixel_to_point(150, 300) == 36.0
        assert _pixel_to_point(0, 300) == 0.0

    def test_pixel_to_point_at_72_dpi(self) -> None:
        """Given a pixel value at 72 DPI, When converting, Then pixel == point."""
        assert _pixel_to_point(100, 72) == 100.0
        assert _pixel_to_point(72, 72) == 72.0

    def test_pixel_to_point_at_custom_dpi(self) -> None:
        """Given a pixel value at a custom DPI, When converting, Then result is correct."""
        assert _pixel_to_point(200, 150) == pytest.approx(96.0)
        assert _pixel_to_point(50, 600) == pytest.approx(6.0)


# ── Parse boxes ─────────────────────────────────────────────────────────────


class TestParseBoxes:
    """Verify ``_parse_boxes()`` handles raw ``image_to_boxes`` strings."""

    def test_parse_boxes_basic(self) -> None:
        """Given a valid boxes string, When parsing, Then entries have expected fields."""
        boxes_str = "H 100 3230 125 3290 1\ne 128 3230 146 3290 1"
        entries = _parse_boxes(boxes_str)

        assert len(entries) == 2
        assert entries[0]["char"] == "H"
        assert entries[0]["left"] == 100
        assert entries[0]["bottom"] == 3230
        assert entries[0]["right"] == 125
        assert entries[0]["top"] == 3290
        assert entries[0]["page_num"] == 1

    def test_parse_boxes_empty(self) -> None:
        """Given an empty boxes string, When parsing, Then empty list is returned."""
        assert _parse_boxes("") == []
        assert _parse_boxes("   ") == []
        assert _parse_boxes("\n\n") == []

    def test_parse_boxes_skips_malformed_lines(self) -> None:
        """Given malformed lines, When parsing, Then they are skipped."""
        boxes_str = "H 100 3230 125 3290 1\nSKIP\n e 128 3230 146 3290 1"
        entries = _parse_boxes(boxes_str)
        assert len(entries) == 2  # Two good lines, one bad


# ── Normalize output ────────────────────────────────────────────────────────


class TestNormalizeOutput:
    """Verify ``normalize()`` produces correct ``NormalizedDocument`` structure."""

    def test_normalize_empty_page(self) -> None:
        """Given raw data with an empty page, When normalizing, Then a valid page with no blocks is produced."""
        raw = _make_minimal_raw(
            pages=[
                _make_mock_raw_page(
                    word_data=_make_mock_image_to_data(), boxes_str=""
                )
            ]
        )
        result = TesseractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert len(validated.pages) == 1
        assert validated.pages[0].page_number == 1
        assert validated.pages[0].blocks == []

    def test_normalize_no_text_detected(self) -> None:
        """Given raw data with all empty text entries, When normalizing, Then no blocks are produced."""
        words = [
            {
                "text": "   ",
                "left": 100,
                "top": 50,
                "width": 50,
                "height": 20,
                "conf": 0,
                "block_num": 0,
                "par_num": 0,
                "line_num": 0,
                "word_num": 0,
            },
        ]
        word_data = _make_mock_image_to_data(words)
        raw = _make_minimal_raw(
            pages=[_make_mock_raw_page(word_data=word_data)]
        )
        result = TesseractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert validated.pages[0].blocks == []

    def test_normalize_basic_page(self) -> None:
        """Given mock word data with characters, When normalizing, Then hierarchy is correct."""
        word_data = _make_mock_image_to_data(SAMPLE_WORDS)
        boxes_str = _make_mock_boxes(SAMPLE_CHARACTERS)
        raw = _make_minimal_raw(
            pages=[
                _make_mock_raw_page(
                    word_data=word_data, boxes_str=boxes_str
                )
            ]
        )
        result = TesseractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert len(validated.pages) == 1
        page = validated.pages[0]

        assert page.page_number == 1
        assert page.width == 612.0
        assert page.height == 792.0

        # Should have one block with one line containing two words
        assert len(page.blocks) == 1
        block = page.blocks[0]
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

        # Words should have matched characters
        for word in line.words:
            assert len(word.chars) == len(word.text)
            for char in word.chars:
                assert isinstance(char, Character)
                assert len(char.char) == 1
                assert 0.0 <= char.confidence <= 1.0
                assert char.order >= 0

    def test_normalize_metadata(self) -> None:
        """Given mock raw data, When normalizing, Then engine metadata is preserved."""
        raw = _make_minimal_raw(
            pages=[
                _make_mock_raw_page(
                    word_data=_make_mock_image_to_data(SAMPLE_WORDS),
                    boxes_str=_make_mock_boxes(SAMPLE_CHARACTERS),
                )
            ]
        )
        result = TesseractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert validated.engine_id == "tesseract"
        assert validated.engine_version == "0.3.13"
        assert validated.config_snapshot["lang"] == "eng"
        assert validated.config_snapshot["dpi"] == 300

    def test_normalize_confidence_zero_for_empty_text(self) -> None:
        """Given confidence -1 entries, When normalizing, Then they become 0.0."""
        words = [
            {
                "text": "Bad",
                "left": 10,
                "top": 10,
                "width": 80,
                "height": 30,
                "conf": -1,  # Tesseract -1 → should become 0
                "block_num": 0,
                "par_num": 0,
                "line_num": 0,
                "word_num": 0,
            },
        ]
        word_data = _make_mock_image_to_data(words)
        raw = _make_minimal_raw(
            pages=[_make_mock_raw_page(word_data=word_data, boxes_str="")]
        )
        result = TesseractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        word = validated.pages[0].blocks[0].lines[0].words[0]
        assert word.confidence == 0.0


# ── Bounding box normalisation ──────────────────────────────────────────────


class TestBoundingBoxNormalization:
    """Verify pixel → page-space coordinate conversion."""

    def test_bounding_box_conversion(self) -> None:
        """Given pixel coords at 300 DPI, When normalizing, Then coords are in points at 72 DPI."""
        scale = 72.0 / 300  # 0.24 per pixel

        words = [
            {
                "text": "Test",
                "left": 100,
                "top": 50,
                "width": 200,
                "height": 40,
                "conf": 95,
                "block_num": 0,
                "par_num": 0,
                "line_num": 0,
                "word_num": 0,
            },
        ]

        # Manually compute expected page-space bbox
        expected_x0 = 100 * scale  # = 24.0
        expected_y0 = 50 * scale  # = 12.0
        expected_x1 = (100 + 200) * scale  # = 72.0
        expected_y1 = (50 + 40) * scale  # = 21.6

        word_data = _make_mock_image_to_data(words)
        raw = _make_minimal_raw(
            pages=[
                _make_mock_raw_page(
                    dpi=300,
                    word_data=word_data,
                    boxes_str="",
                )
            ]
        )
        result = TesseractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        word = validated.pages[0].blocks[0].lines[0].words[0]

        assert word.bbox == pytest.approx(
            [expected_x0, expected_y0, expected_x1, expected_y1]
        )

    def test_bounding_box_at_different_dpi(self) -> None:
        """Given pixel coords at 72 DPI, When normalizing, Then pixel == point."""
        scale = 72.0 / 72  # = 1.0

        words = [
            {
                "text": "OneToOne",
                "left": 50,
                "top": 30,
                "width": 200,
                "height": 40,
                "conf": 90,
                "block_num": 0,
                "par_num": 0,
                "line_num": 0,
                "word_num": 0,
            },
        ]

        expected_x0 = 50 * scale
        expected_y0 = 30 * scale
        expected_x1 = (50 + 200) * scale
        expected_y1 = (30 + 40) * scale

        word_data = _make_mock_image_to_data(words)
        raw = _make_minimal_raw(
            pages=[
                _make_mock_raw_page(
                    dpi=72,
                    word_data=word_data,
                    boxes_str="",
                )
            ]
        )
        result = TesseractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        word = validated.pages[0].blocks[0].lines[0].words[0]

        assert word.bbox == pytest.approx(
            [expected_x0, expected_y0, expected_x1, expected_y1]
        )

    def test_character_bbox_conversion(self) -> None:
        """Given character pixel coords, When normalizing, Then char bboxes are in page-space."""
        scale = 72.0 / 300
        img_height_px = 792.0 / scale  # = 3300

        # Word "Hi" at pixel (100, 50) → page-space: y=12 to y=21.6
        words = [
            {
                "text": "Hi",
                "left": 100,
                "top": 50,
                "width": 60,
                "height": 40,
                "conf": 95,
                "block_num": 0,
                "par_num": 0,
                "line_num": 0,
                "word_num": 0,
            },
        ]

        # Character boxes must overlap the word bbox (pixel 100,50 to 160,90)
        # In bottom-left origin, word spans y=3210 to y=3250 (img_h=3300).
        # Char 'H': pixel left=100, right=128 → page-space x0=24, x1=30.72
        #            bottom=3210, top=3250 → y0=(3300-3250)*s=12, y1=(3300-3210)*s=21.6
        # Char 'i': pixel left=132, right=155 → page-space x0=31.68, x1=37.2
        chars = [
            {"char": "H", "left": 100, "bottom": 3210, "right": 128, "top": 3250, "page": 1},
            {"char": "i", "left": 132, "bottom": 3210, "right": 155, "top": 3250, "page": 1},
        ]

        # Expected page-space bboxes
        expected_char0_x0 = 100 * scale  # = 24.0
        expected_char0_x1 = 128 * scale  # = 30.72
        expected_char0_y0 = (img_height_px - 3250) * scale  # = 12.0
        expected_char0_y1 = (img_height_px - 3210) * scale  # = 21.6

        word_data = _make_mock_image_to_data(words)
        boxes_str = _make_mock_boxes(chars)
        raw = _make_minimal_raw(
            pages=[
                _make_mock_raw_page(
                    dpi=300,
                    word_data=word_data,
                    boxes_str=boxes_str,
                )
            ]
        )
        result = TesseractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        word = validated.pages[0].blocks[0].lines[0].words[0]

        assert len(word.chars) == 2
        assert word.chars[0].char == "H"
        assert word.chars[0].bbox == pytest.approx(
            [expected_char0_x0, expected_char0_y0, expected_char0_x1, expected_char0_y1]
        )

    def test_empty_page_bbox(self) -> None:
        """Given an empty page, When normalizing, Then width/height are preserved."""
        raw = _make_minimal_raw(
            pages=[
                _make_mock_raw_page(
                    width=841.0,  # A4 width in points
                    height=1189.0,  # A4 height in points
                )
            ]
        )
        result = TesseractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        page = validated.pages[0]
        assert page.width == 841.0
        assert page.height == 1189.0
        assert page.blocks == []


# ── Multiple pages and blocks ───────────────────────────────────────────────


class TestNormalizeMultiplePages:
    """Verify ``normalize()`` handles multiple pages and blocks."""

    def test_normalize_two_pages(self) -> None:
        """Given raw data with two pages, When normalizing, Then two pages are produced."""
        words_page1 = [
            {
                "text": "Page1",
                "left": 10,
                "top": 10,
                "width": 120,
                "height": 30,
                "conf": 95,
                "block_num": 0,
                "par_num": 0,
                "line_num": 0,
                "word_num": 0,
            },
        ]
        words_page2 = [
            {
                "text": "Page2",
                "left": 10,
                "top": 10,
                "width": 120,
                "height": 30,
                "conf": 90,
                "block_num": 0,
                "par_num": 0,
                "line_num": 0,
                "word_num": 0,
            },
        ]

        raw = _make_minimal_raw(
            pages=[
                _make_mock_raw_page(
                    page_number=1,
                    word_data=_make_mock_image_to_data(words_page1),
                ),
                _make_mock_raw_page(
                    page_number=2,
                    word_data=_make_mock_image_to_data(words_page2),
                ),
            ]
        )
        result = TesseractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert len(validated.pages) == 2
        assert validated.pages[0].page_number == 1
        assert validated.pages[1].page_number == 2
        assert (
            validated.pages[0].blocks[0].lines[0].words[0].text == "Page1"
        )
        assert (
            validated.pages[1].blocks[0].lines[0].words[0].text == "Page2"
        )

    def test_normalize_multiple_blocks(self) -> None:
        """Given words in different blocks, When normalizing, Then multiple blocks are produced."""
        words = [
            {
                "text": "Block1Word",
                "left": 10,
                "top": 10,
                "width": 200,
                "height": 30,
                "conf": 95,
                "block_num": 0,
                "par_num": 0,
                "line_num": 0,
                "word_num": 0,
            },
            {
                "text": "Block2Word",
                "left": 10,
                "top": 100,
                "width": 200,
                "height": 30,
                "conf": 85,
                "block_num": 1,
                "par_num": 0,
                "line_num": 0,
                "word_num": 0,
            },
        ]

        word_data = _make_mock_image_to_data(words)
        raw = _make_minimal_raw(
            pages=[_make_mock_raw_page(word_data=word_data)]
        )
        result = TesseractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        assert len(validated.pages[0].blocks) == 2
        assert (
            validated.pages[0].blocks[0].lines[0].words[0].text == "Block1Word"
        )
        assert (
            validated.pages[0].blocks[1].lines[0].words[0].text == "Block2Word"
        )


# ── Character fallback (no image_to_boxes) ──────────────────────────────────


class TestNormalizeCharacterFallback:
    """Verify char-level output when no ``image_to_boxes`` data is available."""

    def test_normalize_synthesises_chars_when_no_boxes(self) -> None:
        """Given word data but no boxes string, When normalizing, Then chars are synthesised from word text."""
        words = [
            {
                "text": "Hi",
                "left": 100,
                "top": 50,
                "width": 60,
                "height": 40,
                "conf": 95,
                "block_num": 0,
                "par_num": 0,
                "line_num": 0,
                "word_num": 0,
            },
        ]
        word_data = _make_mock_image_to_data(words)
        raw = _make_minimal_raw(
            pages=[_make_mock_raw_page(word_data=word_data, boxes_str="")]
        )
        result = TesseractEngine.normalize(raw)

        validated = NormalizedDocument(**result)
        word = validated.pages[0].blocks[0].lines[0].words[0]

        assert word.text == "Hi"
        assert len(word.chars) == 2
        assert word.chars[0].char == "H"
        assert word.chars[1].char == "i"
        assert 0.0 <= word.chars[0].confidence <= 1.0


# ── Process PDF (mocked) ────────────────────────────────────────────────────


class TestProcessPDF:
    """Verify ``process_pdf()`` with mocked dependencies.

    These tests do NOT require a Tesseract binary or poppler-utils.
    """

    @pytest.mark.asyncio
    async def test_process_pdf_calls_progress(self) -> None:
        """Given mock dependencies, When processing, Then progress callback is invoked."""
        engine = TesseractEngine()
        captured: list[int] = []

        def track(v: int) -> None:
            captured.append(v)

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "backend.engines.tesseract._pdf2image_convert",
                return_value=[_make_mock_image()],
            ),
            patch(
                "backend.engines.tesseract.pytesseract.image_to_data",
                return_value=_make_mock_image_to_data(SAMPLE_WORDS),
            ),
            patch(
                "backend.engines.tesseract.pytesseract.image_to_boxes",
                return_value=_make_mock_boxes(SAMPLE_CHARACTERS),
            ),
        ):
            await engine.process_pdf(
                "/fake/path.pdf",
                config={"lang": "eng", "dpi": 300},
                progress=track,
            )

        assert len(captured) >= 3
        assert captured[0] == 0
        assert captured[-1] == 100
        assert 10 <= captured[1] <= 99  # Per-page progress

    @pytest.mark.asyncio
    async def test_process_pdf_returns_expected_structure(self) -> None:
        """Given mock dependencies, When processing, Then raw output has expected keys."""
        engine = TesseractEngine()

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "backend.engines.tesseract._pdf2image_convert",
                return_value=[_make_mock_image()],
            ),
            patch(
                "backend.engines.tesseract.pytesseract.image_to_data",
                return_value=_make_mock_image_to_data(SAMPLE_WORDS),
            ),
            patch(
                "backend.engines.tesseract.pytesseract.image_to_boxes",
                return_value=_make_mock_boxes(SAMPLE_CHARACTERS),
            ),
        ):
            result = await engine.process_pdf(
                "/fake/path.pdf",
                config={"lang": "eng", "dpi": 300},
            )

        assert "raw_pages" in result
        assert result["engine_id"] == "tesseract"
        assert "engine_version" in result
        assert "config_snapshot" in result
        assert result["page_count"] == 1
        assert len(result["raw_pages"]) == 1

        page = result["raw_pages"][0]
        assert page["page_number"] == 1
        assert page["dpi"] == 300
        assert "image_to_data" in page
        assert "image_to_boxes" in page

    @pytest.mark.asyncio
    async def test_process_pdf_handles_missing_file(self) -> None:
        """Given a non-existent PDF path, When processing, Then FileNotFoundError is raised."""
        engine = TesseractEngine()

        with (
            patch("pathlib.Path.exists", return_value=False),
            pytest.raises(FileNotFoundError, match="not found"),
        ):
            await engine.process_pdf("/nonexistent/path.pdf")

    @pytest.mark.asyncio
    async def test_process_pdf_multiple_pages(self) -> None:
        """Given a 3-page PDF, When processing, Then each page has correct page_number."""
        engine = TesseractEngine()
        images = [_make_mock_image() for _ in range(3)]

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "backend.engines.tesseract._pdf2image_convert",
                return_value=images,
            ),
            patch(
                "backend.engines.tesseract.pytesseract.image_to_data",
                return_value=_make_mock_image_to_data(SAMPLE_WORDS),
            ),
            patch(
                "backend.engines.tesseract.pytesseract.image_to_boxes",
                return_value=_make_mock_boxes(SAMPLE_CHARACTERS),
            ),
        ):
            result = await engine.process_pdf("/fake/path.pdf")

        assert result["page_count"] == 3
        for i, page in enumerate(result["raw_pages"]):
            assert page["page_number"] == i + 1


# ── Registry integration ────────────────────────────────────────────────────


class TestTesseractRegistry:
    """Verify that TesseractEngine is registered with the global registry."""

    def test_tesseract_registered_in_registry(self) -> None:
        """Given the global registry, When retrieving by ID, Then TesseractEngine is returned."""
        from backend.engine.registry import registry as global_registry

        engine = global_registry.get("tesseract")
        assert isinstance(engine, TesseractEngine)
        assert engine.engine_id == "tesseract"

    def test_tesseract_in_registry_list(self) -> None:
        """Given the global registry, When listing engines, Then tesseract is present."""
        from backend.engine.registry import registry as global_registry

        engine_ids = [e.engine_id for e in global_registry.list()]
        assert "tesseract" in engine_ids
