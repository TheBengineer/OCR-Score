"""Tests for the mock OCR engine interface contract.

Tests verify that MockEngine implements the expected OCREngine protocol:
- get_config_schema() returns valid JSON Schema
- process_pdf() generates synthetic data without reading the PDF
- normalize() produces standardized PageResult output
- Deterministic output for the same seed
- Progress callback invocation
"""

from pathlib import Path

import pytest

from backend.mock_engine import (
    ENGINE_ID,
    PAGE_HEIGHT,
    PAGE_WIDTH,
    VERSION,
    BlockResult,
    BoundingBox,
    CharResult,
    LineResult,
    MockEngine,
    NormalizedResult,
    PageResult,
    WordResult,
)


class TestMockEngineInterface:
    """Verify MockEngine satisfies the OCREngine protocol contract."""

    def test_mock_engine_implements_interface(self, mock_engine: MockEngine) -> None:
        """Given a MockEngine instance, When inspecting its interface, Then all required protocol members exist."""
        # -- class-level attributes
        assert mock_engine.engine_id == "mock"
        assert mock_engine.display_name == "Mock Engine"
        assert mock_engine.version == "0.1.0"

        # -- required methods
        assert callable(mock_engine.get_config_schema)
        assert callable(mock_engine.process_pdf)
        assert callable(mock_engine.normalize)

        # -- method signatures (structural)
        schema = mock_engine.get_config_schema()
        assert isinstance(schema, dict)

        # process_pdf must be a coroutine
        import asyncio
        assert asyncio.iscoroutinefunction(mock_engine.process_pdf)

    @pytest.mark.asyncio
    async def test_mock_engine_returns_expected_schema(
        self, mock_engine: MockEngine, mock_pdf_path: Path,
    ) -> None:
        """Given default config, When processing, Then normalized output matches PageResult schema."""
        raw = await mock_engine.process_pdf(mock_pdf_path)
        normalized = mock_engine.normalize(raw)

        # -- validate with Pydantic models (schema compliance)
        validated = NormalizedResult(**normalized)

        assert validated.engine_id == ENGINE_ID
        assert validated.engine_version == VERSION
        assert validated.config_snapshot == {}

        assert 2 <= len(validated.pages) <= 3

        for page in validated.pages:
            _assert_valid_page(page)

    def test_mock_engine_config_schema(self, mock_engine: MockEngine) -> None:
        """Given a MockEngine, When calling get_config_schema(), Then it returns a valid JSON Schema."""
        schema = mock_engine.get_config_schema()

        # -- valid JSON Schema structure
        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert schema["required"] == []

        # -- seed property
        seed_prop = schema["properties"]["seed"]
        assert seed_prop["type"] == "integer"
        assert seed_prop["default"] == 42

    @pytest.mark.asyncio
    async def test_mock_engine_deterministic(
        self, mock_engine: MockEngine, mock_pdf_path: Path,
    ) -> None:
        """Given a MockEngine with the same seed, When processing twice, Then outputs are identical."""
        config = {"seed": 99}

        raw1 = await mock_engine.process_pdf(mock_pdf_path, config)
        raw2 = await mock_engine.process_pdf(mock_pdf_path, config)

        assert raw1 == raw2

        norm1 = mock_engine.normalize(raw1)
        norm2 = mock_engine.normalize(raw2)

        assert norm1 == norm2

        # -- different seed produces different output
        config2 = {"seed": 100}
        raw3 = await mock_engine.process_pdf(mock_pdf_path, config2)
        assert raw1 != raw3

    @pytest.mark.asyncio
    async def test_mock_engine_progress_callback(
        self, mock_engine: MockEngine, mock_pdf_path: Path,
    ) -> None:
        """Given a MockEngine with a progress callback, When processing, Then callback is invoked with 0 then 100."""
        captured: list[int] = []

        def track_progress(value: int) -> None:
            captured.append(value)

        await mock_engine.process_pdf(mock_pdf_path, progress=track_progress)

        assert len(captured) >= 2
        assert captured[0] == 0
        assert captured[-1] == 100


# ── Schema assertion helpers ─────────────────────────────────────────────────


def _assert_valid_page(page: PageResult) -> None:
    """Assert that a single page result has valid structure and values."""
    assert isinstance(page.page_number, int)
    assert page.page_number >= 1
    assert page.width == PAGE_WIDTH
    assert page.height == PAGE_HEIGHT
    assert isinstance(page.text, str)
    assert len(page.text) > 0
    assert 0.0 <= page.confidence <= 1.0
    assert 1 <= len(page.blocks) <= 3

    for block in page.blocks:
        _assert_valid_block(block)


def _assert_valid_block(block: BlockResult) -> None:
    """Assert that a block result has valid structure."""
    assert isinstance(block.text, str)
    assert len(block.text) > 0
    assert 0.0 <= block.confidence <= 1.0
    assert block.block_type == "text"
    _assert_valid_bbox(block.bbox)
    assert 1 <= len(block.lines) <= 2

    for line in block.lines:
        _assert_valid_line(line)


def _assert_valid_line(line: LineResult) -> None:
    """Assert that a line result has valid structure."""
    assert isinstance(line.text, str)
    assert len(line.text) > 0
    assert 0.0 <= line.confidence <= 1.0
    _assert_valid_bbox(line.bbox)
    assert 3 <= len(line.words) <= 8

    for word in line.words:
        _assert_valid_word(word)


def _assert_valid_word(word: WordResult) -> None:
    """Assert that a word result has valid structure."""
    assert isinstance(word.text, str)
    assert 2 <= len(word.text) <= 10
    assert word.text.islower()
    assert word.text.isascii()
    assert 0.0 <= word.confidence <= 1.0
    _assert_valid_bbox(word.bbox)
    assert 2 <= len(word.chars) <= 10

    for char in word.chars:
        _assert_valid_char(char)


def _assert_valid_char(char: CharResult) -> None:
    """Assert that a character result has valid structure."""
    assert isinstance(char.char, str)
    assert len(char.char) == 1
    assert char.char.islower()
    assert 0.0 <= char.confidence <= 1.0
    _assert_valid_bbox(char.bbox)


def _assert_valid_bbox(bbox: BoundingBox) -> None:
    """Assert that a BoundingBox is valid (positive area, within page bounds)."""
    assert bbox.x0 < bbox.x1
    assert bbox.y0 < bbox.y1
    assert bbox.x0 >= 0.0
    assert bbox.y0 >= 0.0
    assert bbox.x1 <= PAGE_WIDTH
    assert bbox.y1 <= PAGE_HEIGHT
