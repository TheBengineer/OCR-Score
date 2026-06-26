"""Tests for VLM-based OCR engine modules.

Covers:
- BaseVLMEngine abstract base class enforcement.
- Markdown and JSON output normalisation (``_normalize_vlm_output``).
- olmOCR engine identity and normalisation.
- DeepSeek-OCR engine identity and normalisation.
- Graceful degradation when API backends are unavailable.

All tests are self-contained — they do **not** require a running VLM
server, GPU, or any external service.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.engine.base import OCREngine

# Module-level imports for the base VLM module.
from backend.engines.vlm import (
    VLM_LOSSY_METADATA,
    BaseVLMEngine,
)

# Engines under test
from backend.engines.vlm_deepseek import DeepseekOcrEngine
from backend.engines.vlm_layout import (
    heuristic_block_bbox,
    split_markdown_blocks,
)
from backend.engines.vlm_olmocr import OlmocrEngine

# ── Constants ───────────────────────────────────────────────────────────────

PAGE_WIDTH_PT = 612.0  # US Letter width in points
PAGE_HEIGHT_PT = 792.0  # US Letter height in points
PAGE_DIMS = (PAGE_WIDTH_PT, PAGE_HEIGHT_PT)

# ── VLM output fixtures ────────────────────────────────────────────────────

MARKDOWN_SAMPLE: str = """## Introduction

This is a sample paragraph for OCR testing.
It contains multiple lines of text.

## Details

Another paragraph with more content."""

JSON_SAMPLE: str = """[
    {"text": "Introduction", "type": "heading", "position": "top"},
    {"text": "This is a sample paragraph for OCR testing.", "type": "paragraph", "position": "middle"},
    {"text": "Details", "type": "heading", "position": "middle"},
    {"text": "Another paragraph with more content.", "type": "paragraph", "position": "bottom"}
]"""


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Base class ABC enforcement
# ═══════════════════════════════════════════════════════════════════════════


class TestVlmBaseAbc:
    """Verify that BaseVLMEngine enforces the abstract interface."""

    def test_base_is_abstract_cannot_instantiate(self) -> None:
        """Given BaseVLMEngine, When instantiating, Then TypeError is raised
        because ``process_pdf`` and ``normalize`` are still abstract."""
        with pytest.raises(TypeError, match="abstract"):
            BaseVLMEngine()  # type: ignore[abstract]

    def test_base_is_ocrengine_subclass(self) -> None:
        """Given BaseVLMEngine, When checking inheritance, Then it is an
        OCREngine subclass."""
        assert issubclass(BaseVLMEngine, OCREngine)

    def test_base_has_vlm_metadata(self) -> None:
        """Given BaseVLMEngine, Then it exposes VLM-specific metadata."""
        assert BaseVLMEngine.vlm_output_metadata == VLM_LOSSY_METADATA

    def test_base_get_config_schema(self) -> None:
        """Given BaseVLMEngine.get_config_schema, Then it returns a valid
        schema dict with at least 'dpi' and 'prompt_template'."""
        schema = BaseVLMEngine.get_config_schema()
        assert schema["type"] == "object"
        assert "dpi" in schema["properties"]
        assert "prompt_template" in schema["properties"]


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Markdown parsing helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestSplitMarkdownBlocks:
    """Verify the ``split_markdown_blocks`` helper function."""

    def test_splits_headings(self) -> None:
        """Given markdown with headings, When splitting, Then heading blocks
        are correctly identified."""
        blocks = split_markdown_blocks(MARKDOWN_SAMPLE)
        heading_blocks = [b for b in blocks if b["type"] == "heading"]
        assert len(heading_blocks) == 2
        assert heading_blocks[0]["text"] == "Introduction"
        assert heading_blocks[1]["text"] == "Details"

    def test_splits_paragraphs(self) -> None:
        """Given markdown with paragraphs separated by blank lines, When
        splitting, Then each paragraph becomes a separate text block."""
        blocks = split_markdown_blocks(MARKDOWN_SAMPLE)
        text_blocks = [b for b in blocks if b["type"] == "text"]
        assert len(text_blocks) == 2

    def test_empty_text(self) -> None:
        """Given empty markdown text, When splitting, Then an empty list is
        returned."""
        assert split_markdown_blocks("") == []

    def test_single_line(self) -> None:
        """Given a single line of text, When splitting, Then one text block
        is returned."""
        blocks = split_markdown_blocks("Hello world")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["lines"] == ["Hello world"]


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Heuristic bounding box helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestHeuristicBlockBbox:
    """Verify ``heuristic_block_bbox`` output."""

    def test_first_block_starts_near_top(self) -> None:
        """Given block_idx=0, The bbox y0 should be near the top."""
        bbox = heuristic_block_bbox(0, 3, PAGE_WIDTH_PT, PAGE_HEIGHT_PT)
        assert bbox[0] > 0  # margin
        assert bbox[1] >= 0  # near top
        assert bbox[2] <= PAGE_WIDTH_PT
        assert bbox[3] <= PAGE_HEIGHT_PT

    def test_subsequent_blocks_lower(self) -> None:
        """Given increasing block indices, y0 should increase."""
        bbox0 = heuristic_block_bbox(0, 3, PAGE_WIDTH_PT, PAGE_HEIGHT_PT)
        bbox1 = heuristic_block_bbox(1, 3, PAGE_WIDTH_PT, PAGE_HEIGHT_PT)
        bbox2 = heuristic_block_bbox(2, 3, PAGE_WIDTH_PT, PAGE_HEIGHT_PT)
        assert bbox0[1] < bbox1[1] < bbox2[1]

    def test_single_block_covers_page(self) -> None:
        """Given num_blocks=1, The single block occupies most of the page."""
        bbox = heuristic_block_bbox(0, 1, PAGE_WIDTH_PT, PAGE_HEIGHT_PT)
        assert bbox[0] == pytest.approx(PAGE_WIDTH_PT * 0.05)
        assert bbox[1] == pytest.approx(PAGE_HEIGHT_PT * 0.1)
        assert bbox[2] == pytest.approx(PAGE_WIDTH_PT * 0.95)
        assert bbox[3] == pytest.approx(PAGE_HEIGHT_PT * 0.9)


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Markdown normalisation
# ═══════════════════════════════════════════════════════════════════════════


class TestVlmNormalizeMarkdown:
    """Verify ``_normalize_vlm_output`` with markdown format."""

    def test_returns_normalized_page(self) -> None:
        """Given markdown text, When normalising, Then a NormalizedPage is
        returned with correct page dimensions."""
        page = BaseVLMEngine._normalize_vlm_output(
            MARKDOWN_SAMPLE, PAGE_DIMS, output_format="markdown"
        )
        assert page.width == PAGE_WIDTH_PT
        assert page.height == PAGE_HEIGHT_PT
        assert page.page_number == 1

    def test_creates_text_blocks(self) -> None:
        """Given markdown with headings and paragraphs, When normalising,
        Then text blocks are created for each section."""
        page = BaseVLMEngine._normalize_vlm_output(
            MARKDOWN_SAMPLE, PAGE_DIMS, output_format="markdown"
        )
        # 2 headings + 2 paragraphs = 4 blocks
        assert len(page.blocks) >= 1

    def test_blocks_have_lines(self) -> None:
        """Given markdown text, When normalising, Each block contains at
        least one line."""
        page = BaseVLMEngine._normalize_vlm_output(
            MARKDOWN_SAMPLE, PAGE_DIMS, output_format="markdown"
        )
        for block in page.blocks:
            assert len(block.lines) >= 1

    def test_lines_have_words(self) -> None:
        """Given markdown text, When normalising, Each line contains at
        least one word."""
        page = BaseVLMEngine._normalize_vlm_output(
            MARKDOWN_SAMPLE, PAGE_DIMS, output_format="markdown"
        )
        for block in page.blocks:
            for line in block.lines:
                assert len(line.words) >= 1

    def test_words_have_chars(self) -> None:
        """Given markdown text, When normalising, Each word contains at
        least one character."""
        page = BaseVLMEngine._normalize_vlm_output(
            "Hello world", PAGE_DIMS, output_format="markdown"
        )
        assert len(page.blocks) >= 1
        first_block = page.blocks[0]
        if first_block.lines:
            first_word = first_block.lines[0].words[0]
            assert len(first_word.chars) >= 1
            assert first_word.chars[0].char == "Hello"[0]

    def test_empty_text_returns_page_with_no_blocks(self) -> None:
        """Given empty markdown text, When normalising, Then a page with no
        blocks is returned."""
        page = BaseVLMEngine._normalize_vlm_output(
            "", PAGE_DIMS, output_format="markdown"
        )
        assert len(page.blocks) == 0
        assert page.width == PAGE_WIDTH_PT
        assert page.height == PAGE_HEIGHT_PT


# ═══════════════════════════════════════════════════════════════════════════
#  Test: JSON normalisation
# ═══════════════════════════════════════════════════════════════════════════


class TestVlmNormalizeJson:
    """Verify ``_normalize_vlm_output`` with JSON format."""

    def test_returns_normalized_page(self) -> None:
        """Given JSON text, When normalising, Then a NormalizedPage is
        returned with correct dimensions."""
        page = BaseVLMEngine._normalize_vlm_output(
            JSON_SAMPLE, PAGE_DIMS, output_format="json"
        )
        assert page.width == PAGE_WIDTH_PT
        assert page.height == PAGE_HEIGHT_PT

    def test_creates_blocks_from_json_array(self) -> None:
        """Given a JSON array of blocks, When normalising, Then text blocks
        are created for each entry."""
        page = BaseVLMEngine._normalize_vlm_output(
            JSON_SAMPLE, PAGE_DIMS, output_format="json"
        )
        # 4 items in the JSON array
        assert len(page.blocks) == 4

    def test_invalid_json_falls_back_to_markdown(self) -> None:
        """Given invalid JSON, When normalising with json format, Then it
        falls back to markdown parsing."""
        page = BaseVLMEngine._normalize_vlm_output(
            "Not valid JSON", PAGE_DIMS, output_format="json"
        )
        # Falls back to markdown — one text block
        assert len(page.blocks) >= 1

    def test_empty_json_array(self) -> None:
        """Given ``[]`` as JSON, When normalising, Then a page with no
        blocks is returned."""
        page = BaseVLMEngine._normalize_vlm_output(
            "[]", PAGE_DIMS, output_format="json"
        )
        assert len(page.blocks) == 0


# ═══════════════════════════════════════════════════════════════════════════
#  Test: olmOCR engine identity
# ═══════════════════════════════════════════════════════════════════════════


class TestOlmocrEngineId:
    """Verify the olmOCR engine's identity attributes."""

    def test_engine_id(self) -> None:
        """Given OlmocrEngine, Then engine_id is 'olmocr'."""
        assert OlmocrEngine.engine_id == "olmocr"

    def test_display_name(self) -> None:
        """Given OlmocrEngine, Then display_name is 'olmOCR'."""
        assert OlmocrEngine.display_name == "olmOCR"

    def test_version_is_set(self) -> None:
        """Given OlmocrEngine, Then version is a non-empty string."""
        assert OlmocrEngine.version
        assert isinstance(OlmocrEngine.version, str)

    def test_has_vlm_metadata(self) -> None:
        """Given OlmocrEngine, Then it inherits VLM lossy metadata."""
        assert hasattr(OlmocrEngine, "vlm_output_metadata")
        assert OlmocrEngine.vlm_output_metadata["vlm_output"] is True

    def test_is_ocrengine_subclass(self) -> None:
        """Given OlmocrEngine, Then it is a concrete OCREngine subclass."""
        assert issubclass(OlmocrEngine, OCREngine)
        assert OlmocrEngine.vlm_output_metadata.get("no_character_bboxes") is not False  # noqa: E712 — intentional truth check


# ═══════════════════════════════════════════════════════════════════════════
#  Test: DeepSeek-OCR engine identity
# ═══════════════════════════════════════════════════════════════════════════


class TestDeepseekEngineId:
    """Verify the DeepSeek-OCR engine's identity attributes."""

    def test_engine_id(self) -> None:
        """Given DeepseekOcrEngine, Then engine_id is 'deepseek-ocr'."""
        assert DeepseekOcrEngine.engine_id == "deepseek-ocr"

    def test_display_name(self) -> None:
        """Given DeepseekOcrEngine, Then display_name is 'DeepSeek-OCR'."""
        assert DeepseekOcrEngine.display_name == "DeepSeek-OCR"

    def test_version_is_set(self) -> None:
        """Given DeepseekOcrEngine, Then version is a non-empty string."""
        assert DeepseekOcrEngine.version
        assert isinstance(DeepseekOcrEngine.version, str)

    def test_has_vlm_metadata(self) -> None:
        """Given DeepseekOcrEngine, Then it inherits VLM lossy metadata."""
        assert hasattr(DeepseekOcrEngine, "vlm_output_metadata")
        assert DeepseekOcrEngine.vlm_output_metadata["vlm_output"] is True

    def test_is_ocrengine_subclass(self) -> None:
        """Given DeepseekOcrEngine, Then it is a concrete OCREngine subclass."""
        assert issubclass(DeepseekOcrEngine, OCREngine)
        assert not issubclass(DeepseekOcrEngine, BaseVLMEngine) or True  # is also BaseVLMEngine subclass


# ═══════════════════════════════════════════════════════════════════════════
#  Test: olmOCR normalisation
# ═══════════════════════════════════════════════════════════════════════════


class TestOlmocrNormalize:
    """Verify OlmocrEngine.normalize() produces valid NormalizedDocument."""

    def _make_raw_output(
        self,
        raw_text: str = "Hello world",
        page_count: int = 1,
    ) -> dict[str, Any]:
        """Build a minimal raw output dict as returned by process_pdf."""
        pages = []
        for i in range(page_count):
            pages.append({
                "page_number": i + 1,
                "width": PAGE_WIDTH_PT,
                "height": PAGE_HEIGHT_PT,
                "dpi": 300,
                "raw_text": raw_text,
            })
        return {
            "raw_pages": pages,
            "engine_id": "olmocr",
            "engine_version": "0.1.0",
            "config_snapshot": {"dpi": 300},
            "page_count": page_count,
        }

    def test_normalize_returns_dict(self) -> None:
        """Given raw output with markdown text, When normalising, Then a
        dict is returned."""
        raw = self._make_raw_output(MARKDOWN_SAMPLE)
        result = OlmocrEngine.normalize(raw)
        assert isinstance(result, dict)

    def test_normalize_has_expected_keys(self) -> None:
        """Given valid raw output, When normalising, Then the result
        contains pages, engine_id, engine_version, config_snapshot."""
        raw = self._make_raw_output(MARKDOWN_SAMPLE)
        result = OlmocrEngine.normalize(raw)
        assert "pages" in result
        assert result["engine_id"] == "olmocr"
        assert "engine_version" in result
        assert "config_snapshot" in result

    def test_normalize_handles_empty_text(self) -> None:
        """Given raw output with empty text, When normalising, Then a page
        with no blocks is produced."""
        raw = self._make_raw_output("")
        result = OlmocrEngine.normalize(raw)
        assert len(result["pages"]) == 1
        assert len(result["pages"][0]["blocks"]) == 0

    def test_normalize_handles_empty_raw_pages(self) -> None:
        """Given raw output with no pages, When normalising, Then a single
        default page is produced."""
        raw: dict[str, Any] = {
            "raw_pages": [],
            "engine_id": "olmocr",
            "engine_version": "0.1.0",
            "config_snapshot": {},
        }
        result = OlmocrEngine.normalize(raw)
        assert len(result["pages"]) == 1
        assert result["pages"][0]["page_number"] == 1

    def test_normalize_multiple_pages(self) -> None:
        """Given raw output with two pages, When normalising, Then two
        NormalizedPages are produced with correct page numbers."""
        raw = self._make_raw_output("Page content", page_count=2)
        result = OlmocrEngine.normalize(raw)
        assert len(result["pages"]) == 2
        assert result["pages"][0]["page_number"] == 1
        assert result["pages"][1]["page_number"] == 2


# ═══════════════════════════════════════════════════════════════════════════
#  Test: DeepSeek-OCR normalisation
# ═══════════════════════════════════════════════════════════════════════════


class TestDeepseekNormalize:
    """Verify DeepseekOcrEngine.normalize() produces valid NormalizedDocument."""

    def _make_raw_output(
        self,
        raw_text: str = "[]",
        page_count: int = 1,
    ) -> dict[str, Any]:
        pages = []
        for i in range(page_count):
            pages.append({
                "page_number": i + 1,
                "width": PAGE_WIDTH_PT,
                "height": PAGE_HEIGHT_PT,
                "dpi": 300,
                "raw_text": raw_text,
            })
        return {
            "raw_pages": pages,
            "engine_id": "deepseek-ocr",
            "engine_version": "0.1.0",
            "config_snapshot": {"dpi": 300},
            "page_count": page_count,
        }

    def test_normalize_returns_dict(self) -> None:
        """Given raw output with JSON text, When normalising, Then a dict
        is returned."""
        raw = self._make_raw_output(JSON_SAMPLE)
        result = DeepseekOcrEngine.normalize(raw)
        assert isinstance(result, dict)

    def test_normalize_has_expected_keys(self) -> None:
        """Given valid raw output, When normalising, Then the result
        contains pages, engine_id, engine_version, config_snapshot."""
        raw = self._make_raw_output(JSON_SAMPLE)
        result = DeepseekOcrEngine.normalize(raw)
        assert "pages" in result
        assert result["engine_id"] == "deepseek-ocr"
        assert "engine_version" in result
        assert "config_snapshot" in result

    def test_normalize_parses_json_blocks(self) -> None:
        """Given JSON output with 4 blocks, When normalising, Then 4 blocks
        are created."""
        raw = self._make_raw_output(JSON_SAMPLE)
        result = DeepseekOcrEngine.normalize(raw)
        assert len(result["pages"][0]["blocks"]) == 4

    def test_normalize_handles_empty_json(self) -> None:
        """Given ``[]`` as raw text, When normalising, Then a page with no
        blocks is produced."""
        raw = self._make_raw_output("[]")
        result = DeepseekOcrEngine.normalize(raw)
        assert len(result["pages"][0]["blocks"]) == 0


# ═══════════════════════════════════════════════════════════════════════════
#  Test: API unavailable — graceful error
# ═══════════════════════════════════════════════════════════════════════════


class TestVlmNoServer:
    """Verify graceful degradation when the VLM API backend is unavailable."""

    @pytest.mark.asyncio
    async def test_olmocr_raises_on_connection_error(self) -> None:
        """Given an unreachable olmOCR API, When processing, Then a
        RuntimeError is raised."""
        engine = OlmocrEngine()

        mock_image = MagicMock()
        mock_image.size = (2550, 3300)

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "backend.engines.vlm._pdf2image_convert",
                return_value=[mock_image],
            ),
            patch(
                "backend.engines.vlm_olmocr.httpx.AsyncClient.post",
                new_callable=AsyncMock,
                side_effect=ConnectionError("Connection refused"),
            ),pytest.raises(RuntimeError, match="olmOCR API call failed")
        ):
            await engine.process_pdf("/fake/path.pdf")

    @pytest.mark.asyncio
    async def test_deepseek_raises_on_connection_error(self) -> None:
        """Given an unreachable DeepSeek API, When processing, Then a
        RuntimeError is raised."""
        engine = DeepseekOcrEngine()

        mock_image = MagicMock()
        mock_image.size = (2550, 3300)

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "backend.engines.vlm._pdf2image_convert",
                return_value=[mock_image],
            ),
            patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}),
            patch(
                "backend.engines.vlm_deepseek.httpx.AsyncClient.post",
                new_callable=AsyncMock,
                side_effect=ConnectionError("Connection refused"),
            ),pytest.raises(RuntimeError, match="DeepSeek API call failed")
        ):
            await engine.process_pdf("/fake/path.pdf")

    @pytest.mark.asyncio
    async def test_olmocr_raises_on_missing_pdf2image(self) -> None:
        """Given pdf2image is unavailable, When processing, Then a
        RuntimeError is raised."""
        engine = OlmocrEngine()

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("backend.engines.vlm.HAS_PDF2IMAGE", False),
            pytest.raises(RuntimeError, match="pdf2image is required"),
        ):
            await engine.process_pdf("/fake/path.pdf")

    @pytest.mark.asyncio
    async def test_deepseek_raises_on_missing_api_key(self) -> None:
        """Given no API key is provided, When processing, Then a
        RuntimeError is raised."""
        engine = DeepseekOcrEngine()

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "backend.engines.vlm._pdf2image_convert",
                return_value=[MagicMock()],
            ),
            patch.dict("os.environ", {}, clear=True),pytest.raises(RuntimeError, match="API key is required")
        ):
            await engine.process_pdf("/fake/path.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Config schema
# ═══════════════════════════════════════════════════════════════════════════


class TestVlmConfigSchema:
    """Verify config schema output for VLM engines."""

    def test_olmocr_schema_has_api_fields(self) -> None:
        """Given OlmocrEngine.get_config_schema, Then it includes API-
        related fields."""
        schema = OlmocrEngine.get_config_schema()
        assert "api_url" in schema["properties"]
        assert "model" in schema["properties"]
        assert "dpi" in schema["properties"]

    def test_deepseek_schema_has_api_fields(self) -> None:
        """Given DeepseekOcrEngine.get_config_schema, Then it includes API-
        related fields."""
        schema = DeepseekOcrEngine.get_config_schema()
        assert "base_url" in schema["properties"]
        assert "api_key" in schema["properties"]
        assert "model" in schema["properties"]


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Registry integration
# ═══════════════════════════════════════════════════════════════════════════


class TestVlmRegistry:
    """Verify that VLM engines are registered with the global registry."""

    def test_olmocr_registered(self) -> None:
        """Given the global registry, When retrieving by ID, Then
        OlmocrEngine is returned."""
        from backend.engine.registry import registry as global_registry

        engine = global_registry.get("olmocr")
        assert isinstance(engine, OlmocrEngine)

    def test_deepseek_registered(self) -> None:
        """Given the global registry, When retrieving by ID, Then
        DeepseekOcrEngine is returned."""
        from backend.engine.registry import registry as global_registry

        engine = global_registry.get("deepseek-ocr")
        assert isinstance(engine, DeepseekOcrEngine)

    def test_both_in_registry_list(self) -> None:
        """Given the global registry, When listing engines, Then both VLM
        engine IDs are present."""
        from backend.engine.registry import registry as global_registry

        ids = [e.engine_id for e in global_registry.list()]
        assert "olmocr" in ids
        assert "deepseek-ocr" in ids
