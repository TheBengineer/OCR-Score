"""Tests for the OCR engine plugin interface.

Covers:
- ``OCREngine`` ABC cannot be instantiated directly.
- ``MockEngine`` conforms to the ABC contract.
- ``EngineRegistry`` register / get / list / discover.
- ``NormalizedDocument`` schema validation.
"""

from pathlib import Path

import pydantic
import pytest

from backend.engine.base import OCREngine
from backend.engine.normalized_schema import (
    Character,
    NormalizedDocument,
    NormalizedPage,
    TextBlock,
    TextLine,
    Word,
)
from backend.engine.registry import EngineRegistry, EngineRegistryError
from backend.mock_engine import ENGINE_ID, VERSION, MockEngine

# ── ABC contract tests ───────────────────────────────────────────────────────


class TestOCREngineABC:
    """Verify the OCREngine abstract base class contract."""

    def test_abc_cannot_instantiate(self) -> None:
        """When instantiating OCREngine directly, Then it raises TypeError (abstract)."""
        with pytest.raises(TypeError, match="abstract"):
            OCREngine()  # type: ignore[abstract]


class TestMockEngineConformance:
    """Verify MockEngine satisfies the OCREngine ABC."""

    def test_mock_engine_is_ocrenigne_subclass(self) -> None:
        """Given MockEngine, When checking its MRO, Then it subclasses OCREngine."""
        assert issubclass(MockEngine, OCREngine)

    def test_mock_engine_conforms_to_abc(self, mock_engine: MockEngine) -> None:
        """Given a MockEngine instance, When inspecting its interface, Then all ABC members exist."""
        # -- class-level attributes
        assert mock_engine.engine_id == "mock"
        assert mock_engine.display_name == "Mock Engine"
        assert mock_engine.version == "0.1.0"

        # -- required methods
        assert callable(mock_engine.get_config_schema)
        assert callable(mock_engine.process_pdf)
        assert callable(mock_engine.normalize)

        # -- method signatures
        schema = mock_engine.get_config_schema()
        assert isinstance(schema, dict)

        # process_pdf must be a coroutine
        import asyncio

        assert asyncio.iscoroutinefunction(mock_engine.process_pdf)

    def test_mock_engine_config_schema(self, mock_engine: MockEngine) -> None:
        """Given a MockEngine, When calling get_config_schema(), Then it returns valid JSON Schema."""
        schema = mock_engine.get_config_schema()

        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert schema["required"] == []

        seed_prop = schema["properties"]["seed"]
        assert seed_prop["type"] == "integer"
        assert seed_prop["default"] == 42


class TestMockEngineBehavior:
    """Verify MockEngine runtime behaviour (non-regression)."""

    @pytest.mark.asyncio
    async def test_mock_engine_returns_expected_schema(
        self,
        mock_engine: MockEngine,
        mock_pdf_path: Path,
    ) -> None:
        """Given default config, When processing, Then normalized output matches NormalizedDocument schema."""
        raw = await mock_engine.process_pdf(mock_pdf_path)
        normalized = mock_engine.normalize(raw)

        validated = NormalizedDocument(**normalized)

        assert validated.engine_id == ENGINE_ID
        assert validated.engine_version == VERSION
        assert validated.config_snapshot == {}

        assert 2 <= len(validated.pages) <= 3

        for page in validated.pages:
            _assert_valid_normalized_page(page, mock_pdf_path)

    @pytest.mark.asyncio
    async def test_mock_engine_deterministic(
        self,
        mock_engine: MockEngine,
        mock_pdf_path: Path,
    ) -> None:
        """Given a MockEngine with the same seed, When processing twice, Then outputs are identical."""
        config = {"seed": 99}

        raw1 = await mock_engine.process_pdf(mock_pdf_path, config)
        raw2 = await mock_engine.process_pdf(mock_pdf_path, config)

        assert raw1 == raw2

        norm1 = mock_engine.normalize(raw1)
        norm2 = mock_engine.normalize(raw2)

        assert norm1 == norm2

        config2 = {"seed": 100}
        raw3 = await mock_engine.process_pdf(mock_pdf_path, config2)
        assert raw1 != raw3

    @pytest.mark.asyncio
    async def test_mock_engine_progress_callback(
        self,
        mock_engine: MockEngine,
        mock_pdf_path: Path,
    ) -> None:
        """Given a MockEngine with a progress callback, When processing, Then callback is invoked with 0 then 100."""
        captured: list[int] = []

        def track_progress(value: int) -> None:
            captured.append(value)

        await mock_engine.process_pdf(mock_pdf_path, progress=track_progress)

        assert len(captured) >= 2
        assert captured[0] == 0
        assert captured[-1] == 100


# ── Registry tests ───────────────────────────────────────────────────────────


class TestEngineRegistry:
    """Verify EngineRegistry singleton behaviour."""

    def setup_method(self) -> None:
        """Reset the registry singleton before each test."""
        EngineRegistry._instance = None

    def test_engine_registry_register(self) -> None:
        """When registering MockEngine, Then it can be retrieved by engine_id."""
        registry = EngineRegistry()
        registry.register(MockEngine)

        engine = registry.get("mock")
        assert isinstance(engine, MockEngine)
        assert engine.engine_id == "mock"

    def test_engine_registry_register_duplicate(self) -> None:
        """When registering the same engine twice, Then EngineRegistryError is raised."""
        registry = EngineRegistry()
        registry.register(MockEngine)

        with pytest.raises(EngineRegistryError, match="already registered"):
            registry.register(MockEngine)

    def test_engine_registry_register_abstract(self) -> None:
        """When registering OCREngine itself, Then EngineRegistryError is raised."""
        registry = EngineRegistry()

        with pytest.raises(EngineRegistryError, match="abstract"):
            registry.register(OCREngine)  # type: ignore[abstract]

    def test_engine_registry_get_unknown(self) -> None:
        """When getting an unregistered engine, Then EngineRegistryError is raised."""
        registry = EngineRegistry()
        registry.register(MockEngine)

        with pytest.raises(EngineRegistryError, match="No engine registered"):
            registry.get("nonexistent")

    def test_engine_registry_list(self) -> None:
        """When listing engines, Then all registered engines are returned."""
        registry = EngineRegistry()
        registry.register(MockEngine)

        engines = registry.list()
        assert len(engines) == 1
        assert isinstance(engines[0], MockEngine)

    def test_engine_registry_list_empty(self) -> None:
        """When no engines are registered, Then list is empty."""
        registry = EngineRegistry()
        assert registry.list() == []

    def test_engine_registry_discover(self) -> None:
        """When discovering engines, Then MockEngine is found in the engine/ package."""
        registry = EngineRegistry()
        registry.discover()

        engines = registry.list()
        engine_ids = [e.engine_id for e in engines]
        assert "mock" in engine_ids

    def test_engine_registry_singleton(self) -> None:
        """When creating multiple EngineRegistry instances, Then they share the same state."""
        r1 = EngineRegistry()
        r2 = EngineRegistry()

        assert r1 is r2

        r1.register(MockEngine)
        assert len(r2.list()) == 1


# ── Normalized schema validation tests ───────────────────────────────────────


class TestNormalizedSchemaValidation:
    """Verify NormalizedDocument schema validation."""

    def test_normalized_schema_validation(self) -> None:
        """Given valid normalized data, When validating through NormalizedDocument, Then it passes."""
        doc = NormalizedDocument(
            pages=[
                NormalizedPage(
                    page_number=1,
                    width=612.0,
                    height=792.0,
                    blocks=[
                        TextBlock(
                            type="text",
                            bbox=[72.0, 72.0, 540.0, 88.0],
                            confidence=0.95,
                            order=0,
                            lines=[
                                TextLine(
                                    text="hello world",
                                    bbox=[72.0, 72.0, 200.0, 88.0],
                                    confidence=0.95,
                                    order=0,
                                    words=[
                                        Word(
                                            text="hello",
                                            bbox=[72.0, 72.0, 120.0, 88.0],
                                            confidence=0.95,
                                            order=0,
                                            chars=[
                                                Character(
                                                    char="h",
                                                    bbox=[72.0, 72.0, 79.0, 88.0],
                                                    confidence=0.95,
                                                    order=0,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                    tables=[],
                ),
            ],
            engine_id="test",
            engine_version="1.0.0",
            config_snapshot={},
        )

        serialised = doc.model_dump()
        roundtripped = NormalizedDocument(**serialised)

        assert roundtripped.engine_id == "test"
        assert len(roundtripped.pages) == 1
        assert roundtripped.pages[0].blocks[0].type == "text"

    def test_normalized_schema_invalid_data(self) -> None:
        """Given invalid normalized data, When validating through NormalizedDocument, Then it raises ValidationError."""
        with pytest.raises(Exception) as exc_info:
            NormalizedDocument(
                pages=[
                    NormalizedPage(
                        page_number=0,  # ge=1
                        width=-1,  # gt=0
                        height=792.0,
                        blocks=[],
                        tables=[],
                    ),
                ],
                engine_id="test",
                engine_version="1.0.0",
                config_snapshot={},
            )

        # Pydantic v2 raises ValidationError
        import pydantic

        assert issubclass(type(exc_info.value), pydantic.ValidationError)

    def test_normalized_schema_confidence_bounds(self) -> None:
        """Given confidence outside [0,1], When validating, Then it raises ValidationError."""
        with pytest.raises(pydantic.ValidationError):
            Character(
                char="a",
                bbox=[0.0, 0.0, 10.0, 10.0],
                confidence=1.5,  # ge=0.0, le=1.0
                order=0,
            )

    def test_normalized_schema_bbox_length(self) -> None:
        """Given bbox with wrong number of elements, When validating, Then it raises ValidationError."""
        with pytest.raises(pydantic.ValidationError):
            Character(
                char="a",
                bbox=[0.0, 0.0, 10.0],  # only 3 elements, needs 4
                confidence=0.9,
                order=0,
            )


# ── Normalized page assertion helpers ────────────────────────────────────────


def _assert_valid_normalized_page(page: NormalizedPage, path: Path) -> None:  # noqa: ARG001 — fixture kept for future use
    """Assert that a single NormalizedPage has valid structure and values."""
    assert isinstance(page.page_number, int)
    assert page.page_number >= 1
    assert page.width > 0
    assert page.height > 0

    assert 1 <= len(page.blocks) <= 3

    for block in page.blocks:
        _assert_valid_normalized_block(block)


def _assert_valid_normalized_block(block: TextBlock) -> None:
    """Assert that a block has valid structure."""
    assert block.type in ("text", "table", "figure", "math", "separator")
    assert 0.0 <= block.confidence <= 1.0
    _assert_valid_bbox_list(block.bbox)

    if isinstance(block, TextBlock):
        assert 1 <= len(block.lines) <= 2
        for line in block.lines:
            _assert_valid_normalized_line(line)


def _assert_valid_normalized_line(line: TextLine) -> None:
    """Assert that a line has valid structure."""
    assert isinstance(line.text, str)
    assert len(line.text) > 0
    assert 0.0 <= line.confidence <= 1.0
    _assert_valid_bbox_list(line.bbox)
    assert 3 <= len(line.words) <= 8

    for word in line.words:
        _assert_valid_normalized_word(word)


def _assert_valid_normalized_word(word: Word) -> None:
    """Assert that a word has valid structure."""
    assert isinstance(word.text, str)
    assert 2 <= len(word.text) <= 10
    assert word.text.islower()
    assert word.text.isascii()
    assert 0.0 <= word.confidence <= 1.0
    _assert_valid_bbox_list(word.bbox)
    assert 2 <= len(word.chars) <= 10

    for char in word.chars:
        _assert_valid_normalized_char(char)


def _assert_valid_normalized_char(char: Character) -> None:
    """Assert that a character has valid structure."""
    assert isinstance(char.char, str)
    assert len(char.char) == 1
    assert char.char.islower()
    assert 0.0 <= char.confidence <= 1.0
    _assert_valid_bbox_list(char.bbox)


def _assert_valid_bbox_list(bbox: list[float]) -> None:
    """Assert that a bbox list is valid (positive area, within page bounds)."""
    assert len(bbox) == 4
    x0, y0, x1, y1 = bbox
    assert x0 < x1
    assert y0 < y1
    assert x0 >= 0.0
    assert y0 >= 0.0
    assert x1 <= 612.0  # PAGE_WIDTH
    assert y1 <= 792.0  # PAGE_HEIGHT
