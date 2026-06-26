"""Mock OCR engine for CI testing — returns synthetic normalized page results without external APIs.

The engine generates deterministic page content from a seed value,
enabling pipeline testing in CI without cloud dependencies.
"""

from collections.abc import Callable
from pathlib import Path
from random import Random
from string import ascii_lowercase
from typing import Any, ClassVar, Final

from pydantic import BaseModel, ConfigDict

from backend.engine.base import OCREngine
from backend.engine.normalized_schema import (
    Character as NormalizedCharacter,
)
from backend.engine.normalized_schema import (
    NormalizedDocument,
    NormalizedPage,
)
from backend.engine.normalized_schema import (
    TableBlock as NormalizedTableBlock,
)
from backend.engine.normalized_schema import (
    TextBlock as NormalizedTextBlock,
)
from backend.engine.normalized_schema import (
    TextLine as NormalizedTextLine,
)
from backend.engine.normalized_schema import (
    Word as NormalizedWord,
)

ENGINE_ID: Final[str] = "mock"
DISPLAY_NAME: Final[str] = "Mock Engine"
VERSION: Final[str] = "0.1.0"
PAGE_WIDTH: Final[float] = 612.0
PAGE_HEIGHT: Final[float] = 792.0
MARGIN: Final[float] = 72.0
LINE_HEIGHT: Final[float] = 16.0
LINE_SPACING: Final[float] = 4.0
BLOCK_SPACING: Final[float] = 8.0
CHAR_WIDTH: Final[float] = 7.0
WORD_SPACING: Final[float] = 4.0


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in page-space coordinates (points, 72 DPI, top-left origin)."""

    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float


class CharResult(BaseModel):
    """Single character OCR result."""

    model_config = ConfigDict(frozen=True)

    char: str
    bbox: BoundingBox
    confidence: float


class WordResult(BaseModel):
    """Word-level OCR result with per-character details."""

    model_config = ConfigDict(frozen=True)

    text: str
    bbox: BoundingBox
    confidence: float
    chars: list[CharResult]


class LineResult(BaseModel):
    """Line-level OCR result with per-word details."""

    model_config = ConfigDict(frozen=True)

    text: str
    bbox: BoundingBox
    confidence: float
    words: list[WordResult]


class BlockResult(BaseModel):
    """Block-level OCR result with per-line details."""

    model_config = ConfigDict(frozen=True)

    text: str
    bbox: BoundingBox
    confidence: float
    lines: list[LineResult]
    block_type: str


class PageResult(BaseModel):
    """Single page OCR result."""

    model_config = ConfigDict(frozen=True)

    page_number: int
    width: float
    height: float
    blocks: list[BlockResult]
    text: str
    confidence: float


class NormalizedResult(BaseModel):
    """Top-level normalized OCR result containing all pages and metadata."""

    model_config = ConfigDict(frozen=True)

    pages: list[PageResult]
    engine_id: str
    engine_version: str
    config_snapshot: dict[str, Any]


# ── Synthetic data generators ────────────────────────────────────────────────


def _generate_word(rng: Random, x_start: float, y_start: float) -> tuple[WordResult, float]:
    """Generate a single synthetic word and return (word, next_x).

    The word contains 2-10 random lowercase ASCII characters.
    Each character gets an evenly-spaced bounding box within the word.
    """
    char_count = rng.randint(2, 10)
    chars: list[CharResult] = []
    word_chars: list[str] = []
    x = x_start

    for _ in range(char_count):
        c = rng.choice(ascii_lowercase)
        word_chars.append(c)
        conf = round(rng.uniform(0.7, 1.0), 4)
        char_bbox = BoundingBox(x0=x, y0=y_start, x1=x + CHAR_WIDTH, y1=y_start + LINE_HEIGHT)
        chars.append(CharResult(char=c, bbox=char_bbox, confidence=conf))
        x += CHAR_WIDTH

    word_text = "".join(word_chars)
    word_bbox = BoundingBox(x0=x_start, y0=y_start, x1=x, y1=y_start + LINE_HEIGHT)
    word_conf = round(sum(c.confidence for c in chars) / len(chars), 4)

    return WordResult(text=word_text, bbox=word_bbox, confidence=word_conf, chars=chars), x + WORD_SPACING


def _generate_line(rng: Random, x_start: float, y_start: float) -> tuple[LineResult, float]:
    """Generate a single synthetic line of 3-8 words and return (line, next_y)."""
    word_count = rng.randint(3, 8)
    words: list[WordResult] = []
    x = x_start

    for _ in range(word_count):
        word, x = _generate_word(rng, x, y_start)
        words.append(word)

    line_text = " ".join(w.text for w in words)
    line_conf = round(sum(w.confidence for w in words) / len(words), 4)
    line_bbox = BoundingBox(
        x0=words[0].bbox.x0,
        y0=y_start,
        x1=words[-1].bbox.x1,
        y1=y_start + LINE_HEIGHT,
    )

    return (
        LineResult(text=line_text, bbox=line_bbox, confidence=line_conf, words=words),
        y_start + LINE_HEIGHT + LINE_SPACING,
    )


def _generate_block(rng: Random, x_start: float, y_start: float) -> tuple[BlockResult, float]:
    """Generate a single synthetic block of 1-2 lines and return (block, next_y)."""
    line_count = rng.randint(1, 2)
    lines: list[LineResult] = []
    y = y_start

    for _ in range(line_count):
        line, y = _generate_line(rng, x_start, y)
        lines.append(line)

    block_text = "\n".join(line.text for line in lines)
    block_conf = round(sum(line.confidence for line in lines) / len(lines), 4)
    block_bbox = BoundingBox(
        x0=lines[0].bbox.x0,
        y0=y_start,
        x1=max(line.bbox.x1 for line in lines),
        y1=y,
    )

    return BlockResult(
        text=block_text,
        bbox=block_bbox,
        confidence=block_conf,
        lines=lines,
        block_type="text",
    ), y + BLOCK_SPACING


def _generate_page(rng: Random, page_number: int) -> PageResult:
    """Generate a single synthetic page with 1-3 blocks."""
    block_count = rng.randint(1, 3)
    blocks: list[BlockResult] = []
    y = MARGIN

    for _ in range(block_count):
        block, y = _generate_block(rng, MARGIN, y)
        blocks.append(block)

    page_text = "\n".join(b.text for b in blocks)
    page_conf = round(sum(b.confidence for b in blocks) / len(blocks), 4)

    return PageResult(
        page_number=page_number,
        width=PAGE_WIDTH,
        height=PAGE_HEIGHT,
        blocks=blocks,
        text=page_text,
        confidence=page_conf,
    )


def _generate_pages(rng: Random) -> list[PageResult]:
    """Generate 2-3 synthetic pages."""
    page_count = rng.randint(2, 3)
    return [_generate_page(rng, i + 1) for i in range(page_count)]


# ── Conversion helpers ────────────────────────────────────────────────────────


def _bbox_obj_to_list(bbox: dict[str, float]) -> list[float]:
    """Convert a ``{"x0": …, "y0": …, "x1": …, "y1": …}`` dict to a ``[x0, y0, x1, y1]`` list."""
    return [bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]]


# ── Engine class ─────────────────────────────────────────────────────────────


class MockEngine(OCREngine):
    """Synthetic OCR engine that generates deterministic page results without external APIs.

    Used for testing the pipeline in CI without cloud dependencies.
    The engine does not read the PDF file — it generates data purely from the config seed.
    """

    engine_id: ClassVar[str] = ENGINE_ID
    display_name: ClassVar[str] = DISPLAY_NAME
    version: ClassVar[str] = VERSION

    @staticmethod
    def get_config_schema() -> dict[str, Any]:
        """Return the JSON Schema for engine configuration.

        Returns:
            A JSON Schema dict describing the config parameters:
            - seed (integer, default 42): Random seed for deterministic output.
        """
        return {
            "type": "object",
            "properties": {
                "seed": {"type": "integer", "default": 42},
            },
            "required": [],
        }

    async def process_pdf(
        self,
        pdf_path: str | Path,  # noqa: ARG002 — interface contract parameter, not read
        config: dict[str, Any] | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> dict[str, Any]:
        """Generate synthetic OCR results without reading the PDF.

        Args:
            pdf_path: Path to the PDF file (not read, may not exist in CI).
            config: Engine configuration dict. May include 'seed' for determinism.
            progress: Optional callback for progress reporting (0-100).

        Returns:
            Raw engine output dict with:
            - raw_pages: list of per-page OCR data dicts
            - engine_id: engine identifier
            - engine_version: version string
            - config_snapshot: the config used for this run
        """
        if progress is not None:
            progress(0)

        resolved_config = config if config is not None else {}
        seed = resolved_config.get("seed", 42)
        rng = Random(seed)

        pages = _generate_pages(rng)
        raw_pages = [p.model_dump() for p in pages]

        if progress is not None:
            progress(100)

        return {
            "raw_pages": raw_pages,
            "engine_id": ENGINE_ID,
            "engine_version": VERSION,
            "config_snapshot": resolved_config,
        }

    @staticmethod
    def normalize(raw: dict[str, Any]) -> dict[str, Any]:
        """Convert raw engine output to standardized NormalizedDocument structure.

        Args:
            raw: Raw output dict from process_pdf.

        Returns:
            Normalized dict conforming to NormalizedDocument schema.
        """
        raw_pages: list[dict[str, Any]] = raw.get("raw_pages", [])
        normalized_pages: list[NormalizedPage] = []

        for page_data in raw_pages:
            blocks: list[NormalizedTextBlock | NormalizedTableBlock] = []
            for idx, block_data in enumerate(page_data.get("blocks", [])):
                block_bbox = _bbox_obj_to_list(block_data.get("bbox", {}))
                lines: list[NormalizedTextLine] = []

                for line_idx, line_data in enumerate(block_data.get("lines", [])):
                    line_bbox = _bbox_obj_to_list(line_data.get("bbox", {}))
                    words: list[NormalizedWord] = []

                    for word_idx, word_data in enumerate(line_data.get("words", [])):
                        word_bbox = _bbox_obj_to_list(word_data.get("bbox", {}))
                        chars: list[NormalizedCharacter] = []

                        for char_idx, char_data in enumerate(word_data.get("chars", [])):
                            char_bbox = _bbox_obj_to_list(char_data.get("bbox", {}))
                            chars.append(
                                NormalizedCharacter(
                                    char=char_data["char"],
                                    bbox=char_bbox,
                                    confidence=char_data["confidence"],
                                    order=char_idx,
                                )
                            )

                        words.append(
                            NormalizedWord(
                                text=word_data["text"],
                                bbox=word_bbox,
                                confidence=word_data["confidence"],
                                order=word_idx,
                                chars=chars,
                            )
                        )

                    lines.append(
                        NormalizedTextLine(
                            text=line_data["text"],
                            bbox=line_bbox,
                            confidence=line_data["confidence"],
                            order=line_idx,
                            words=words,
                        )
                    )

                block_type = block_data.get("block_type", "text")
                if block_type == "table":
                    blocks.append(
                        NormalizedTableBlock(
                            bbox=block_bbox,
                            confidence=block_data["confidence"],
                            order=idx,
                        )
                    )
                else:
                    blocks.append(
                        NormalizedTextBlock(
                            type=block_type,
                            bbox=block_bbox,
                            confidence=block_data["confidence"],
                            order=idx,
                            lines=lines,
                        )
                    )

            normalized_pages.append(
                NormalizedPage(
                    page_number=page_data["page_number"],
                    width=page_data["width"],
                    height=page_data["height"],
                    blocks=blocks,
                    tables=[],
                )
            )

        doc = NormalizedDocument(
            pages=normalized_pages,
            engine_id=raw.get("engine_id", ENGINE_ID),
            engine_version=raw.get("engine_version", VERSION),
            config_snapshot=raw.get("config_snapshot", {}),
        )
        return doc.model_dump()

