"""Tesseract OCR engine module.

Wraps ``pytesseract`` to process PDFs, extracting text with bounding boxes
at the character, word, line, and block levels.  Converts raw Tesseract
output into the canonical ``NormalizedDocument`` schema for cross-engine
comparison.

Dependencies
------------
- ``pytesseract`` (required) — Python wrapper around the Tesseract C API.
- ``pdf2image`` (required) — Renders PDF pages to PIL Images at a configurable DPI.
- ``Pillow`` (required via pdf2image) — Image format support.
- The system **tesseract** binary must be installed for runtime operation.
- The system **poppler-utils** (``pdftoppm``) must be installed for PDF rendering.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytesseract
from pytesseract import Output

from backend.engine.base import OCREngine
from backend.engine.normalized_schema import (
    Character as NormalizedCharacter,
)
from backend.engine.normalized_schema import (
    NormalizedDocument,
    NormalizedPage,
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
from backend.engine.registry import EngineRegistryError, registry

logger = logging.getLogger(__name__)

# ── Optional PDF rendering support ──────────────────────────────────────────

try:
    from pdf2image import convert_from_path as _pdf2image_convert

    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False
    _pdf2image_convert = None  # type: ignore[assignment]

# ── Default configuration ───────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    "lang": "eng",
    "psm": 3,
    "oem": 3,
    "dpi": 300,
}


# ── Coordinate helpers ──────────────────────────────────────────────────────


def _pixel_to_point(pixel: float, dpi: int) -> float:
    """Convert a pixel coordinate to points at 72 DPI.

    Args:
        pixel: Value in pixels.
        dpi: The DPI the pixel measurement was taken at.

    Returns:
        Equivalent value in points (1/72 inch).
    """
    return pixel * 72.0 / dpi


def _parse_confidence(conf: Any) -> float:
    """Parse a Tesseract confidence value into a 0‑1 float.

    Tesseract returns ``-1`` for non-word elements (pages, blocks,
    paragraphs, lines) or when detection fails.  Any negative value,
    ``None``, or unparseable input is treated as 0.

    Returns:
        Normalised confidence in ``[0.0, 1.0]``.
    """
    if conf is None:
        return 0.0
    try:
        val = float(conf)
    except (ValueError, TypeError):
        return 0.0
    if val < 0:
        return 0.0
    # Tesseract reports confidence as 0‑100; normalise to 0‑1.
    return min(val / 100.0, 1.0)


def _parse_boxes(boxes_str: str) -> list[dict[str, Any]]:
    """Parse Tesseract ``image_to_boxes`` output into char entries.

    The input string has one character per line in the format::

        <char> <left> <bottom> <right> <top> <page_num>

    Coordinates use a **bottom-left origin** pixel space.  This function
    returns entries with page-space bounding boxes ([x0, y0, x1, y1] in
    points at 72 DPI, top-left origin).  The caller must supply
    *page_height_pts* and *dpi* to perform the coordinate conversion.

    Args:
        boxes_str: Raw ``image_to_boxes`` output string.

    Returns:
        List of dicts with keys ``char``, ``bbox`` (as ``[x0, y0, x1, y1]``
        in **pixel** space, top-left origin), and ``page_num``.
    """
    entries: list[dict[str, Any]] = []
    if not boxes_str or not boxes_str.strip():
        return entries

    for line in boxes_str.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        char = parts[0]
        left = int(parts[1])
        bottom = int(parts[2])
        right = int(parts[3])
        top = int(parts[4])
        page_num = int(parts[5])

        entries.append({
            "char": char,
            # Store as pixel coords for now; conversion to page-space happens
            # during word-matching when we know the page height and DPI.
            "left": left,
            "bottom": bottom,
            "right": right,
            "top": top,
            "page_num": page_num,
        })
    return entries


def _boxes_to_page_bbox(
    entry: dict[str, Any],
    img_height_px: float,
    scale: float,
) -> list[float]:
    """Convert a ``image_to_boxes`` char entry to a page-space bounding box.

    Args:
        entry: A single char entry from ``_parse_boxes()``.
        img_height_px: Image height in pixels.
        scale: ``72.0 / dpi`` conversion factor.

    Returns:
        ``[x0, y0, x1, y1]`` in page-space points, top-left origin.
    """
    left = entry["left"]
    bottom = entry["bottom"]
    right = entry["right"]
    top = entry["top"]

    # Convert bottom-left origin → top-left origin pixel space …
    y0_px = img_height_px - top
    y1_px = img_height_px - bottom
    # … then to page-space points.
    return [
        left * scale,
        y0_px * scale,
        right * scale,
        y1_px * scale,
    ]


# ── Word/block/line groupers ────────────────────────────────────────────────


def _extract_word_entries(word_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract word-level entries (level == 5) with non-empty text.

    Args:
        word_data: The ``image_to_data`` dict with parallel lists.

    Returns:
        A list of per-word dicts with pixel-level bounding box fields and
        confidence.
    """
    levels = word_data.get("level", [])
    block_nums = word_data.get("block_num", [])
    par_nums = word_data.get("par_num", [])
    line_nums = word_data.get("line_num", [])
    word_nums = word_data.get("word_num", [])
    lefts = word_data.get("left", [])
    tops = word_data.get("top", [])
    widths = word_data.get("width", [])
    heights = word_data.get("height", [])
    confs = word_data.get("conf", [])
    texts = word_data.get("text", [])

    entries: list[dict[str, Any]] = []
    for i in range(len(levels)):
        if levels[i] != 5:
            continue
        text = texts[i] if texts[i] else ""
        if not text.strip():
            continue
        entries.append({
            "block_num": block_nums[i],
            "par_num": par_nums[i],
            "line_num": line_nums[i],
            "word_num": word_nums[i],
            "left": lefts[i],
            "top": tops[i],
            "width": widths[i],
            "height": heights[i],
            "conf": _parse_confidence(confs[i]),
            "text": text.strip(),
        })
    return entries


def _group_words_into_lines(
    word_entries: list[dict[str, Any]],
) -> tuple[dict[tuple[int, int, int], list[dict[str, Any]]], list[tuple[int, int, int]]]:
    """Group word entries into lines by ``(block_num, par_num, line_num)``.

    Returns:
        A ``(line_groups, line_order)`` tuple.  ``line_groups`` maps
        ``(block, par, line)`` keys to lists of word entries, and
        ``line_order`` preserves insertion order.
    """
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    order: list[tuple[int, int, int]] = []
    for entry in word_entries:
        key = (entry["block_num"], entry["par_num"], entry["line_num"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(entry)
    return groups, order


def _group_lines_into_blocks(
    line_order: list[tuple[int, int, int]],
) -> tuple[dict[int, list[tuple[int, int, int]]], list[int]]:
    """Group line keys into blocks by ``block_num``.

    Returns:
        A ``(block_groups, block_order)`` tuple.  ``block_groups`` maps
        block numbers to lists of line keys, and ``block_order`` preserves
        insertion order.
    """
    groups: dict[int, list[tuple[int, int, int]]] = {}
    order: list[int] = []
    for line_key in line_order:
        block_num = line_key[0]
        if block_num not in groups:
            groups[block_num] = []
            order.append(block_num)
        groups[block_num].append(line_key)
    return groups, order


def _match_chars_to_words(
    char_entries: list[dict[str, Any]],
    word_bbox: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition char entries into those inside/outside *word_bbox*.

    Matching is done by checking whether the character's **centre point**
    falls inside the word's bounding box.

    Args:
        char_entries: List of char entries with page-space ``bbox``.
        word_bbox: ``[x0, y0, x1, y1]`` in page-space points.

    Returns:
        ``(inside, outside)`` — characters matched to this word, and
        the remaining characters for subsequent words.
    """
    inside: list[dict[str, Any]] = []
    outside: list[dict[str, Any]] = []
    cx0, cy0, cx1, cy1 = word_bbox
    for ch in char_entries:
        ch_bbox = ch["bbox"]
        centre_x = (ch_bbox[0] + ch_bbox[2]) / 2.0
        centre_y = (ch_bbox[1] + ch_bbox[3]) / 2.0
        if cx0 <= centre_x <= cx1 and cy0 <= centre_y <= cy1:
            inside.append(ch)
        else:
            outside.append(ch)
    return inside, outside


# ── Engine class ────────────────────────────────────────────────────────────


class TesseractEngine(OCREngine):
    """OCR engine that wraps Tesseract via ``pytesseract``.

    Processes PDF files by:
    1. Rendering each page to a PIL Image at the configured DPI.
    2. Running ``pytesseract.image_to_data`` for word-level results.
    3. Running ``pytesseract.image_to_boxes`` for character-level results.
    4. Normalising all coordinates to page-space (points at 72 DPI,
       top-left origin).
    """

    engine_id: ClassVar[str] = "tesseract"
    display_name: ClassVar[str] = "Tesseract OCR"
    version: ClassVar[str] = getattr(pytesseract, "__version__", "unknown")

    # ── Config schema ──────────────────────────────────────────────────────

    @staticmethod
    def get_config_schema() -> dict[str, Any]:
        """Return JSON Schema for Tesseract-specific configuration.

        Supported parameters:
            - **lang** (string, default ``"eng"``): OCR language(s).
            - **psm** (integer, default ``3``): Page segmentation mode.
            - **oem** (integer, default ``3``): OCR engine mode.
            - **dpi** (integer, default ``300``): DPI for PDF rendering.
        """
        return {
            "type": "object",
            "properties": {
                "lang": {
                    "type": "string",
                    "default": "eng",
                    "description": "OCR language(s), e.g. 'eng' or 'eng+fra'",
                },
                "psm": {
                    "type": "integer",
                    "default": 3,
                    "minimum": 0,
                    "maximum": 13,
                    "description": "Tesseract page segmentation mode",
                },
                "oem": {
                    "type": "integer",
                    "default": 3,
                    "minimum": 0,
                    "maximum": 3,
                    "description": "Tesseract OCR engine mode (0=legacy, 1=LSTM, 2=legacy+LSTM, 3=default)",
                },
                "dpi": {
                    "type": "integer",
                    "default": 300,
                    "minimum": 72,
                    "maximum": 1200,
                    "description": "DPI for PDF page rendering",
                },
            },
            "required": [],
        }

    # ── PDF processing ─────────────────────────────────────────────────────

    async def process_pdf(
        self,
        pdf_path: str | Path,
        config: dict[str, Any] | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> dict[str, Any]:
        """Run Tesseract OCR on a PDF file.

        Args:
            pdf_path: Path to the PDF file to process.
            config: Engine configuration (merged with defaults).
            progress: Optional progress callback (0‑100).

        Returns:
            Raw engine output dict with:
            - ``raw_pages``: per-page dicts containing ``image_to_data`` and
              ``image_to_boxes`` results.
            - ``engine_id``, ``engine_version``: engine identification.
            - ``config_snapshot``: the resolved configuration.
            - ``page_count``: number of pages processed.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            RuntimeError: If pdf2image is unavailable, poppler is missing,
                or Tesseract processing fails.
        """
        if progress is not None:
            progress(0)

        resolved = {**DEFAULT_CONFIG, **(config or {})}
        pdf_path_obj = Path(pdf_path)

        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if not HAS_PDF2IMAGE:
            raise RuntimeError(
                "pdf2image is required for PDF rendering. "
                "Install it with: pip install pdf2image"
            )

        import asyncio

        dpi = int(resolved.get("dpi", 300))
        lang = str(resolved.get("lang", "eng"))
        psm = int(resolved.get("psm", 3))
        oem = int(resolved.get("oem", 3))
        tesseract_config = f"--psm {psm} --oem {oem}"

        # ── Render PDF pages to images ─────────────────────────────────
        try:
            images = await asyncio.to_thread(
                _pdf2image_convert, str(pdf_path_obj), dpi=dpi
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to render PDF pages: {exc}. "
                "Ensure poppler-utils (pdftoppm) is installed."
            ) from exc

        total_pages = len(images)
        raw_pages: list[dict[str, Any]] = []

        for page_idx, image in enumerate(images):
            img_width_px, img_height_px = image.size
            page_width = _pixel_to_point(img_width_px, dpi)
            page_height = _pixel_to_point(img_height_px, dpi)

            # ── Word-level data ────────────────────────────────────────
            word_data = await asyncio.to_thread(
                pytesseract.image_to_data,
                image,
                lang=lang,
                config=tesseract_config,
                output_type=Output.DICT,
            )

            # ── Character-level data ───────────────────────────────────
            boxes_str = await asyncio.to_thread(
                pytesseract.image_to_boxes,
                image,
                lang=lang,
                config=tesseract_config,
            )

            raw_pages.append({
                "page_number": page_idx + 1,
                "width": page_width,
                "height": page_height,
                "dpi": dpi,
                "image_to_data": word_data,
                "image_to_boxes": boxes_str,
            })

            if progress is not None:
                pct = 10 + int((page_idx + 1) / total_pages * 80)
                progress(min(pct, 99))

        if progress is not None:
            progress(100)

        return {
            "raw_pages": raw_pages,
            "engine_id": "tesseract",
            "engine_version": self.version,
            "config_snapshot": resolved,
            "page_count": total_pages,
        }

    # ── Normalisation ──────────────────────────────────────────────────────

    @staticmethod
    def normalize(raw: dict[str, Any]) -> dict[str, Any]:
        """Convert raw Tesseract output to ``NormalizedDocument``.

        Args:
            raw: The raw output dict from ``process_pdf()``.

        Returns:
            A dict conforming to ``NormalizedDocument`` with all bounding
            boxes in page-space coordinates (points at 72 DPI, top-left
            origin).
        """
        raw_pages: list[dict[str, Any]] = raw.get("raw_pages", [])
        normalized_pages: list[NormalizedPage] = []

        for page_data in raw_pages:
            page_number = page_data["page_number"]
            width = page_data["width"]
            height = page_data["height"]
            dpi = page_data.get("dpi", 300)
            scale = 72.0 / dpi

            word_data: dict[str, Any] = page_data.get("image_to_data", {})
            boxes_str: str = page_data.get("image_to_boxes", "")

            # ── Extract word entries ───────────────────────────────────
            word_entries = _extract_word_entries(word_data)

            # Short-circuit for empty/blank pages.
            if not word_entries:
                normalized_pages.append(
                    NormalizedPage(
                        page_number=page_number,
                        width=width,
                        height=height,
                        blocks=[],
                        tables=[],
                    )
                )
                continue

            # ── Parse character boxes ──────────────────────────────────
            # image_to_boxes uses bottom-left origin pixel space.
            # Convert to top-left origin pixel space first, then to
            # page-space points during word matching.
            raw_char_entries = _parse_boxes(boxes_str)
            img_height_px = height / scale  # total image height in pixels

            # Convert each char entry to page-space bbox once.
            char_entries: list[dict[str, Any]] = []
            for ch in raw_char_entries:
                char_entries.append({
                    "char": ch["char"],
                    "bbox": _boxes_to_page_bbox(ch, img_height_px, scale),
                })

            # ── Group into blocks / lines / words ──────────────────────
            line_groups, line_order = _group_words_into_lines(word_entries)
            block_groups, block_order = _group_lines_into_blocks(line_order)

            blocks: list[NormalizedTextBlock] = []

            for block_idx, block_num in enumerate(block_order):
                line_keys = block_groups[block_num]
                lines: list[NormalizedTextLine] = []

                for line_idx, line_key in enumerate(line_keys):
                    word_entries_in_line = line_groups[line_key]

                    # Sort words in visual reading order (top-to-bottom,
                    # left-to-right).
                    word_entries_in_line.sort(
                        key=lambda w: (w["top"], w["left"])
                    )

                    words: list[NormalizedWord] = []

                    for word_idx, entry in enumerate(word_entries_in_line):
                        word_text = entry["text"]

                        # Pixel → page-space coordinates
                        x0 = entry["left"] * scale
                        y0 = entry["top"] * scale
                        x1 = (entry["left"] + entry["width"]) * scale
                        y1 = (entry["top"] + entry["height"]) * scale
                        word_bbox = [x0, y0, x1, y1]

                        wconf = entry["conf"]

                        # ── Match characters to this word ──────────────
                        matched, char_entries = _match_chars_to_words(
                            char_entries, word_bbox
                        )

                        chars: list[NormalizedCharacter] = []
                        if matched:
                            for ci, ch_entry in enumerate(matched):
                                ch_bbox = ch_entry["bbox"]
                                chars.append(
                                    NormalizedCharacter(
                                        char=ch_entry["char"],
                                        bbox=ch_bbox,
                                        confidence=wconf,
                                        order=ci,
                                    )
                                )
                        else:
                            # No character boxes matched — synthesise
                            # evenly-spaced characters from the word text.
                            char_width = (x1 - x0) / max(len(word_text), 1)
                            for ci, ch in enumerate(word_text):
                                cx0 = x0 + ci * char_width
                                cx1 = cx0 + char_width
                                chars.append(
                                    NormalizedCharacter(
                                        char=ch,
                                        bbox=[cx0, y0, cx1, y1],
                                        confidence=wconf,
                                        order=ci,
                                    )
                                )

                        words.append(
                            NormalizedWord(
                                text=word_text,
                                bbox=word_bbox,
                                confidence=wconf,
                                order=word_idx,
                                chars=chars,
                            )
                        )

                    # ── Compute line bbox from constituent words ───────
                    if words:
                        line_x0 = min(w.bbox[0] for w in words)
                        line_y0 = min(w.bbox[1] for w in words)
                        line_x1 = max(w.bbox[2] for w in words)
                        line_y1 = max(w.bbox[3] for w in words)
                        line_text = " ".join(w.text for w in words)
                        line_conf = sum(w.confidence for w in words) / len(words)
                    else:
                        line_x0 = line_y0 = line_x1 = line_y1 = 0.0
                        line_text = ""
                        line_conf = 0.0

                    lines.append(
                        NormalizedTextLine(
                            text=line_text,
                            bbox=[line_x0, line_y0, line_x1, line_y1],
                            confidence=line_conf,
                            order=line_idx,
                            words=words,
                        )
                    )

                # ── Compute block bbox from constituent lines ──────────
                if lines:
                    block_x0 = min(line_.bbox[0] for line_ in lines)
                    block_y0 = min(line_.bbox[1] for line_ in lines)
                    block_x1 = max(line_.bbox[2] for line_ in lines)
                    block_y1 = max(line_.bbox[3] for line_ in lines)
                    block_conf = sum(line_.confidence for line_ in lines) / len(lines)
                else:
                    block_x0 = block_y0 = block_x1 = block_y1 = 0.0
                    block_conf = 0.0

                blocks.append(
                    NormalizedTextBlock(
                        type="text",
                        bbox=[block_x0, block_y0, block_x1, block_y1],
                        confidence=block_conf,
                        order=block_idx,
                        lines=lines,
                    )
                )

            normalized_pages.append(
                NormalizedPage(
                    page_number=page_number,
                    width=width,
                    height=height,
                    blocks=blocks,
                    tables=[],
                )
            )

        doc = NormalizedDocument(
            pages=normalized_pages,
            engine_id=raw.get("engine_id", "tesseract"),
            engine_version=raw.get("engine_version", "unknown"),
            config_snapshot=raw.get("config_snapshot", {}),
        )
        return doc.model_dump()


# ── Import-time registration ────────────────────────────────────────────────

with contextlib.suppress(EngineRegistryError):
    registry.register(TesseractEngine)
