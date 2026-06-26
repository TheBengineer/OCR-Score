"""VLM (Vision-Language Model) base OCR engine.

Provides ``BaseVLMEngine`` — an abstract base class for VLM-based OCR
engines with shared PDF-to-image conversion, prompt templates, and lossy
output metadata.

Layout heuristics live in ``vlm_layout.py`` and output normalisation in
``vlm_normalize.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from backend.engine.base import OCREngine
from backend.engine.normalized_schema import (
    NormalizedPage,
)
from backend.engines.vlm_normalize import normalize_vlm_output

logger = logging.getLogger(__name__)

# ── Optional PDF rendering support ──────────────────────────────────────────

try:
    from pdf2image import convert_from_path as _pdf2image_convert  # noqa: F401

    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

# ── Constants ───────────────────────────────────────────────────────────────

VLM_LOSSY_METADATA: dict[str, Any] = {
    "vlm_output": True,
    "no_character_bboxes": True,
    "position_heuristic": True,
}

DEFAULT_VLM_PROMPT: str = (
    "You are an OCR engine. Extract all text from this page image precisely.\n\n"
    "Rules:\n"
    "1. Preserve reading order (top-to-bottom, left-to-right).\n"
    "2. Preserve paragraph structure with blank lines.\n"
    '3. Use "###" for section headings, "##" for subsections.\n'
    "4. Extract text verbatim — do not summarise or paraphrase.\n"
    "5. Only output the extracted text, no commentary.\n\n"
    "Page: {width_px}x{height_px}px at {dpi} DPI."
)

DEFAULT_VLM_JSON_PROMPT: str = (
    "You are an OCR engine. Extract all text from this page image.\n\n"
    "Return a JSON list of text blocks. Each block is an object with:\n"
    '- "text": the extracted text content\n'
    '- "type": "heading" | "paragraph" | "list_item" | "table_caption"\n'
    '- "position": approximate page position ("top", "middle", "bottom", '
    '"left", "right", "center")\n\n'
    "Rules:\n"
    "1. Preserve reading order from top to bottom.\n"
    "2. Extract text verbatim — do not summarise.\n"
    "3. Only output the JSON array, no commentary.\n\n"
    "Page: {width_px}x{height_px}px at {dpi} DPI."
)


# ── Base class ──────────────────────────────────────────────────────────────


class BaseVLMEngine(OCREngine):
    """Abstract base class for VLM-based OCR engines.

    Provides shared infrastructure:
    - ``_pdf_to_images()`` — Render PDF pages to PIL Images.
    - ``_normalize_vlm_output()`` — Convert free-form VLM text output to
      ``NormalizedPage`` with heuristic bounding boxes (delegates to
      ``vlm_normalize``).
    - ``DEFAULT_VLM_PROMPT`` / ``DEFAULT_VLM_JSON_PROMPT`` — Prompt templates.
    - ``vlm_output_metadata`` — Flags indicating output is lossy (no
      character-level bounding boxes).

    Subclasses **must** implement:
    - ``engine_id``, ``display_name``, ``version``
    - ``process_pdf()`` — Call the VLM API or local model.
    - ``normalize()`` — Normalise raw output via ``_normalize_vlm_output()``.
    """

    vlm_output_metadata: ClassVar[dict[str, Any]] = VLM_LOSSY_METADATA

    # ── Config schema ──────────────────────────────────────────────────────

    @staticmethod
    def get_config_schema() -> dict[str, Any]:
        """Return a JSON Schema for common VLM engine configuration.

        Shared parameters:
            - **dpi** (int, default ``300``): PDF rendering resolution.
            - **prompt_template** (str): Custom prompt sent to the VLM.
        """
        return {
            "type": "object",
            "properties": {
                "dpi": {
                    "type": "integer",
                    "default": 300,
                    "minimum": 72,
                    "maximum": 1200,
                    "description": "DPI for PDF page rendering",
                },
                "prompt_template": {
                    "type": "string",
                    "default": DEFAULT_VLM_PROMPT,
                    "description": "Custom prompt template for the VLM",
                },
            },
            "required": [],
        }

    # ── PDF rendering ───────────────────────────────────────────────────────

    @staticmethod
    def _pdf_to_images(
        pdf_path: str | Path,
        dpi: int = 300,
    ) -> list[Any]:
        """Render PDF pages to PIL Image objects.

        Args:
            pdf_path: Path to the PDF file.
            dpi: Rendering resolution (default 300).

        Returns:
            List of PIL Image objects, one per page.

        Raises:
            RuntimeError: If ``pdf2image`` is unavailable or rendering fails.
        """
        if not HAS_PDF2IMAGE:
            raise RuntimeError(
                "pdf2image is required for PDF rendering. "
                "Install it with: pip install pdf2image"
            )

        try:
            return _pdf2image_convert(str(pdf_path), dpi=dpi)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to render PDF pages: {exc}. "
                "Ensure poppler-utils (pdftoppm) is installed."
            ) from exc

    # ── VLM output normalisation (dispatcher) ───────────────────────────────

    @staticmethod
    def _normalize_vlm_output(
        raw_text: str,
        page_dims: tuple[float, float],
        dpi: int = 300,
        output_format: str = "markdown",
    ) -> NormalizedPage:
        """Convert VLM text output to a ``NormalizedPage``.

        Delegates to ``vlm_normalize.normalize_vlm_output()``.

        Args:
            raw_text: The raw text output from the VLM.
            page_dims: ``(width_pts, height_pts)`` — page dimensions in points.
            dpi: DPI used for PDF rendering.
            output_format: ``"markdown"`` (default) or ``"json"``.

        Returns:
            A ``NormalizedPage`` with heuristic bounding boxes.
        """
        return normalize_vlm_output(raw_text, page_dims, dpi, output_format)
