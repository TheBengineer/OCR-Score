"""DeepSeek-OCR VLM engine module.

Wraps the `DeepSeek Vision API <https://platform.deepseek.com/>`_ for
OCR.  DeepSeek's VL model can return structured JSON output with text
block classification, enabling richer normalisation than raw markdown.

The engine sends PDF page images to the DeepSeek chat/completions API and
normalises the returned JSON into the canonical ``NormalizedDocument``
structure.

Dependencies
------------
- ``pdf2image`` (required) — Renders PDF pages to PIL Images.
- ``httpx`` (required) — Async HTTP client for API calls.

The engine registers itself with ``EngineRegistry`` at import time.
"""

# allow: SIZE_OK — cohesive engine module (config + process_pdf + normalize)

from __future__ import annotations

import base64
import contextlib
import logging
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar

from backend.engine.base import SecretDef
from backend.engine.normalized_schema import (
    NormalizedDocument,
    NormalizedPage,
)
from backend.engine.registry import EngineRegistryError, registry
from backend.engines.vlm import (
    DEFAULT_VLM_JSON_PROMPT,
    VLM_LOSSY_METADATA,
    BaseVLMEngine,
)
from backend.engines.vlm_layout import pixel_to_point

logger = logging.getLogger(__name__)

# ── Optional HTTP client ────────────────────────────────────────────────────

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# ── Default configuration ───────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    "api_key": "",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-vl2",
    "dpi": 300,
    "timeout_seconds": 120,
    "prompt_template": DEFAULT_VLM_JSON_PROMPT,
}


# ── Image encoding helper ───────────────────────────────────────────────────


def _pil_image_to_base64(image: Any) -> str:
    """Encode a PIL Image as a base64-encoded PNG string.

    Args:
        image: A PIL Image instance.

    Returns:
        Data URI string (``data:image/png;base64,…``).
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ── Engine class ────────────────────────────────────────────────────────────


class DeepseekOcrEngine(BaseVLMEngine):
    """OCR engine that wraps the DeepSeek Vision API.

    Processes PDF files by:
    1. Rendering each page to a PNG image at the configured DPI.
    2. Sending each image to DeepSeek's chat/completions API with a
       prompt requesting JSON-structured output.
    3. Parsing the returned JSON into text blocks.
    4. Normalising into the canonical ``NormalizedDocument`` structure
       with heuristic bounding boxes.

    The engine requires a valid DeepSeek API key.  Provide one via the
    ``api_key`` config field or the ``DEEPSEEK_API_KEY`` environment
    variable.

    .. note::

        DeepSeek-OCR output is **lossy** — the VLM does not provide
        character-level bounding boxes.  The normaliser distributes
        blocks/lines/words heuristically across each page.
    """

    engine_id: ClassVar[str] = "deepseek-ocr"
    display_name: ClassVar[str] = "DeepSeek-OCR"
    version: ClassVar[str] = "0.1.0"

    required_secrets: ClassVar[list] = [
        SecretDef(
            key="api_key",
            env_var="DEEPSEEK_API_KEY",
            display_name="DeepSeek API Key",
            description="DeepSeek API authentication key",
        ),
    ]

    # ── Config schema ──────────────────────────────────────────────────────

    @staticmethod
    def get_config_schema() -> dict[str, Any]:
        """Return JSON Schema for DeepSeek-OCR configuration.

        Extends the base VLM schema with:
            - **api_key** (string, required): DeepSeek API key.
            - **base_url** (string, default ``"https://api.deepseek.com"``).
            - **model** (string, default ``"deepseek-vl2"``).
            - **timeout_seconds** (int, default ``120``).
        """
        base_schema = BaseVLMEngine.get_config_schema()
        base_schema["properties"].update({
            "api_key": {
                "type": "string",
                "default": "",
                "description": "DeepSeek API key (or set DEEPSEEK_API_KEY env var)",
            },
            "base_url": {
                "type": "string",
                "default": "https://api.deepseek.com",
                "description": "DeepSeek API base URL",
            },
            "model": {
                "type": "string",
                "default": "deepseek-vl2",
                "description": "DeepSeek model identifier",
            },
            "timeout_seconds": {
                "type": "integer",
                "default": 120,
                "minimum": 10,
                "maximum": 600,
                "description": "HTTP request timeout in seconds",
            },
        })
        return base_schema

    # ── PDF processing ─────────────────────────────────────────────────────

    async def process_pdf(
        self,
        pdf_path: str | Path,
        config: dict[str, Any] | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> dict[str, Any]:
        """Run DeepSeek-OCR on a PDF file.

        Args:
            pdf_path: Path to the PDF file to process.
            config: Engine configuration (merged with defaults).
            progress: Optional progress callback (0‑100).

        Returns:
            Raw engine output dict with:
            - ``raw_pages``: Per-page dicts with ``page_number``,
              ``width``, ``height``, ``dpi``, and ``raw_text`` (JSON).
            - ``engine_id``, ``engine_version``: Engine identification.
            - ``config_snapshot``: The resolved configuration.
            - ``page_count``: Number of pages processed.
            - ``vlm_metadata``: Lossy output flags.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            RuntimeError: If pdf2image or httpx is unavailable, or the
                API call fails.
        """
        if progress is not None:
            progress(0)

        resolved = {**DEFAULT_CONFIG, **(config or {})}
        pdf_path_obj = Path(pdf_path)

        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if not HAS_HTTPX:
            raise RuntimeError(
                "httpx is required for DeepSeek-OCR API calls. "
                "Install it with: pip install httpx"
            )

        # ── Resolve API key ───────────────────────────────────────────
        import os

        api_key = str(resolved.get("api_key", "")) or os.environ.get(
            "DEEPSEEK_API_KEY", ""
        )
        if not api_key:
            raise RuntimeError(
                "DeepSeek API key is required. Provide it via the "
                "'api_key' config field or the DEEPSEEK_API_KEY "
                "environment variable."
            )

        # ── Render PDF pages ──────────────────────────────────────────
        dpi = int(resolved.get("dpi", 300))
        images = self._pdf_to_images(pdf_path_obj, dpi=dpi)

        total_pages = len(images)
        if progress is not None:
            progress(20)

        # ── Call DeepSeek API for each page ───────────────────────────
        base_url = str(resolved.get("base_url", DEFAULT_CONFIG["base_url"]))
        model = str(resolved.get("model", DEFAULT_CONFIG["model"]))
        timeout = float(resolved.get("timeout_seconds", 120))
        prompt_template = str(
            resolved.get("prompt_template", DEFAULT_VLM_JSON_PROMPT)
        )

        chat_url = f"{base_url.rstrip('/')}/chat/completions"
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        raw_pages: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            for page_idx, image in enumerate(images):
                img_width_px, img_height_px = image.size
                page_width = pixel_to_point(float(img_width_px), dpi)
                page_height = pixel_to_point(float(img_height_px), dpi)

                image_b64 = _pil_image_to_base64(image)
                prompt = prompt_template.format(
                    width_px=img_width_px,
                    height_px=img_height_px,
                    dpi=dpi,
                )

                payload: dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_b64},
                                },
                            ],
                        },
                    ],
                    "response_format": {"type": "json_object"},
                }

                try:
                    response = await client.post(
                        chat_url, json=payload, headers=headers
                    )
                    response.raise_for_status()
                    result: dict[str, Any] = response.json()

                    # Extract the JSON content from the choice.
                    raw_text = (
                        result.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "[]")
                    )

                except Exception as exc:
                    raise RuntimeError(
                        f"DeepSeek API call failed for page "
                        f"{page_idx + 1}: {exc}. "
                        "Ensure your API key is valid and the endpoint "
                        "is accessible."
                    ) from exc

                raw_pages.append({
                    "page_number": page_idx + 1,
                    "width": page_width,
                    "height": page_height,
                    "dpi": dpi,
                    "raw_text": raw_text,
                })

                if progress is not None:
                    pct = 20 + int((page_idx + 1) / total_pages * 70)
                    progress(min(pct, 99))

        if progress is not None:
            progress(100)

        return {
            "raw_pages": raw_pages,
            "engine_id": "deepseek-ocr",
            "engine_version": self.version,
            "config_snapshot": resolved,
            "page_count": total_pages,
            "vlm_metadata": VLM_LOSSY_METADATA,
        }

    # ── Normalisation ──────────────────────────────────────────────────────

    @staticmethod
    def normalize(raw: dict[str, Any]) -> dict[str, Any]:
        """Convert raw DeepSeek-OCR output to ``NormalizedDocument``.

        DeepSeek-OCR returns JSON-structured output.  This method
        parses it into the canonical page hierarchy.

        Args:
            raw: The raw output dict from ``process_pdf()``.

        Returns:
            A dict conforming to ``NormalizedDocument``.
        """
        raw_pages: list[dict[str, Any]] = raw.get("raw_pages", [])
        normalized_pages: list[NormalizedPage] = []

        for page_data in raw_pages:
            page_number = page_data["page_number"]
            width = page_data["width"]
            height = page_data["height"]
            dpi = page_data.get("dpi", 300)
            raw_text = page_data.get("raw_text", "[]")

            page_dims = (width, height)

            if raw_text and raw_text.strip():
                normalized_page = BaseVLMEngine._normalize_vlm_output(
                    raw_text, page_dims, dpi=dpi, output_format="json"
                )
                normalized_page = NormalizedPage(
                    page_number=page_number,
                    width=normalized_page.width,
                    height=normalized_page.height,
                    blocks=normalized_page.blocks,
                    tables=normalized_page.tables,
                )
            else:
                normalized_page = NormalizedPage(
                    page_number=page_number,
                    width=width or 1.0,
                    height=height or 1.0,
                    blocks=[],
                    tables=[],
                )

            normalized_pages.append(normalized_page)

        if not normalized_pages:
            normalized_pages.append(
                NormalizedPage(
                    page_number=1, width=612.0, height=792.0, blocks=[], tables=[]
                )
            )

        doc = NormalizedDocument(
            pages=normalized_pages,
            engine_id=raw.get("engine_id", "deepseek-ocr"),
            engine_version=raw.get("engine_version", "0.1.0"),
            config_snapshot=raw.get("config_snapshot", {}),
        )
        return doc.model_dump()


# ── Import-time registration ────────────────────────────────────────────────

with contextlib.suppress(EngineRegistryError):
    registry.register(DeepseekOcrEngine)
