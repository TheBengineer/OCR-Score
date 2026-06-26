"""olmOCR VLM engine module.

Wraps the `olmOCR <https://olmocr.allenai.org/>`_ vision-language model for
OCR.  olmOCR outputs markdown-formatted text with headings, paragraphs, and
code blocks.

The engine sends PDF page images to an olmOCR inference endpoint (API or
local) and normalises the returned markdown into the canonical
``NormalizedDocument`` structure using heuristic bounding boxes.

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
    DEFAULT_VLM_PROMPT,
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
    "api_url": "http://localhost:8001/v1/ocr",
    "api_key": "",
    "model": "olmocr-7b",
    "dpi": 300,
    "timeout_seconds": 120,
    "prompt_template": DEFAULT_VLM_PROMPT,
}


# ── Image encoding helper ───────────────────────────────────────────────────


def _pil_image_to_base64(image: Any) -> str:
    """Encode a PIL Image as a base64-encoded JPEG string.

    Args:
        image: A PIL Image instance.

    Returns:
        Data URI string (``data:image/jpeg;base64,…``).
    """
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


# ── Engine class ────────────────────────────────────────────────────────────


class OlmocrEngine(BaseVLMEngine):
    """OCR engine that wraps the olmOCR VLM API.

    Processes PDF files by:
    1. Rendering each page to a JPEG image at the configured DPI.
    2. Sending each image to the olmOCR API endpoint.
    3. Collecting the returned markdown text.
    4. Normalising the markdown into the canonical ``NormalizedDocument``
       structure with heuristic bounding boxes.

    .. note::

        olmOCR output is **lossy** — the VLM does not provide character-level
        bounding boxes.  The normaliser distributes blocks/lines/words
        heuristically across each page.  See ``vlm_output_metadata``.
    """

    engine_id: ClassVar[str] = "olmocr"
    display_name: ClassVar[str] = "olmOCR"
    version: ClassVar[str] = "0.1.0"

    required_secrets: ClassVar[list] = [
        SecretDef(
            key="api_key",
            env_var="OLMOCR_API_KEY",
            display_name="olmOCR API Key",
            description="API key for the olmOCR inference endpoint",
        ),
    ]

    # ── Config schema ──────────────────────────────────────────────────────

    @staticmethod
    def get_config_schema() -> dict[str, Any]:
        """Return JSON Schema for olmOCR-specific configuration.

        Extends the base VLM schema with:
            - **api_url** (string, default ``"http://localhost:8001/v1/ocr"``).
            - **api_key** (string, optional).
            - **model** (string, default ``"olmocr-7b"``).
            - **timeout_seconds** (int, default ``120``).
        """
        base_schema = BaseVLMEngine.get_config_schema()
        base_schema["properties"].update({
            "api_url": {
                "type": "string",
                "default": "http://localhost:8001/v1/ocr",
                "description": "olmOCR API endpoint URL",
            },
            "api_key": {
                "type": "string",
                "default": "",
                "description": "API key (if required by the endpoint)",
            },
            "model": {
                "type": "string",
                "default": "olmocr-7b",
                "description": "Model identifier",
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
        """Run olmOCR on a PDF file.

        Args:
            pdf_path: Path to the PDF file to process.
            config: Engine configuration (merged with defaults).
            progress: Optional progress callback (0‑100).

        Returns:
            Raw engine output dict with:
            - ``raw_pages``: Per-page dicts with ``page_number``,
              ``width``, ``height``, ``dpi``, and ``raw_text`` (markdown).
            - ``engine_id``, ``engine_version``: Engine identification.
            - ``config_snapshot``: The resolved configuration.
            - ``page_count``: Number of pages processed.
            - ``vlm_metadata``: Lossy output flags.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            RuntimeError: If pdf2image, httpx is unavailable, or the API
                call fails.
        """
        if progress is not None:
            progress(0)

        resolved = {**DEFAULT_CONFIG, **(config or {})}
        pdf_path_obj = Path(pdf_path)

        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if not HAS_HTTPX:
            raise RuntimeError(
                "httpx is required for olmOCR API calls. "
                "Install it with: pip install httpx"
            )

        # ── Render PDF pages ──────────────────────────────────────────
        dpi = int(resolved.get("dpi", 300))
        images = self._pdf_to_images(pdf_path_obj, dpi=dpi)

        total_pages = len(images)
        if progress is not None:
            progress(20)

        # ── Call olmOCR API for each page ─────────────────────────────
        api_url = str(resolved.get("api_url", DEFAULT_CONFIG["api_url"]))
        api_key = str(resolved.get("api_key", ""))
        model = str(resolved.get("model", DEFAULT_CONFIG["model"]))
        timeout = float(resolved.get("timeout_seconds", 120))
        prompt_template = str(
            resolved.get("prompt_template", DEFAULT_VLM_PROMPT)
        )

        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

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
                }

                try:
                    response = await client.post(
                        api_url, json=payload, headers=headers
                    )
                    response.raise_for_status()
                    result: dict[str, Any] = response.json()

                    # olmOCR returns markdown text in the response.
                    raw_text = (
                        result.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    if not raw_text:
                        raw_text = result.get("text", "")

                except Exception as exc:
                    raise RuntimeError(
                        f"olmOCR API call failed for page {page_idx + 1}: {exc}. "
                        "Ensure the olmOCR endpoint is running and accessible."
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
            "engine_id": "olmocr",
            "engine_version": self.version,
            "config_snapshot": resolved,
            "page_count": total_pages,
            "vlm_metadata": VLM_LOSSY_METADATA,
        }

    # ── Normalisation ──────────────────────────────────────────────────────

    @staticmethod
    def normalize(raw: dict[str, Any]) -> dict[str, Any]:
        """Convert raw olmOCR output to ``NormalizedDocument``.

        olmOCR returns markdown text.  This method converts it to the
        canonical page hierarchy using heuristic bounding boxes since the
        VLM does not provide positional data.

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
            raw_text = page_data.get("raw_text", "")

            page_dims = (width, height)

            if raw_text and raw_text.strip():
                normalized_page = BaseVLMEngine._normalize_vlm_output(
                    raw_text, page_dims, dpi=dpi, output_format="markdown"
                )
                # Preserve the original page number from raw data.
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
            engine_id=raw.get("engine_id", "olmocr"),
            engine_version=raw.get("engine_version", "0.1.0"),
            config_snapshot=raw.get("config_snapshot", {}),
        )
        return doc.model_dump()


# ── Import-time registration ────────────────────────────────────────────────

with contextlib.suppress(EngineRegistryError):
    registry.register(OlmocrEngine)
