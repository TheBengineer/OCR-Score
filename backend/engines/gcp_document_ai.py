"""GCP Document AI OCR engine module.

Wraps ``google.cloud.documentai`` to process PDFs via the Document AI API,
extracting text with bounding boxes at the character, word, line, and block
levels.  Converts raw Document AI output into the canonical
``NormalizedDocument`` schema for cross-engine comparison.

The engine registers itself with ``EngineRegistry`` at import time and works
when GCP credentials are available (via ADC or explicit key file), gracefully
raising ``RuntimeError`` with a clear message when they are not.

Dependencies
------------
- ``google-cloud-documentai`` (required) — GCP Document AI Python SDK.

Coordinate system
-----------------
Document AI returns coordinates as either:
- **Normalised** (0.0–1.0) relative to page dimensions — stored in
  ``bounding_poly.normalized_vertices``.
- **Points** in page-space — stored in ``bounding_poly.vertices``.

This module converts both to the canonical page-space (points at 72 DPI,
top-left origin) by scaling normalised coordinates with
``page.dimensions.width`` / ``height``.

Document AI hierarchy mapping
-----------------------------
Document AI provides flat per-page lists:
    ``paragraphs → lines → tokens → symbols``

These are mapped to the canonical hierarchy using **text_anchor** offsets
(byte ranges into ``document.text``) to establish containment:
    ``paragraph (TextBlock) → line (TextLine) → token (Word) → symbol (Character)``
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Final

import anyio

from backend.engine.base import OCREngine
from backend.engine.normalized_schema import (
    Character as NormalizedCharacter,
)
from backend.engine.normalized_schema import (
    NormalizedDocument,
    NormalizedPage,
)
from backend.engine.normalized_schema import (
    Table as NormalizedTable,
)
from backend.engine.normalized_schema import (
    TableBlock as NormalizedTableBlock,
)
from backend.engine.normalized_schema import (
    TableCell as NormalizedTableCell,
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

# ── Optional SDK import ─────────────────────────────────────────────────────

try:
    from google.cloud import documentai
    from google.protobuf.json_format import MessageToDict

    HAS_DOCUMENT_AI: bool = True
except ImportError:
    HAS_DOCUMENT_AI = False

# ── Default configuration ───────────────────────────────────────────────────

DEFAULT_CONFIG: Final[dict[str, Any]] = {
    "processor_id": "",
    "location": "us",
    "project_id": "",
    "credentials_path": "",
    "timeout_seconds": 300,
    "mime_type": "application/pdf",
}


# ── SDK version helper ──────────────────────────────────────────────────────


def _get_sdk_version() -> str:
    """Return the ``google-cloud-documentai`` SDK version, or ``"unknown"``."""
    if HAS_DOCUMENT_AI:
        try:
            return documentai.__version__  # type: ignore[attr-defined,unused-ignore]  # noqa: PGH003
        except AttributeError:
            pass
    return "unknown"


_DOCUMENT_AI_VERSION: Final[str] = _get_sdk_version()


# ── Proto → dict conversion ─────────────────────────────────────────────────


def _document_to_dict(
    document: Any,
) -> dict[str, Any]:
    """Convert a Document AI ``Document`` proto to a plain dict.

    Uses protobuf JSON serialisation with snake_case field names to match
    the proto field naming convention.

    Args:
        document: A ``google.cloud.documentai.Document`` instance.

    Returns:
        A plain dict with snake_case keys matching the proto structure.
    """
    if document is None:  # type: ignore[unused-ignore]  # noqa: PGH003
        return {"text": "", "pages": []}
    try:
        result: dict[str, Any] = MessageToDict(
            document,
            preserving_proto_field_name=True,
        )
        return result
    except (ValueError, TypeError, AttributeError):
        return {"text": "", "pages": []}


# ── Coordinate and layout helpers ───────────────────────────────────────────


def _layout_to_bbox(
    layout: dict[str, Any] | None,
    page_width: float,
    page_height: float,
) -> list[float]:
    """Convert a Document AI layout dict to a page-space bounding box.

    Document AI returns coordinates as either normalised (0.0–1.0) in
    ``normalized_vertices`` or absolute points in ``vertices``.  This function
    handles both and returns page-space points at 72 DPI.

    Args:
        layout: The ``layout`` dict from a Document AI page element.
        page_width: Page width in points.
        page_height: Page height in points.

    Returns:
        ``[x0, y0, x1, y1]`` in page-space points, top-left origin.
    """
    if not layout:
        return [0.0, 0.0, 0.0, 0.0]

    bounding_poly = (layout.get("bounding_poly") or {}) or {}
    if not bounding_poly:
        return [0.0, 0.0, 0.0, 0.0]

    # Try normalised vertices first (0.0–1.0), scale by page dimensions.
    norm_verts: list[dict[str, Any]] = (bounding_poly.get("normalized_vertices") or []) or []
    if norm_verts:
        xs = [v.get("x", 0.0) for v in norm_verts]
        ys = [v.get("y", 0.0) for v in norm_verts]
        return [
            min(xs) * page_width,
            min(ys) * page_height,
            max(xs) * page_width,
            max(ys) * page_height,
        ]

    # Fall back to vertices (already in points).
    verts: list[dict[str, Any]] = (bounding_poly.get("vertices") or []) or []
    if verts:
        xs = [v.get("x", 0.0) for v in verts]
        ys = [v.get("y", 0.0) for v in verts]
        return [min(xs), min(ys), max(xs), max(ys)]

    return [0.0, 0.0, 0.0, 0.0]


def _extract_text(full_text: str, element: dict[str, Any]) -> str:
    """Extract text from the document's full text using element's text_anchor.

    Each Document AI element (paragraph, line, token, symbol) has a
    ``text_anchor`` with ``text_segments`` containing ``start_index`` and
    ``end_index`` byte offsets into ``document.text``.

    Args:
        full_text: The ``document.text`` string.
        element: A dict representing a Document AI page element.

    Returns:
        The concatenated text from all segments.
    """
    if not full_text:
        return ""

    layout = (element.get("layout") or {}) or {}
    text_anchor = (layout.get("text_anchor") or {}) or {}
    segments: list[dict[str, Any]] = (text_anchor.get("text_segments") or []) or []

    parts: list[str] = []
    for seg in segments:
        start = int(seg.get("start_index", 0) or 0)
        end = int(seg.get("end_index", len(full_text)) or len(full_text))
        if start < 0 or end > len(full_text) or start >= end:
            continue
        parts.append(full_text[start:end])

    return "".join(parts)


def _get_text_range(element: dict[str, Any]) -> tuple[int, int]:
    """Get the text range ``[start, end)`` from an element's text_anchor.

    Returns:
        A ``(start, end)`` tuple.  ``(0, 0)`` if no text segments exist.
    """
    layout = (element.get("layout") or {}) or {}
    text_anchor = (layout.get("text_anchor") or {}) or {}
    segments: list[dict[str, Any]] = (text_anchor.get("text_segments") or []) or []
    if not segments:
        return (0, 0)
    return (
        int(segments[0].get("start_index", 0) or 0),
        int(segments[-1].get("end_index", 0) or 0),
    )


def _find_contained(
    children: list[dict[str, Any]],
    parent: dict[str, Any],
) -> list[dict[str, Any]]:
    """Find all children whose text ranges fall within the parent's text range.

    Uses text_anchor byte offsets to establish containment.  This is reliable
    for well-formed Document AI output where text ranges nest properly.

    Args:
        children: List of potential child element dicts.
        parent: The parent element dict.

    Returns:
        List of child elements whose text range is within the parent's range.
    """
    p_start, p_end = _get_text_range(parent)
    if p_start == 0 and p_end == 0:
        return list(children)

    result: list[dict[str, Any]] = []
    for child in children:
        c_start, c_end = _get_text_range(child)
        if c_start >= p_start and c_end <= p_end:
            result.append(child)
    return result


# ── Table builders ──────────────────────────────────────────────────────────


def _build_tables(
    tables_data: list[dict[str, Any]],
    full_text: str,
    page_width: float,
    page_height: float,
) -> list[NormalizedTable]:
    """Convert Document AI table data to ``NormalizedTable`` objects.

    Args:
        tables_data: List of table dicts from a Document AI page.
        full_text: The document's full text for cell content extraction.
        page_width: Page width in points.
        page_height: Page height in points.

    Returns:
        List of ``NormalizedTable`` objects.
    """
    normalized_tables: list[NormalizedTable] = []

    for table_data in tables_data:
        table_layout = (table_data.get("layout") or {}) or {}
        table_bbox = _layout_to_bbox(table_layout, page_width, page_height)

        # Collect cells from header rows and body rows.
        header_rows: list[dict[str, Any]] = (table_data.get("header_rows") or []) or []
        body_rows: list[dict[str, Any]] = (table_data.get("body_rows") or []) or []

        cells: list[NormalizedTableCell] = []
        num_rows = 0
        num_cols = 0

        for row_idx, row in enumerate(header_rows):
            row_cells: list[dict[str, Any]] = (row.get("cells") or []) or []
            for cell in row_cells:
                cell_text = _extract_text(full_text, cell)
                cell_layout = (cell.get("layout") or {}) or {}
                cell_bbox = _layout_to_bbox(cell_layout, page_width, page_height)
                cell_conf = float(cell_layout.get("confidence", 0.0) or 0.0)
                col_span = int(cell.get("col_span", 1) or 1)
                row_span = int(cell.get("row_span", 1) or 1)
                col_index = int(cell.get("col_index", 0) or 0)

                cells.append(
                    NormalizedTableCell(
                        row=row_idx,
                        col=col_index,
                        row_span=row_span,
                        col_span=col_span,
                        text=cell_text,
                        bbox=cell_bbox,
                        confidence=cell_conf,
                    )
                )
                num_cols = max(num_cols, col_index + col_span)

            num_rows = max(num_rows, row_idx + 1)

        body_start_row = num_rows
        for row_idx, row in enumerate(body_rows):
            row_cells = (row.get("cells") or []) or []
            for cell in row_cells:
                cell_text = _extract_text(full_text, cell)
                cell_layout = (cell.get("layout") or {}) or {}
                cell_bbox = _layout_to_bbox(cell_layout, page_width, page_height)
                cell_conf = float(cell_layout.get("confidence", 0.0) or 0.0)
                col_span = int(cell.get("col_span", 1) or 1)
                row_span = int(cell.get("row_span", 1) or 1)
                col_index = int(cell.get("col_index", 0) or 0)

                cells.append(
                    NormalizedTableCell(
                        row=body_start_row + row_idx,
                        col=col_index,
                        row_span=row_span,
                        col_span=col_span,
                        text=cell_text,
                        bbox=cell_bbox,
                        confidence=cell_conf,
                    )
                )
                num_cols = max(num_cols, col_index + col_span)

            num_rows = body_start_row + row_idx + 1

        normalized_tables.append(
            NormalizedTable(
                bbox=table_bbox,
                num_rows=num_rows,
                num_cols=num_cols,
                caption="",
                cells=cells,
            )
        )

    return normalized_tables


# ── Engine class ────────────────────────────────────────────────────────────


class GcpDocumentAiEngine(OCREngine):
    """OCR engine that wraps GCP Document AI.

    Processes PDF files by:
    1. Creating a ``DocumentProcessorServiceClient`` (with optional key-file
       credentials or Application Default Credentials).
    2. Sending the PDF via ``ProcessRequest`` to the configured processor.
    3. Converting the returned ``Document`` proto to a plain dict.
    4. Normalising all coordinates to page-space (points at 72 DPI,
       top-left origin).

    The engine requires a configured Document AI processor and GCP project.
    It works with Application Default Credentials or an explicit service
    account key file.
    """

    engine_id: ClassVar[str] = "gcp-document-ai"
    display_name: ClassVar[str] = "GCP Document AI"
    version: ClassVar[str] = _DOCUMENT_AI_VERSION

    # ── Config schema ──────────────────────────────────────────────────────

    @staticmethod
    def get_config_schema() -> dict[str, Any]:
        """Return JSON Schema for GCP Document AI-specific configuration.

        Supported parameters:
            - **processor_id** (string, required): Document AI processor ID.
            - **location** (string, default ``"us"``): GCP location.
            - **project_id** (string, required): GCP project ID.
            - **credentials_path** (string, optional): Path to service account
              JSON key file.
            - **timeout_seconds** (int, default ``300``): API timeout.
            - **mime_type** (string, default ``"application/pdf"``): Document
              MIME type.
        """
        return {
            "type": "object",
            "properties": {
                "processor_id": {
                    "type": "string",
                    "description": "Document AI processor ID (e.g. 'abc123...')",
                },
                "location": {
                    "type": "string",
                    "default": "us",
                    "description": "GCP location (e.g. 'us', 'eu')",
                },
                "project_id": {
                    "type": "string",
                    "description": "GCP project ID",
                },
                "credentials_path": {
                    "type": "string",
                    "description": "Path to service account JSON key file (optional, uses ADC if empty)",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "default": 300,
                    "minimum": 30,
                    "maximum": 3600,
                    "description": "API request timeout in seconds",
                },
                "mime_type": {
                    "type": "string",
                    "default": "application/pdf",
                    "description": "Document MIME type (e.g. 'application/pdf', 'image/png')",
                },
            },
            "required": ["processor_id", "project_id"],
        }

    # ── PDF processing ─────────────────────────────────────────────────────

    async def process_pdf(
        self,
        pdf_path: str | Path,
        config: dict[str, Any] | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> dict[str, Any]:
        """Run GCP Document AI OCR on a PDF file.

        Args:
            pdf_path: Path to the PDF file to process.
            config: Engine configuration (merged with defaults).
            progress: Optional progress callback (0‑100).

        Returns:
            Raw engine output dict with:
            - ``document``: Dict version of the Document AI ``Document`` proto.
            - ``engine_id``, ``engine_version``: Engine identification.
            - ``config_snapshot``: The resolved configuration.
            - ``page_count``: Number of pages processed.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            RuntimeError: If the Document AI SDK is unavailable or processing
                fails (e.g. missing credentials, API error).
        """
        if progress is not None:
            progress(0)

        resolved = {**DEFAULT_CONFIG, **(config or {})}
        pdf_path_obj = Path(pdf_path)

        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if not HAS_DOCUMENT_AI:
            raise RuntimeError(
                "google-cloud-documentai is required for GCP Document AI. "
                "Install it with: uv add google-cloud-documentai"
            )

        if progress is not None:
            progress(15)

        # Read the PDF content.
        pdf_content = pdf_path_obj.read_bytes()

        if progress is not None:
            progress(30)

        # Build the processor name.
        processor_name = (
            f"projects/{resolved['project_id']}"
            f"/locations/{resolved['location']}"
            f"/processors/{resolved['processor_id']}"
        )

        # Create the client — with optional explicit credentials.
        client_kwargs: dict[str, Any] = {}
        credentials_path = str(resolved.get("credentials_path") or "")
        if credentials_path:
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            client_kwargs["credentials"] = credentials

        client = documentai.DocumentProcessorServiceClient(**client_kwargs)

        if progress is not None:
            progress(50)

        # Build and send the processing request.
        request = documentai.ProcessRequest(
            name=processor_name,
            raw_document=documentai.RawDocument(
                content=pdf_content,
                mime_type=str(resolved.get("mime_type", "application/pdf")),
            ),
        )

        timeout = int(resolved.get("timeout_seconds", 300))

        try:
            result = await anyio.to_thread.run_sync(
                partial(client.process_document, request=request, timeout=timeout),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Document AI processing failed: {exc}. "
                "Ensure your processor is configured and credentials are valid."
            ) from exc

        if progress is not None:
            progress(80)

        document = result.document
        document_dict = _document_to_dict(document)
        page_count = len(document_dict.get("pages", []))

        if progress is not None:
            progress(100)

        return {
            "document": document_dict,
            "engine_id": "gcp-document-ai",
            "engine_version": self.version,
            "config_snapshot": resolved,
            "page_count": page_count,
        }

    # ── Normalisation ──────────────────────────────────────────────────────

    @staticmethod
    def normalize(
        raw: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert raw GCP Document AI output to ``NormalizedDocument``.

        Builds the canonical hierarchy using text_anchor offsets:
            1. Iterate pages → paragraphs → lines → tokens → symbols.
            2. Extract text from ``document.text`` using offset ranges.
            3. Build ``NormalizedPage`` with blocks → lines → words → characters.
            4. Convert Document AI table structures to ``NormalizedTable``.

        Args:
            raw: The raw output dict from ``process_pdf()``.

        Returns:
            A dict conforming to ``NormalizedDocument`` with all bounding
            boxes in page-space coordinates (points at 72 DPI, top-left
            origin).
        """
        document = raw.get("document", {}) or {}
        full_text = document.get("text", "") or ""
        raw_pages: list[dict[str, Any]] = (document.get("pages") or []) or []
        normalized_pages: list[NormalizedPage] = []

        for page_data in raw_pages:
            dimensions = (page_data.get("dimensions") or {}) or {}
            width = float(dimensions.get("width", 0) or 0)
            height = float(dimensions.get("height", 0) or 0)
            page_number_val = int(page_data.get("page_number", 0) or 0)

            if page_number_val < 1:
                page_number_val = len(normalized_pages) + 1

            # Extract flat lists from the page.
            paragraphs: list[dict[str, Any]] = (page_data.get("paragraphs") or []) or []
            lines: list[dict[str, Any]] = (page_data.get("lines") or []) or []
            tokens: list[dict[str, Any]] = (page_data.get("tokens") or []) or []
            symbols: list[dict[str, Any]] = (page_data.get("symbols") or []) or []
            tables_data: list[dict[str, Any]] = (page_data.get("tables") or []) or []

            blocks: list[NormalizedTextBlock | NormalizedTableBlock] = []

            # ── Build blocks from paragraphs ───────────────────────────
            if paragraphs:
                blocks = GcpDocumentAiEngine._build_paragraph_blocks(
                    paragraphs, lines, tokens, symbols, full_text, width, height
                )
            elif lines:
                # Fallback: group lines as blocks.
                blocks = GcpDocumentAiEngine._build_line_blocks(
                    lines, tokens, symbols, full_text, width, height
                )

            # ── Build tables ───────────────────────────────────────────
            normalized_tables = _build_tables(tables_data, full_text, width, height)

            normalized_pages.append(
                NormalizedPage(
                    page_number=page_number_val,
                    width=width,
                    height=height,
                    blocks=blocks,
                    tables=normalized_tables,
                )
            )

        # Handle completely empty input — use 1pt defaults for page dimensions
        # since NormalizedPage enforces width > 0, height > 0.
        if not normalized_pages:
            normalized_pages.append(
                NormalizedPage(
                    page_number=1,
                    width=1.0,
                    height=1.0,
                    blocks=[],
                    tables=[],
                )
            )

        doc = NormalizedDocument(
            pages=normalized_pages,
            engine_id=raw.get("engine_id", "gcp-document-ai"),
            engine_version=raw.get("engine_version", "unknown"),
            config_snapshot=raw.get("config_snapshot", {}),
        )
        return doc.model_dump()

    # ── Internal hierarchy builders ─────────────────────────────────────────

    @staticmethod
    def _build_paragraph_blocks(
        paragraphs: list[dict[str, Any]],
        lines: list[dict[str, Any]],
        tokens: list[dict[str, Any]],
        symbols: list[dict[str, Any]],
        full_text: str,
        page_width: float,
        page_height: float,
    ) -> list[NormalizedTextBlock]:
        """Build ``TextBlock`` objects from Document AI paragraphs.

        Maps:
            paragraph → TextBlock
            line → TextLine
            token → Word
            symbol → Character
        """
        blocks: list[NormalizedTextBlock] = []

        for p_idx, paragraph in enumerate(paragraphs):
            p_bbox = _layout_to_bbox(
                (paragraph.get("layout") or {}) or {}, page_width, page_height
            )

            # Find lines contained in this paragraph.
            contained_lines = _find_contained(lines, paragraph)
            contained_lines.sort(
                key=lambda ln: _layout_to_bbox(
                    (ln.get("layout") or {}) or {}, page_width, page_height
                )[1]
            )

            line_objects = GcpDocumentAiEngine._build_lines(
                contained_lines,
                tokens,
                symbols,
                full_text,
                page_width,
                page_height,
            )

            # Compute block bbox from constituent lines.
            if line_objects:
                block_x0 = min(ln.bbox[0] for ln in line_objects)
                block_y0 = min(ln.bbox[1] for ln in line_objects)
                block_x1 = max(ln.bbox[2] for ln in line_objects)
                block_y1 = max(ln.bbox[3] for ln in line_objects)
                block_conf = sum(ln.confidence for ln in line_objects) / len(line_objects)
            else:
                block_x0, block_y0, block_x1, block_y1 = p_bbox
                block_conf = float(
                    (paragraph.get("layout") or {}).get("confidence", 0.0) or 0.0
                )

            blocks.append(
                NormalizedTextBlock(
                    type="text",
                    bbox=[block_x0, block_y0, block_x1, block_y1],
                    confidence=block_conf,
                    order=p_idx,
                    lines=line_objects,
                )
            )

        return blocks

    @staticmethod
    def _build_line_blocks(
        lines: list[dict[str, Any]],
        tokens: list[dict[str, Any]],
        symbols: list[dict[str, Any]],
        full_text: str,
        page_width: float,
        page_height: float,
    ) -> list[NormalizedTextBlock]:
        """Build ``TextBlock`` objects directly from lines (no paragraphs).

        When a page has lines but no paragraphs, each line becomes a block
        containing a single line.
        """
        blocks: list[NormalizedTextBlock] = []

        # Sort lines top-to-bottom.
        sorted_lines = sorted(
            lines,
            key=lambda ln: _layout_to_bbox(
                (ln.get("layout") or {}) or {}, page_width, page_height
            )[1],
        )

        for p_idx, line in enumerate(sorted_lines):
            line_objects = GcpDocumentAiEngine._build_lines(
                [line], tokens, symbols, full_text, page_width, page_height
            )

            if line_objects:
                line_bbox = line_objects[0].bbox
                line_conf = line_objects[0].confidence
            else:
                line_bbox = _layout_to_bbox(
                    (line.get("layout") or {}) or {}, page_width, page_height
                )
                line_conf = float(
                    (line.get("layout") or {}).get("confidence", 0.0) or 0.0
                )

            blocks.append(
                NormalizedTextBlock(
                    type="text",
                    bbox=line_bbox,
                    confidence=line_conf,
                    order=p_idx,
                    lines=line_objects,
                )
            )

        return blocks

    @staticmethod
    def _build_lines(
        contained_lines: list[dict[str, Any]],
        tokens: list[dict[str, Any]],
        symbols: list[dict[str, Any]],
        full_text: str,
        page_width: float,
        page_height: float,
    ) -> list[NormalizedTextLine]:
        """Build ``TextLine`` objects from Document AI lines.

        Maps:
            line → TextLine
            token → Word
            symbol → Character
        """
        line_objects: list[NormalizedTextLine] = []

        for l_idx, line in enumerate(contained_lines):
            line_layout = (line.get("layout") or {}) or {}
            l_bbox = _layout_to_bbox(line_layout, page_width, page_height)
            l_conf = float(line_layout.get("confidence", 0.0) or 0.0)

            # Find tokens contained in this line.
            contained_tokens = _find_contained(tokens, line)
            contained_tokens.sort(
                key=lambda tk: _layout_to_bbox(
                    (tk.get("layout") or {}) or {}, page_width, page_height
                )[0]
            )

            word_objects = GcpDocumentAiEngine._build_words(
                contained_tokens, symbols, full_text, page_width, page_height
            )

            # Compute line bbox from constituent words.
            if word_objects:
                line_x0 = min(w.bbox[0] for w in word_objects)
                line_y0 = min(w.bbox[1] for w in word_objects)
                line_x1 = max(w.bbox[2] for w in word_objects)
                line_y1 = max(w.bbox[3] for w in word_objects)
                line_avg_conf = sum(w.confidence for w in word_objects) / len(word_objects)
                line_text = " ".join(w.text for w in word_objects)
            else:
                line_x0, line_y0, line_x1, line_y1 = l_bbox
                line_avg_conf = l_conf
                line_text = _extract_text(full_text, line)

            line_objects.append(
                NormalizedTextLine(
                    text=line_text,
                    bbox=[line_x0, line_y0, line_x1, line_y1],
                    confidence=line_avg_conf,
                    order=l_idx,
                    words=word_objects,
                )
            )

        return line_objects

    @staticmethod
    def _build_words(
        contained_tokens: list[dict[str, Any]],
        symbols: list[dict[str, Any]],
        full_text: str,
        page_width: float,
        page_height: float,
    ) -> list[NormalizedWord]:
        """Build ``Word`` objects from Document AI tokens.

        Maps:
            token → Word
            symbol → Character

        If no symbols are available for a token, characters are synthesised
        from the token text with evenly-spaced bounding boxes.
        """
        word_objects: list[NormalizedWord] = []

        for w_idx, token in enumerate(contained_tokens):
            token_layout = (token.get("layout") or {}) or {}
            w_text = _extract_text(full_text, token)
            w_bbox = _layout_to_bbox(token_layout, page_width, page_height)
            w_conf = float(token_layout.get("confidence", 0.0) or 0.0)

            # Find symbols contained in this token.
            contained_symbols = _find_contained(symbols, token)
            contained_symbols.sort(
                key=lambda sym: _layout_to_bbox(
                    (sym.get("layout") or {}) or {}, page_width, page_height
                )[0]
            )

            char_objects = GcpDocumentAiEngine._build_characters(
                contained_symbols, w_text, w_bbox, w_conf, full_text, page_width, page_height
            )

            word_objects.append(
                NormalizedWord(
                    text=w_text,
                    bbox=w_bbox,
                    confidence=w_conf,
                    order=w_idx,
                    chars=char_objects,
                )
            )

        return word_objects

    @staticmethod
    def _build_characters(
        contained_symbols: list[dict[str, Any]],
        word_text: str,
        word_bbox: list[float],
        word_conf: float,
        full_text: str,
        page_width: float,
        page_height: float,
    ) -> list[NormalizedCharacter]:
        """Build ``Character`` objects from Document AI symbols.

        If symbols are empty, characters are synthesised from the word text
        with evenly-spaced bounding boxes across the word's bbox.
        """
        char_objects: list[NormalizedCharacter] = []

        if contained_symbols:
            for c_idx, symbol in enumerate(contained_symbols):
                symbol_layout = (symbol.get("layout") or {}) or {}
                c_char = _extract_text(full_text, symbol)
                c_bbox = _layout_to_bbox(symbol_layout, page_width, page_height)
                c_conf = float(symbol_layout.get("confidence", 0.0) or 0.0)

                char_objects.append(
                    NormalizedCharacter(
                        char=c_char,
                        bbox=c_bbox,
                        confidence=c_conf,
                        order=c_idx,
                    )
                )
        else:
            # Synthesise characters from token text when no symbols exist.
            char_width = (word_bbox[2] - word_bbox[0]) / max(len(word_text), 1)
            for c_idx, ch in enumerate(word_text):
                cx0 = word_bbox[0] + c_idx * char_width
                cx1 = cx0 + char_width
                char_objects.append(
                    NormalizedCharacter(
                        char=ch,
                        bbox=[cx0, word_bbox[1], cx1, word_bbox[3]],
                        confidence=word_conf,
                        order=c_idx,
                    )
                )

        return char_objects


# ── Import-time registration ────────────────────────────────────────────────

with contextlib.suppress(EngineRegistryError):
    registry.register(GcpDocumentAiEngine)
