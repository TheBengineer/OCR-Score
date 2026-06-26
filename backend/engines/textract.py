"""AWS Textract OCR engine module.

Wraps ``boto3`` to process PDFs via the AWS Textract API, extracting text
with bounding boxes at the word, line, and block levels.  Converts raw
Textract output into the canonical ``NormalizedDocument`` schema for
cross-engine comparison.

The engine registers itself with ``EngineRegistry`` at import time and works
when AWS credentials are available (via explicit keys, environment variables,
or IAM roles), gracefully raising ``RuntimeError`` with a clear message when
they are not.

Dependencies
------------
- ``boto3`` (required) — AWS SDK for Python.

Coordinate system
-----------------
Textract returns geometry as **normalised** values (0.0–1.0) relative to the
page dimensions in both ``BoundingBox`` and ``Polygon``.  The PAGE block
carries ``Width`` and ``Height`` in **inches**.  This module converts these
to the canonical page-space (points at 72 DPI, top-left origin):

    point_x = normalised_x * page_width_inches * 72
    point_y = normalised_y * page_height_inches * 72

Textract block hierarchy
------------------------
Textract returns a flat list of ``Block`` dicts linked by ``Relationships``:

    PAGE
    ├── LINE (via CHILD relationship)
    │   └── WORD (via CHILD relationship of LINE)
    ├── TABLE (via CHILD relationship)
    │   └── CELL (via CHILD relationship of TABLE)
    │       └── WORD (via CHILD relationship of CELL)
    ├── KEY_VALUE_SET (via CHILD relationship)
    └── SELECTION_ELEMENT (via CHILD relationship)

This module:
1. Indexes all blocks by ``Id``.
2. Finds PAGE blocks and follows their CHILD relationships.
3. Groups WORD blocks by LINE parent → builds ``Word`` objects.
4. Groups CELL blocks by TABLE parent → builds ``Table`` objects.
5. Converts all coordinates to page-space points at 72 DPI.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

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
    import boto3
    from botocore.config import Config as BotoConfig

    HAS_BOTO3: bool = True
except ImportError:
    HAS_BOTO3 = False

# ── SDK version helper ──────────────────────────────────────────────────────


def _get_sdk_version() -> str:
    """Return the ``boto3`` SDK version, or ``"unknown"``."""
    if HAS_BOTO3:
        try:
            return boto3.__version__  # type: ignore[attr-defined,unused-ignore]  # noqa: PGH003
        except AttributeError:
            pass
    return "unknown"


_TEXTRACT_VERSION: str = _get_sdk_version()

# ── Default configuration ───────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    "access_key_id": "",
    "secret_access_key": "",
    "region": "us-east-1",
    "timeout_seconds": 300,
}

# ── Confidence helper ───────────────────────────────────────────────────────


def _normalize_confidence(conf: Any) -> float:
    """Normalise a Textract confidence value (0‑100) to a ``[0.0, 1.0]`` float.

    Args:
        conf: Raw confidence value from a Textract block.

    Returns:
        Confidence in ``[0.0, 1.0]``, or ``0.0`` for ``None`` / negative /
        unparseable inputs.
    """
    if conf is None:
        return 0.0
    try:
        val = float(conf)
    except (ValueError, TypeError):
        return 0.0
    if val < 0.0:
        return 0.0
    return min(val / 100.0, 1.0)


# ── Coordinate helpers ──────────────────────────────────────────────────────


def _polygon_to_bbox(
    polygon: list[dict[str, float]] | None,
    page_width_pts: float,
    page_height_pts: float,
) -> list[float]:
    """Convert a Textract ``Polygon`` (normalised 0‑1 vertices) to a bbox.

    Textract ``Polygon`` vertices have ``X`` and ``Y`` values in the
    normalised ``[0.0, 1.0]`` range relative to page dimensions.  This
    function returns a bounding box in page-space points (72 DPI).

    Args:
        polygon: List of ``{"X": float, "Y": float}`` vertices, or ``None``.
        page_width_pts: Page width in points.
        page_height_pts: Page height in points.

    Returns:
        ``[x0, y0, x1, y1]`` in page-space points, top-left origin.
    """
    if not polygon:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [v.get("X", 0.0) for v in polygon]
    ys = [v.get("Y", 0.0) for v in polygon]
    return [
        min(xs) * page_width_pts,
        min(ys) * page_height_pts,
        max(xs) * page_width_pts,
        max(ys) * page_height_pts,
    ]


def _bbox_to_points(
    bbox: dict[str, float] | None,
    page_width_pts: float,
    page_height_pts: float,
) -> list[float]:
    """Convert a Textract ``BoundingBox`` (normalised 0‑1) to page-space points.

    Textract ``BoundingBox`` has ``Left``, ``Top``, ``Width``, ``Height`` —
    all in the normalised ``[0.0, 1.0]`` range relative to page dimensions.

    Args:
        bbox: A ``{"Left": …, "Top": …, "Width": …, "Height": …}`` dict,
            or ``None``.
        page_width_pts: Page width in points.
        page_height_pts: Page height in points.

    Returns:
        ``[x0, y0, x1, y1]`` in page-space points, top-left origin.
    """
    if not bbox:
        return [0.0, 0.0, 0.0, 0.0]
    left = float(bbox.get("Left", 0.0) or 0.0)
    top = float(bbox.get("Top", 0.0) or 0.0)
    width = float(bbox.get("Width", 0.0) or 0.0)
    height = float(bbox.get("Height", 0.0) or 0.0)
    return [
        left * page_width_pts,
        top * page_height_pts,
        (left + width) * page_width_pts,
        (top + height) * page_height_pts,
    ]


# ── Block index and relationship helpers ────────────────────────────────────


def _build_block_index(
    blocks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index Textract blocks by ``Id`` for ``O(1)`` lookup.

    Args:
        blocks: Flat list of Textract block dicts.

    Returns:
        A ``{block_id: block_dict}`` mapping.
    """
    return {b["Id"]: b for b in blocks if "Id" in b}


def _get_child_ids(
    block: dict[str, Any],
    rel_type: str = "CHILD",
) -> list[str]:
    """Get child block IDs from a block's relationships of a given type.

    Args:
        block: A Textract block dict.
        rel_type: The relationship type to extract (e.g. ``"CHILD"``,
            ``"MERGED_CELL"``, ``"VALUE"``, ``"TITLE"``, ``"ANSWER"``).

    Returns:
        List of child block IDs.  Empty list if the relationship type is
        not present.
    """
    relationships: list[dict[str, Any]] = block.get("Relationships") or []
    for rel in relationships:
        if rel.get("Type") == rel_type:
            return rel.get("Ids", [])
    return []


# ── Hierarchy builders ──────────────────────────────────────────────────────


def _build_word(
    block: dict[str, Any],
    page_width_pts: float,
    page_height_pts: float,
    order: int,
) -> NormalizedWord:
    """Build a ``NormalizedWord`` from a Textract ``WORD`` block.

    Textract does not provide character-level geometry, so characters are
    synthesised from the word text with evenly-spaced bounding boxes across
    the word's bbox.

    Args:
        block: A Textract ``WORD`` block dict.
        page_width_pts: Page width in points.
        page_height_pts: Page height in points.
        order: Reading order position within the parent line.

    Returns:
        A ``NormalizedWord`` with synthesised characters.
    """
    text = block.get("Text", "") or ""
    conf = _normalize_confidence(block.get("Confidence"))
    bbox = _bbox_to_points(
        (block.get("Geometry") or {}).get("BoundingBox"),
        page_width_pts,
        page_height_pts,
    )

    # Synthesise characters from the word text.
    char_width = (bbox[2] - bbox[0]) / max(len(text), 1) if text else 0.0
    chars: list[NormalizedCharacter] = []
    for i, ch in enumerate(text):
        chars.append(
            NormalizedCharacter(
                char=ch,
                bbox=[
                    bbox[0] + i * char_width,
                    bbox[1],
                    bbox[0] + (i + 1) * char_width,
                    bbox[3],
                ],
                confidence=conf,
                order=i,
            )
        )

    return NormalizedWord(
        text=text,
        bbox=bbox,
        confidence=conf,
        order=order,
        chars=chars,
    )


def _build_text_line(
    block: dict[str, Any],
    block_index: dict[str, dict[str, Any]],
    page_width_pts: float,
    page_height_pts: float,
) -> NormalizedTextLine:
    """Build a ``NormalizedTextLine`` from a Textract ``LINE`` block.

    Traverses the LINE's CHILD relationships to find WORD blocks and
    builds ``NormalizedWord`` objects.  The line's bounding box is
    recomputed from its constituent words for tighter bounds.

    Args:
        block: A Textract ``LINE`` block dict.
        block_index: ``{block_id: block_dict}`` mapping for this page.
        page_width_pts: Page width in points.
        page_height_pts: Page height in points.

    Returns:
        A ``NormalizedTextLine`` with word children.
    """
    child_ids = _get_child_ids(block)

    words: list[NormalizedWord] = []
    for order, child_id in enumerate(child_ids):
        child = block_index.get(child_id)
        if child is None or child.get("BlockType") != "WORD":
            continue
        words.append(
            _build_word(child, page_width_pts, page_height_pts, order)
        )

    if words:
        line_bbox = [
            min(w.bbox[0] for w in words),
            min(w.bbox[1] for w in words),
            max(w.bbox[2] for w in words),
            max(w.bbox[3] for w in words),
        ]
        line_confidence = sum(w.confidence for w in words) / len(words)
        line_text = " ".join(w.text for w in words)
    else:
        line_bbox = _bbox_to_points(
            (block.get("Geometry") or {}).get("BoundingBox"),
            page_width_pts,
            page_height_pts,
        )
        line_confidence = _normalize_confidence(block.get("Confidence"))
        line_text = block.get("Text", "") or ""

    return NormalizedTextLine(
        text=line_text,
        bbox=line_bbox,
        confidence=line_confidence,
        order=0,
        words=words,
    )


def _build_table(
    block: dict[str, Any],
    block_index: dict[str, dict[str, Any]],
    page_width_pts: float,
    page_height_pts: float,
) -> NormalizedTable | None:
    """Build a ``NormalizedTable`` from a Textract ``TABLE`` block.

    Traverses the TABLE's CHILD relationships to find CELL blocks and
    builds ``NormalizedTableCell`` objects.  Textract uses **1-based**
    ``RowIndex`` / ``ColumnIndex``; the normalised schema uses 0-based.

    Args:
        block: A Textract ``TABLE`` block dict.
        block_index: ``{block_id: block_dict}`` mapping for this page.
        page_width_pts: Page width in points.
        page_height_pts: Page height in points.

    Returns:
        A ``NormalizedTable`` with cells, or ``None`` if no cells are found.
    """
    child_ids = _get_child_ids(block)

    cells: list[NormalizedTableCell] = []
    num_rows = 0
    num_cols = 0

    for child_id in child_ids:
        child = block_index.get(child_id)
        if child is None or child.get("BlockType") != "CELL":
            continue

        # Textract uses 1-based indices → convert to 0-based.
        row = int(child.get("RowIndex", 1) or 1) - 1
        col = int(child.get("ColumnIndex", 1) or 1) - 1
        row_span = int(child.get("RowSpan", 1) or 1)
        col_span = int(child.get("ColumnSpan", 1) or 1)
        cell_text = child.get("Text", "") or ""
        cell_conf = _normalize_confidence(child.get("Confidence"))
        cell_bbox = _bbox_to_points(
            (child.get("Geometry") or {}).get("BoundingBox"),
            page_width_pts,
            page_height_pts,
        )

        cells.append(
            NormalizedTableCell(
                row=row,
                col=col,
                row_span=row_span,
                col_span=col_span,
                text=cell_text,
                bbox=cell_bbox,
                confidence=cell_conf,
            )
        )
        num_rows = max(num_rows, row + row_span)
        num_cols = max(num_cols, col + col_span)

    if not cells:
        return None

    table_bbox = _bbox_to_points(
        (block.get("Geometry") or {}).get("BoundingBox"),
        page_width_pts,
        page_height_pts,
    )

    return NormalizedTable(
        bbox=table_bbox,
        num_rows=num_rows,
        num_cols=num_cols,
        caption="",
        cells=cells,
    )


def _build_table_block(
    block: dict[str, Any],
    page_width_pts: float,
    page_height_pts: float,
    order: int,
) -> NormalizedTableBlock:
    """Build a ``NormalizedTableBlock`` from a Textract ``TABLE`` block.

    This lightweight block marker is placed in the page's ``blocks`` list
    alongside text blocks, while the detailed cell data goes into the
    page's ``tables`` list.

    Args:
        block: A Textract ``TABLE`` block dict.
        page_width_pts: Page width in points.
        page_height_pts: Page height in points.
        order: Reading order position within the page.

    Returns:
        A ``NormalizedTableBlock``.
    """
    table_bbox = _bbox_to_points(
        (block.get("Geometry") or {}).get("BoundingBox"),
        page_width_pts,
        page_height_pts,
    )
    table_conf = _normalize_confidence(block.get("Confidence"))

    return NormalizedTableBlock(
        type="table",
        bbox=table_bbox,
        confidence=table_conf,
        order=order,
    )


# ── Engine class ────────────────────────────────────────────────────────────


class TextractEngine(OCREngine):
    """OCR engine that wraps AWS Textract via ``boto3``.

    Processes PDF files by:
    1. Reading the PDF bytes.
    2. Sending to Textract via ``analyze_document`` with ``TABLES`` and
       ``FORMS`` features.
    3. Building a block index and reconstructing the hierarchy from
       relationship links.
    4. Normalising all coordinates to page-space (points at 72 DPI,
       top-left origin).

    .. note::

        The sync ``analyze_document`` API processes only the **first page**
        of a PDF.  For multi-page support, use the async
        ``start_document_analysis`` API with an S3 ``DocumentLocation``
        (available in a future enhancement).
    """

    engine_id: ClassVar[str] = "aws-textract"
    display_name: ClassVar[str] = "AWS Textract"
    version: ClassVar[str] = _TEXTRACT_VERSION

    # ── Config schema ──────────────────────────────────────────────────────

    @staticmethod
    def get_config_schema() -> dict[str, Any]:
        """Return JSON Schema for Textract-specific configuration.

        Supported parameters:
            - **access_key_id** (string, optional): AWS access key ID.
            - **secret_access_key** (string, optional): AWS secret access key.
            - **region** (string, default ``"us-east-1"``): AWS region name.
            - **timeout_seconds** (int, default ``300``): API request timeout.
        """
        return {
            "type": "object",
            "properties": {
                "access_key_id": {
                    "type": "string",
                    "description": "AWS access key ID (optional, uses default chain if empty)",
                },
                "secret_access_key": {
                    "type": "string",
                    "description": "AWS secret access key (optional, uses default chain if empty)",
                },
                "region": {
                    "type": "string",
                    "default": "us-east-1",
                    "description": "AWS region name (e.g. 'us-east-1', 'eu-west-1')",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "default": 300,
                    "minimum": 30,
                    "maximum": 3600,
                    "description": "API request timeout in seconds",
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
        """Run Textract OCR on a PDF file.

        Uses the sync ``analyze_document`` API wrapped in
        ``anyio.to_thread.run_sync`` to avoid blocking the event loop.

        Args:
            pdf_path: Path to the PDF file to process.
            config: Engine configuration (merged with defaults).
            progress: Optional progress callback (0‑100).

        Returns:
            Raw engine output dict with:
            - ``blocks``: Flat list of Textract block dicts.
            - ``document_metadata``: Textract ``DocumentMetadata`` dict.
            - ``engine_id``, ``engine_version``: Engine identification.
            - ``config_snapshot``: The resolved configuration.
            - ``page_count``: Number of pages in the response.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            RuntimeError: If boto3 is unavailable or Textract processing
                fails (e.g. missing credentials, API error).
        """
        if progress is not None:
            progress(0)

        resolved = {**DEFAULT_CONFIG, **(config or {})}
        pdf_path_obj = Path(pdf_path)

        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if not HAS_BOTO3:
            raise RuntimeError(
                "boto3 is required for AWS Textract. "
                "Install it with: uv add boto3"
            )

        if progress is not None:
            progress(10)

        pdf_bytes = pdf_path_obj.read_bytes()

        if progress is not None:
            progress(20)

        # ── Create Textract client ─────────────────────────────────
        client_kwargs: dict[str, Any] = {
            "region_name": str(resolved.get("region", "us-east-1")),
            "config": BotoConfig(
                connect_timeout=int(resolved.get("timeout_seconds", 300)),
                read_timeout=int(resolved.get("timeout_seconds", 300)),
                retries={"max_attempts": 3},
            ),
        }

        ak = str(resolved.get("access_key_id") or "")
        sk = str(resolved.get("secret_access_key") or "")
        if ak and sk:
            client_kwargs["aws_access_key_id"] = ak
            client_kwargs["aws_secret_access_key"] = sk

        if progress is not None:
            progress(30)

        try:
            client = boto3.client("textract", **client_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create Textract client: {exc}. "
                "Ensure your AWS credentials are configured."
            ) from exc

        if progress is not None:
            progress(40)

        # ── Send document to Textract ─────────────────────────────
        try:
            response = await anyio.to_thread.run_sync(
                lambda: client.analyze_document(
                    Document={"Bytes": pdf_bytes},
                    FeatureTypes=["TABLES", "FORMS"],
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Textract processing failed: {exc}. "
                "Ensure your AWS credentials are configured and the "
                "document is valid."
            ) from exc

        if progress is not None:
            progress(80)

        blocks: list[dict[str, Any]] = response.get("Blocks", [])
        page_count = sum(
            1 for b in blocks if b.get("BlockType") == "PAGE"
        )
        if page_count < 1:
            page_count = 1

        if progress is not None:
            progress(100)

        return {
            "blocks": blocks,
            "document_metadata": response.get("DocumentMetadata", {}),
            "engine_id": "aws-textract",
            "engine_version": self.version,
            "config_snapshot": resolved,
            "page_count": page_count,
        }

    # ── Normalisation ──────────────────────────────────────────────────────

    @staticmethod
    def normalize(raw: dict[str, Any]) -> dict[str, Any]:
        """Convert raw Textract output to ``NormalizedDocument``.

        Reconstructs the hierarchy from the flat block list using
        relationship links:

        1. Index all blocks by ``Id``.
        2. Find PAGE blocks and extract their CHILD relationships.
        3. For each child of a PAGE:
           - ``LINE`` → ``TextBlock`` with one ``TextLine`` of ``Word`` objects.
           - ``TABLE`` → ``TableBlock`` + ``NormalizedTable`` with cells.
           - ``KEY_VALUE_SET`` → preserved for future forms support.
           - ``SELECTION_ELEMENT`` → handled but not yet mapped to output.

        Args:
            raw: The raw output dict from ``process_pdf()``.

        Returns:
            A dict conforming to ``NormalizedDocument`` with all bounding
            boxes in page-space coordinates (points at 72 DPI, top-left
            origin).
        """
        blocks: list[dict[str, Any]] = raw.get("blocks", [])

        # ── Group blocks by page ──────────────────────────────────
        page_groups: dict[int, list[dict[str, Any]]] = {}
        for block in blocks:
            page_num = int(block.get("Page", 1) or 1)
            page_groups.setdefault(page_num, []).append(block)

        normalized_pages: list[NormalizedPage] = []

        for page_num in sorted(page_groups):
            page_blocks = page_groups[page_num]
            block_index = _build_block_index(page_blocks)

            # Find the PAGE block for this page.
            page_block: dict[str, Any] | None = None
            for b in page_blocks:
                if b.get("BlockType") == "PAGE":
                    page_block = b
                    break

            if page_block is None:
                continue

            # Page dimensions: Textract stores Width/Height in inches.
            page_width_pts = float(
                page_block.get("Width", 8.5) or 8.5
            ) * 72.0
            page_height_pts = float(
                page_block.get("Height", 11.0) or 11.0
            ) * 72.0

            child_ids = _get_child_ids(page_block)

            page_text_blocks: list[NormalizedTextBlock | NormalizedTableBlock] = []
            page_tables: list[NormalizedTable] = []
            block_order = 0

            for child_id in child_ids:
                child_block = block_index.get(child_id)
                if child_block is None:
                    continue

                block_type = child_block.get("BlockType")

                if block_type == "LINE":
                    line = _build_text_line(
                        child_block, block_index,
                        page_width_pts, page_height_pts,
                    )
                    page_text_blocks.append(
                        NormalizedTextBlock(
                            type="text",
                            bbox=line.bbox,
                            confidence=line.confidence,
                            order=block_order,
                            lines=[line],
                        )
                    )
                    block_order += 1

                elif block_type == "TABLE":
                    # Add the table block marker.
                    table_block = _build_table_block(
                        child_block, page_width_pts,
                        page_height_pts, block_order,
                    )
                    if table_block is not None:
                        page_text_blocks.append(table_block)
                        block_order += 1

                    # Build the full table with cells.
                    table = _build_table(
                        child_block, block_index,
                        page_width_pts, page_height_pts,
                    )
                    if table is not None:
                        page_tables.append(table)

                elif block_type == "KEY_VALUE_SET":
                    # Future: build form entries from key / value pairs.
                    entity_types: list[str] = (
                        child_block.get("EntityTypes") or []
                    )
                    logger.debug(
                        "KEY_VALUE_SET (%s) on page %d — forms support "
                        "not yet implemented",
                        ",".join(entity_types),
                        page_num,
                    )

                elif block_type == "SELECTION_ELEMENT":
                    # Future: map to form field values.
                    selection_status = child_block.get(
                        "SelectionStatus", ""
                    )
                    logger.debug(
                        "SELECTION_ELEMENT (%s) on page %d — selection "
                        "support not yet mapped",
                        selection_status,
                        page_num,
                    )

                # Other block types (e.g. MERGED_CELL, TITLE, etc.)
                # are handled implicitly through the CHILD relationships
                # of their parent blocks.

            normalized_pages.append(
                NormalizedPage(
                    page_number=page_num,
                    width=page_width_pts,
                    height=page_height_pts,
                    blocks=page_text_blocks,
                    tables=page_tables,
                )
            )

        # Handle completely empty input — use default Letter dimensions.
        if not normalized_pages:
            normalized_pages.append(
                NormalizedPage(
                    page_number=1,
                    width=612.0,
                    height=792.0,
                    blocks=[],
                    tables=[],
                )
            )

        doc = NormalizedDocument(
            pages=normalized_pages,
            engine_id=raw.get("engine_id", "aws-textract"),
            engine_version=raw.get("engine_version", "unknown"),
            config_snapshot=raw.get("config_snapshot", {}),
        )
        return doc.model_dump()


# ── Import-time registration ────────────────────────────────────────────────

with contextlib.suppress(EngineRegistryError):
    registry.register(TextractEngine)
