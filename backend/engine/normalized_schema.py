"""Canonical normalized output schema for OCR engines.

All engine ``normalize()`` implementations MUST return a dict conforming to
``NormalizedDocument`` — the standardised structure that gets stored as
``PageResult.data`` JSONB in the database.

Coordinate system
-----------------
All bounding boxes use **page-space coordinates**:
    - Unit: points (1/72 inch) at **72 DPI**
    - Origin: **top-left** corner of the page
    - Format: ``[x0, y0, x1, y1]``

The JSONB column expects:
    .. code-block:: python

        {
            "blocks": [
                {
                    "type": "text|table|figure|math|separator",
                    "bbox": [x0, y0, x1, y1],
                    "confidence": 0.95,
                    "order": 0,
                    "lines": [
                        {
                            "text": "example text",
                            "bbox": [x0, y0, x1, y1],
                            "confidence": 0.95,
                            "order": 0,
                            "words": [
                                {
                                    "text": "example",
                                    "bbox": [x0, y0, x1, y1],
                                    "confidence": 0.95,
                                    "order": 0,
                                    "chars": [
                                        {
                                            "char": "e",
                                            "bbox": [x0, y0, x1, y1],
                                            "confidence": 0.95,
                                            "order": 0
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ],
            "tables": [
                {
                    "bbox": [x0, y0, x1, y1],
                    "num_rows": 3,
                    "num_cols": 4,
                    "caption": "Table 1",
                    "cells": [
                        {
                            "row": 0,
                            "col": 0,
                            "row_span": 1,
                            "col_span": 1,
                            "text": "Header",
                            "bbox": [x0, y0, x1, y1],
                            "confidence": 0.95
                        }
                    ]
                }
            ]
        }
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Text hierarchy ────────────────────────────────────────────────────────────


class Character(BaseModel):
    """A single character with its bounding box and confidence score.

    This is the finest granularity in the OCR output hierarchy.
    """

    char: str = Field(..., description="The individual character")
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x0, y0, x1, y1] in points at 72 DPI, top-left origin",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    order: int = Field(..., ge=0, description="Reading order within the parent word")


class Word(BaseModel):
    """A word composed of one or more characters."""

    text: str = Field(..., description="The full word text")
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x0, y0, x1, y1] in points at 72 DPI, top-left origin",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    order: int = Field(..., ge=0, description="Reading order within the parent line")
    chars: list[Character] = Field(default_factory=list)


class TextLine(BaseModel):
    """A single line of text composed of words."""

    text: str = Field(..., description="Concatenated text of all words in the line")
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x0, y0, x1, y1] in points at 72 DPI, top-left origin",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    order: int = Field(..., ge=0, description="Reading order within the parent block")
    words: list[Word] = Field(default_factory=list)


# ── Block types ───────────────────────────────────────────────────────────────


class TextBlock(BaseModel):
    """A text-type block on the page — the most common OCR block type.

    Contains a list of ``TextLine`` objects with their word and character
    children.
    """

    type: str = Field(
        default="text",
        pattern=r"^(text|table|figure|math|separator)$",
        description="Block type identifier",
    )
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x0, y0, x1, y1] in points at 72 DPI, top-left origin",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    order: int = Field(..., ge=0, description="Reading order within the page")
    lines: list[TextLine] = Field(default_factory=list)


class TableBlock(BaseModel):
    """A table-type structural block on the page.

    Table blocks mark regions identified as tabular content. The detailed
    cell structure lives in the ``Table`` objects within the page-level
    ``tables`` list.
    """

    type: str = Field(
        default="table",
        pattern=r"^(text|table|figure|math|separator)$",
        description="Block type identifier",
    )
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x0, y0, x1, y1] in points at 72 DPI, top-left origin",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    order: int = Field(..., ge=0, description="Reading order within the page")


# ── Table types ───────────────────────────────────────────────────────────────


class TableCell(BaseModel):
    """A single cell within a detected table."""

    row: int = Field(..., ge=0, description="Row index (0-based)")
    col: int = Field(..., ge=0, description="Column index (0-based)")
    row_span: int = Field(default=1, ge=1, description="Number of rows the cell spans")
    col_span: int = Field(default=1, ge=1, description="Number of columns the cell spans")
    text: str = Field(default="", description="Cell text content")
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x0, y0, x1, y1] in points at 72 DPI, top-left origin",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")


class Table(BaseModel):
    """A detected table structure with its cells."""

    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x0, y0, x1, y1] in points at 72 DPI, top-left origin",
    )
    num_rows: int = Field(..., ge=0, description="Number of rows in the table")
    num_cols: int = Field(..., ge=0, description="Number of columns in the table")
    caption: str = Field(default="", description="Table caption or title text")
    cells: list[TableCell] = Field(default_factory=list)


# ── Page and document containers ──────────────────────────────────────────────


class NormalizedPage(BaseModel):
    """A single page's OCR output, containing blocks and tables.

    When serialised, this dict is stored as one entry in the ``pages`` list
    of ``NormalizedDocument``.   Engine implementers should produce one
    ``NormalizedPage`` per page of the input PDF.
    """

    page_number: int = Field(..., ge=1, description="1-based page number")
    width: float = Field(..., gt=0, description="Page width in points (72 DPI)")
    height: float = Field(..., gt=0, description="Page height in points (72 DPI)")
    blocks: list[TextBlock | TableBlock] = Field(
        default_factory=list,
        description="Ordered list of content blocks on the page",
    )
    tables: list[Table] = Field(
        default_factory=list,
        description="Detected table structures with cell-level detail",
    )


class NormalizedDocument(BaseModel):
    """Top-level container for the normalised output of an entire PDF.

    This is the return type contract for ``OCREngine.normalize()``.
    The ``model_dump()`` of this object is the authoritative representation
    that gets persisted as the engine output for a run.

    Engine implementers should build one ``NormalizedDocument`` from the raw
    engine output, then call ``model_dump()`` to produce the final dict.
    """

    pages: list[NormalizedPage] = Field(
        ...,
        min_length=1,
        description="All pages in reading order",
    )
    engine_id: str = Field(
        ...,
        description="Unique engine identifier (e.g. 'tesseract', 'gcp-document-ai')",
    )
    engine_version: str = Field(
        ...,
        description="Engine plugin version string",
    )
    config_snapshot: dict = Field(
        default_factory=dict,
        description="Snapshot of the engine configuration used for this run",
    )
