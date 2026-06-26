"""Pydantic schemas for PageResult — including the canonical JSONB hierarchy."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


# ── Canonical JSONB sub-models ───────────────────────────────────────────────
class PageResultChar(BaseModel):
    """A single character with its bounding box and confidence."""

    char: str = Field(..., description="The character text")
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x0, y0, x1, y1] in points at 72 DPI, top-left origin",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    order: int = Field(..., ge=0, description="Reading order within the word")


class PageResultWord(BaseModel):
    """A word composed of characters."""

    text: str = Field(..., description="The full word text")
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x0, y0, x1, y1] in points at 72 DPI",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    order: int = Field(..., ge=0, description="Reading order within the line")
    chars: list[PageResultChar] = Field(default_factory=list)


class PageResultLine(BaseModel):
    """A line of text composed of words."""

    text: str = Field(..., description="Concatenated line text")
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x0, y0, x1, y1] in points at 72 DPI",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    order: int = Field(..., ge=0, description="Reading order within the block")
    words: list[PageResultWord] = Field(default_factory=list)


class PageTableBlock(BaseModel):
    """A block-level element (text, table, figure, math, separator)."""

    type: str = Field(
        ...,
        pattern=r"^(text|table|figure|math|separator)$",
        description="Block type",
    )
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x0, y0, x1, y1] in points at 72 DPI",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    order: int = Field(..., ge=0, description="Reading order within the page")
    lines: list[PageResultLine] = Field(default_factory=list)


class PageTableCell(BaseModel):
    """A single cell within a table."""

    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)
    row_span: int = Field(default=1, ge=1)
    col_span: int = Field(default=1, ge=1)
    text: str = Field(default="")
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x0, y0, x1, y1] in points at 72 DPI",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)


class PageTable(BaseModel):
    """A detected table structure."""

    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x0, y0, x1, y1] in points at 72 DPI",
    )
    num_rows: int = Field(..., ge=0)
    num_cols: int = Field(..., ge=0)
    caption: str = Field(default="")
    cells: list[PageTableCell] = Field(default_factory=list)


class PageResultData(BaseModel):
    """Canonical JSONB content for a page result."""

    blocks: list[PageTableBlock] = Field(default_factory=list)
    tables: list[PageTable] = Field(default_factory=list)


# ── Create ───────────────────────────────────────────────────────────────────
class PageResultCreate(BaseModel):
    """Schema for creating a page result."""

    run_id: uuid.UUID
    page_number: int = Field(..., ge=1, description="1-based page number")
    width: float | None = Field(
        default=None,
        gt=0,
        description="Page width in points (72 DPI)",
    )
    height: float | None = Field(
        default=None,
        gt=0,
        description="Page height in points (72 DPI)",
    )
    data: dict = Field(
        default_factory=dict,
        description="Canonical JSONB page hierarchy",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Aggregate per-page confidence",
    )


# ── Read ─────────────────────────────────────────────────────────────────────
class PageResultRead(BaseModel):
    """Schema for reading a page result."""

    id: uuid.UUID
    run_id: uuid.UUID
    page_number: int
    width: float | None = None
    height: float | None = None
    data: dict
    confidence: float | None = None

    model_config = {"from_attributes": True}
