"""Shared enumerations for OCRScore data models."""

from __future__ import annotations

import enum


class PDFStatus(enum.StrEnum):
    """Lifecycle status of an uploaded PDF document."""

    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"
    DELETED = "deleted"


class RunStatus(enum.StrEnum):
    """Lifecycle status of an OCR processing run."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GroundTruthSource(enum.StrEnum):
    """Provenance of a ground-truth version."""

    MANUAL = "manual"
    CONSENSUS = "consensus"
    IMPORTED = "imported"


class ScoreLevel(enum.StrEnum):
    """Granularity level at which a score is computed."""

    CHARACTER = "character"
    WORD = "word"
    LINE = "line"
    PARAGRAPH = "paragraph"
    BLOCK = "block"
    TABLE = "table"
    PAGE = "page"
    DOCUMENT = "document"


class ScoreMetric(enum.StrEnum):
    """Metric used to compute a score value."""

    CER = "cer"  # Character Error Rate
    WER = "wer"  # Word Error Rate
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    EDIT_DISTANCE = "edit_distance"
    CONFIDENCE = "confidence"
    TABLE_STRUCTURE = "table_structure"
