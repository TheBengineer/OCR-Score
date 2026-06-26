"""OCR engine implementations.

This package contains concrete OCR engine plugins that extend the
``OCREngine`` abstract base class defined in ``backend.engine.base``.

Each module implements at minimum:
- ``get_config_schema()`` — JSON Schema for engine-specific configuration.
- ``process_pdf()`` — Runs OCR on a PDF, returns raw engine output.
- ``normalize()`` — Converts raw output to the canonical ``NormalizedDocument``
  structure.

Engines are registered with the global ``EngineRegistry`` at import time.
"""

from backend.engines.gcp_document_ai import GcpDocumentAiEngine
from backend.engines.tesseract import TesseractEngine
from backend.engines.textract import TextractEngine

__all__ = [
    "GcpDocumentAiEngine",
    "TesseractEngine",
    "TextractEngine",
]
