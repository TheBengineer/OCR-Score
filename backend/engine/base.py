"""Abstract base class for all OCR engine plugins.

All OCR engine modules (Tesseract, GCP Document AI, AWS Textract, etc.)
must subclass ``OCREngine`` and implement its three abstract methods:

1. ``get_config_schema()`` — Declare valid engine configuration via JSON Schema.
2. ``process_pdf()`` — Run OCR on a PDF and return *raw* engine output.
3. ``normalize()`` — Convert raw output into the standardised ``NormalizedDocument``
   structure (defined in ``backend.engine.normalized_schema``).

Engine subclasses also set three **class-level attributes** that identify them:

- ``engine_id``: Short machine-readable slug (e.g. ``"tesseract"``).
- ``display_name``: Human-readable name (e.g. ``"Tesseract OCR"``).
- ``version``: Engine plugin version (e.g. ``"0.1.0"``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any


class OCREngine(ABC):
    """Abstract base class for all OCR engine plugins.

    Every concrete engine must define the class attributes ``engine_id``,
    ``display_name``, and ``version``, and implement all abstract methods.

    Typical lifecycle::

        engine = MyEngine()
        raw = await engine.process_pdf("/path/to.pdf", {"option": "value"}, progress_fn)
        normalised = engine.normalize(raw)
        # normalised is a dict ready for PageResult.data JSONB
    """

    # ── Class-level identifiers (set by subclasses) ────────────────────────
    engine_id: str
    display_name: str
    version: str

    # ── Abstract methods ──────────────────────────────────────────────────

    @abstractmethod
    def get_config_schema(self) -> dict[str, Any]:
        """Return a `JSON Schema <https://json-schema.org/>`_ dict describing
        valid configuration parameters for this engine.

        The schema is used by the UI and API to validate user-provided
        ``config`` before passing it to ``process_pdf()``.

        Returns:
            A JSON Schema object (draft 2020-12 or compatible) that describes
            the engine-specific config.  At minimum this must be::

                {"type": "object", "properties": {}, "required": []}
        """
        ...

    @abstractmethod
    async def process_pdf(
        self,
        pdf_path: str | Path,
        config: dict[str, Any],
        progress: Callable[[int], None],
    ) -> dict[str, Any]:
        """Execute OCR on a PDF file and return **raw** engine-specific output.

        This method preserves the engine's native output verbatim so that
        debugging and migrations can always access the original data.

        Args:
            pdf_path: Path to the PDF file to process.
            config: Engine-specific configuration (validated against
                ``get_config_schema()`` upstream).
            progress: Callback invoked with an ``int`` between 0 and 100
                to report processing progress.  Engines should call this
                at meaningful checkpoints.

        Returns:
            A **raw** output dict whose structure is engine-specific.
            This dict MUST contain enough information for ``normalize()``
            to produce a complete ``NormalizedDocument``.

        Raises:
            FileNotFoundError: If ``pdf_path`` does not exist.
            RuntimeError: If OCR processing fails for any reason.
        """
        ...

    @abstractmethod
    def normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Convert **raw** engine output to the standardised normalized schema.

        This is a **pure transformation** — no I/O, no side effects.

        The returned dict MUST validate against
        ``backend.engine.normalized_schema.NormalizedDocument`` and produce
        ``PageResultData``-compatible JSONB when its ``pages`` entries are
        stored individually.

        Args:
            raw: The dict returned by ``process_pdf()``.

        Returns:
            A dict conforming to ``NormalizedDocument`` with all pages,
            blocks, lines, words, characters, and tables in the canonical
            page-space coordinate system (points at 72 DPI, top-left origin).
        """
        ...
