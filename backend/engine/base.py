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

Engines that need API keys or credentials declare them via the
``required_secrets`` class attribute — a list of ``SecretDef`` entries
that the user must configure before running that engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── Secret definition ──────────────────────────────────────────────────────────


@dataclass
class SecretDef:
    """Declares a secret (API key, credential) an engine needs to operate.

    Each entry becomes a labelled input field in the Security settings page.
    Secrets are stored in the ``EngineSecret`` table, keyed by engine + key,
    and are injected into the engine's ``config`` dict before ``process_pdf``
    is called (they override any user-supplied value with the same key).

    Attributes:
        key: Machine key used to store and inject the secret (e.g.
            ``"aws_access_key_id"``).  This same key appears in the ``config``
            dict the engine receives.
        env_var: Optional environment variable name that also supplies this
            value (e.g. ``"AWS_ACCESS_KEY_ID"``).  If the env var is set it
            takes precedence, but the UI still shows the field.
        display_name: Short human-readable label for the UI.
        description: Longer explanation shown as placeholder / help text.
    """

    key: str
    env_var: str | None = None
    display_name: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.key.replace("_", " ").title()


# ── Base engine class ──────────────────────────────────────────────────────────


class OCREngine(ABC):
    """Abstract base class for all OCR engine plugins.

    Every concrete engine must define the class attributes ``engine_id``,
    ``display_name``, and ``version``, and implement all abstract methods.
    Optionally define ``required_secrets`` to declare API keys / credentials.

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

    # ── Secrets this engine needs (override in subclasses) ─────────────────
    required_secrets: list[SecretDef] = []

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
                ``get_config_schema()`` upstream).  Any secrets declared
                in ``required_secrets`` that have been configured by the
                user are already merged into this dict.
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
            blocks, lines, words, and characters in the canonical
            page-space coordinate system (points at 72 DPI, top-left origin).
        """
        ...

    @classmethod
    def get_secret_schema(cls) -> list[SecretDef]:
        """Return the list of secrets this engine requires.

        The default implementation returns ``cls.required_secrets``.
        Subclasses may override to compute the list dynamically.
        """
        return list(cls.required_secrets)
