"""Tests for Tesseract OCR engine hardening features (Wave 2 contract).

These tests define the expected behaviour of hardening features that will
be implemented in Wave 2 of the Tesseract engine: page-level timeout,
deterministic classification (``classify_enable_learning=0``),
``user_defined_dpi`` passthrough, custom ``tesseract_config`` flag
passthrough, and optional image pre-processing.

Several tests are expected to **fail on Wave 1** — they define a contract
that the Wave 2 implementation must satisfy.

All tests are self-contained — they do **not** require a Tesseract binary,
poppler-utils, OpenCV, or any external service.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.engines.tesseract import TesseractEngine

# Reuse test-data factories from the existing test suite so we do not
# duplicate mock infrastructure.
from backend.engines.test_tesseract import (
    SAMPLE_CHARACTERS,
    SAMPLE_WORDS,
    _make_mock_boxes,
    _make_mock_image,
    _make_mock_image_to_data,
)

# ── Timeout hardening ────────────────────────────────────────────────────────


class TestTesseractTimeout:
    """Verify ``process_pdf`` respects *page_timeout* configuration."""

    @pytest.mark.asyncio
    async def test_timeout_hangs(self) -> None:
        """Given a Tesseract call that would hang, When page_timeout is set,
        Then a TimeoutError is raised within the configured deadline."""
        engine = TesseractEngine()
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "backend.engines.tesseract._pdf2image_convert",
                return_value=[_make_mock_image()],
            ),
            patch(
                "backend.engines.tesseract.pytesseract.image_to_data",
                return_value=_make_mock_image_to_data(SAMPLE_WORDS),
            ),
            patch(
                "backend.engines.tesseract.pytesseract.image_to_boxes",
                return_value=_make_mock_boxes(SAMPLE_CHARACTERS),
            ),
            # Simulate a timeout by making every wait_for call fail.
            patch(
                "asyncio.wait_for",
                side_effect=TimeoutError("Tesseract page timed out"),
            ),
            pytest.raises(TimeoutError),
        ):
            await engine.process_pdf(
                "/fake/path.pdf",
                config={"page_timeout": 0.1},
            )

    @pytest.mark.asyncio
    async def test_page_timeout_config(self) -> None:
        """Given page_timeout in config, When process_pdf runs,
        Then asyncio.wait_for is called with the configured timeout."""
        engine = TesseractEngine()
        captured_timeouts: list[float] = []

        async def _capture_wait_for(
            coro: Any,
            timeout: float | None = None,
        ) -> Any:
            captured_timeouts.append(timeout if timeout is not None else -1)
            return await coro

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "backend.engines.tesseract._pdf2image_convert",
                return_value=[_make_mock_image()],
            ),
            patch(
                "backend.engines.tesseract.pytesseract.image_to_data",
                return_value=_make_mock_image_to_data(SAMPLE_WORDS),
            ),
            patch(
                "backend.engines.tesseract.pytesseract.image_to_boxes",
                return_value=_make_mock_boxes(SAMPLE_CHARACTERS),
            ),
            patch("asyncio.wait_for", side_effect=_capture_wait_for),
        ):
            await engine.process_pdf(
                "/fake/path.pdf",
                config={"page_timeout": 42},
            )

        assert len(captured_timeouts) > 0
        assert all(t == 42 for t in captured_timeouts)


# ── Deterministic classification ─────────────────────────────────────────────


class TestTesseractClassifyEnableLearning:
    """Verify ``classify_enable_learning=0`` is sent to the Tesseract CLI."""

    @pytest.mark.asyncio
    async def test_classify_enable_learning(self) -> None:
        """Given a classify_enable_learning flag, When process_pdf runs,
        Then ``-c classify_enable_learning=0`` appears in the config string
        passed to both ``image_to_data`` and ``image_to_boxes``."""
        engine = TesseractEngine()
        mock_image_to_data = MagicMock(
            return_value=_make_mock_image_to_data(SAMPLE_WORDS),
        )
        mock_image_to_boxes = MagicMock(
            return_value=_make_mock_boxes(SAMPLE_CHARACTERS),
        )

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "backend.engines.tesseract._pdf2image_convert",
                return_value=[_make_mock_image()],
            ),
            patch(
                "backend.engines.tesseract.pytesseract.image_to_data",
                mock_image_to_data,
            ),
            patch(
                "backend.engines.tesseract.pytesseract.image_to_boxes",
                mock_image_to_boxes,
            ),
        ):
            await engine.process_pdf(
                "/fake/path.pdf",
                config={"classify_enable_learning": False},
            )

        data_config: str = mock_image_to_data.call_args[1].get("config", "")
        boxes_config: str = mock_image_to_boxes.call_args[1].get("config", "")
        assert "classify_enable_learning=0" in data_config
        assert "classify_enable_learning=0" in boxes_config


# ── User-defined DPI passthrough ─────────────────────────────────────────────


class TestTesseractUserDefinedDPI:
    """Verify ``user_defined_dpi`` is sent to the Tesseract CLI."""

    @pytest.mark.asyncio
    async def test_user_defined_dpi(self) -> None:
        """Given a DPI config, When process_pdf runs,
        Then ``-c user_defined_dpi=<dpi>`` is present in the config string."""
        engine = TesseractEngine()
        mock_image_to_data = MagicMock(
            return_value=_make_mock_image_to_data(SAMPLE_WORDS),
        )
        mock_image_to_boxes = MagicMock(
            return_value=_make_mock_boxes(SAMPLE_CHARACTERS),
        )

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "backend.engines.tesseract._pdf2image_convert",
                return_value=[_make_mock_image()],
            ),
            patch(
                "backend.engines.tesseract.pytesseract.image_to_data",
                mock_image_to_data,
            ),
            patch(
                "backend.engines.tesseract.pytesseract.image_to_boxes",
                mock_image_to_boxes,
            ),
        ):
            await engine.process_pdf(
                "/fake/path.pdf",
                config={"dpi": 300},
            )

        data_config: str = mock_image_to_data.call_args[1].get("config", "")
        boxes_config: str = mock_image_to_boxes.call_args[1].get("config", "")
        assert "user_defined_dpi=300" in data_config
        assert "user_defined_dpi=300" in boxes_config


# ── Custom tesseract_config passthrough ──────────────────────────────────────


class TestTesseractConfigPassthrough:
    """Verify custom *tesseract_config* flags are appended to the CLI call."""

    @pytest.mark.asyncio
    async def test_tesseract_config_passthrough(self) -> None:
        """Given a custom config flag via tesseract_config, When process_pdf runs,
        Then the flag appears in the Tesseract CLI config string."""
        engine = TesseractEngine()
        mock_image_to_data = MagicMock(
            return_value=_make_mock_image_to_data(SAMPLE_WORDS),
        )
        mock_image_to_boxes = MagicMock(
            return_value=_make_mock_boxes(SAMPLE_CHARACTERS),
        )

        extra_flag = "--tessdata-dir /custom/path"
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "backend.engines.tesseract._pdf2image_convert",
                return_value=[_make_mock_image()],
            ),
            patch(
                "backend.engines.tesseract.pytesseract.image_to_data",
                mock_image_to_data,
            ),
            patch(
                "backend.engines.tesseract.pytesseract.image_to_boxes",
                mock_image_to_boxes,
            ),
        ):
            await engine.process_pdf(
                "/fake/path.pdf",
                config={"tesseract_config": extra_flag},
            )

        data_config: str = mock_image_to_data.call_args[1].get("config", "")
        boxes_config: str = mock_image_to_boxes.call_args[1].get("config", "")
        assert extra_flag in data_config
        assert extra_flag in boxes_config


# ── Optional image pre-processing ────────────────────────────────────────────


class TestTesseractOptionalPreprocessing:
    """Verify optional image pre-processing is applied based on the config."""

    @pytest.mark.asyncio
    async def test_optional_preprocessing_enabled(self) -> None:
        """Given preprocess=True, When process_pdf runs,
        Then ``_preprocess_image`` is called for each page image.

        This test will **fail on Wave 1** because the preprocessing hook
        does not exist yet — it serves as the contract for Wave 2.
        """
        engine = TesseractEngine()

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "backend.engines.tesseract._pdf2image_convert",
                return_value=[_make_mock_image(), _make_mock_image()],
            ),
            patch(
                "backend.engines.tesseract.pytesseract.image_to_data",
                return_value=_make_mock_image_to_data(SAMPLE_WORDS),
            ),
            patch(
                "backend.engines.tesseract.pytesseract.image_to_boxes",
                return_value=_make_mock_boxes(SAMPLE_CHARACTERS),
            ),
            patch(
                "backend.engines.tesseract._preprocess_image",
                create=True,
            ) as mock_preprocess,
        ):
            await engine.process_pdf(
                "/fake/path.pdf",
                config={"preprocess": True},
            )

        mock_preprocess.assert_called()

    @pytest.mark.asyncio
    async def test_optional_preprocessing_disabled(self) -> None:
        """Given no preprocess flag, When process_pdf runs,
        Then ``_preprocess_image`` is **not** called.

        This test passes on both Wave 1 (no call possible) and Wave 2
        (bypassed when preprocess is absent).
        """
        engine = TesseractEngine()

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "backend.engines.tesseract._pdf2image_convert",
                return_value=[_make_mock_image()],
            ),
            patch(
                "backend.engines.tesseract.pytesseract.image_to_data",
                return_value=_make_mock_image_to_data(SAMPLE_WORDS),
            ),
            patch(
                "backend.engines.tesseract.pytesseract.image_to_boxes",
                return_value=_make_mock_boxes(SAMPLE_CHARACTERS),
            ),
            patch(
                "backend.engines.tesseract._preprocess_image",
                create=True,
            ) as mock_preprocess,
        ):
            await engine.process_pdf("/fake/path.pdf")

        mock_preprocess.assert_not_called()
