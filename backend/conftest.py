"""Pytest fixtures for OCR engine interface tests."""

from pathlib import Path

import pytest

from backend.mock_engine import MockEngine


@pytest.fixture
def mock_pdf_path() -> Path:
    """Return a dummy PDF path for mock engine testing.

    The mock engine never reads the file, so the path need not exist.
    """
    return Path("/tmp/nonexistent/test_document.pdf")


@pytest.fixture
def mock_engine() -> MockEngine:
    """Return a MockEngine instance for testing."""
    return MockEngine()
