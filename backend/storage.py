"""Content-addressable storage for PDF files, keyed by SHA-256 hash.

Files are stored at ``store/pdfs/{sha256[:2]}/{sha256[2:4]}/{sha256}.pdf``.
Content-addressing guarantees that identical content maps to the same path,
providing automatic deduplication.
"""

from __future__ import annotations

from contextlib import suppress
from functools import partial
from pathlib import Path

import anyio


class ContentAddressableStorage:
    """Stores and retrieves files at content-addressable paths.

    Each file is stored under a three-level directory tree derived from its
    SHA-256 hash, ensuring that the same content always maps to the same path.

    Attributes:
        base_path: Root directory for all stored content.
    """

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path.resolve()

    def _file_path(self, sha256: str) -> Path:
        """Compute the content-addressable file path for a given SHA-256 hash.

        Args:
            sha256: The SHA-256 hex digest of the file contents.

        Returns:
            Absolute path where the file should be stored.
        """
        return self._base_path / "pdfs" / sha256[:2] / sha256[2:4] / f"{sha256}.pdf"

    async def store(self, file_bytes: bytes, sha256: str) -> Path:
        """Store file bytes at the content-addressable path.

        If a file with the same SHA-256 hash already exists, the write is
        skipped (content-addressed deduplication).

        Args:
            file_bytes: The raw file content to store.
            sha256: The SHA-256 hex digest of the file contents.

        Returns:
            The absolute path to the stored file.
        """
        path = self._file_path(sha256)
        exists = await anyio.to_thread.run_sync(path.exists)
        if exists:
            return path
        await anyio.to_thread.run_sync(partial(path.parent.mkdir, parents=True, exist_ok=True))
        await anyio.to_thread.run_sync(path.write_bytes, file_bytes)
        return path

    async def retrieve(self, sha256: str) -> bytes:
        """Retrieve file bytes by SHA-256 hash.

        Args:
            sha256: The SHA-256 hex digest identifying the file.

        Returns:
            The raw file content as bytes.

        Raises:
            FileNotFoundError: If no file exists for the given hash.
        """
        path = self._file_path(sha256)
        return await anyio.to_thread.run_sync(path.read_bytes)

    async def delete(self, sha256: str) -> None:
        """Delete the file for the given SHA-256 hash.

        If the file does not exist, the operation silently succeeds.

        Args:
            sha256: The SHA-256 hex digest identifying the file to delete.
        """
        path = self._file_path(sha256)
        with suppress(FileNotFoundError):
            await anyio.to_thread.run_sync(path.unlink)
