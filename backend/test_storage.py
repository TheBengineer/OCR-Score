"""Tests for the content-addressable storage layer.

These are pure unit tests — no database or external services required.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from backend.storage import ContentAddressableStorage


@pytest.fixture
def storage(tmp_path: Path) -> ContentAddressableStorage:
    """Provide a ``ContentAddressableStorage`` backed by a temporary directory."""
    return ContentAddressableStorage(tmp_path)


class TestContentAddressableStorage:
    """Verify that ``ContentAddressableStorage`` correctly de-duplicates,
    round-trips, and cleans up files by SHA-256 hash."""

    async def test_content_addressable_storage_same_hash(self, storage: ContentAddressableStorage) -> None:
        """Given the same file bytes with the same SHA-256 hash,
        When storing them twice, Then the returned paths are identical."""
        data = b"hello world, this is a test PDF-like content"
        sha256 = hashlib.sha256(data).hexdigest()

        path_a = await storage.store(data, sha256)
        path_b = await storage.store(data, sha256)

        assert path_a == path_b

    async def test_content_addressable_storage_different_hash(self, storage: ContentAddressableStorage) -> None:
        """Given two different byte sequences with different SHA-256 hashes,
        When storing them, Then the returned paths differ."""
        data_a = b"content A with distinct bytes"
        data_b = b"content B with distinct bytes"
        sha256_a = hashlib.sha256(data_a).hexdigest()
        sha256_b = hashlib.sha256(data_b).hexdigest()

        path_a = await storage.store(data_a, sha256_a)
        path_b = await storage.store(data_b, sha256_b)

        assert path_a != path_b

    async def test_store_retrieve_roundtrip(self, storage: ContentAddressableStorage) -> None:
        """Given a sequence of bytes,
        When stored and then retrieved, Then the original bytes are returned unchanged."""
        original = b"roundtrip test content for PDF storage"
        sha256 = hashlib.sha256(original).hexdigest()

        await storage.store(original, sha256)
        retrieved = await storage.retrieve(sha256)

        assert retrieved == original

    async def test_delete_removes_file(self, storage: ContentAddressableStorage) -> None:
        """Given a stored file,
        When deleted and then retrieved, Then ``FileNotFoundError`` is raised."""
        data = b"content to be deleted"
        sha256 = hashlib.sha256(data).hexdigest()

        await storage.store(data, sha256)
        await storage.delete(sha256)

        with pytest.raises(FileNotFoundError):
            await storage.retrieve(sha256)

    async def test_store_idempotent(self, storage: ContentAddressableStorage) -> None:
        """Given the same file bytes,
        When stored multiple times, Then all operations succeed without error."""
        data = b"idempotent storage test"
        sha256 = hashlib.sha256(data).hexdigest()

        path_a = await storage.store(data, sha256)
        path_b = await storage.store(data, sha256)

        assert await storage.retrieve(sha256) == data
        assert path_a == path_b

    async def test_delete_missing_file_noop(self, storage: ContentAddressableStorage) -> None:
        """Given a non-existent SHA-256,
        When delete is called, Then no exception is raised."""
        sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
        await storage.delete(sha256)  # should not raise
