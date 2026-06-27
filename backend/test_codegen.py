"""Tests for the code generation endpoint.

Uses a fake subprocess to avoid calling the real LLM during unit tests.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _fake_run_opencode_success(
    goal: str,  # noqa: ARG001
    workdir: Path,
) -> str:
    """Simulate a successful codegen run by writing a test file."""
    (workdir / "hello.py").write_text("print('Hello from test')")
    return "mock log output"


async def _fake_run_opencode_async(goal: str, workdir: Path) -> str:
    """Async wrapper for the fake run_opencode."""
    return _fake_run_opencode_success(goal, workdir)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestCodeGenValidation:
    """Verify request validation."""

    def test_empty_goal_rejected(self) -> None:
        """Empty goal string should return 422."""
        resp = client.post("/api/v1/codegen", json={"goal": ""})
        assert resp.status_code == 422

    def test_missing_goal_rejected(self) -> None:
        """Missing goal field should return 422."""
        resp = client.post("/api/v1/codegen", json={})
        assert resp.status_code == 422

    def test_non_string_goal_rejected(self) -> None:
        """Non-string goal should return 422."""
        resp = client.post("/api/v1/codegen", json={"goal": 42})
        assert resp.status_code == 422


class TestCodeGenSuccess:
    """Verify successful code generation."""

    @patch("backend.routers.codegen._run_opencode", new_callable=AsyncMock)
    def test_basic_generation(self, mock_run: AsyncMock) -> None:
        """Happy path: goal → generated file → committed git repo."""
        async def fake(goal: str, workdir: Path) -> str:
            (workdir / "app.py").write_text("def main():\n    pass\n")
            return "mock log"

        mock_run.side_effect = fake

        resp = client.post("/api/v1/codegen", json={"goal": "Create app.py with a main function"})
        assert resp.status_code == 200, resp.text

        data = resp.json()
        assert data["commit_hash"] is not None
        assert "app.py" in data["files"]
        assert "def main()" in data["contents"].get("app.py", "")

    @patch("backend.routers.codegen._run_opencode", new_callable=AsyncMock)
    def test_git_log_shows_two_commits(self, mock_run: AsyncMock) -> None:
        """Repo should have 'Initial commit' + 'Generated code'."""
        async def fake(goal: str, workdir: Path) -> str:
            (workdir / "test.txt").write_text("hello")
            return "mock log"

        mock_run.side_effect = fake

        resp = client.post("/api/v1/codegen", json={"goal": "Create test.txt"})
        assert resp.status_code == 200

        # Verify project dir exists and has git history
        project_dir = resp.json()["project_dir"]
        log = (
            subprocess_run := __import__("subprocess").run(
                ["git", "log", "--oneline"],
                cwd=project_dir, capture_output=True, text=True,
            )
        ).stdout.strip().splitlines()
        assert len(log) >= 2  # Initial commit + Generated code

    @patch("backend.routers.codegen._run_opencode", new_callable=AsyncMock)
    def test_gitignore_excludes_opencode(self, mock_run: AsyncMock) -> None:
        """.opencode/ and .omo/ should not appear in tracked files."""
        async def fake(goal: str, workdir: Path) -> str:
            # Simulate opencode creating its runtime files
            (workdir / ".opencode").mkdir(exist_ok=True)
            (workdir / ".opencode" / "cfg.json").write_text("{}")
            (workdir / ".omo").mkdir(exist_ok=True)
            (workdir / ".omo" / "state.json").write_text("{}")
            # And the user's file
            (workdir / "index.html").write_text("<h1>Hi</h1>")
            return "mock log"

        mock_run.side_effect = fake

        resp = client.post("/api/v1/codegen", json={"goal": "Create index.html"})
        assert resp.status_code == 200
        data = resp.json()

        assert "index.html" in data["files"]
        assert not any(f.startswith(".opencode") for f in data["files"])
        assert not any(f.startswith(".omo") for f in data["files"])


class TestCodeGenFailure:
    """Verify error handling."""

    @patch("backend.routers.codegen._run_opencode", new_callable=AsyncMock)
    def test_subprocess_nonzero_exit(self, mock_run: AsyncMock) -> None:
        """When opencode exits with non-zero, should return 502."""
        from backend.routers.codegen import _run_opencode as real_run

        # We need to check that _run_opencode raises RuntimeError on non-zero
        # Instead, just verify that the endpoint handles errors properly
        async def fake(goal: str, workdir: Path) -> str:  # noqa: ARG001
            raise RuntimeError("opencode exited with code 1")

        mock_run.side_effect = fake

        resp = client.post("/api/v1/codegen", json={"goal": "do something"})
        assert resp.status_code == 502

    @patch("backend.routers.codegen._run_opencode", new_callable=AsyncMock)
    def test_no_files_generated(self, mock_run: AsyncMock) -> None:
        """When LLM generates no files, should return 400."""
        async def fake(goal: str, workdir: Path) -> str:  # noqa: ARG001
            return "llm responded but no files"

        mock_run.side_effect = fake

        resp = client.post("/api/v1/codegen", json={"goal": "do nothing"})
        assert resp.status_code == 400

    @patch("backend.routers.codegen._run_opencode", new_callable=AsyncMock)
    def test_timeout_returns_504(self, mock_run: AsyncMock) -> None:
        """When opencode times out, should return 504."""
        async def fake(goal: str, workdir: Path) -> str:  # noqa: ARG001
            raise TimeoutError("timed out")

        mock_run.side_effect = fake

        resp = client.post("/api/v1/codegen", json={"goal": "slow task"})
        assert resp.status_code == 504
