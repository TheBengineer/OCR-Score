"""Code generation endpoint — takes a natural-language goal and generates
code using the deepseek-v4-flash LLM via the opencode CLI, placing the
result in a fresh git repo under ``generated/``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_OPENCODE = os.environ.get("OPENCODE_BIN", "/home/bengi/.opencode/bin/opencode")

codegen_router = APIRouter(prefix="/api/v1/codegen", tags=["codegen"])

# ── Root for generated projects (writable temp dir) ────────────────────────
_BASE = Path(tempfile.gettempdir()) / "ocrscore-codegen"
_BASE.mkdir(parents=True, exist_ok=True)

_GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

# Cleanup: delete projects older than 1 hour
_TTL_SECONDS = 3600
_last_cleanup: float = 0


def _cleanup_old_projects() -> None:
    """Remove generated project directories older than ``_TTL_SECONDS``."""
    global _last_cleanup  # noqa: PLW0603
    now = time.time()
    if now - _last_cleanup < _TTL_SECONDS / 2:
        return
    _last_cleanup = now
    for child in _BASE.iterdir():
        if not child.is_dir():
            continue
        try:
            age = now - child.stat().st_mtime
            if age > _TTL_SECONDS:
                shutil.rmtree(child, ignore_errors=True)
                logger.info("cleaned up expired project %s", child.name)
        except OSError:
            pass


# ── Schemas ─────────────────────────────────────────────────────────────────


class CodeGenRequest(BaseModel):
    goal: str = Field(..., min_length=1, description="Natural-language description of what to build")


class CodeGenResponse(BaseModel):
    project_dir: str
    commit_hash: str | None = None
    files: list[str] = []
    contents: dict[str, str] = {}
    log: str = ""


# ── Helpers ─────────────────────────────────────────────────────────────────


async def _run_opencode(goal: str, workdir: Path) -> str:
    """Run ``opencode run <goal>`` inside *workdir* and return stderr+stdout.

    Uses the ``opencode-go`` provider (which has a valid API key in
    ``~/.local/share/opencode/auth.json``) rather than the default LM Studio
    provider whose endpoint is unreachable.
    """
    proc = await asyncio.create_subprocess_exec(
        _OPENCODE,
        "run", goal,
        "--dangerously-skip-permissions",
        "--print-logs",
        "--dir", str(workdir),
        "--model", "opencode-go/deepseek-v4-flash",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_GIT_ENV,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=300,
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise

    combined = (stdout or b"").decode("utf-8", errors="replace")
    combined += "\n" + (stderr or b"").decode("utf-8", errors="replace")

    if proc.returncode != 0:
        msg = f"opencode exited with code {proc.returncode}"
        logger.warning("%s\n%s", msg, combined[-500:])
        raise RuntimeError(f"{msg}: {combined[-500:]}")

    return combined


def _git_commit(workdir: Path) -> str | None:
    """Stage all changes and commit.  Returns the commit hash or ``None``."""
    subprocess.run(
        ["git", "add", "-A"],
        cwd=str(workdir), capture_output=True, env=_GIT_ENV,
    )
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=str(workdir), capture_output=True, env=_GIT_ENV,
    )
    if result.returncode == 0:
        return None  # nothing to commit

    subprocess.run(
        ["git", "commit", "-m", "Generated code"],
        cwd=str(workdir), capture_output=True, env=_GIT_ENV,
    )
    hash_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(workdir), capture_output=True, text=True, env=_GIT_ENV,
    )
    return hash_result.stdout.strip() or None


def _walk_files(workdir: Path) -> list[str]:
    """Return relative paths of all tracked files (only user-generated files)."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=str(workdir), capture_output=True, text=True, env=_GIT_ENV,
    )
    return sorted(
        f for f in result.stdout.strip().splitlines()
        if f and not f.startswith(".opencode") and not f.startswith(".omo")
    )


# ── Endpoint ────────────────────────────────────────────────────────────────


@codegen_router.post("")
async def generate_code(body: CodeGenRequest) -> CodeGenResponse:
    """Generate code for a natural-language goal using the deepseek LLM.

    The goal is sent to the LLM which writes files into a fresh git repo
    under ``generated/``.  The response lists every file created.
    """
    _cleanup_old_projects()

    project_id = uuid.uuid4().hex[:12]
    workdir = _BASE / project_id
    workdir.mkdir(parents=True, exist_ok=True)

    # Initialise git repo
    subprocess.run(
        ["git", "init"],
        cwd=str(workdir), capture_output=True, env=_GIT_ENV,
    )
    subprocess.run(
        ["git", "config", "user.email", "codegen@ocrscore.local"],
        cwd=str(workdir), capture_output=True, env=_GIT_ENV,
    )
    subprocess.run(
        ["git", "config", "user.name", "OCRScore CodeGen"],
        cwd=str(workdir), capture_output=True, env=_GIT_ENV,
    )
    # Prevent opencode runtime files from being tracked
    (workdir / ".gitignore").write_text(
        ".opencode/\n.omo/\n"
    )
    subprocess.run(
        ["git", "add", ".gitignore"],
        cwd=str(workdir), capture_output=True, env=_GIT_ENV,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=str(workdir), capture_output=True, env=_GIT_ENV,
    )

    try:
        log = await _run_opencode(body.goal, workdir)
    except TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Code generation timed out after 300s",
        ) from None
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    commit_hash = _git_commit(workdir)
    files = _walk_files(workdir)

    if not [f for f in files if f != ".gitignore"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="LLM did not generate any files for the given goal",
        )

    # Read generated file contents
    contents: dict[str, str] = {}
    for rel in files:
        try:
            contents[rel] = (workdir / rel).read_text(encoding="utf-8", errors="replace")
        except Exception:
            contents[rel] = ""

    return CodeGenResponse(
        project_dir=str(workdir),
        commit_hash=commit_hash,
        files=files,
        contents=contents,
        log=log[-3000:],
    )
