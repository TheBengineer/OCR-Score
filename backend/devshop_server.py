#!/usr/bin/env python3
"""Devshop — autonomous remote agent server.

Accepts high-level goals over HTTP, executes them using the opencode CLI
(which leverages deepseek-v4-flash LLM), and returns compact status updates.

Designed for low-bandwidth links: minimal JSON in, minimal JSON out.

Usage (remote):
    python3 devshop_server.py [--port 8900] [--opencode /path/to/opencode]

Usage (local — send a goal):
    curl -X POST http://remote:8900/task \\
      -H "Content-Type: application/json" \\
      -d '{"goal": "build a simple TODO app in Python"}'

Usage (local — poll status):
    curl http://remote:8900/task/<task_id>

Architecture:
    POST /task       — Submit a goal → returns {"task_id": "...", "status": "pending"}
    GET  /task/<id>  — Poll task status → {"status": "done"|"running"|"failed", "summary": {...}}
    GET  /status     — Server health → {"ok": true, "queue_depth": N, "uptime": "..."}
    POST /plan       — Submit a plan for execution (advanced) → {"plan_id": "...", "steps": [...]}

    Tasks run sequentially (one at a time) to avoid overwhelming the LLM.
    Each task is stored as a JSON file in ./devshop_tasks/ for persistence.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_PORT = 8900
OPENCODE_BIN = os.environ.get(
    "DEVSHOP_OPENCODE_BIN",
    subprocess.run(
        ["which", "opencode"], capture_output=True, text=True
    ).stdout.strip() or "/home/bengi/.opencode/bin/opencode",
)
TASKS_DIR = Path(os.environ.get("DEVSHOP_TASKS_DIR", "/tmp/devshop_tasks"))
TASKS_DIR.mkdir(exist_ok=True)
START_TIME = time.time()
_lock = threading.Lock()


# ── Task persistence ──────────────────────────────────────────────────────────


def _task_path(task_id: str) -> Path:
    return TASKS_DIR / f"{task_id}.json"


def _load_task(task_id: str) -> dict[str, Any] | None:
    path = _task_path(task_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _save_task(task: dict[str, Any]) -> None:
    path = _task_path(task["task_id"])
    path.write_text(json.dumps(task, indent=2, default=str))


def _next_task_id() -> str:
    return uuid.uuid4().hex[:12]


# ── Background worker ─────────────────────────────────────────────────────────


_task_queue: list[dict[str, Any]] = []
_current_task: dict[str, Any] | None = None
_worker_thread: threading.Thread | None = None
_worker_event = threading.Event()


def _worker_loop() -> None:
    """Background thread: pick tasks from queue, execute, store result."""
    global _current_task  # noqa: PLW0603
    while True:
        _worker_event.wait()
        _worker_event.clear()

        with _lock:
            if not _task_queue:
                _current_task = None
                continue
            task = _task_queue.pop(0)
            _current_task = task

        task["status"] = "running"
        task["started_at"] = datetime.now(timezone.utc).isoformat()
        _save_task(task)

        try:
            goal = task["goal"]
            workdir = Path(task["workdir"])

            # Execute the goal via opencode
            result = subprocess.run(
                [OPENCODE_BIN, "run", goal,
                 "--dangerously-skip-permissions", "--print-logs",
                 "--dir", str(workdir),
                 "--model", "opencode-go/deepseek-v4-flash"],
                capture_output=True, text=True,
                timeout=600,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )

            log = (result.stdout or "") + "\n" + (result.stderr or "")

            if result.returncode != 0:
                task["status"] = "failed"
                task["error"] = log[-1000:]
            else:
                # Collect generated files from the workdir
                files = _collect_files(workdir)
                task["status"] = "done"
                task["summary"] = {
                    "files": files,
                    "file_count": len(files),
                    "returncode": result.returncode,
                }

        except subprocess.TimeoutExpired:
            task["status"] = "failed"
            task["error"] = "Timed out after 600s"
        except Exception as exc:
            task["status"] = "failed"
            task["error"] = str(exc)

        task["finished_at"] = datetime.now(timezone.utc).isoformat()
        _save_task(task)

        with _lock:
            _current_task = None


def _collect_files(workdir: Path) -> list[dict[str, Any]]:
    """Walk *workdir* and return compact info about generated files."""
    files: list[dict[str, Any]] = []
    for f in sorted(workdir.rglob("*")):
        if not f.is_file():
            continue
        rel = str(f.relative_to(workdir))
        # Skip opencode runtime files
        if rel.startswith(".opencode") or rel.startswith(".omo") or rel.startswith(".git"):
            continue
        try:
            size = f.stat().st_size
            # Only include first 200 chars of content for summary
            text = f.read_text(errors="replace")[:200]
        except Exception:
            size = 0
            text = ""
        files.append({"path": rel, "size": size, "preview": text})
    return files


def _start_worker() -> None:
    global _worker_thread  # noqa: PLW0603
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
        _worker_thread.start()


def submit_task(goal: str) -> dict[str, Any]:
    """Create a task, enqueue it, and return immediately."""
    task_id = _next_task_id()
    workdir = TASKS_DIR / task_id
    workdir.mkdir(parents=True, exist_ok=True)

    # Init git repo
    subprocess.run(["git", "init"], cwd=str(workdir), capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "devshop@local"],
        cwd=str(workdir), capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Devshop"],
        cwd=str(workdir), capture_output=True,
    )
    (workdir / ".gitignore").write_text(".opencode/\n.omo/\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=str(workdir), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(workdir), capture_output=True)

    task: dict[str, Any] = {
        "task_id": task_id,
        "goal": goal,
        "status": "pending",
        "workdir": str(workdir),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_task(task)

    with _lock:
        _task_queue.append(task)

    _worker_event.set()
    _start_worker()

    return {"task_id": task_id, "status": "pending", "goal": goal}


# ── HTTP handler ──────────────────────────────────────────────────────────────


class DevshopHandler(BaseHTTPRequestHandler):
    """Minimal JSON-only HTTP handler."""

    def log_message(self, fmt: str, *args: Any) -> None:
        """Quiet logging — only log errors."""
        if args and args[0].startswith("4") or args[0].startswith("5"):
            super().log_message(fmt, *args)

    def _send_json(self, data: dict[str, Any], code: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/task":
            body = self._read_body()
            if not body or "goal" not in body:
                self._send_json({"error": "missing 'goal' in request body"}, 400)
                return
            goal = str(body["goal"]).strip()
            if not goal:
                self._send_json({"error": "'goal' must be non-empty"}, 400)
                return
            result = submit_task(goal)
            self._send_json(result, 202)

        elif self.path == "/plan":
            body = self._read_body()
            if not body or "plan" not in body:
                self._send_json({"error": "missing 'plan' in request body"}, 400)
                return
            # Plans are multi-step — submit the first step as a task
            plan = body["plan"]
            if isinstance(plan, list) and len(plan) > 0:
                first_goal = plan[0] if isinstance(plan[0], str) else plan[0].get("goal", str(plan[0]))
                result = submit_task(first_goal)
                result["plan_steps"] = len(plan)
                self._send_json(result, 202)
            else:
                self._send_json({"error": "plan must be a non-empty list"}, 400)

        else:
            self._send_json({"error": "not found"}, 404)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/status":
            with _lock:
                queue_depth = len(_task_queue)
                current = _current_task["task_id"] if _current_task else None
            uptime = time.time() - START_TIME
            self._send_json({
                "ok": True,
                "queue_depth": queue_depth,
                "current_task": current,
                "uptime_seconds": round(uptime),
                "version": "0.1.0",
            })

        elif self.path.startswith("/task/"):
            task_id = self.path[6:]
            if not re.match(r"^[a-f0-9]{12}$", task_id):
                self._send_json({"error": "invalid task_id"}, 400)
                return
            task = _load_task(task_id)
            if task is None:
                self._send_json({"error": "task not found"}, 404)
                return
            # Return compact response
            resp: dict[str, Any] = {
                "task_id": task["task_id"],
                "status": task["status"],
                "goal": task["goal"],
            }
            if task["status"] == "done":
                resp["summary"] = task.get("summary", {})
            if task["status"] == "failed":
                resp["error"] = task.get("error", "unknown error")
            self._send_json(resp)

        else:
            self._send_json({"error": "not found"}, 404)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else DEFAULT_PORT

    if "--opencode" in sys.argv:
        idx = sys.argv.index("--opencode")
        global OPENCODE_BIN  # noqa: PLW0603
        OPENCODE_BIN = sys.argv[idx + 1]

    # Verify opencode is available
    if not Path(OPENCODE_BIN).exists():
        print(f"Warning: opencode not found at {OPENCODE_BIN}", file=sys.stderr)
        print("Set DEVSHOP_OPENCODE_BIN env var or pass --opencode <path>", file=sys.stderr)

    server = HTTPServer(("0.0.0.0", port), DevshopHandler)
    print(f"Devshop server listening on http://0.0.0.0:{port}", flush=True)
    print(f"  OPENCODE_BIN={OPENCODE_BIN}", flush=True)
    print(f"  TASKS_DIR={TASKS_DIR.resolve()}", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
