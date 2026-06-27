#!/usr/bin/env python3
"""Devshop — autonomous remote agent server.

Accepts high-level goals over HTTP, executes them using the opencode CLI
(which leverages deepseek-v4-flash LLM), and returns compact status updates.
Designed for low-bandwidth links: minimal JSON in, minimal JSON out.

Usage (remote):
    python3 devshop_server.py [--port 8900] [--opencode /path/to/opencode]
                              [--allow-dangerous] [--tasks-dir /path/to/tasks]

Usage (local — send a goal):
    curl -X POST http://remote:8900/task \\
      -H "Content-Type: application/json" \\
      -d '{"goal": "build a simple TODO app in Python"}'

Architecture:
    POST /task            — Submit a goal → {"task_id": "...", "status": "pending"}
    GET  /task/<id>       — Poll task status + file summary
    GET  /task/<id>/files — List generated files with full contents
    GET  /task/<id>/wait  — Long-poll: blocks until task completes (up to 120s)
    POST /plan            — Multi-step sequential plan → auto-chains steps
    POST /cancel/<id>     — Cancel a running/pending task
    GET  /status          — Server health

    --allow-dangerous:  Enables the opencode --dangerously-skip-permissions flag.
                        Without this flag, the server runs with guardrails.
                        NEVER enable on a publicly-reachable port.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

# ── Configuration (set from CLI args in main()) ──────────────────────────────

DEFAULT_PORT = 8900
OPENCODE_BIN = ""
TASKS_DIR: Path = Path("/tmp/devshop_tasks")
ALLOW_DANGEROUS = False
MODEL = "opencode-go/deepseek-v4-flash"
START_TIME = time.time()
_lock = threading.Lock()

# Background worker state
_task_queue: list[dict[str, Any]] = []
_current_task: dict[str, Any] | None = None
_current_proc: subprocess.Popen | None = None
_worker_thread: threading.Thread | None = None
_worker_event = threading.Event()
_completion_events: dict[str, threading.Event] = {}
_PLAN_SEQUENTIAL = True


# ── Task persistence ──────────────────────────────────────────────────────────


def _task_path(task_id: str) -> Path:
    return TASKS_DIR / f"{task_id}.json"


def _task_workdir(task_id: str) -> Path:
    return TASKS_DIR / task_id


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


def _worker_loop() -> None:
    """Background thread: pick tasks from queue, execute, store result."""
    global _current_task, _current_proc  # noqa: PLW0603
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

            cmd = [OPENCODE_BIN, "run", goal, "--print-logs", "--dir", str(workdir),
                   "--model", MODEL]
            if ALLOW_DANGEROUS:
                cmd.insert(3, "--dangerously-skip-permissions")

            with _lock:
                _current_proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                )
                proc = _current_proc

            stdout, stderr = proc.communicate(timeout=600)
            log = (stdout or "") + "\n" + (stderr or "")

            with _lock:
                _current_proc = None

            # If task was cancelled mid-flight via POST /cancel, skip overwrite
            if task.get("status") == "cancelled":
                _signal_completion(task["task_id"])
                _save_task(task)
            elif proc.returncode != 0:
                task["status"] = "failed"
                task["error"] = log[-1000:]
            else:
                files = _collect_files(workdir)
                task["status"] = "done"
                task["summary"] = {
                    "files": files,
                    "file_count": len(files),
                    "returncode": proc.returncode,
                }

        except subprocess.TimeoutExpired:
            with _lock:
                if _current_proc:
                    _current_proc.kill()
                    _current_proc = None
            task["status"] = "failed"
            task["error"] = "Timed out after 600s"
        except Exception as exc:
            task["status"] = "failed"
            task["error"] = str(exc)

        # Only set finished_at for non-cancelled tasks (cancel already saved)
        if task.get("status") != "cancelled":
            task["finished_at"] = datetime.now(timezone.utc).isoformat()
            _save_task(task)

        _signal_completion(task["task_id"])

        with _lock:
            _current_task = None
            # Re-arm worker event if more tasks are queued (avoids backlog)
            if _task_queue:
                _worker_event.set()

        # If this task is part of a plan chain, submit the next step
        plan_queue = task.get("_plan_queue")
        if plan_queue and task["status"] == "done":
            _do_submit_task(plan_queue[0], plan_queue[0], parent_id=task["task_id"],
                            plan_queue=plan_queue[1:])


def _collect_files(workdir: Path) -> list[dict[str, Any]]:
    """Walk *workdir* and return compact info about generated files."""
    files: list[dict[str, Any]] = []
    for f in sorted(workdir.rglob("*")):
        if not f.is_file():
            continue
        # Security: reject path traversal
        try:
            rel = str(f.relative_to(workdir))
        except ValueError:
            continue
        if rel.startswith(".opencode") or rel.startswith(".omo") or rel.startswith(".git"):
            continue
        try:
            size = f.stat().st_size
            text = f.read_text(errors="replace")[:200]
        except Exception:
            size = 0
            text = ""
        files.append({"path": rel, "size": size, "preview": text})
    return files


def _read_file_content(workdir: Path, file_rel: str) -> str | None:
    """Read a full file, validating it's within the workdir."""
    target = (workdir / file_rel).resolve()
    workdir_resolved = workdir.resolve()
    if not str(target).startswith(str(workdir_resolved) + "/"):
        return None  # path traversal
    if not target.is_file():
        return None
    try:
        return target.read_text(errors="replace")[:65536]  # cap at 64KB per file
    except Exception:
        return None


def _signal_completion(task_id: str) -> None:
    """Signal any long-poll waiters that a task completed."""
    with _lock:
        ev = _completion_events.get(task_id)
        if ev:
            ev.set()


def _start_worker() -> None:
    global _worker_thread  # noqa: PLW0603
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
        _worker_thread.start()


def _do_submit_task(goal: str, step_label: str, *,
                    parent_id: str | None = None,
                    plan_queue: list[str] | None = None) -> dict[str, Any]:
    """Internal: create a task, enqueue, return immediately."""
    task_id = _next_task_id()
    workdir = TASKS_DIR / task_id
    workdir.mkdir(parents=True, exist_ok=True)

    subprocess.run(["git", "init"], cwd=str(workdir), capture_output=True)
    subprocess.run(["git", "config", "user.email", "devshop@local"],
                   cwd=str(workdir), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Devshop"],
                   cwd=str(workdir), capture_output=True)
    (workdir / ".gitignore").write_text(".opencode/\n.omo/\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=str(workdir), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(workdir), capture_output=True)

    task: dict[str, Any] = {
        "task_id": task_id,
        "goal": goal,
        "step_label": step_label,
        "status": "pending",
        "workdir": str(workdir),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if parent_id:
        task["parent_task_id"] = parent_id
    if plan_queue:
        task["_plan_queue"] = plan_queue

    _save_task(task)

    with _lock:
        _task_queue.append(task)
        _completion_events[task_id] = threading.Event()

    _worker_event.set()
    _start_worker()

    return {"task_id": task_id, "status": "pending", "goal": goal}


def submit_task(goal: str) -> dict[str, Any]:
    """Public: create a single task."""
    return _do_submit_task(goal, goal)


def submit_plan(steps: list[str]) -> dict[str, Any]:
    """Submit a multi-step plan. Each step executes sequentially on success."""
    if not steps:
        return {"error": "plan must be a non-empty list"}
    first = steps[0]
    remaining = steps[1:] if len(steps) > 1 else None
    result = _do_submit_task(first, first, plan_queue=remaining)
    result["total_steps"] = len(steps)
    return result


def cancel_task(task_id: str) -> bool:
    """Cancel a pending or running task. Returns True if cancelled."""
    global _current_proc  # noqa: PLW0603
    with _lock:
        # Remove from queue if pending
        for i, t in enumerate(_task_queue):
            if t["task_id"] == task_id:
                _task_queue.pop(i)
                if task_id in _completion_events:
                    _completion_events[task_id].set()
                t["status"] = "cancelled"
                t["finished_at"] = datetime.now(timezone.utc).isoformat()
                _save_task(t)
                return True
        # Kill if running
        if _current_task and _current_task["task_id"] == task_id:
            if _current_proc:
                _current_proc.terminate()
                _current_proc = None
            _current_task["status"] = "cancelled"
            _current_task["finished_at"] = datetime.now(timezone.utc).isoformat()
            _save_task(_current_task)
            if task_id in _completion_events:
                _completion_events[task_id].set()
            return True
    return False


# ── HTTP handler ──────────────────────────────────────────────────────────────


class DevshopHandler(BaseHTTPRequestHandler):
    """Minimal JSON-only HTTP handler."""

    def log_message(self, fmt: str, *args: Any) -> None:
        if args and str(args[0]).startswith(("4", "5")):
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

    def _task_status(self, task: dict[str, Any]) -> dict[str, Any]:
        resp: dict[str, Any] = {
            "task_id": task["task_id"],
            "status": task["status"],
            "goal": task.get("step_label", task["goal"]),
        }
        if task["status"] == "done":
            resp["summary"] = task.get("summary", {})
        if task["status"] == "failed":
            resp["error"] = task.get("error", "unknown error")
        resp["created_at"] = task.get("created_at", "")
        return resp

    # ── POST ─────────────────────────────────────────────────────────────

    def do_POST(self) -> None:  # noqa: N802
        path = self.path
        body = self._read_body()

        if path == "/task":
            if not body or "goal" not in body:
                self._send_json({"error": "missing 'goal' in request body"}, 400)
                return
            goal = str(body["goal"]).strip()
            if not goal:
                self._send_json({"error": "'goal' must be non-empty"}, 400)
                return
            result = submit_task(goal)
            self._send_json(result, 202)

        elif path == "/plan":
            if not body or "plan" not in body:
                self._send_json({"error": "missing 'plan' in request body"}, 400)
                return
            plan = body["plan"]
            if not isinstance(plan, list) or len(plan) == 0:
                self._send_json({"error": "plan must be a non-empty list"}, 400)
                return
            if not all(isinstance(s, str) for s in plan):
                self._send_json({"error": "each plan step must be a string"}, 400)
                return
            result = submit_plan(plan)
            if "error" in result:
                self._send_json(result, 400)
            else:
                self._send_json(result, 202)

        elif path.startswith("/cancel/"):
            task_id = path[8:]
            if not re.match(r"^[a-f0-9]{12}$", task_id):
                self._send_json({"error": "invalid task_id"}, 400)
                return
            if cancel_task(task_id):
                self._send_json({"task_id": task_id, "status": "cancelled"})
            else:
                self._send_json({"error": "task not found or already finished"}, 404)

        else:
            self._send_json({"error": "not found"}, 404)

    # ── GET ──────────────────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802
        path = self.path

        if path == "/status":
            with _lock:
                queue_depth = len(_task_queue)
                current = _current_task["task_id"] if _current_task else None
            uptime = time.time() - START_TIME
            self._send_json({
                "ok": True, "queue_depth": queue_depth,
                "current_task": current, "uptime_seconds": round(uptime),
                "allow_dangerous": ALLOW_DANGEROUS, "version": "0.1.0",
            })

        elif path.startswith("/task/"):
            remainder = path[6:]
            parts = remainder.split("/", 1)
            task_id = parts[0]

            if not re.match(r"^[a-f0-9]{12}$", task_id):
                self._send_json({"error": "invalid task_id"}, 400)
                return

            task = _load_task(task_id)
            if task is None:
                self._send_json({"error": "task not found"}, 404)
                return

            # /task/<id>/files or /task/<id>/files/<path>
            if len(parts) == 2 and parts[1] == "files":
                workdir = _task_workdir(task_id)
                generated = _collect_files(workdir)
                # Return full content for each file
                for f in generated:
                    full = _read_file_content(workdir, f["path"])
                    if full:
                        f["content"] = full
                self._send_json({"task_id": task_id, "files": generated})

            elif len(parts) == 2 and parts[1].startswith("files/"):
                file_path = parts[1][6:]  # strip "files/"
                workdir = _task_workdir(task_id)
                content = _read_file_content(workdir, file_path)
                if content is None:
                    self._send_json({"error": "file not found"}, 404)
                else:
                    self._send_json({"path": file_path, "content": content})

            elif len(parts) == 2 and parts[1] == "wait":
                # Long-poll: block until task completes (up to 120s)
                ev = _completion_events.get(task_id)
                if task["status"] in ("pending", "running") and ev:
                    ev.wait(timeout=120)
                    # Re-read after event
                    task = _load_task(task_id) or task
                self._send_json(self._task_status(task))

            else:
                # /task/<id> — compact status
                self._send_json(self._task_status(task))

        else:
            self._send_json({"error": "not found"}, 404)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    global ALLOW_DANGEROUS, OPENCODE_BIN, TASKS_DIR  # noqa: PLW0603

    args = sys.argv[1:]
    port = DEFAULT_PORT

    if "--port" in args:
        idx = args.index("--port")
        port = int(args[idx + 1])

    if "--opencode" in args:
        idx = args.index("--opencode")
        OPENCODE_BIN = args[idx + 1]
    else:
        OPENCODE_BIN = os.environ.get(
            "DEVSHOP_OPENCODE_BIN",
            subprocess.run(["which", "opencode"], capture_output=True, text=True
                          ).stdout.strip() or "",
        )

    if "--allow-dangerous" in args:
        ALLOW_DANGEROUS = True

    if "--model" in args:
        idx = args.index("--model")
        global MODEL  # noqa: PLW0603
        MODEL = args[idx + 1]
    elif os.environ.get("DEVSHOP_MODEL"):
        MODEL = os.environ.get("DEVSHOP_MODEL", MODEL)

    if "--tasks-dir" in args:
        idx = args.index("--tasks-dir")
        TASKS_DIR = Path(args[idx + 1])
    else:
        TASKS_DIR = Path(os.environ.get("DEVSHOP_TASKS_DIR", "/tmp/devshop_tasks"))

    TASKS_DIR.mkdir(parents=True, exist_ok=True)

    if not OPENCODE_BIN or not Path(OPENCODE_BIN).exists():
        print("ERROR: opencode not found. Install it or set DEVSHOP_OPENCODE_BIN.", file=sys.stderr)
        sys.exit(1)

    if not ALLOW_DANGEROUS:
        print("ERROR: --allow-dangerous is required for non-interactive operation.", file=sys.stderr)
        print("  Without it, opencode prompts for permission on every file write", file=sys.stderr)
        print("  and tasks will hang indefinitely waiting for stdin input.", file=sys.stderr)
        print("  Pass --allow-dangerous to enable (only on trusted networks).", file=sys.stderr)
        sys.exit(1)

    server = ThreadingHTTPServer(("0.0.0.0", port), DevshopHandler)
    print(f"Devshop server listening on http://0.0.0.0:{port}", flush=True)
    print(f"  OPENCODE_BIN={OPENCODE_BIN}", flush=True)
    print(f"  ALLOW_DANGEROUS={ALLOW_DANGEROUS}", flush=True)
    print(f"  TASKS_DIR={TASKS_DIR.resolve()}", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
