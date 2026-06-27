#!/usr/bin/env python3
# Devshop v0.1.0
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
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


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


# ── Self-editing helpers ───────────────────────────────────────────────────────


_SELF_PATH = Path(__file__).resolve()


def _self_edit(change_desc: str) -> dict[str, Any]:
    """Modify this server's source code using opencode to implement *change_desc*.

    The LLM reads the current source, plans the modification, applies it,
    and validates syntax.  On success the file is overwritten and the server
    should be restarted.
    """
    source = _SELF_PATH.read_text()
    plan_dir = TASKS_DIR / "_self_edit"
    plan_dir.mkdir(parents=True, exist_ok=True)

    prompt = (
        f"Modify the Python file at {_SELF_PATH} to implement this change:\n\n"
        f"{change_desc}\n\n"
        f"RULES:\n"
        f"1. Output ONLY the COMPLETE modified file content.\n"
        f"2. Preserve ALL existing functionality.\n"
        f"3. Use the same style, imports, and conventions.\n"
        f"4. The file is a standalone stdlib-only HTTP server.\n"
        f"5. Write the result to {plan_dir / 'devshop_server.py'}.\n"
        f"6. Ensure `python3 -c \"import ast; ast.parse(open('{plan_dir / 'devshop_server.py'}').read())\"` passes."
    )

    result = subprocess.run(
        [OPENCODE_BIN, "run", prompt, "--print-logs",
         "--dir", str(plan_dir), "--model", MODEL,
         "--dangerously-skip-permissions"],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )

    generated_path = plan_dir / "devshop_server.py"
    if not generated_path.exists():
        return {"success": False, "error": "LLM did not generate a modified file",
                "log": (result.stdout or "")[-500:]}

    modified = generated_path.read_text()

    # Validate syntax
    try:
        compile(modified, str(_SELF_PATH), "exec")
    except SyntaxError as e:
        return {"success": False, "error": f"Syntax error in generated code: {e}"}

    # Backup and apply
    backup = _SELF_PATH.with_suffix(".py.bak")
    _SELF_PATH.rename(backup)
    _SELF_PATH.write_text(modified)

    return {
        "success": True,
        "backup": str(backup),
        "restart_required": True,
        "bytes_changed": len(modified) - len(source),
    }


# ── Project management (oversight loops) ───────────────────────────────────────

_PROJECTS: dict[str, dict[str, Any]] = {}
_PROJECT_LOCK = threading.Lock()


def _llm_plan(goal: str) -> list[str]:
    """Ask the LLM to break *goal* into concrete, actionable steps."""
    plan_dir = TASKS_DIR / "_planner"
    plan_dir.mkdir(parents=True, exist_ok=True)
    prompt = (
        f"Break this goal into 3-5 concrete, sequential steps:\n\n"
        f"{goal}\n\n"
        f"Output each step on a separate line starting with '- '. "
        f"Each step must be a specific, actionable instruction that an AI agent "
        f"can execute to generate code. No numbering, no commentary."
    )
    result = subprocess.run(
        [OPENCODE_BIN, "run", prompt, "--print-logs", "--dir", str(plan_dir),
         "--model", MODEL, "--dangerously-skip-permissions"],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    lines = [
        ln.strip().lstrip("- ").strip()
        for ln in (result.stdout or "").splitlines()
        if ln.strip().startswith("-")
    ]
    return lines[:8] if lines else [goal]  # fallback: single step


def _llm_review(goal: str, step_label: str, output_summary: dict[str, Any]) -> tuple[bool, str]:
    """Ask the LLM to review whether *step_label* was successfully completed.

    Returns (passed: bool, feedback: str).
    """
    files_summary = json.dumps(output_summary.get("files", []), indent=2)
    review_dir = TASKS_DIR / "_reviewer"
    review_dir.mkdir(parents=True, exist_ok=True)
    prompt = (
        f"Goal: {goal}\n"
        f"Step completed: {step_label}\n"
        f"Generated files:\n{files_summary[:2000]}\n\n"
        f"Did this step successfully achieve its objective? "
        f"Answer with exactly one line: PASS or FAIL. "
        f"Then on the next line give 1-2 sentences of feedback."
    )
    result = subprocess.run(
        [OPENCODE_BIN, "run", prompt, "--print-logs", "--dir", str(review_dir),
         "--model", MODEL, "--dangerously-skip-permissions"],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    passed = "PASS" in output.upper() and "FAIL" not in output.upper()
    return passed, output[:500]


def create_project(goal: str) -> dict[str, Any]:
    """Create a project: plan → execute each step → review → loop on failure."""
    project_id = _next_task_id()

    project: dict[str, Any] = {
        "project_id": project_id,
        "goal": goal,
        "status": "planning",
        "steps": [],
        "current_step": 0,
        "total_steps": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_project(project)

    # Run planning + execution in a background thread (LLM calls take time)
    threading.Thread(target=_run_project_loop, args=(project_id,), daemon=True).start()

    return {"project_id": project_id, "status": "planning", "goal": goal}


def _project_path(project_id: str) -> Path:
    return TASKS_DIR / f"_project_{project_id}.json"


def _save_project(project: dict[str, Any]) -> None:
    _project_path(project["project_id"]).write_text(json.dumps(project, indent=2, default=str))


def _load_project(project_id: str) -> dict[str, Any] | None:
    path = _project_path(project_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _run_project_loop(project_id: str) -> None:
    """Background: plan, execute project steps with oversight review loop."""
    project = _load_project(project_id)
    if not project:
        return

    # Step 1: Plan — break goal into steps using LLM
    steps_raw = _llm_plan(project["goal"])
    project["steps"] = [
        {"label": s, "status": "pending", "attempts": 0, "max_attempts": 3,
         "task_id": None, "review": None}
        for s in steps_raw
    ]
    project["total_steps"] = len(project["steps"])
    project["status"] = "running"
    _save_project(project)

    for idx, step in enumerate(project["steps"]):
        project["current_step"] = idx
        step["status"] = "running"
        _save_project(project)

        for attempt in range(step["max_attempts"]):
            step["attempts"] = attempt + 1
            # Execute the step as a task
            task_result = submit_task(step["label"])
            step["task_id"] = task_result.get("task_id")

            # Wait for task completion (poll with timeout)
            task_data = None
            for _ in range(120):
                task_data = _load_task(step["task_id"]) if step.get("task_id") else None
                if task_data and task_data["status"] in ("done", "failed"):
                    break
                time.sleep(5)

            # Review
            task_data = _load_task(step["task_id"])
            if task_data and task_data["status"] == "done":
                passed, feedback = _llm_review(
                    project["goal"], step["label"],
                    task_data.get("summary", {}),
                )
                step["review"] = {"passed": passed, "feedback": feedback}
                if passed:
                    step["status"] = "done"
                    _save_project(project)
                    break
                else:
                    step["status"] = "review_failed"
                    _save_project(project)
                    if attempt < step["max_attempts"] - 1:
                        # Re-plan the step based on feedback
                        revised_goal = f"{step['label']} (REVISION: {feedback[:200]})"
                        step["label"] = revised_goal
            else:
                step["status"] = "failed"
                step["error"] = task_data.get("error", "unknown") if task_data else "no task data"
                _save_project(project)
                break
        else:
            # Exhausted all attempts
            step["status"] = "failed"
            step["error"] = f"Failed after {step['max_attempts']} attempts"
            _save_project(project)
            break

    # Final status
    all_done = all(s["status"] == "done" for s in project["steps"])
    project["status"] = "completed" if all_done else "failed"
    _save_project(project)


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

        elif path == "/self/edit":
            if not body or "change" not in body:
                self._send_json({"error": "missing 'change' in request body"}, 400)
                return
            change = str(body["change"]).strip()
            if not change:
                self._send_json({"error": "'change' must be non-empty"}, 400)
                return
            result = _self_edit(change)
            if result.get("success"):
                # Delay restart so the HTTP response is sent first
                threading.Thread(
                    target=lambda: (time.sleep(1), os._exit(0)),
                    daemon=True,
                ).start()
            self._send_json(result, 200 if result.get("success") else 500)

        elif path == "/project":
            if not body or "goal" not in body:
                self._send_json({"error": "missing 'goal' in request body"}, 400)
                return
            goal = str(body["goal"]).strip()
            if not goal:
                self._send_json({"error": "'goal' must be non-empty"}, 400)
                return
            result = create_project(goal)
            self._send_json(result, 202)

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

        elif path == "/capabilities":
            self._send_json({
                "endpoints": [
                    {"method": "GET", "path": "/status", "desc": "Server health and queue depth"},
                    {"method": "GET", "path": "/capabilities", "desc": "List all available endpoints"},
                    {"method": "GET", "path": "/task/<id>", "desc": "Compact task status"},
                    {"method": "GET", "path": "/task/<id>/wait", "desc": "Long-poll task completion"},
                    {"method": "GET", "path": "/task/<id>/files", "desc": "List generated files with content"},
                    {"method": "GET", "path": "/task/<id>/files/<path>", "desc": "Single file content"},
                    {"method": "GET", "path": "/project/<id>", "desc": "Project status with step breakdown"},
                    {"method": "POST", "path": "/task", "desc": "Submit a goal"},
                    {"method": "POST", "path": "/plan", "desc": "Multi-step plan"},
                    {"method": "POST", "path": "/cancel/<id>", "desc": "Cancel a task"},
                    {"method": "POST", "path": "/self/edit", "desc": "Self-modify server code"},
                    {"method": "POST", "path": "/project", "desc": "Create project with oversight loop"},
                ],
                "model": MODEL,
                "allow_dangerous": ALLOW_DANGEROUS,
                "version": "0.1.0",
            })

        elif path == "/project/list":
            projects = sorted(TASKS_DIR.glob("_project_*.json"))
            ids = [p.stem.replace("_project_", "") for p in projects]
            self._send_json({"projects": ids})

        elif path.startswith("/project/"):
            project_id = path[9:]
            project = _load_project(project_id)
            if project is None:
                self._send_json({"error": "project not found"}, 404)
                return
            # Return compact view (omit large fields)
            view = {
                "project_id": project["project_id"],
                "goal": project["goal"],
                "status": project["status"],
                "current_step": project["current_step"],
                "total_steps": project["total_steps"],
                "steps": [
                    {
                        "label": s["label"],
                        "status": s["status"],
                        "attempts": s["attempts"],
                        "task_id": s["task_id"],
                        "review_passed": s.get("review", {}).get("passed") if s.get("review") else None,
                        "review_feedback": (s.get("review", {}).get("feedback", "")[:200]
                                            if s.get("review") else None),
                    }
                    for s in project["steps"]
                ],
            }
            self._send_json(view)
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
