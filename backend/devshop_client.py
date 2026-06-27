#!/usr/bin/env python3
"""Devshop client — reference for communicating with the remote Devshop server.

This file is a reference / tutorial for how to talk to the Devshop server
from the local side (where shell access is limited).  Commands here use
only ``curl``, which can be invoked via MCP ``bash`` or ``webfetch`` tools.

Quick reference
---------------

# 1. Check server health
curl http://<remote>:8900/status

# 2. Submit a high-level goal
curl -X POST http://<remote>:8900/task \\
  -H "Content-Type: application/json" \\
  -d '{"goal": "build a simple TODO app in Python with flask"}

# 3. Poll for status (replace <task_id> with returned ID)
curl http://<remote>:8900/task/<task_id>

# 4. Submit a multi-step plan
curl -X POST http://<remote>:8900/plan \\
  -H "Content-Type: application/json" \\
  -d '{"plan": ["step 1 description", "step 2 description"]}'

Expected responses
-----------------

POST /task → 202
  {"task_id": "a1b2c3d4e5f6", "status": "pending", "goal": "..."}

GET /task/<id> → 200 (running)
  {"task_id": "...", "status": "running", "goal": "..."}

GET /task/<id> → 200 (done)
  {"task_id": "...", "status": "done", "goal": "...",
   "summary": {"files": [{"path": "app.py", "size": 123, "preview": "..."}],
               "file_count": 3}}

GET /task/<id> → 200 (failed)
  {"task_id": "...", "status": "failed", "goal": "...",
   "error": "error message here"}

GET /status → 200
  {"ok": true, "queue_depth": 0, "current_task": null,
   "uptime_seconds": 3600, "version": "0.1.0"}

Bandwidth profile
-----------------
- Request: ~80 bytes  ({"goal": "build a TODO app"})
- Success response: ~300-800 bytes  (task_id + file list + previews)
- Status poll: ~150 bytes  (just task_id + status)
- Error response: ~200 bytes  (task_id + error message)

This is designed to work efficiently over very low-bandwidth links.
"""

if __name__ == "__main__":
    import textwrap
    print(textwrap.dedent(__doc__))
