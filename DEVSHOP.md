# Devshop — Low-Bandwidth Remote Agent System

## Overview

Devshop is a two-part system designed for managing a remote development server
over a very low-bandwidth SSH link.

- **Remote side** (`devshop_server.py`): A single-file, stdlib-only Python HTTP
  server that accepts high-level goals, executes them using the `opencode` CLI
  (which leverages the deepseek-v4-flash LLM), and returns compact status updates.
- **Local side** (`devshop_client.py`): A reference showing how to communicate
  with the remote server using `curl` commands (compatible with MCP `bash`/`webfetch`).

**Bandwidth profile:**
| Direction | Size | Content |
|-----------|------|---------|
| Request (goal) | ~80 bytes | `{"goal": "build a TODO app"}` |
| Response (success) | ~300-800 bytes | task_id + file list + previews |
| Status poll | ~150 bytes | task_id + status |
| Error response | ~200 bytes | task_id + error message |

---

## Deploy to Remote

### Prerequisites (remote)
- Python 3.10+
- `opencode` CLI installed (`which opencode`)
- SSH access

### Steps

**1. Copy the server file to the remote machine**
```bash
# From your local machine:
scp backend/devshop_server.py user@remote:/path/to/
```

**2. SSH into the remote and start the server**
```bash
ssh user@remote
python3 /path/to/devshop_server.py --port 8900
```

You should see:
```
Devshop server listening on http://0.0.0.0:8900
  OPENCODE_BIN=/home/user/.opencode/bin/opencode
  TASKS_DIR=/tmp/devshop_tasks
```

**3. (Optional) Run as a service**

Create a systemd service file `/etc/systemd/system/devshop.service`:
```ini
[Unit]
Description=Devshop Remote Agent
After=network.target

[Service]
ExecStart=/usr/bin/python3 /path/to/devshop_server.py --port 8900
Restart=always
User=youruser
Environment=DEVSHOP_OPENCODE_BIN=/home/youruser/.opencode/bin/opencode
Environment=DEVSHOP_TASKS_DIR=/tmp/devshop_tasks

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable devshop
sudo systemctl start devshop
```

**4. (Optional) SSH tunnel for local access**

If the remote port isn't directly accessible, create an SSH tunnel:
```bash
ssh -L 8900:localhost:8900 user@remote -N
```

---

## Usage (from local side)

Once the server is running on the remote, communicate with it using `curl`
(or MCP `webfetch`).

### 1. Check server health
```bash
curl http://<remote>:8900/status
```
→ `{"ok": true, "queue_depth": 0, "current_task": null, "uptime_seconds": 123}`

### 2. Submit a high-level goal
```bash
curl -X POST http://<remote>:8900/task \
  -H "Content-Type: application/json" \
  -d '{"goal": "build a simple TODO app in Python with flask"}'
```
→ `{"task_id": "a1b2c3d4e5f6", "status": "pending", "goal": "..."}` (202)

### 3. Poll for task status
```bash
curl http://<remote>:8900/task/a1b2c3d4e5f6
```
When running: `{"task_id": "...", "status": "running", "goal": "..."}`
When done: `{"task_id": "...", "status": "done", "summary": {"files": [...], "file_count": 3}}`
On failure: `{"task_id": "...", "status": "failed", "error": "error message"}`

### 4. Submit a multi-step plan
```bash
curl -X POST http://<remote>:8900/plan \
  -H "Content-Type: application/json" \
  -d '{"plan": ["step 1: build backend API", "step 2: build frontend"]}'
```
→ `{"task_id": "...", "status": "pending", "goal": "step 1: ...", "plan_steps": 2}`

---

## Communication Protocol

The entire protocol is JSON over HTTP. No special headers, no authentication
(assumes the port is not exposed to the public internet — use SSH tunneling).

| Method | Path | Body | Response |
|--------|------|------|----------|
| GET | `/status` | — | Server health |
| POST | `/task` | `{"goal": "..."}` | Task created (202) |
| GET | `/task/<id>` | — | Task status + result |
| POST | `/plan` | `{"plan": ["...", "..."]}` | First step enqueued (202) |

---

## Architecture

```
┌──────────────────────┐        Low-bandwidth link         ┌──────────────────────┐
│   Local side (AI)     │  ◄─────── JSON over HTTP ──────►  │   Remote server       │
│                       │                                    │                      │
│  - No shell access    │    POST /task {"goal": "..."}      │  devshop_server.py   │
│  - MCP tools only     │    ───────────────────────────►   │  (stdlib only)       │
│  - curl / webfetch    │                                    │                      │
│                       │    GET /task/<id> {"status":"..."} │  ┌────────────────┐  │
│                       │    ◄────────────────────────────   │  │  opencode CLI   │  │
│                       │                                    │  │  (deepseek-v4)  │  │
│                       │                                    │  └────────────────┘  │
└──────────────────────┘                                    └──────────────────────┘
```

The remote server runs tasks sequentially using the `opencode` CLI, which
connects to the deepseek-v4-flash LLM. Each task:
1. Creates an isolated git repo in `/tmp/devshop_tasks/<task_id>/`
2. Runs `opencode run <goal> --model opencode-go/deepseek-v4-flash`
3. Collects generated files (excluding opencode runtime files)
4. Returns a compact file summary with previews

---

## Troubleshooting

**opencode not found**: Set `DEVSHOP_OPENCODE_BIN` env var:
```bash
export DEVSHOP_OPENCODE_BIN=/custom/path/opencode
```

**Port in use**: Change the port:
```bash
python3 devshop_server.py --port 8901
```

**Tasks directory filling up**: Tasks are stored in `/tmp/devshop_tasks/`.
Add a cron job to clean up old tasks:
```bash
find /tmp/devshop_tasks -maxdepth 1 -name "*.json" -mtime +1 -delete
```
