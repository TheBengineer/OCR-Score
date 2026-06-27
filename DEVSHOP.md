# Devshop — Low-Bandwidth Remote Agent System

## Overview

Devshop is a two-part system for managing a remote development server over a
very low-bandwidth SSH link.

- **Remote side** (`devshop_server.py`): A single-file, stdlib-only Python HTTP
  server that accepts high-level goals and executes them using the `opencode` CLI.
- **Local side** (`devshop_client.py`): curl commands for the local (no-shell) side.

**Bandwidth profile:**
| Direction | Size | Content |
|-----------|------|---------|
| Request (goal) | ~80 bytes | `{"goal": "build a TODO app"}` |
| Response (success) | ~300-800 bytes | task_id + file list + previews |
| Long-poll wait | 1 round-trip | Blocks until done |
| File download | ~file size | Full file content via `/files` |

## Deploy to Remote

### Steps

1. Copy server to remote:
   ```
   scp backend/devshop_server.py user@remote:/path/to/
   ```

2. SSH in and start:
   ```
   ssh user@remote
   python3 /path/to/devshop_server.py --port 8900 --allow-dangerous
   ```

3. (Optional) systemd service — create `/etc/systemd/system/devshop.service`:
   ```ini
   [Unit]
   Description=Devshop Remote Agent
   After=network.target
   [Service]
   ExecStart=/usr/bin/python3 /path/to/devshop_server.py --port 8900 --allow-dangerous
   Restart=always
   User=youruser
   WorkingDirectory=/tmp
   RestartSec=5
   [Install]
   WantedBy=multi-user.target
   ```
   Then: `sudo systemctl enable devshop && sudo systemctl start devshop`

4. (Optional) SSH tunnel:
   ```
   ssh -L 8900:localhost:8900 user@remote -N
   ```

## Usage (from local side)

### Check health
```
curl http://<remote>:8900/status
```

### Submit a goal
```
curl -X POST http://<remote>:8900/task \
  -H "Content-Type: application/json" \
  -d '{"goal": "build a simple TODO app in Python"}'
```

### Wait for completion (single round-trip)
```
curl http://<remote>:8900/task/<id>/wait --max-time 180
```

### Get generated files
```
curl http://<remote>:8900/task/<id>/files          # all files with content
curl http://<remote>:8900/task/<id>/files/app.py    # single file
```

### Multi-step plan
```
curl -X POST http://<remote>:8900/plan \
  -H "Content-Type: application/json" \
  -d '{"plan": ["step 1", "step 2"]}'
```

### Cancel a task
```
curl -X POST http://<remote>:8900/cancel/<id>
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/status` | Server health, queue depth, uptime |
| POST | `/task` | Submit a goal → task_id (202) |
| GET | `/task/<id>` | Compact status |
| GET | `/task/<id>/wait` | Long-poll: blocks until done (120s) |
| GET | `/task/<id>/files` | List all files with full contents |
| GET | `/task/<id>/files/<path>` | Single file content |
| POST | `/plan` | Multi-step plan (auto-chains) |
| POST | `/cancel/<id>` | Cancel pending/running task |

## Server flags

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8900 | HTTP port |
| `--opencode <path>` | auto-detect | Path to opencode binary |
| `--allow-dangerous` | off | Enable dangerous permissions |
| `--tasks-dir <path>` | /tmp/devshop_tasks | Task storage dir |

Without `--allow-dangerous`, opencode prompts on every file write, causing
tasks to hang. Only enable on trusted networks (SSH tunnel).

## Architecture

```
Local (AI, no shell)         Low-bandwidth HTTP        Remote (full shell)
                         
  POST /task {"goal":"..."}  ─────────────────────►  devshop_server.py
  GET /task/<id>/wait        ◄── {status:"done"}       ┌───────────────┐
  GET /task/<id>/files       ◄── {files:[...]}         │ opencode CLI  │
                                                       │ deepseek-v4   │
                                                       └───────────────┘
```
