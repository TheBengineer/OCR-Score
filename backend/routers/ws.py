"""WebSocket endpoint for real-time OCR run progress streaming.

Provides a single endpoint:

- ``WS /api/v1/ws/runs/{run_id}`` — Subscribe to progress updates for a run.

Messages
--------
All messages are JSON-encoded dicts.

**Server → Client:**

- ``{"type": "connected", "run_id": "...", "run_status": "..."}``
- ``{"type": "progress", "run_id": "...", "progress": 50, "status": "running", "message": "Processing page 5/10"}``
- ``{"type": "status_change", "run_id": "...", "status": "completed", "progress": 100}``
- ``{"type": "error", "run_id": "...", "error": "Engine timed out"}``

**Client → Server:**

- ``"ping"`` (text) — Keep-alive; server responds with ``{"type": "pong"}``.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from backend.database import async_session_factory
from backend.models.run import OCRRun
from backend.websocket_manager import manager

ws_router = APIRouter()


async def _get_run_status(run_id: str) -> str | None:
    """Fetch the current status of a run from the database.

    Returns the status string or ``None`` if the run does not exist or an
    error occurs.
    """
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        return None

    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(OCRRun.status).where(OCRRun.id == run_uuid),
            )
            status = result.scalar_one_or_none()
            return status.value if status is not None else None  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        return None


@ws_router.websocket("/api/v1/ws/runs/{run_id}")
async def websocket_run_progress(websocket: WebSocket, run_id: str) -> None:
    """WebSocket endpoint for real-time run progress.

    On connect: subscribes to *run_id*, sends current run status.
    On disconnect: cleans up subscription.
    While connected: responds to ``ping`` with ``pong``.
    """
    await manager.connect(websocket, run_id)
    try:
        # Send current status immediately on connect
        status = await _get_run_status(run_id)
        await websocket.send_json({
            "type": "connected",
            "run_id": run_id,
            "run_status": status or "unknown",
        })

        # Keep connection alive — handle incoming ping/pong
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket, run_id)
