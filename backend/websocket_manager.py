"""WebSocket connection manager for OCRScore real-time progress streaming.

Manages per-run_id WebSocket connections and provides broadcast methods
for progress updates, status changes, and error messages. Thread-safe via
:class:`asyncio.Lock`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    """Manages WebSocket connections grouped by run_id.

    Stores connections as ``{run_id: [websocket1, websocket2, ...]}``.
    All mutations to the connection dict are protected by an :class:`asyncio.Lock`
    to prevent race conditions when multiple tasks broadcast concurrently.
    """

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, run_id: str) -> None:
        """Accept a new WebSocket connection and subscribe it to *run_id*.

        Args:
            websocket: The incoming WebSocket connection.
            run_id: The run identifier to subscribe to.
        """
        await websocket.accept()
        async with self._lock:
            if run_id not in self._connections:
                self._connections[run_id] = []
            self._connections[run_id].append(websocket)

    async def disconnect(self, websocket: WebSocket, run_id: str) -> None:
        """Remove a WebSocket connection from the *run_id* subscription list.

        If the list becomes empty the run_id key is removed entirely.

        Args:
            websocket: The WebSocket connection to remove.
            run_id: The run identifier the connection was subscribed to.
        """
        async with self._lock:
            if run_id in self._connections:
                self._connections[run_id] = [
                    ws for ws in self._connections[run_id] if ws is not websocket
                ]
                if not self._connections[run_id]:
                    del self._connections[run_id]

    async def broadcast(self, run_id: str, message: dict[str, Any]) -> None:
        """Send a JSON message to every subscriber of *run_id*.

        Dead connections (disconnected clients) are silently removed during
        the broadcast pass.

        Args:
            run_id: Target run identifier.
            message: Serializable dict to send as JSON.
        """
        async with self._lock:
            connections = list(self._connections.get(run_id, []))

        dead: list[WebSocket] = []
        for ws in connections:
            try:
                await ws.send_json(message)
            except (WebSocketDisconnect, RuntimeError):
                dead.append(ws)

        if dead:
            async with self._lock:
                if run_id in self._connections:
                    self._connections[run_id] = [
                        ws for ws in self._connections[run_id] if ws not in dead
                    ]
                    if not self._connections[run_id]:
                        del self._connections[run_id]

    async def broadcast_progress(
        self,
        run_id: str,
        progress: int,
        status: str,
        message: str = "",
    ) -> None:
        """Convenience: broadcast a progress update message.

        Args:
            run_id: Target run identifier.
            progress: Integer 0-100 indicating completion percentage.
            status: One of ``pending``, ``queued``, ``running``, ``completed``,
                ``failed``, ``cancelled``.
            message: Human-readable progress description.
        """
        await self.broadcast(
            run_id,
            {
                "type": "progress",
                "run_id": run_id,
                "progress": progress,
                "status": status,
                "message": message,
            },
        )

    async def broadcast_status_change(
        self,
        run_id: str,
        status: str,
        progress: int = 0,
    ) -> None:
        """Convenience: broadcast a status-change notification.

        Args:
            run_id: Target run identifier.
            status: The new run status.
            progress: Current progress percentage (default 0).
        """
        await self.broadcast(
            run_id,
            {
                "type": "status_change",
                "run_id": run_id,
                "status": status,
                "progress": progress,
            },
        )

    async def broadcast_error(self, run_id: str, error: str) -> None:
        """Convenience: broadcast an error message.

        Args:
            run_id: Target run identifier.
            error: Human-readable error description.
        """
        await self.broadcast(
            run_id,
            {
                "type": "error",
                "run_id": run_id,
                "error": error,
            },
        )

    def active_connections(self) -> int:
        """Return the total number of active WebSocket connections."""
        return sum(len(conns) for conns in self._connections.values())


# Module-level singleton — imported by both the WS router and RunOrchestrator.
manager = ConnectionManager()
