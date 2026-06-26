"""Unit tests for the ``ConnectionManager`` — connect, disconnect, broadcast.

Uses mock WebSocket objects to verify the connection manager's behaviour
without requiring a running FastAPI application.
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.websocket_manager import ConnectionManager

# ── Mock WebSocket ───────────────────────────────────────────────────────────


class MockWebSocket:
    """Fake WebSocket for testing ConnectionManager in isolation.

    Records all sent messages so tests can assert on them.
    Optionally simulates a broken connection for error-path tests.
    """

    def __init__(self, *, broken: bool = False) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self._broken = broken
        self._accepted = False

    async def accept(self) -> None:
        self._accepted = True

    async def send_json(self, data: dict[str, Any]) -> None:
        if self._broken:
            msg = "Connection is broken"
            raise RuntimeError(msg)
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True

    def __repr__(self) -> str:
        return f"<MockWebSocket sent={len(self.sent)} closed={self.closed}>"


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def manager() -> ConnectionManager:
    """Provide a fresh ConnectionManager for each test."""
    return ConnectionManager()


# ── Connect / Disconnect ─────────────────────────────────────────────────────


class TestConnectDisconnect:
    """Verify basic connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect(self, manager: ConnectionManager) -> None:
        """Given a fresh manager, When connecting a WS to a run_id,
        Then active_connections increases and the WS is accepted."""
        ws = MockWebSocket()
        await manager.connect(ws, "run-1")
        assert manager.active_connections() == 1
        assert ws._accepted

    @pytest.mark.asyncio
    async def test_connect_multiple_to_same_run(self, manager: ConnectionManager) -> None:
        """Given a run with one subscriber, When a second connects,
        Then both are tracked under the same run_id."""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        await manager.connect(ws1, "run-1")
        await manager.connect(ws2, "run-1")
        assert manager.active_connections() == 2

    @pytest.mark.asyncio
    async def test_connect_multiple_runs(self, manager: ConnectionManager) -> None:
        """Given distinct run_ids, When connecting WSs,
        Then they are tracked under separate keys."""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        await manager.connect(ws1, "run-a")
        await manager.connect(ws2, "run-b")
        assert manager.active_connections() == 2

    @pytest.mark.asyncio
    async def test_disconnect(self, manager: ConnectionManager) -> None:
        """Given a connected WS, When disconnecting,
        Then active_connections decreases and run_id key may be cleaned up."""
        ws = MockWebSocket()
        await manager.connect(ws, "run-1")
        assert manager.active_connections() == 1

        await manager.disconnect(ws, "run-1")
        assert manager.active_connections() == 0

    @pytest.mark.asyncio
    async def test_disconnect_partial(self, manager: ConnectionManager) -> None:
        """Given two WSs on the same run, When one disconnects,
        Then the other remains."""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        await manager.connect(ws1, "run-1")
        await manager.connect(ws2, "run-1")
        assert manager.active_connections() == 2

        await manager.disconnect(ws1, "run-1")
        assert manager.active_connections() == 1


# ── Broadcast ────────────────────────────────────────────────────────────────


class TestBroadcast:
    """Verify broadcast delivery semantics."""

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_subscribers(self, manager: ConnectionManager) -> None:
        """Given a run with two subscribers, When broadcasting a message,
        Then both receive it."""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        await manager.connect(ws1, "run-1")
        await manager.connect(ws2, "run-1")

        msg = {"type": "progress", "run_id": "run-1", "progress": 50, "status": "running"}
        await manager.broadcast("run-1", msg)

        assert len(ws1.sent) == 1
        assert ws1.sent[0] == msg
        assert len(ws2.sent) == 1
        assert ws2.sent[0] == msg

    @pytest.mark.asyncio
    async def test_broadcast_does_not_cross_streams(self, manager: ConnectionManager) -> None:
        """Given two runs with different subscribers, When broadcasting to one,
        Then only that run's subscribers receive the message."""
        ws_a = MockWebSocket()
        ws_b = MockWebSocket()
        await manager.connect(ws_a, "run-a")
        await manager.connect(ws_b, "run-b")

        msg_a = {"type": "progress", "run_id": "run-a", "progress": 100}
        await manager.broadcast("run-a", msg_a)

        assert len(ws_a.sent) == 1
        assert ws_a.sent[0] == msg_a
        assert len(ws_b.sent) == 0  # no cross-talk

    @pytest.mark.asyncio
    async def test_broadcast_no_subscribers(self, manager: ConnectionManager) -> None:
        """Given no subscribers for a run_id, When broadcasting,
        Then no error is raised."""
        msg = {"type": "progress", "run_id": "ghost-run", "progress": 0}
        # Should not raise
        await manager.broadcast("ghost-run", msg)

    @pytest.mark.asyncio
    async def test_broadcast_with_broken_connection(self, manager: ConnectionManager) -> None:
        """Given a subscriber whose connection is broken, When broadcasting,
        Then the dead connection is removed and the others still receive."""
        ws_good = MockWebSocket()
        ws_broken = MockWebSocket(broken=True)
        await manager.connect(ws_good, "run-1")
        await manager.connect(ws_broken, "run-1")
        assert manager.active_connections() == 2

        msg = {"type": "progress", "run_id": "run-1", "progress": 75}
        await manager.broadcast("run-1", msg)

        # Good WS received the message
        assert len(ws_good.sent) == 1
        # Broken WS was removed
        assert manager.active_connections() == 1


# ── Convenience methods ──────────────────────────────────────────────────────


class TestConvenienceMethods:
    """Verify broadcast_progress, broadcast_status_change, broadcast_error."""

    @pytest.mark.asyncio
    async def test_broadcast_progress(self, manager: ConnectionManager) -> None:
        """Given a subscriber, When broadcast_progress is called,
        Then the subscriber receives a properly formatted progress message."""
        ws = MockWebSocket()
        await manager.connect(ws, "run-1")

        await manager.broadcast_progress("run-1", 50, "running", "Processing page 5/10")

        assert len(ws.sent) == 1
        msg = ws.sent[0]
        assert msg["type"] == "progress"
        assert msg["run_id"] == "run-1"
        assert msg["progress"] == 50
        assert msg["status"] == "running"
        assert msg["message"] == "Processing page 5/10"

    @pytest.mark.asyncio
    async def test_broadcast_progress_default_message(self, manager: ConnectionManager) -> None:
        """Given a subscriber, When broadcast_progress is called without message,
        Then the message field defaults to empty string."""
        ws = MockWebSocket()
        await manager.connect(ws, "run-1")

        await manager.broadcast_progress("run-1", 100, "completed")

        msg = ws.sent[0]
        assert msg["type"] == "progress"
        assert msg["message"] == ""

    @pytest.mark.asyncio
    async def test_broadcast_status_change(self, manager: ConnectionManager) -> None:
        """Given a subscriber, When broadcast_status_change is called,
        Then the subscriber receives a properly formatted status_change message."""
        ws = MockWebSocket()
        await manager.connect(ws, "run-1")

        await manager.broadcast_status_change("run-1", "completed", 100)

        assert len(ws.sent) == 1
        msg = ws.sent[0]
        assert msg["type"] == "status_change"
        assert msg["run_id"] == "run-1"
        assert msg["status"] == "completed"
        assert msg["progress"] == 100

    @pytest.mark.asyncio
    async def test_broadcast_error(self, manager: ConnectionManager) -> None:
        """Given a subscriber, When broadcast_error is called,
        Then the subscriber receives a properly formatted error message."""
        ws = MockWebSocket()
        await manager.connect(ws, "run-1")

        await manager.broadcast_error("run-1", "Engine timed out")

        assert len(ws.sent) == 1
        msg = ws.sent[0]
        assert msg["type"] == "error"
        assert msg["run_id"] == "run-1"
        assert msg["error"] == "Engine timed out"


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Verify manager handles boundary conditions."""

    def test_active_connections_empty(self, manager: ConnectionManager) -> None:
        """Given a fresh manager, When querying active connections,
        Then the count is zero."""
        assert manager.active_connections() == 0

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent(self, manager: ConnectionManager) -> None:
        """Given a run_id that has no connections, When disconnecting,
        Then no error is raised."""
        ws = MockWebSocket()
        await manager.disconnect(ws, "nonexistent-run")

    @pytest.mark.asyncio
    async def test_disconnect_removes_subscriber(self, manager: ConnectionManager) -> None:
        """Given a connected subscriber, When disconnecting,
        Then the subscriber stops receiving messages."""
        ws = MockWebSocket()
        await manager.connect(ws, "run-1")
        await manager.disconnect(ws, "run-1")

        msg = {"type": "progress", "run_id": "run-1", "progress": 100}
        await manager.broadcast("run-1", msg)

        assert len(ws.sent) == 0  # no longer subscribed
