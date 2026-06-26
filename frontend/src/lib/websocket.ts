import { useState, useEffect, useRef, useCallback } from "react";
import type { RunStatus } from "./types.ts";

// ── Constants ─────────────────────────────────────────────────────────────

const API_PREFIX = "/api/v1";
const MAX_RETRIES = 3;
const BASE_DELAY_MS = 1_000;
const POLL_INTERVAL_MS = 5_000;

// ── Types ─────────────────────────────────────────────────────────────────

export interface RunProgressState {
  progress: number;
  status: RunStatus | null;
  message: string;
  connected: boolean;
  error: string | null;
}

type WsIncomingMessage =
  | { type: "connected"; run_id: string; run_status: string }
  | { type: "progress"; run_id: string; progress: number; status: string; message?: string }
  | { type: "status_change"; run_id: string; status: string; progress: number }
  | { type: "error"; run_id: string; error: string }
  | { type: "pong" };

// ── Helpers ───────────────────────────────────────────────────────────────

function getWsBaseUrl(): string {
  const configured = import.meta.env.VITE_WS_BASE_URL;
  if (configured) {
    return configured.replace(/\/+$/, "");
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}`;
}

// ── Hook ──────────────────────────────────────────────────────────────────

const INITIAL_STATE: RunProgressState = {
  progress: 0,
  status: null,
  message: "",
  connected: false,
  error: null,
};

export function useRunProgress(runId: string | null): RunProgressState {
  const [state, setState] = useState<RunProgressState>(INITIAL_STATE);

  // Refs to keep values stable across closures
  const runIdRef = useRef(runId);
  runIdRef.current = runId;

  const wsRef = useRef<WebSocket | null>(null);
  const retryCountRef = useRef(0);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Polling fallback ──────────────────────────────────────────────────

  const startPolling = useCallback(() => {
    const id = runIdRef.current;
    if (!id) return;

    stopPolling();
    pollingTimerRef.current = setInterval(async () => {
      const currentId = runIdRef.current;
      if (!currentId) return;

      try {
        const { getRun } = await import("./api.ts");
        const run = await getRun(currentId);
        setState((prev) => ({
          ...prev,
          progress:
            run.status === "completed"
              ? 100
              : run.status === "running"
                ? 50
                : run.status === "failed"
                  ? 0
                  : prev.progress,
          status: run.status as RunStatus,
          message: run.error_message ?? "",
          error: run.error_message,
        }));
      } catch {
        // Silently retry on next interval
      }
    }, POLL_INTERVAL_MS);
  }, []);

  const stopPolling = useCallback(() => {
    if (pollingTimerRef.current) {
      clearInterval(pollingTimerRef.current);
      pollingTimerRef.current = null;
    }
  }, []);

  // ── WebSocket connection ──────────────────────────────────────────────

  const cleanupConnection = useCallback(() => {
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    retryCountRef.current = 0;
  }, []);

  const connect = useCallback(() => {
    const id = runIdRef.current;
    if (!id) return;

    // Clean up any existing connection before creating a new one
    cleanupConnection();
    stopPolling();

    const url = `${getWsBaseUrl()}${API_PREFIX}/ws/runs/${encodeURIComponent(id)}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setState((prev) => ({ ...prev, connected: true, error: null }));
      retryCountRef.current = 0;
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data) as WsIncomingMessage;

        switch (data.type) {
          case "connected":
            setState((prev) => ({
              ...prev,
              status: data.run_status as RunStatus,
              connected: true,
            }));
            break;

          case "progress":
            setState((prev) => ({
              ...prev,
              progress: data.progress,
              status: data.status as RunStatus,
              message: data.message ?? "",
            }));
            break;

          case "status_change":
            setState((prev) => ({
              ...prev,
              progress: data.progress,
              status: data.status as RunStatus,
              message:
                data.status === "completed"
                  ? "Processing complete"
                  : data.status === "failed"
                    ? "Processing failed"
                    : "",
            }));
            break;

          case "error":
            setState((prev) => ({
              ...prev,
              error: data.error,
              status: "failed" as RunStatus,
              progress: 0,
              message: data.error,
            }));
            break;

          // pong — no state update needed
        }
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
      setState((prev) => ({ ...prev, connected: false }));

      const currentId = runIdRef.current;
      if (!currentId) return;

      if (retryCountRef.current < MAX_RETRIES) {
        const delay = BASE_DELAY_MS * 2 ** retryCountRef.current;
        retryCountRef.current++;
        retryTimerRef.current = setTimeout(connect, delay);
      } else {
        // Max retries exceeded — fall back to HTTP polling
        startPolling();
      }
    };

    ws.onerror = () => {
      // onclose will fire after this, triggering reconnect logic
      ws.close();
    };
  }, [cleanupConnection, startPolling, stopPolling]);

  // ── Effect — connect when runId changes ───────────────────────────────

  useEffect(() => {
    if (!runId) {
      setState(INITIAL_STATE);
      return;
    }

    connect();

    return () => {
      cleanupConnection();
      stopPolling();
      setState(INITIAL_STATE);
    };
  }, [runId, connect, cleanupConnection, stopPolling]);

  return state;
}
