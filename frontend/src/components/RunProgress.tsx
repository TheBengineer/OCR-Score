import { useRunProgress, type RunProgressState } from "../lib/websocket.ts";
import type { RunStatus } from "../lib/types.ts";

// ── Status config ─────────────────────────────────────────────────────────

interface StatusConfig {
  label: string;
  bgColor: string;
  textColor: string;
  dotColor: string;
}

const STATUS_CONFIG: Record<RunStatus, StatusConfig> = {
  pending: {
    label: "Pending",
    bgColor: "bg-surface-100",
    textColor: "text-surface-600",
    dotColor: "bg-surface-400",
  },
  queued: {
    label: "Queued",
    bgColor: "bg-primary-100",
    textColor: "text-primary-700",
    dotColor: "bg-primary-500",
  },
  running: {
    label: "Running",
    bgColor: "bg-blue-100",
    textColor: "text-blue-700",
    dotColor: "bg-blue-500",
  },
  completed: {
    label: "Completed",
    bgColor: "bg-emerald-100",
    textColor: "text-emerald-700",
    dotColor: "bg-emerald-500",
  },
  failed: {
    label: "Failed",
    bgColor: "bg-red-100",
    textColor: "text-red-700",
    dotColor: "bg-red-500",
  },
  cancelled: {
    label: "Cancelled",
    bgColor: "bg-amber-100",
    textColor: "text-amber-700",
    dotColor: "bg-amber-500",
  },
};

function defaultConfig(status: string | null): StatusConfig {
  if (status && status in STATUS_CONFIG) {
    return STATUS_CONFIG[status as RunStatus];
  }
  return {
    label: status ?? "Unknown",
    bgColor: "bg-surface-100",
    textColor: "text-surface-600",
    dotColor: "bg-surface-400",
  };
}

// ── Status badge ──────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: RunStatus | null }) {
  const cfg = defaultConfig(status);

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ${cfg.bgColor} ${cfg.textColor}`}
    >
      <span className={`inline-block size-1.5 rounded-full ${cfg.dotColor}`} />
      {cfg.label}
    </span>
  );
}

// ── Connection indicator ──────────────────────────────────────────────────

function ConnectionIndicator({ connected }: { connected: boolean }) {
  return (
    <span
      className={`inline-flex items-center gap-1 text-xs ${
        connected ? "text-emerald-600" : "text-surface-400"
      }`}
      title={connected ? "Connected" : "Disconnected (polling)"}
    >
      <span
        className={`inline-block size-1.5 rounded-full ${
          connected ? "bg-emerald-500" : "bg-surface-300"
        }`}
      />
      {connected ? "Live" : "Polling"}
    </span>
  );
}

// ── Progress bar ──────────────────────────────────────────────────────────

function ProgressBar({ progress, status }: { progress: number; status: RunStatus | null }) {
  const clamped = Math.max(0, Math.min(100, progress));

  const barColor =
    status === "failed" || status === "cancelled"
      ? "bg-red-500"
      : status === "completed"
        ? "bg-emerald-500"
        : "bg-primary-500";

  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-surface-200">
      <div
        className={`h-full rounded-full transition-all duration-300 ease-out ${barColor}`}
        style={{ width: `${clamped}%` }}
        role="progressbar"
        aria-valuenow={clamped}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Progress: ${clamped}%`}
      />
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────

interface RunProgressProps {
  runId: string | null;
  compact?: boolean;
}

export default function RunProgress({ runId, compact = false }: RunProgressProps) {
  const { progress, status, message, connected, error }: RunProgressState =
    useRunProgress(runId);

  if (!runId) {
    return null;
  }

  if (compact) {
    return (
      <div className="flex items-center gap-3">
        <div className="flex-1">
          <ProgressBar progress={progress} status={status} />
        </div>
        <StatusBadge status={status} />
        <ConnectionIndicator connected={connected} />
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-surface-200 bg-white p-4 shadow-sm">
      {/* Header */}
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-surface-700">OCR Progress</span>
          <StatusBadge status={status} />
        </div>
        <ConnectionIndicator connected={connected} />
      </div>

      {/* Progress bar */}
      <div className="mb-2">
        <ProgressBar progress={progress} status={status} />
      </div>

      {/* Percentage */}
      <div className="mb-1 flex items-center justify-between text-xs text-surface-500">
        <span>{progress}%</span>
        <span>
          {status === "completed"
            ? "Complete"
            : status === "failed"
              ? "Failed"
              : status === "running"
                ? "In progress"
                : ""}
        </span>
      </div>

      {/* Message */}
      {(message || error) && (
        <p
          className={`mt-2 text-xs ${
            error ? "text-red-600" : "text-surface-500"
          }`}
        >
          {error || message}
        </p>
      )}
    </div>
  );
}
