import { useEffect, useState } from "react";
import {
  Download,
  FileSpreadsheet,
  FileJson,
  FileText,
  Loader2,
  AlertCircle,
  CheckSquare,
  Square,
  Search,
} from "lucide-react";
import { listRuns, getReportDownloadUrl } from "../lib/api.ts";
import type { Run } from "../lib/types.ts";

// ── Format definitions ────────────────────────────────────────────────────

interface ExportFormat {
  key: "csv" | "json" | "html";
  label: string;
  icon: typeof FileSpreadsheet;
  color: string;
  description: string;
}

const FORMATS: ExportFormat[] = [
  {
    key: "csv",
    label: "CSV",
    icon: FileSpreadsheet,
    color:
      "border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100",
    description: "Tabular data, one row per run per page",
  },
  {
    key: "json",
    label: "JSON",
    icon: FileJson,
    color: "border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100",
    description: "Full structured data dump with engine summaries",
  },
  {
    key: "html",
    label: "HTML",
    icon: FileText,
    color: "border-purple-200 bg-purple-50 text-purple-700 hover:bg-purple-100",
    description: "Self-contained report with inline CSS",
  },
];

// ── Progress bar for downloads ─────────────────────────────────────────────

function DownloadProgress({ format }: { format: string }) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-primary-200 bg-primary-50 px-4 py-3">
      <Loader2 className="h-5 w-5 animate-spin text-primary-500" />
      <span className="text-sm font-medium text-primary-700">
        Downloading {format.toUpperCase()} report…
      </span>
    </div>
  );
}

// ── Run row with checkbox ──────────────────────────────────────────────────

function RunCheckRow({
  run,
  selected,
  onToggle,
}: {
  run: Run;
  selected: boolean;
  onToggle: () => void;
}) {
  const statusColors: Record<string, string> = {
    completed: "text-emerald-600",
    failed: "text-red-600",
    running: "text-blue-600",
    pending: "text-amber-600",
    queued: "text-purple-600",
    cancelled: "text-surface-400",
  };
  const color = statusColors[run.status] ?? "text-surface-500";

  return (
    <div
      className={`flex cursor-pointer items-center gap-3 rounded-lg border bg-white px-4 py-3 transition-colors hover:bg-surface-50 ${
        selected ? "border-primary-300 ring-1 ring-primary-200" : "border-surface-200"
      }`}
      onClick={onToggle}
    >
      <button
        type="button"
        className="shrink-0 text-surface-400 hover:text-primary-600"
        onClick={(e) => {
          e.stopPropagation();
          onToggle();
        }}
      >
        {selected ? (
          <CheckSquare className="h-5 w-5 text-primary-600" />
        ) : (
          <Square className="h-5 w-5" />
        )}
      </button>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-surface-900">
          Run {run.id.slice(0, 8)}…
        </p>
        <p className="text-xs text-surface-400">
          {new Date(run.created_at).toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
          })}
        </p>
      </div>
      <span className={`shrink-0 text-xs font-medium ${color}`}>
        {run.status}
      </span>
    </div>
  );
}

// ── Main Reports Page ─────────────────────────────────────────────────────

export default function Reports() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);

  // Fetch runs on mount
  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const result = await listRuns({ limit: 100 });
        if (!cancelled) setRuns(result.items);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load runs");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  // Toggle a single run
  function toggleRun(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }

  // Select / deselect all completed runs
  function toggleAllCompleted() {
    const completedIds = runs
      .filter((r) => r.status === "completed")
      .map((r) => r.id);
    const allSelected = completedIds.every((id) => selectedIds.has(id));

    setSelectedIds((prev) => {
      const next = new Set(prev);
      for (const id of completedIds) {
        if (allSelected) {
          next.delete(id);
        } else {
          next.add(id);
        }
      }
      return next;
    });
  }

  // Trigger a download
  function handleDownload(format: "csv" | "json" | "html") {
    setDownloading(format);
    const selectedArray =
      selectedIds.size > 0 ? Array.from(selectedIds) : undefined;
    const url = getReportDownloadUrl(format, selectedArray);

    // Trigger download via hidden anchor
    const a = document.createElement("a");
    a.href = url;
    a.download = `ocrscore_report.${format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    // Reset downloading state after a brief delay
    setTimeout(() => setDownloading(null), 1500);
  }

  // ── Render states ───────────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="mx-auto flex max-w-3xl items-center justify-center py-24">
        <Loader2 className="h-8 w-8 animate-spin text-primary-500" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="mx-auto max-w-3xl">
        <div className="flex items-center gap-3 rounded-xl border border-red-200 bg-red-50 p-4">
          <AlertCircle className="h-5 w-5 shrink-0 text-red-500" />
          <p className="text-sm text-red-700">{error}</p>
        </div>
      </div>
    );
  }

  const completedRuns = runs.filter((r) => r.status === "completed");
  const allCompletedSelected = completedRuns.every((r) => selectedIds.has(r.id));

  return (
    <div className="mx-auto max-w-3xl">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-surface-900">Reports</h1>
        <p className="mt-1 text-sm text-surface-500">
          Aggregate rankings, score breakdowns, and exportable evaluation
          reports.
        </p>
      </div>

      {/* Export section */}
      <div className="mb-8 rounded-xl border border-surface-200 bg-white p-6 shadow-sm">
        <h2 className="mb-1 text-sm font-semibold text-surface-900">
          Export Data
        </h2>
        <p className="mb-4 text-xs text-surface-400">
          Select runs below or export all completed runs.
        </p>

        {/* Download progress */}
        {downloading && <DownloadProgress format={downloading} />}

        {/* Format buttons */}
        <div className="grid gap-3 sm:grid-cols-3">
          {FORMATS.map((fmt) => (
            <button
              key={fmt.key}
              type="button"
              disabled={downloading !== null}
              onClick={() => handleDownload(fmt.key)}
              className={`flex items-center gap-3 rounded-lg border px-4 py-3 text-left text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${fmt.color}`}
            >
              <fmt.icon className="h-5 w-5 shrink-0" />
              <div className="min-w-0">
                <p>{fmt.label}</p>
                <p className="text-xs font-normal opacity-75">
                  {fmt.description}
                </p>
              </div>
              <Download className="ml-auto h-4 w-4 shrink-0 opacity-60" />
            </button>
          ))}
        </div>
      </div>

      {/* Run selection */}
      <div className="mb-8">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-surface-900">
            Runs ({runs.length})
          </h2>
          {completedRuns.length > 0 && (
            <button
              type="button"
              onClick={toggleAllCompleted}
              className="flex items-center gap-1.5 text-xs font-medium text-primary-600 hover:text-primary-700"
            >
              {allCompletedSelected ? (
                <>
                  <Square className="h-3.5 w-3.5" />
                  Deselect all completed
                </>
              ) : (
                <>
                  <CheckSquare className="h-3.5 w-3.5" />
                  Select all completed
                </>
              )}
            </button>
          )}
        </div>

        {/* Search / filter hint */}
        <div className="mb-3 flex items-center gap-2 text-xs text-surface-400">
          <Search className="h-3.5 w-3.5" />
          <span>
            {selectedIds.size > 0
              ? `${selectedIds.size} run(s) selected`
              : "No runs selected — export will include all completed runs"}
          </span>
        </div>

        {/* List */}
        <div className="space-y-2">
          {runs.length > 0 ? (
            runs.map((run) => (
              <RunCheckRow
                key={run.id}
                run={run}
                selected={selectedIds.has(run.id)}
                onToggle={() => toggleRun(run.id)}
              />
            ))
          ) : (
            <p className="rounded-xl border border-surface-200 bg-white p-8 text-center text-sm text-surface-400">
              No runs yet. Upload a PDF and run an OCR evaluation.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
