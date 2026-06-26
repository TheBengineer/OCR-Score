import { useEffect, useState } from "react";
import { X, Play, CheckCircle, XCircle, Clock } from "lucide-react";
import { createBatch, listEngines } from "../lib/api.ts";
import type { BatchProgressResponse, Engine } from "../lib/types.ts";

// ── Props ─────────────────────────────────────────────────────────────────

interface BatchDialogProps {
  selectedPdfIds: string[];
  onClose: () => void;
  onBatchCreated: (batchId: string) => void;
  progress: BatchProgressResponse | null;
}

// ── Item status icon ──────────────────────────────────────────────────────

function ItemStatusIcon({ status }: { status: string }) {
  switch (status) {
    case "completed":
      return <CheckCircle className="h-4 w-4 text-emerald-500" />;
    case "failed":
      return <XCircle className="h-4 w-4 text-red-500" />;
    case "processing":
      return (
        <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary-500 border-t-transparent" />
      );
    default:
      return <Clock className="h-4 w-4 text-surface-400" />;
  }
}

// ── Color for percent ─────────────────────────────────────────────────────

function percentColor(pct: number): string {
  if (pct >= 100) return "text-emerald-600";
  if (pct > 0) return "text-primary-600";
  return "text-surface-500";
}

// ── Component ─────────────────────────────────────────────────────────────

export default function BatchDialog({
  selectedPdfIds,
  onClose,
  onBatchCreated,
  progress,
}: BatchDialogProps) {
  const [engines, setEngines] = useState<Engine[]>([]);
  const [selectedEngines, setSelectedEngines] = useState<Set<string>>(new Set());
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load engines
  useEffect(() => {
    listEngines()
      .then((list) => {
        setEngines(list);
        // Pre-select all enabled engines
        const enabled = new Set(
          list.filter((e) => e.enabled).map((e) => e.slug),
        );
        setSelectedEngines(enabled);
      })
      .catch(() => {
        // If endpoint doesn't exist, provide defaults
        setEngines([
          {
            id: "mock",
            slug: "mock",
            display_name: "Mock Engine",
            version: "0.1.0",
            enabled: true,
            config_schema: null,
            description: "Test engine",
            created_at: "",
            updated_at: "",
          },
        ]);
        setSelectedEngines(new Set(["mock"]));
      });
  }, []);

  // Toggle engine selection
  const toggleEngine = (slug: string) => {
    setSelectedEngines((prev) => {
      const next = new Set(prev);
      if (next.has(slug)) next.delete(slug);
      else next.add(slug);
      return next;
    });
  };

  // Start batch
  const handleStart = async () => {
    if (selectedEngines.size === 0) {
      setError("Select at least one engine");
      return;
    }
    setStarting(true);
    setError(null);
    try {
      const resp = await createBatch(
        selectedPdfIds,
        Array.from(selectedEngines),
      );
      onBatchCreated(resp.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start batch");
    } finally {
      setStarting(false);
    }
  };

  const inProgress = progress && progress.status !== "completed" && progress.status !== "failed";
  const finished = progress && (progress.status === "completed" || progress.status === "failed");

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="mx-4 w-full max-w-lg rounded-xl bg-white shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-surface-200 px-6 py-4">
          <h2 className="text-lg font-semibold text-surface-900">
            {progress ? "Batch Progress" : "Batch Process PDFs"}
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1.5 text-surface-400 hover:bg-surface-100 hover:text-surface-600"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Body */}
        <div className="space-y-5 px-6 py-5">
          {/* PDF count */}
          <div>
            <p className="text-sm font-medium text-surface-700">Selected PDFs</p>
            <p className="mt-1 text-sm text-surface-500">
              {selectedPdfIds.length} document{selectedPdfIds.length !== 1 ? "s" : ""}
            </p>
          </div>

          {/* Engine selection */}
          {!progress && (
            <div>
              <p className="text-sm font-medium text-surface-700">Engines</p>
              <div className="mt-2 space-y-2">
                {engines.map((engine) => (
                  <label
                    key={engine.slug}
                    className="flex cursor-pointer items-center gap-3 rounded-lg border border-surface-200 px-3 py-2.5 transition-colors hover:bg-surface-50"
                  >
                    <input
                      type="checkbox"
                      checked={selectedEngines.has(engine.slug)}
                      onChange={() => toggleEngine(engine.slug)}
                      className="h-4 w-4 rounded border-surface-300 text-primary-600 focus:ring-primary-500"
                    />
                    <div className="flex-1">
                      <p className="text-sm font-medium text-surface-900">
                        {engine.display_name}
                      </p>
                      <p className="text-xs text-surface-400">{engine.slug}</p>
                    </div>
                    <span className="text-xs text-surface-400">
                      v{engine.version}
                    </span>
                  </label>
                ))}
              </div>
            </div>
          )}

          {/* Progress display */}
          {progress && (
            <div>
              {/* Overall progress */}
              <div className="mb-4">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium text-surface-700">Overall</span>
                  <span className={percentColor(progress.percent)}>
                    {progress.percent.toFixed(1)}%
                  </span>
                </div>
                <div className="mt-1.5 h-2.5 w-full overflow-hidden rounded-full bg-surface-200">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ease-out ${
                      progress.status === "failed"
                        ? "bg-red-500"
                        : progress.status === "completed"
                          ? "bg-emerald-500"
                          : "bg-primary-500"
                    }`}
                    style={{ width: `${progress.percent}%` }}
                  />
                </div>
                <div className="mt-1 flex gap-4 text-xs text-surface-500">
                  <span>{progress.completed} completed</span>
                  <span>{progress.failed} failed</span>
                  <span>{progress.pending} pending</span>
                </div>
              </div>

              {/* Per-item details */}
              <div className="max-h-64 space-y-1.5 overflow-y-auto">
                {progress.items.map((item, idx) => (
                  <div
                    key={`${item.pdf_id}-${item.engine_slug}-${idx}`}
                    className="flex items-center gap-2 rounded-lg bg-surface-50 px-3 py-2"
                  >
                    <ItemStatusIcon status={item.status} />
                    <span className="flex-1 truncate text-sm text-surface-700">
                      {item.engine_slug}
                    </span>
                    {item.message && (
                      <span className="max-w-[120px] truncate text-xs text-surface-400" title={item.message}>
                        {item.message}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 border-t border-surface-200 px-6 py-4">
          {!progress && (
            <>
              <button
                type="button"
                onClick={onClose}
                className="rounded-lg px-4 py-2 text-sm font-medium text-surface-600 hover:bg-surface-100 transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleStart}
                disabled={starting || selectedEngines.size === 0}
                className="inline-flex items-center gap-2 rounded-lg bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700 transition-colors disabled:opacity-50"
              >
                <Play className="h-4 w-4" />
                {starting ? "Starting..." : "Start Batch"}
              </button>
            </>
          )}
          {finished && (
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg bg-surface-900 px-4 py-2 text-sm font-medium text-white hover:bg-surface-800 transition-colors"
            >
              Done
            </button>
          )}
          {inProgress && (
            <p className="text-sm text-surface-500">Processing...</p>
          )}
        </div>
      </div>
    </div>
  );
}
