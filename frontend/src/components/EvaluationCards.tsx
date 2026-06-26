import type { EngineScoreEntry } from "../lib/types.ts";

/* ── Helpers ───────────────────────────────────────────────────────────── */

export function formatPct(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(2)}%`;
}

function formatNum(value: number | null | undefined, decimals = 4): string {
  if (value === null || value === undefined) return "—";
  return value.toFixed(decimals);
}

export function cerColorClass(cer: number): string {
  if (cer < 0.05) return "text-green-600 bg-green-50";
  if (cer < 0.15) return "text-yellow-600 bg-yellow-50";
  return "text-red-600 bg-red-50";
}

/* ── MetricCard ─────────────────────────────────────────────────────────── */

export function MetricCard({
  title,
  value,
  subtitle,
  color,
}: {
  title: string;
  value: string;
  subtitle?: string;
  color?: "green" | "yellow" | "red";
}) {
  const colorStyles = {
    green: "text-green-700",
    yellow: "text-yellow-700",
    red: "text-red-700",
  };

  return (
    <div className="rounded-xl border border-surface-200 bg-white p-5 shadow-sm">
      <p className="text-sm font-medium text-surface-500">{title}</p>
      <p
        className={`mt-1 text-2xl font-bold ${color ? colorStyles[color] : "text-surface-900"}`}
      >
        {value}
      </p>
      {subtitle && (
        <p className="mt-0.5 text-xs text-surface-400">{subtitle}</p>
      )}
    </div>
  );
}

/* ── Engine Comparison Card ──────────────────────────────────────────────── */

export function EngineCard({
  entry,
}: {
  entry: EngineScoreEntry;
}) {
  return (
    <div className="rounded-lg border border-surface-200 bg-white p-4">
      <p className="mb-2 truncate font-mono text-xs font-medium text-surface-500">
        {entry.engine_id.slice(0, 8)}…
      </p>
      {entry.scores ? (
        <div className="space-y-1.5 text-sm">
          <div className="flex justify-between">
            <span className="text-surface-500">CER</span>
            <span className="font-mono font-medium">{formatPct(entry.scores.cer)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-surface-500">WER</span>
            <span className="font-mono font-medium">{formatPct(entry.scores.wer)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-surface-500">Char F1</span>
            <span className="font-mono font-medium">{formatPct(entry.scores.char_f1)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-surface-500">Word F1</span>
            <span className="font-mono font-medium">{formatPct(entry.scores.word_f1)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-surface-500">Pages</span>
            <span className="font-mono font-medium">{entry.scores.pages}</span>
          </div>
        </div>
      ) : (
        <p className="text-sm text-surface-400">{entry.message ?? "No data"}</p>
      )}
    </div>
  );
}

export { formatNum };
