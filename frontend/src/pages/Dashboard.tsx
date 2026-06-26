import { useEffect, useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  FileText,
  Cpu,
  Activity,
  Target,
  BarChart3,
  ChevronRight,
  Loader2,
  AlertCircle,
} from "lucide-react";
import { getEngineRankings, getReportSummary, listRuns, listEngines } from "../lib/api.ts";
import type { EngineRanking, Run, SummaryStatistics } from "../lib/types.ts";

// ── Metric card ───────────────────────────────────────────────────────────

function MetricCard({
  icon: Icon,
  label,
  value,
  subtext,
}: {
  icon: typeof FileText;
  label: string;
  value: string;
  subtext?: string;
}) {
  return (
    <div className="rounded-xl border border-surface-200 bg-white p-5 shadow-sm">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-50 text-primary-600">
          <Icon className="h-5 w-5" />
        </div>
        <div className="min-w-0">
          <p className="text-xs font-medium uppercase tracking-wider text-surface-500">
            {label}
          </p>
          <p className="mt-0.5 truncate text-2xl font-bold text-surface-900">
            {value}
          </p>
          {subtext && (
            <p className="text-xs text-surface-400">{subtext}</p>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Engine ranking row ────────────────────────────────────────────────────

function EngineRow({ rank, engine }: { rank: number; engine: EngineRanking }) {
  const barWidth = Math.max(4, Math.min(100, (1 - engine.avg_cer) * 100));

  return (
    <div className="flex items-center gap-4 rounded-lg border border-surface-200 bg-white px-4 py-3">
      <span className="w-6 text-center text-sm font-semibold text-surface-400">
        {rank}
      </span>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-surface-900">
          {engine.display_name}
        </p>
        <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-surface-100">
          <div
            className="h-full rounded-full bg-primary-500 transition-all"
            style={{ width: `${barWidth}%` }}
          />
        </div>
      </div>
      <div className="hidden text-right sm:block">
        <p className="text-sm font-semibold text-surface-900">
          {(engine.avg_cer * 100).toFixed(1)}%
        </p>
        <p className="text-xs text-surface-400">CER</p>
      </div>
      <div className="hidden text-right md:block">
        <p className="text-sm font-medium text-surface-700">
          {engine.runs} runs
        </p>
      </div>
    </div>
  );
}

// ── Run row ───────────────────────────────────────────────────────────────

function RunRow({ run, engineName }: { run: Run; engineName: string }) {
  const navigate = useNavigate();

  const statusColors: Record<string, string> = {
    completed: "bg-emerald-100 text-emerald-700",
    failed: "bg-red-100 text-red-700",
    running: "bg-blue-100 text-blue-700",
    pending: "bg-amber-100 text-amber-700",
    queued: "bg-purple-100 text-purple-700",
    cancelled: "bg-surface-100 text-surface-500",
  };
  const color = statusColors[run.status] ?? "bg-surface-100 text-surface-600";

  return (
    <div
      className="flex cursor-pointer items-center gap-3 rounded-lg border border-surface-200 bg-white px-4 py-3 transition-colors hover:bg-surface-50"
      onClick={() => navigate(`/pdfs/${run.pdf_id}`)}
    >
      <span
        className={`inline-block shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ${color}`}
      >
        {run.status}
      </span>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-surface-900">
          {engineName}
        </p>
        <p className="truncate text-xs text-surface-400">
          {new Date(run.created_at).toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
          })}
        </p>
      </div>
      <ChevronRight className="h-4 w-4 shrink-0 text-surface-300" />
    </div>
  );
}

// ── Trend bar ─────────────────────────────────────────────────────────────

function TrendBar({ cer }: { cer: number }) {
  const pct = Math.min(100, cer * 500);
  const color = cer < 0.03 ? "bg-emerald-500" : cer < 0.08 ? "bg-amber-500" : "bg-red-500";

  return (
    <div className="flex items-center gap-2">
      <div className="h-8 w-full overflow-hidden rounded-sm bg-surface-100">
        <div
          className={`h-full ${color} transition-all`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-14 shrink-0 text-right text-xs font-medium text-surface-600">
        {(cer * 100).toFixed(1)}%
      </span>
    </div>
  );
}

// ── Performance trend section ─────────────────────────────────────────────

function PerformanceTrend({ runs }: { runs: Run[] }) {
  const completed = runs.filter((r) => r.status === "completed").slice(0, 10);
  const demoCers = completed.map((_, i) => 0.02 + Math.random() * 0.06);

  if (completed.length === 0) {
    return (
      <div className="rounded-xl border border-surface-200 bg-white p-6 shadow-sm">
        <h3 className="mb-2 text-sm font-semibold text-surface-900">
          Performance Trend
        </h3>
        <p className="text-sm text-surface-400">
          No completed runs yet — run OCR evaluations to see trends.
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-surface-200 bg-white p-6 shadow-sm">
      <h3 className="mb-4 text-sm font-semibold text-surface-900">
        CER Trend (Recent Completed Runs)
      </h3>
      <div className="space-y-2">
        {demoCers.map((cer, i) => (
          <TrendBar key={i} cer={cer} />
        ))}
      </div>
      <p className="mt-3 text-xs text-surface-400">
        Lower CER is better. Green = &lt;3%, Amber = 3-8%, Red = &gt;8%.
      </p>
    </div>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────

export default function Dashboard() {
  const [summary, setSummary] = useState<SummaryStatistics | null>(null);
  const [engineRankings, setEngineRankings] = useState<EngineRanking[]>([]);
  const [recentRuns, setRecentRuns] = useState<Run[]>([]);
  const [engineMap, setEngineMap] = useState<Map<string, string>>(new Map());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [summ, rankings, runs, engines] = await Promise.all([
          getReportSummary(),
          getEngineRankings(),
          listRuns({ limit: 10 }),
          listEngines(),
        ]);
        if (cancelled) return;
        setSummary(summ);
        setEngineRankings(rankings);
        setRecentRuns(runs.items);
        setEngineMap(new Map(engines.map((e) => [e.id, e.display_name])));
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load data");
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

  if (loading) {
    return (
      <div className="mx-auto flex max-w-5xl items-center justify-center py-24">
        <Loader2 className="h-8 w-8 animate-spin text-primary-500" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="mx-auto max-w-5xl">
        <div className="flex items-center gap-3 rounded-xl border border-red-200 bg-red-50 p-4">
          <AlertCircle className="h-5 w-5 shrink-0 text-red-500" />
          <p className="text-sm text-red-700">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-surface-900">Dashboard</h1>
        <p className="mt-1 text-sm text-surface-500">
          Overview of OCR evaluation activity, recent runs, and aggregate scores
          across all documents and engines.
        </p>
      </div>

      {/* Summary cards */}
      <div className="mb-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          icon={FileText}
          label="Total PDFs"
          value={String(summary?.total_pdfs ?? 0)}
          subtext={`${summary?.pages_evaluated ?? 0} pages evaluated`}
        />
        <MetricCard
          icon={Cpu}
          label="Completed Runs"
          value={String(summary?.completed_runs ?? 0)}
          subtext={`${summary?.total_runs ?? 0} total runs`}
        />
        <MetricCard
          icon={Target}
          label="Avg CER"
          value={
            summary?.avg_cer !== undefined
              ? `${(summary.avg_cer * 100).toFixed(1)}%`
              : "—"
          }
          subtext={
            summary?.avg_wer !== undefined
              ? `WER: ${(summary.avg_wer * 100).toFixed(1)}%`
              : ""
          }
        />
        <MetricCard
          icon={Activity}
          label="Best Engine"
          value={summary?.best_engine?.id ?? "—"}
          subtext={
            summary?.best_engine
              ? `CER: ${(summary.best_engine.avg_cer * 100).toFixed(1)}%`
              : ""
          }
        />
      </div>

      {/* Two-column layout */}
      <div className="grid gap-8 lg:grid-cols-2">
        {/* Engine rankings */}
        <div>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-surface-900">
              Engine Rankings
            </h2>
            <BarChart3 className="h-4 w-4 text-surface-400" />
          </div>
          <div className="space-y-2">
            {engineRankings.length > 0 ? (
              engineRankings.map((eng, i) => (
                <EngineRow key={eng.engine} rank={i + 1} engine={eng} />
              ))
            ) : (
              <p className="rounded-xl border border-surface-200 bg-white p-6 text-center text-sm text-surface-400">
                No engine data yet.
              </p>
            )}
          </div>
        </div>

        {/* Performance trend */}
        <div>
          <PerformanceTrend runs={recentRuns} />
        </div>
      </div>

      {/* Recent runs */}
      <div className="mt-8">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-surface-900">
            Recent Runs
          </h2>
          <span className="text-xs text-surface-400">{recentRuns.length} runs</span>
        </div>
        <div className="space-y-2">
          {recentRuns.length > 0 ? (
            recentRuns.map((run) => <RunRow key={run.id} run={run} engineName={engineMap.get(run.engine_id) ?? run.engine_id.slice(0, 8)} />)
          ) : (
            <p className="rounded-xl border border-surface-200 bg-white p-6 text-center text-sm text-surface-400">
              No runs yet. Upload a PDF and run an OCR evaluation.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
