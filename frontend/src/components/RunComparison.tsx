import { useEffect, useState } from "react";
import { BarChart3, RefreshCw } from "lucide-react";
import { compareEngines, compareRuns } from "../lib/api.ts";
import type {
  ComparisonEnginesResponse,
  ComparisonRunEntry,
  ComparisonRunsResponse,
  EngineComparisonEntry,
} from "../lib/types.ts";
import { MetricCard, formatPct } from "./EvaluationCards.tsx";

interface RunComparisonProps {
  runIds?: string[];
  engineIds?: string[];
  pdfIds?: string[];
  gtVersionId?: string;
}

function cerColor(cer: number): "green" | "yellow" | "red" {
  if (cer < 0.05) return "green";
  if (cer < 0.15) return "yellow";
  return "red";
}

function scoreColorClass(value: number, lowerIsBetter: boolean): string {
  if (lowerIsBetter) {
    if (value < 0.05) return "text-emerald-600";
    if (value < 0.15) return "text-amber-600";
    return "text-red-600";
  }
  if (value > 0.95) return "text-emerald-600";
  if (value > 0.85) return "text-amber-600";
  return "text-red-600";
}

interface ChartBar {
  label: string;
  cer: number;
  wer: number;
}

function SimpleBarChart({ bars }: { bars: ChartBar[] }) {
  if (!bars.length) return null;
  const maxVal = Math.max(...bars.map((b) => Math.max(b.cer, b.wer)), 0.01);

  return (
    <div className="space-y-3">
      {bars.map((bar) => (
        <div key={bar.label}>
          <p className="mb-1 text-xs font-medium text-surface-600">{bar.label}</p>
          <div className="flex items-center gap-3">
            <span className="w-6 text-xs text-surface-400">CER</span>
            <div className="flex-1">
              <div className="h-3 w-full overflow-hidden rounded-full bg-surface-200">
                <div
                  className="h-full rounded-full bg-primary-500 transition-all"
                  style={{ width: `${(bar.cer / maxVal) * 100}%` }}
                  title={`CER: ${formatPct(bar.cer)}`}
                />
              </div>
            </div>
            <span className={`w-14 text-right font-mono text-xs ${scoreColorClass(bar.cer, true)}`}>
              {formatPct(bar.cer)}
            </span>
          </div>
          <div className="flex items-center gap-3">
            <span className="w-6 text-xs text-surface-400">WER</span>
            <div className="flex-1">
              <div className="h-3 w-full overflow-hidden rounded-full bg-surface-200">
                <div
                  className="h-full rounded-full bg-amber-500 transition-all"
                  style={{ width: `${(bar.wer / maxVal) * 100}%` }}
                  title={`WER: ${formatPct(bar.wer)}`}
                />
              </div>
            </div>
            <span className={`w-14 text-right font-mono text-xs ${scoreColorClass(bar.wer, true)}`}>
              {formatPct(bar.wer)}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

function ScoreCard({
  label,
  engineSlug,
  scores,
  message,
  showPages,
}: {
  label: string;
  engineSlug: string;
  scores: ComparisonRunEntry["scores"];
  message: string | undefined;
  showPages?: boolean;
}) {
  return (
    <div className="rounded-xl border border-surface-200 bg-white p-5 shadow-sm">
      <div className="mb-3">
        <p className="font-mono text-xs text-surface-400">{label}</p>
        <p className="text-sm font-medium text-surface-700">
          Engine: {engineSlug}
        </p>
      </div>
      {scores ? (
        <div className="space-y-2">
          <MetricCard
            title="CER"
            value={formatPct(scores.cer)}
            color={cerColor(scores.cer)}
          />
          <MetricCard
            title="WER"
            value={formatPct(scores.wer)}
            color={cerColor(scores.wer)}
          />
          <div className="flex justify-between text-sm">
            <span className="text-surface-500">Char F1</span>
            <span className={`font-mono font-medium ${scoreColorClass(scores.char_f1 ?? 0, false)}`}>
              {formatPct(scores.char_f1 ?? 0)}
            </span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-surface-500">Word F1</span>
            <span className={`font-mono font-medium ${scoreColorClass(scores.word_f1 ?? 0, false)}`}>
              {formatPct(scores.word_f1 ?? 0)}
            </span>
          </div>
          {showPages && "pages" in scores && (
            <div className="flex justify-between text-sm">
              <span className="text-surface-500">Pages</span>
              <span className="font-mono text-surface-700">
                {(scores as { pages: number }).pages}
              </span>
            </div>
          )}
          {"bootstrap_ci" in scores && scores.bootstrap_ci && (
            <div className="mt-2 rounded-lg bg-surface-50 px-3 py-2 text-xs text-surface-500">
              <p>
                95% CI: {formatPct(scores.bootstrap_ci.cer_lower)} –{" "}
                {formatPct(scores.bootstrap_ci.cer_upper)}
              </p>
            </div>
          )}
        </div>
      ) : (
        <p className="text-sm text-surface-400">{message ?? "No scores"}</p>
      )}
    </div>
  );
}

function RunsComparisonView({ data }: { data: ComparisonRunsResponse }) {
  const entries = data.entries;
  const bars: ChartBar[] = entries
    .filter((e) => e.scores)
    .map((e) => ({
      label: e.engine_slug,
      cer: e.scores!.cer,
      wer: e.scores!.wer,
    }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {entries.map((entry) => (
          <ScoreCard
            key={entry.run_id}
            label={entry.run_id.slice(0, 8) + "..."}
            engineSlug={entry.engine_slug}
            scores={entry.scores}
            message={entry.message}
            showPages
          />
        ))}
      </div>

      {bars.length > 0 && (
        <div className="rounded-xl border border-surface-200 bg-white p-5 shadow-sm">
          <h3 className="mb-4 text-sm font-semibold text-surface-700">
            CER / WER Comparison
          </h3>
          <SimpleBarChart bars={bars} />
        </div>
      )}
    </div>
  );
}

function EnginesComparisonView({ data }: { data: ComparisonEnginesResponse }) {
  return (
    <div className="space-y-6">
      {data.engines.map((engineEntry) => {
        const bars: ChartBar[] = engineEntry.pdfs
          .filter((p) => p.scores)
          .map((p) => ({
            label: p.pdf_id.slice(0, 8),
            cer: p.scores!.cer,
            wer: p.scores!.wer,
          }));

        return (
          <div
            key={engineEntry.engine_id}
            className="rounded-xl border border-surface-200 bg-white p-5 shadow-sm"
          >
            <h3 className="mb-4 text-sm font-semibold text-surface-700">
              Engine {engineEntry.engine_id.slice(0, 8)}...
            </h3>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {engineEntry.pdfs.map((pdfEntry) => (
                <div
                  key={pdfEntry.pdf_id}
                  className="rounded-lg border border-surface-200 bg-surface-50 p-4"
                >
                  <p className="mb-2 truncate font-mono text-xs text-surface-400">
                    PDF {pdfEntry.pdf_id.slice(0, 8)}...
                  </p>
                  {pdfEntry.scores ? (
                    <div className="space-y-1.5 text-sm">
                      <div className="flex justify-between">
                        <span className="text-surface-500">CER</span>
                        <span className={`font-mono font-medium ${scoreColorClass(pdfEntry.scores.cer, true)}`}>
                          {formatPct(pdfEntry.scores.cer)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-surface-500">WER</span>
                        <span className={`font-mono font-medium ${scoreColorClass(pdfEntry.scores.wer, true)}`}>
                          {formatPct(pdfEntry.scores.wer)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-surface-500">Char F1</span>
                        <span className={`font-mono font-medium ${scoreColorClass(pdfEntry.scores.char_f1 ?? 0, false)}`}>
                          {formatPct(pdfEntry.scores.char_f1 ?? 0)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-surface-500">Word F1</span>
                        <span className={`font-mono font-medium ${scoreColorClass(pdfEntry.scores.word_f1 ?? 0, false)}`}>
                          {formatPct(pdfEntry.scores.word_f1 ?? 0)}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <p className="text-sm text-surface-400">
                      {pdfEntry.message ?? "No data"}
                    </p>
                  )}
                </div>
              ))}
            </div>

            {bars.length > 0 && (
              <div className="mt-4">
                <SimpleBarChart bars={bars} />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default function RunComparison({
  runIds,
  engineIds,
  pdfIds,
  gtVersionId,
}: RunComparisonProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<
    ComparisonRunsResponse | ComparisonEnginesResponse | null
  >(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      if (runIds && runIds.length >= 2) {
        setData(await compareRuns(runIds, gtVersionId));
      } else if (engineIds && pdfIds) {
        setData(await compareEngines(engineIds, pdfIds, gtVersionId));
      } else {
        setError("Provide at least 2 run IDs or engine IDs + PDF IDs");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Comparison failed");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (runIds || (engineIds && pdfIds)) {
      fetchData();
    }
  }, [runIds, engineIds, pdfIds, gtVersionId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center rounded-xl border border-surface-200 bg-white p-12">
        <RefreshCw className="h-6 w-6 animate-spin text-surface-400" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-6 text-center">
        <p className="text-sm text-red-700">{error}</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="rounded-xl border border-dashed border-surface-300 bg-surface-50 p-12 text-center">
        <BarChart3 className="mx-auto h-10 w-10 text-surface-300" />
        <p className="mt-3 text-sm text-surface-500">
          Select runs or engines to compare
        </p>
      </div>
    );
  }

  if ("entries" in data) {
    return <RunsComparisonView data={data as ComparisonRunsResponse} />;
  }

  return <EnginesComparisonView data={data as ComparisonEnginesResponse} />;
}
