import { useCallback, useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  getRunScores,
  getRunScoresByPage,
  listRuns,
  getEngineComparison,
} from "../lib/api.ts";
import type {
  BootstrapCI,
  EngineComparisonResponse,
  RunOverallScores,
  PageScoreEntry,
} from "../lib/types.ts";
import ScoreChart from "../components/ScoreChart.tsx";
import {
  EngineCard,
  MetricCard,
  cerColorClass,
  formatNum,
  formatPct,
} from "../components/EvaluationCards.tsx";

export default function Evaluation() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [runIds, setRunIds] = useState<string[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [pageScores, setPageScores] = useState<PageScoreEntry[]>([]);
  const [overall, setOverall] = useState<RunOverallScores | null>(null);
  const [bootstrapCi, setBootstrapCi] = useState<BootstrapCI | null>(null);
  const [comparison, setComparison] = useState<EngineComparisonResponse | null>(null);

  const loadRunInfo = useCallback(async () => {
    if (!id) return;
    setLoading(true);
    setError(null);
    try {
      const runsResp = await listRuns({
        pdf_id: id,
        status: "completed",
        limit: 20,
      });

      const runIdList = runsResp.items.map((r) => r.id);
      setRunIds(runIdList);

      if (runIdList.length > 0) {
        setSelectedRunId(runIdList[0] ?? null);
      } else {
        setLoading(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load runs");
      setLoading(false);
    }
  }, [id]);

  const loadScores = useCallback(async () => {
    if (!selectedRunId || !id) return;
    setLoading(true);
    setError(null);
    try {
      const [scoresResp, byPageResp] = await Promise.all([
        getRunScores(selectedRunId, "00000000-0000-0000-0000-000000000000").catch(
          () => null,
        ),
        getRunScoresByPage(
          selectedRunId,
          "00000000-0000-0000-0000-000000000000",
        ).catch(() => null),
      ]);

      if (scoresResp && scoresResp.overall) {
        setOverall(scoresResp.overall);
        setBootstrapCi(scoresResp.bootstrap_ci);
      }

      if (byPageResp && byPageResp.pages) {
        setPageScores(byPageResp.pages);
      }

      const compResp = await getEngineComparison(id, [], "00000000-0000-0000-0000-000000000000").catch(
        () => null,
      );
      if (compResp) setComparison(compResp);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load scores");
    } finally {
      setLoading(false);
    }
  }, [selectedRunId, id]);

  useEffect(() => {
    loadRunInfo();
  }, [loadRunInfo]);

  useEffect(() => {
    if (selectedRunId) {
      loadScores();
    }
  }, [selectedRunId, loadScores]);

  const handlePageClick = useCallback(
    (page: number) => {
      if (id) {
        navigate(`/pdfs/${id}?page=${page}`);
      }
    },
    [id, navigate],
  );

  if (!id) {
    return (
      <div className="mx-auto max-w-6xl">
        <h1 className="text-3xl font-bold text-surface-900">Evaluation</h1>
        <p className="mt-2 text-surface-500">No document specified.</p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-7xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-surface-900">Evaluation</h1>
        <p className="mt-1 text-surface-500">
          Document <span className="font-mono text-surface-700">{id.slice(0, 8)}…</span>
        </p>
      </div>

      {loading && !overall && (
        <div className="flex items-center justify-center py-20">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary-200 border-t-primary-600" />
        </div>
      )}

      {error && (
        <div className="mb-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {runIds.length > 0 && (
        <div className="mb-6 flex items-center gap-3">
          <label className="text-sm font-medium text-surface-600">Run:</label>
          <select
            value={selectedRunId ?? ""}
            onChange={(e) => setSelectedRunId(e.target.value)}
            className="rounded-lg border border-surface-300 bg-white px-3 py-1.5 text-sm text-surface-700"
          >
            {runIds.map((rid) => (
              <option key={rid} value={rid}>
                {rid.slice(0, 8)}…
              </option>
            ))}
          </select>
        </div>
      )}

      {!loading && runIds.length === 0 && !error && (
        <div className="rounded-lg border border-dashed border-surface-300 bg-surface-50 p-12 text-center text-surface-400">
          <p className="text-lg font-medium">No completed runs yet</p>
          <p className="mt-1 text-sm">
            Start an OCR run for this document to see evaluation scores.
          </p>
        </div>
      )}

      {overall && (
        <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            title="CER"
            value={formatPct(overall.cer)}
            {...(bootstrapCi ? { subtitle: `95% CI: ${formatPct(bootstrapCi.cer_lower)} – ${formatPct(bootstrapCi.cer_upper)}` } : {})}
            color={
              overall.cer < 0.05
                ? "green"
                : overall.cer < 0.15
                  ? "yellow"
                  : "red"
            }
          />
          <MetricCard title="WER" value={formatPct(overall.wer)} />
          <MetricCard title="Char F1" value={formatPct(overall.char_f1)} />
          <MetricCard title="Word F1" value={formatPct(overall.word_f1)} />
        </div>
      )}

      <div className="mb-6 grid gap-6 lg:grid-cols-4">
        <div className="lg:col-span-3 rounded-xl border border-surface-200 bg-white p-6 shadow-sm">
          <h2 className="mb-4 text-lg font-semibold text-surface-800">
            Per-Page CER
          </h2>
          <ScoreChart pages={pageScores} onPageClick={handlePageClick} />
        </div>

        <div className="rounded-xl border border-surface-200 bg-white p-5 shadow-sm">
          <h2 className="mb-3 text-sm font-semibold text-surface-700">
            Engine Comparison
          </h2>
          {comparison && comparison.engines.length > 0 ? (
            <div className="space-y-3">
              {comparison.engines.map((entry) => (
                <EngineCard key={entry.engine_id} entry={entry} />
              ))}
            </div>
          ) : (
            <p className="text-xs text-surface-400">
              Run multiple engines to see comparison.
            </p>
          )}
        </div>
      </div>

      {pageScores.length > 0 && (
        <div className="overflow-hidden rounded-xl border border-surface-200 bg-white shadow-sm">
          <div className="border-b border-surface-200 px-6 py-4">
            <h2 className="text-lg font-semibold text-surface-800">
              Per-Page Details
            </h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-surface-100 bg-surface-50 text-surface-500">
                  <th className="px-6 py-3 font-medium">Page</th>
                  <th className="px-6 py-3 font-medium">CER</th>
                  <th className="px-6 py-3 font-medium">WER</th>
                  <th className="px-6 py-3 font-medium">Char F1</th>
                  <th className="px-6 py-3 font-medium">Word F1</th>
                </tr>
              </thead>
              <tbody>
                {pageScores.map((ps) => (
                  <tr
                    key={ps.page}
                    className="cursor-pointer border-b border-surface-100 transition-colors last:border-0 hover:bg-surface-50"
                    onClick={() => handlePageClick(ps.page)}
                  >
                    <td className="px-6 py-3 font-medium text-surface-700">
                      {ps.page}
                    </td>
                    <td className="px-6 py-3">
                      <span
                        className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${cerColorClass(ps.cer)}`}
                      >
                        {formatPct(ps.cer)}
                      </span>
                    </td>
                    <td className="px-6 py-3 font-mono text-surface-600">
                      {formatPct(ps.wer)}
                    </td>
                    <td className="px-6 py-3 font-mono text-surface-600">
                      {formatNum(ps.char_f1)}
                    </td>
                    <td className="px-6 py-3 font-mono text-surface-600">
                      {formatNum(ps.word_f1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
