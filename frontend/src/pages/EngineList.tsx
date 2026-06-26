import { useEffect, useState } from "react";
import { listEngines } from "@/lib/api";
import type { Engine } from "@/lib/types";

export default function EngineList() {
  const [engines, setEngines] = useState<Engine[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    listEngines()
      .then((data) => {
        if (!cancelled) setEngines(data);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return (
      <div className="mx-auto flex max-w-4xl items-center justify-center py-24">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-surface-200 border-t-primary-600" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="mx-auto max-w-4xl">
        <h1 className="text-3xl font-bold text-surface-900">Engines</h1>
        <div className="mt-8 rounded-xl border border-red-200 bg-red-50 p-6 text-center">
          <p className="text-red-600">Failed to load engines: {error}</p>
          <button
            onClick={() => window.location.reload()}
            className="mt-3 rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-surface-900">Engines</h1>
          <p className="mt-2 text-surface-500">
            {engines.length} OCR engine{engines.length !== 1 ? "s" : ""} available
          </p>
        </div>
      </div>

      <div className="mt-8 grid gap-4 sm:grid-cols-2">
        {engines.map((engine) => (
          <div
            key={engine.slug}
            className="rounded-xl border border-surface-200 bg-white p-5 shadow-sm transition-shadow hover:shadow-md"
          >
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-lg font-semibold text-surface-900">
                  {engine.display_name}
                </h3>
                <span className="mt-0.5 inline-block rounded-full bg-surface-100 px-2.5 py-0.5 text-xs font-medium text-surface-600">
                  {engine.slug}
                </span>
              </div>
              <span className="text-xs text-surface-400">v{engine.version}</span>
            </div>

            {engine.config_schema &&
              typeof engine.config_schema === "object" &&
              "properties" in engine.config_schema && (
              <details className="mt-3">
                <summary className="cursor-pointer text-xs font-medium text-surface-500 hover:text-surface-700">
                  Configuration (
                  {Object.keys(
                    (engine.config_schema as Record<string, unknown>).properties as Record<
                      string,
                      unknown
                    >,
                  ).length}{" "}
                  params)
                </summary>
                <div className="mt-2 space-y-1.5">
                  {Object.entries(
                    (engine.config_schema as Record<string, unknown>).properties as Record<
                      string,
                      { type?: string; default?: unknown; description?: string }
                    >,
                  ).map(([key, val]) => (
                    <div key={key} className="flex items-center gap-2 text-xs">
                      <code className="rounded bg-surface-50 px-1.5 py-0.5 font-mono text-surface-700">
                        {key}
                      </code>
                      <span className="text-surface-400">{val.type ?? "string"}</span>
                      {val.default !== undefined && (
                        <span className="ml-auto text-surface-400">
                          = {String(val.default)}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </details>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
