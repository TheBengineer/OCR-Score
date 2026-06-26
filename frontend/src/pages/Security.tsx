import { useState, useEffect, useCallback } from "react";
import { Shield, Eye, EyeOff, Loader2, CheckCircle, AlertCircle } from "lucide-react";
import type { Engine } from "../lib/types.ts";
import { listEngines } from "../lib/api.ts";

const SECRETS_URL = "/api/v1/engines";

async function getSecrets(slug: string): Promise<Record<string, string>> {
  const res = await fetch(`${SECRETS_URL}/${slug}/secrets`);
  if (!res.ok) return {};
  return res.json();
}

async function putSecrets(slug: string, secrets: Record<string, string>): Promise<void> {
  await fetch(`${SECRETS_URL}/${slug}/secrets`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(secrets),
  });
}

interface EngineWithSecrets extends Engine {
  secret_schema: { key: string; env_var: string | null; display_name: string; description: string }[];
}

export default function Security() {
  const [engines, setEngines] = useState<EngineWithSecrets[]>([]);
  const [values, setValues] = useState<Record<string, Record<string, string>>>({});
  const [visible, setVisible] = useState<Record<string, boolean>>({});
  const [saving, setSaving] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const list = await listEngines() as unknown as EngineWithSecrets[];
        if (cancelled) return;
        setEngines(list);

        // Load stored secrets for each engine
        const vals: Record<string, Record<string, string>> = {};
        for (const engine of list) {
          vals[engine.slug] = await getSecrets(engine.slug);
        }
        if (!cancelled) setValues(vals);
      } catch {
        // silently fail
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, []);

  const handleChange = useCallback((slug: string, key: string, value: string) => {
    setValues((prev) => ({
      ...prev,
      [slug]: { ...(prev[slug] || {}), [key]: value },
    }));
  }, []);

  const handleSave = useCallback(async (slug: string) => {
    setSaving(slug);
    setSaved(null);
    try {
      await putSecrets(slug, values[slug] || {});
      setSaved(slug);
      setTimeout(() => setSaved(null), 2000);
    } catch {
      // silently fail
    } finally {
      setSaving(null);
    }
  }, [values]);

  const toggleVisible = useCallback((key: string) => {
    setVisible((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  if (loading) {
    return (
      <div className="mx-auto flex max-w-3xl items-center justify-center py-24">
        <Loader2 className="h-8 w-8 animate-spin text-primary-500" />
      </div>
    );
  }

  const enginesWithSecrets = engines.filter((e) => e.secret_schema?.length > 0);

  return (
    <div className="mx-auto max-w-3xl">
      <div className="flex items-center gap-3">
        <Shield className="h-6 w-6 text-surface-500" />
        <div>
          <h1 className="text-3xl font-bold text-surface-900">Security</h1>
          <p className="mt-1 text-surface-500">
            Manage API keys and credentials for OCR engines that need them.
            Values are stored in the database and injected automatically when
            a run is executed.
          </p>
        </div>
      </div>

      {enginesWithSecrets.length === 0 ? (
        <div className="mt-8 rounded-xl border border-surface-200 bg-white p-12 text-center shadow-sm">
          <Shield className="mx-auto h-10 w-10 text-surface-300" />
          <p className="mt-3 text-sm text-surface-400">
            No engines require secrets.
          </p>
        </div>
      ) : (
        <div className="mt-8 space-y-6">
          {enginesWithSecrets.map((engine) => (
            <div
              key={engine.slug}
              className="rounded-xl border border-surface-200 bg-white shadow-sm"
            >
              <div className="border-b border-surface-100 px-6 py-4">
                <h2 className="text-lg font-semibold text-surface-900">
                  {engine.display_name}
                </h2>
                <p className="mt-0.5 text-sm text-surface-400">{engine.slug}</p>
              </div>

              <div className="space-y-4 px-6 py-4">
                {engine.secret_schema.map((secret) => {
                  const fieldId = `${engine.slug}-${secret.key}`;
                  const val = values[engine.slug]?.[secret.key] ?? "";
                  const isVisible = visible[fieldId];

                  return (
                    <div key={secret.key}>
                      <label
                        htmlFor={fieldId}
                        className="block text-sm font-medium text-surface-700"
                      >
                        {secret.display_name}
                        {secret.env_var && (
                          <span className="ml-2 text-xs text-surface-400">
                            env: {secret.env_var}
                          </span>
                        )}
                      </label>
                      <p className="mt-0.5 text-xs text-surface-400">
                        {secret.description}
                      </p>
                      <div className="mt-1.5 flex items-center gap-2">
                        <div className="relative flex-1">
                          <input
                            id={fieldId}
                            type={isVisible ? "text" : "password"}
                            value={val}
                            onChange={(e) =>
                              handleChange(engine.slug, secret.key, e.target.value)
                            }
                            placeholder={
                              secret.env_var
                                ? `Set via ${secret.env_var} or enter here`
                                : `Enter ${secret.display_name}`
                            }
                            className="w-full rounded-lg border border-surface-300 bg-white px-3 py-2 pr-10 text-sm text-surface-900 placeholder-surface-400 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                          />
                          <button
                            type="button"
                            onClick={() => toggleVisible(fieldId)}
                            className="absolute right-2 top-1/2 -translate-y-1/2 text-surface-400 hover:text-surface-600"
                            tabIndex={-1}
                          >
                            {isVisible ? <EyeOff size={16} /> : <Eye size={16} />}
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="flex items-center justify-end gap-3 border-t border-surface-100 px-6 py-3">
                {saved === engine.slug && (
                  <span className="flex items-center gap-1 text-xs text-emerald-600">
                    <CheckCircle size={14} />
                    Saved
                  </span>
                )}
                <button
                  type="button"
                  onClick={() => handleSave(engine.slug)}
                  disabled={saving === engine.slug}
                  className="inline-flex items-center gap-2 rounded-lg bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700 disabled:opacity-50 transition-colors"
                >
                  {saving === engine.slug ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    "Save"
                  )}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
