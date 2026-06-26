export default function Reports() {
  return (
    <div className="mx-auto max-w-4xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-surface-900">Reports</h1>
          <p className="mt-2 text-surface-500">
            Aggregate rankings, score breakdowns, and exportable evaluation
            reports.
          </p>
        </div>
      </div>
      <div className="mt-8 rounded-xl border border-surface-200 bg-white p-12 text-center shadow-sm">
        <p className="text-surface-400">
          No reports available yet. Run OCR evaluations to generate reports.
        </p>
      </div>
    </div>
  );
}
