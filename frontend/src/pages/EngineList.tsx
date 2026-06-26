export default function EngineList() {
  return (
    <div className="mx-auto max-w-4xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-surface-900">Engines</h1>
          <p className="mt-2 text-surface-500">
            Register, configure, and manage OCR engine plugins.
          </p>
        </div>
      </div>
      <div className="mt-8 rounded-xl border border-surface-200 bg-white p-12 text-center shadow-sm">
        <p className="text-surface-400">
          No engines configured yet. Engine management will be available here.
        </p>
      </div>
    </div>
  );
}
