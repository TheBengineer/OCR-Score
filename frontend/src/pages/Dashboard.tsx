export default function Dashboard() {
  return (
    <div className="mx-auto max-w-4xl">
      <h1 className="text-3xl font-bold text-surface-900">Dashboard</h1>
      <p className="mt-2 text-surface-500">
        Overview of OCR evaluation activity, recent runs, and aggregate scores
        across all documents and engines.
      </p>
      <div className="mt-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        <MetricCard title="Documents" value="—" />
        <MetricCard title="Engines" value="—" />
        <MetricCard title="Runs" value="—" />
      </div>
    </div>
  );
}

function MetricCard({ title, value }: { title: string; value: string }) {
  return (
    <div className="rounded-xl border border-surface-200 bg-white p-6 shadow-sm">
      <p className="text-sm font-medium text-surface-500">{title}</p>
      <p className="mt-2 text-3xl font-bold text-surface-900">{value}</p>
    </div>
  );
}
