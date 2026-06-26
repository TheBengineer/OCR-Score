import { useParams } from "react-router-dom";

export default function Evaluation() {
  const { id } = useParams<{ id: string }>();

  return (
    <div className="mx-auto max-w-6xl">
      <h1 className="text-3xl font-bold text-surface-900">Evaluation</h1>
      <p className="mt-2 text-surface-500">
        Detailed evaluation scores for document {id} across all OCR engines.
      </p>
      <div className="mt-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard title="CER" value="—" />
        <MetricCard title="WER" value="—" />
        <MetricCard title="Accuracy" value="—" />
        <MetricCard title="F1" value="—" />
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
