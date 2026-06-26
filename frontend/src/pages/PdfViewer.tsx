import { useParams } from "react-router-dom";

export default function PdfViewer() {
  const { id } = useParams<{ id: string }>();

  return (
    <div className="mx-auto max-w-6xl">
      <h1 className="text-3xl font-bold text-surface-900">PDF Viewer</h1>
      <p className="mt-2 text-surface-500">
        View OCR results overlaid on the original document.
      </p>
      <div className="mt-8 rounded-xl border border-surface-200 bg-white p-12 text-center shadow-sm">
        <p className="text-surface-400">
          Document <code className="rounded bg-surface-100 px-1.5 py-0.5 font-mono text-sm text-surface-600">{id ?? "—"}</code>
        </p>
        <p className="mt-2 text-sm text-surface-400">
          PDF rendering and OCR overlay layers will be available here.
        </p>
      </div>
    </div>
  );
}
