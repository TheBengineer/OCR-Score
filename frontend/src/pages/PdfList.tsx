export default function PdfList() {
  return (
    <div className="mx-auto max-w-4xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-surface-900">PDFs</h1>
          <p className="mt-2 text-surface-500">
            Upload, browse, and manage PDF documents for OCR evaluation.
          </p>
        </div>
      </div>
      <div className="mt-8 rounded-xl border border-surface-200 bg-white p-12 text-center shadow-sm">
        <p className="text-surface-400">
          No documents uploaded yet. Use the upload button to get started.
        </p>
      </div>
    </div>
  );
}
