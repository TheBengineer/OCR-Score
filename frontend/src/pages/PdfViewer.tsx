import { useParams } from "react-router-dom";
import { PdfViewer } from "@/components/PdfViewer";
import { useCallback } from "react";

/**
 * Sample PDF for development/testing when no document ID is available.
 * Mozilla's compressed.tracemonkey-pldi-09.pdf — a standard PDF.js test file.
 */
const DEV_SAMPLE_PDF =
  "https://raw.githubusercontent.com/mozilla/pdf.js/ba2edeae/web/compressed.tracemonkey-pldi-09.pdf";

export default function PdfViewerPage() {
  const { id } = useParams<{ id: string }>();

  const pdfUrl = id
    ? `/api/v1/documents/${id}/file`
    : DEV_SAMPLE_PDF;

  const handlePageChange = useCallback((pageNumber: number) => {
    console.info("Page changed to", pageNumber);
  }, []);

  return (
    <div className="mx-auto max-w-6xl">
      <h1 className="text-3xl font-bold text-surface-900">PDF Viewer</h1>
      <p className="mt-2 text-surface-500">
        View OCR results overlaid on the original document.
      </p>

      <div className="mt-6">
        <PdfViewer
          pdfUrl={pdfUrl}
          onPageChange={handlePageChange}
        />
      </div>
    </div>
  );
}
