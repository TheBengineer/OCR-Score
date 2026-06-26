import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Upload, Search, Trash2, Play, CheckSquare, Square, FileText } from "lucide-react";
import {
  deleteDocument,
  listDocuments,
  uploadDocument,
} from "../lib/api.ts";
import type {
  BatchProgressResponse,
  Document,
} from "../lib/types.ts";
import BatchDialog from "../components/BatchDialog.tsx";

// ── PDF status config ────────────────────────────────────────────────────

const STATUS_STYLES: Record<string, { bg: string; text: string }> = {
  uploading: { bg: "bg-amber-100", text: "text-amber-700" },
  uploaded: { bg: "bg-blue-100", text: "text-blue-700" },
  processing: { bg: "bg-purple-100", text: "text-purple-700" },
  ready: { bg: "bg-emerald-100", text: "text-emerald-700" },
  error: { bg: "bg-red-100", text: "text-red-700" },
  deleted: { bg: "bg-surface-100", text: "text-surface-500" },
};

function StatusBadge({ status }: { status: string }) {
  const style = STATUS_STYLES[status] ?? { bg: "bg-surface-100", text: "text-surface-600" };
  return (
    <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${style.bg} ${style.text}`}>
      {status}
    </span>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// ── Main component ────────────────────────────────────────────────────────

export default function PdfList() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [uploading, setUploading] = useState(false);
  const [showBatchDialog, setShowBatchDialog] = useState(false);
  const [batchProgress, setBatchProgress] = useState<BatchProgressResponse | null>(null);
  const [pollTimer, setPollTimer] = useState<ReturnType<typeof setInterval> | null>(null);

  // Load documents
  const loadDocs = useCallback(async () => {
    try {
      const resp = await listDocuments();
      setDocuments(resp.items);
    } catch {
      // silently fail
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDocs();
  }, [loadDocs]);

  // Upload handler
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      await uploadDocument(file);
      await loadDocs();
    } catch {
      // silently fail
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  // Delete handler
  const handleDelete = async (id: string) => {
    try {
      await deleteDocument(id);
      setDocuments((prev) => prev.filter((d) => d.id !== id));
      setSelectedIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    } catch {
      // silently fail
    }
  };

  // Selection
  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleSelectAll = () => {
    if (selectedIds.size === filtered.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(filtered.map((d) => d.id)));
    }
  };

  const clearSelection = () => setSelectedIds(new Set());

  // Filter
  const filtered = documents.filter(
    (d) =>
      d.original_filename.toLowerCase().includes(search.toLowerCase()) ||
      d.filename.toLowerCase().includes(search.toLowerCase()),
  );

  // Batch completion polling
  const handleBatchCreated = (batchId: string) => {
    setShowBatchDialog(true);
    const timer = setInterval(async () => {
      try {
        const { getBatchProgress } = await import("../lib/api.ts");
        const progress = await getBatchProgress(batchId);
        setBatchProgress(progress);
        if (progress.status === "completed" || progress.status === "failed") {
          clearInterval(timer);
          setPollTimer(null);
          await loadDocs();
        }
      } catch {
        clearInterval(timer);
        setPollTimer(null);
      }
    }, 2000);
    setPollTimer(timer);
  };

  const handleCloseBatchDialog = () => {
    setShowBatchDialog(false);
    setBatchProgress(null);
    if (pollTimer) {
      clearInterval(pollTimer);
      setPollTimer(null);
    }
  };

  // Cleanup poll timer on unmount
  useEffect(() => {
    return () => {
      if (pollTimer) clearInterval(pollTimer);
    };
  }, [pollTimer]);

  return (
    <div className="mx-auto max-w-6xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-surface-900">PDFs</h1>
          <p className="mt-2 text-surface-500">
            Upload, browse, and manage PDF documents for OCR evaluation.
          </p>
        </div>
        <div className="flex items-center gap-3">
          {selectedIds.size > 0 && (
            <button
              type="button"
              onClick={() => setShowBatchDialog(true)}
              className="inline-flex items-center gap-2 rounded-lg bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700 transition-colors"
            >
              <Play className="h-4 w-4" />
              Batch Process ({selectedIds.size})
            </button>
          )}
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="inline-flex items-center gap-2 rounded-lg border border-surface-300 bg-white px-4 py-2 text-sm font-medium text-surface-700 hover:bg-surface-50 transition-colors disabled:opacity-50"
          >
            <Upload className="h-4 w-4" />
            {uploading ? "Uploading..." : "Upload"}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            onChange={handleUpload}
            className="hidden"
          />
        </div>
      </div>

      {/* Search & filter bar */}
      <div className="mt-6 flex items-center gap-4">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-surface-400" />
          <input
            type="text"
            placeholder="Search by filename..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full rounded-lg border border-surface-300 bg-white py-2 pl-10 pr-4 text-sm text-surface-900 placeholder-surface-400 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
          />
        </div>
        {selectedIds.size > 0 && (
          <button
            type="button"
            onClick={clearSelection}
            className="text-sm text-surface-500 hover:text-surface-700"
          >
            Clear selection
          </button>
        )}
      </div>

      {/* Content */}
      {loading ? (
        <div className="mt-8 flex items-center justify-center py-20">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-surface-200 border-t-primary-600" />
        </div>
      ) : documents.length === 0 ? (
        <div className="mt-8 rounded-xl border border-dashed border-surface-300 bg-surface-50 p-16 text-center">
          <FileText className="mx-auto h-12 w-12 text-surface-300" />
          <p className="mt-4 text-surface-500">
            No documents uploaded yet. Click "Upload" to get started.
          </p>
        </div>
      ) : filtered.length === 0 ? (
        <div className="mt-8 rounded-xl border border-surface-200 bg-white p-12 text-center shadow-sm">
          <p className="text-surface-400">No documents match your search.</p>
        </div>
      ) : (
        <div className="mt-6 overflow-hidden rounded-xl border border-surface-200 bg-white shadow-sm">
          <table className="min-w-full divide-y divide-surface-200">
            <thead className="bg-surface-50">
              <tr>
                <th className="w-10 px-4 py-3 text-left">
                  <button
                    type="button"
                    onClick={toggleSelectAll}
                    className="text-surface-400 hover:text-surface-600"
                  >
                    {selectedIds.size === filtered.length ? (
                      <CheckSquare className="h-4 w-4" />
                    ) : (
                      <Square className="h-4 w-4" />
                    )}
                  </button>
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-surface-500">
                  Filename
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-surface-500">
                  Status
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-surface-500">
                  Pages
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-surface-500">
                  Size
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-surface-500">
                  Uploaded
                </th>
                <th className="w-20 px-4 py-3" />
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-100">
              {filtered.map((doc) => (
                <tr
                  key={doc.id}
                  className="group transition-colors hover:bg-surface-50 cursor-pointer"
                  onClick={() => navigate(`/pdfs/${doc.id}`)}
                >
                  <td
                    className="px-4 py-3"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <button
                      type="button"
                      onClick={() => toggleSelect(doc.id)}
                      className="text-surface-400 hover:text-surface-600"
                    >
                      {selectedIds.has(doc.id) ? (
                        <CheckSquare className="h-4 w-4 text-primary-600" />
                      ) : (
                        <Square className="h-4 w-4" />
                      )}
                    </button>
                  </td>
                  <td className="max-w-xs truncate px-4 py-3 text-sm font-medium text-surface-900">
                    {doc.original_filename}
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={doc.status} />
                  </td>
                  <td className="px-4 py-3 text-sm text-surface-600">
                    {doc.page_count}
                  </td>
                  <td className="px-4 py-3 text-sm text-surface-600">
                    {formatSize(doc.file_size_bytes)}
                  </td>
                  <td className="px-4 py-3 text-sm text-surface-500">
                    {formatDate(doc.upload_timestamp)}
                  </td>
                  <td
                    className="px-4 py-3 text-right"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <button
                      type="button"
                      onClick={() => handleDelete(doc.id)}
                      className="rounded p-1.5 text-surface-400 opacity-0 transition-all hover:bg-red-50 hover:text-red-600 group-hover:opacity-100"
                      title="Delete document"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Footer stats */}
          <div className="border-t border-surface-200 px-4 py-3 text-xs text-surface-400">
            {filtered.length} of {documents.length} document{documents.length !== 1 ? "s" : ""}
            {selectedIds.size > 0 && (
              <span> &middot; {selectedIds.size} selected</span>
            )}
          </div>
        </div>
      )}

      {/* Batch processing dialog */}
      {showBatchDialog && (
        <BatchDialog
          selectedPdfIds={Array.from(selectedIds)}
          onClose={handleCloseBatchDialog}
          onBatchCreated={handleBatchCreated}
          progress={batchProgress}
        />
      )}
    </div>
  );
}
