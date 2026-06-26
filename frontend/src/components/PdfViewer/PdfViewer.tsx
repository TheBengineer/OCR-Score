import {
  useState,
  useCallback,
  useRef,
  useEffect,
  createContext,
  useContext,
  type ReactNode,
} from "react";
import { Document, Page, pdfjs } from "react-pdf";
import { ViewerControls } from "./ViewerControls";
import { OverlayLayer } from "./OverlayLayer";

/* ── PdfViewerContext ──────────────────────────────────────────────────────
 * Lets overlay components access the current render state (scale,
 * dimensions, rotation) without drilling props through intermediate
 * consumers. */

export interface PdfViewerRenderState {
  pageWidthPts: number;
  pageHeightPts: number;
  scale: number;
  rotation: number;
}

export const PdfViewerContext = createContext<PdfViewerRenderState>({
  pageWidthPts: 0,
  pageHeightPts: 0,
  scale: 1,
  rotation: 0,
});

export function usePdfViewerState(): PdfViewerRenderState {
  return useContext(PdfViewerContext);
}

/* ── PDF.js worker setup ───────────────────────────────────────────────────
 * Must be set in the same module where <Document> / <Page> are used.
 * Vite transforms `new URL(…, import.meta.url)` into a properly resolved
 * static asset URL at both dev and build time. */
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

/* ── Local types (avoid pulling pdfjs-dist peer dep types) ──────────────── */

interface PDFPageViewport {
  width: number;
  height: number;
}

interface PDFPage {
  getViewport(options: { scale: number; rotation: number }): PDFPageViewport;
}

interface PDFDocument {
  numPages: number;
  getPage(pageNumber: number): Promise<PDFPage>;
}

/* ── Constants ──────────────────────────────────────────────────────────── */

const MIN_ZOOM = 0.1;
const MAX_ZOOM = 5;
const ZOOM_STEP = 1.25;

/* ── Props ──────────────────────────────────────────────────────────────── */

export interface PdfViewerProps {
  /** URL (or null/undefined for empty state) of the PDF to display. */
  pdfUrl: string | undefined | null;
  /** Called when the user navigates to a different page. */
  onPageChange?: (pageNumber: number) => void;
  /** Page to show on first render (default: 1). */
  initialPage?: number;
  /** Whether reading order overlay is visible. */
  showReadingOrder?: boolean;
  /** Called when reading order toggle is clicked. */
  onToggleReadingOrder?: () => void;
  /** Optional overlay content rendered inside OverlayLayer. */
  children?: ReactNode;
}

/* ── Component ──────────────────────────────────────────────────────────── */

export function PdfViewer({
  pdfUrl,
  onPageChange,
  initialPage = 1,
  showReadingOrder = false,
  onToggleReadingOrder,
  children,
}: PdfViewerProps) {
  /* ── State ── */
  const [numPages, setNumPages] = useState<number | null>(null);
  const [pageNumber, setPageNumber] = useState(initialPage);
  const [zoom, setZoom] = useState(1); // 1 = fit-to-container-width
  const [rotation, setRotation] = useState(0);
  const [pageWidthPts, setPageWidthPts] = useState(0);
  const [pageHeightPts, setPageHeightPts] = useState(0);
  const [containerWidth, setContainerWidth] = useState(0);
  const [error, setError] = useState<Error | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  /* ── ResizeObserver for responsive width ── */
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const width =
          entry.contentBoxSize[0]?.inlineSize ?? entry.contentRect.width;
        setContainerWidth(width);
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  /* ── PDF document callbacks ── */
  const handleDocLoadSuccess = useCallback(
    async (pdf: PDFDocument) => {
      setNumPages(pdf.numPages);
      setError(null);

      try {
        const page = await pdf.getPage(1);
        const viewport = page.getViewport({ scale: 1, rotation: 0 });
        setPageWidthPts(viewport.width);
        setPageHeightPts(viewport.height);
      } catch {
        // Page dimensions unavailable — viewer still works with default scale
      }
    },
    [],
  );

  const handleDocLoadError = useCallback((err: Error) => {
    setError(err);
  }, []);

  /* ── Computed scale ──
   *   zoom=1  → page fills container width (fit-to-width)
   *   zoom>1  → page is magnified relative to container
   *   zoom<1  → page is shrunk relative to container */
  const pageScale =
    containerWidth > 0 && pageWidthPts > 0
      ? (containerWidth / pageWidthPts) * zoom
      : zoom;

  const renderedWidth = pageWidthPts * pageScale;
  const renderedHeight = pageHeightPts * pageScale;

  /* ── Page navigation ── */
  const goToPage = useCallback(
    (page: number) => {
      const clamped = Math.max(1, Math.min(page, numPages ?? 1));
      setPageNumber(clamped);
      onPageChange?.(clamped);
    },
    [numPages, onPageChange],
  );

  const goToPrevPage = useCallback(() => {
    goToPage(pageNumber - 1);
  }, [pageNumber, goToPage]);

  const goToNextPage = useCallback(() => {
    goToPage(pageNumber + 1);
  }, [pageNumber, goToPage]);

  /* ── Zoom ── */
  const zoomIn = useCallback(() => {
    setZoom((prev) => Math.min(prev * ZOOM_STEP, MAX_ZOOM));
  }, []);

  const zoomOut = useCallback(() => {
    setZoom((prev) => Math.max(prev / ZOOM_STEP, MIN_ZOOM));
  }, []);

  const fitToWidth = useCallback(() => {
    setZoom(1);
  }, []);

  /* ── Rotation ── */
  const rotateCw = useCallback(() => {
    setRotation((prev) => (prev + 90) % 360);
  }, []);

  const rotateCcw = useCallback(() => {
    setRotation((prev) => (prev - 90 + 360) % 360);
  }, []);

  /* ── Render helpers ── */

  const zoomPercent = Math.round(zoom * 100);

  const spinner = (
    <div className="flex items-center justify-center p-12">
      <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary-200 border-t-primary-500" />
    </div>
  );

  const docErrorEl = (
    <div className="flex flex-col items-center gap-2 p-12">
      <p className="text-sm font-medium text-red-600">Failed to load PDF</p>
      <p className="max-w-md text-center text-sm text-surface-400">
        {error?.message ?? "An unknown error occurred. Please try again."}
      </p>
    </div>
  );

  const pageLoadingEl = (
    <div
      className="flex items-center justify-center"
      style={{
        width: containerWidth || 600,
        height: containerWidth ? containerWidth * 1.4 : 800,
      }}
    >
      <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary-200 border-t-primary-500" />
    </div>
  );

  /* ── Render ── */
  return (
    <div className="flex flex-col gap-3">
      <ViewerControls
        pageNumber={pageNumber}
        numPages={numPages ?? 0}
        zoomPercent={zoomPercent}
        showReadingOrder={showReadingOrder}
        onPrevPage={goToPrevPage}
        onNextPage={goToNextPage}
        onPageChange={goToPage}
        onToggleReadingOrder={onToggleReadingOrder ?? (() => {})}
        onZoomIn={zoomIn}
        onZoomOut={zoomOut}
        onFitToWidth={fitToWidth}
        onRotateCw={rotateCw}
        onRotateCcw={rotateCcw}
      />

      <div
        ref={containerRef}
        className="flex min-h-[300px] justify-center overflow-auto rounded-lg border border-surface-200 bg-surface-100"
      >
        {/* ── Empty state (no PDF URL) ── */}
        {!pdfUrl && (
          <div className="flex items-center justify-center p-12">
            <div className="text-center">
              <p className="text-sm font-medium text-surface-400">
                No PDF selected
              </p>
              <p className="mt-1 text-xs text-surface-300">
                Upload or select a document to view it here.
              </p>
            </div>
          </div>
        )}

        {/* ── Document / Viewer ── */}
        {pdfUrl && (
          <div className="relative">
            <Document
              file={pdfUrl}
              onLoadSuccess={handleDocLoadSuccess}
              onLoadError={handleDocLoadError}
              loading={spinner}
              error={docErrorEl}
            >
              {pageWidthPts > 0 && (
                <div className="relative shadow-lg">
                  <Page
                    pageNumber={pageNumber}
                    scale={pageScale}
                    rotate={rotation}
                    renderTextLayer={false}
                    renderAnnotationLayer={false}
                    loading={pageLoadingEl}
                  />

                  {renderedWidth > 0 && renderedHeight > 0 && (
                    <PdfViewerContext.Provider
                      value={{
                        pageWidthPts,
                        pageHeightPts,
                        scale: pageScale,
                        rotation,
                      }}
                    >
                      <OverlayLayer
                        pageWidth={pageWidthPts}
                        pageHeight={pageHeightPts}
                        scale={pageScale}
                        rotation={rotation}
                      >
                        {children}
                      </OverlayLayer>
                    </PdfViewerContext.Provider>
                  )}
                </div>
              )}
            </Document>
          </div>
        )}
      </div>
    </div>
  );
}
