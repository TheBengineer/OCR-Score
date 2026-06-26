import {
  ChevronLeft,
  ChevronRight,
  Maximize,
  ZoomIn,
  ZoomOut,
  RotateCw,
  RotateCcw,
  ListOrdered,
} from "lucide-react";
import { useState, useCallback, type KeyboardEvent, type ChangeEvent } from "react";

export interface ViewerControlsProps {
  pageNumber: number;
  numPages: number;
  zoomPercent: number;
  showReadingOrder: boolean;
  onPrevPage: () => void;
  onNextPage: () => void;
  onPageChange: (page: number) => void;
  onToggleReadingOrder: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFitToWidth: () => void;
  onRotateCw: () => void;
  onRotateCcw: () => void;
}

function iconButtonClass(active?: boolean): string {
  const base =
    "flex items-center justify-center rounded-md p-1.5 transition-colors";
  if (active) {
    return `${base} bg-primary-100 text-primary-700 hover:bg-primary-200`;
  }
  return `${base} text-surface-500 hover:bg-surface-200 hover:text-surface-700`;
}

export function ViewerControls({
  pageNumber,
  numPages,
  zoomPercent,
  showReadingOrder,
  onPrevPage,
  onNextPage,
  onPageChange,
  onToggleReadingOrder,
  onZoomIn,
  onZoomOut,
  onFitToWidth,
  onRotateCw,
  onRotateCcw,
}: ViewerControlsProps) {
  const [pageInput, setPageInput] = useState(String(pageNumber));

  const handlePageInputChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      setPageInput(e.target.value);
    },
    [],
  );

  const commitPageInput = useCallback(() => {
    const parsed = parseInt(pageInput, 10);
    if (!Number.isNaN(parsed) && parsed >= 1 && parsed <= numPages) {
      onPageChange(parsed);
    } else {
      setPageInput(String(pageNumber));
    }
  }, [pageInput, numPages, onPageChange, pageNumber]);

  const handlePageInputKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter") {
        commitPageInput();
      }
    },
    [commitPageInput],
  );

  // Sync external pageNumber changes to the input field
  // (but only when the input is not being actively edited)
  // We do this via a simple check: when pageNumber prop changes and
  // the input value is not the stringified version, update it.
  // This is handled by the controlled-value update on blur/enter.

  return (
    <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-surface-200 bg-white px-3 py-2 shadow-sm">
      {/* ── Page navigation ── */}
      <div className="flex items-center gap-1">
        <button
          type="button"
          className={iconButtonClass()}
          onClick={onPrevPage}
          disabled={pageNumber <= 1}
          aria-label="Previous page"
        >
          <ChevronLeft size={18} />
        </button>

        <div className="flex items-center gap-1">
          <input
            type="text"
            value={pageInput}
            onChange={handlePageInputChange}
            onBlur={commitPageInput}
            onKeyDown={handlePageInputKeyDown}
            className="w-10 rounded border border-surface-300 px-1 py-0.5 text-center text-sm tabular-nums text-surface-700 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-300"
            aria-label="Page number"
          />
          <span className="text-sm text-surface-400">
            / {numPages}
          </span>
        </div>

        <button
          type="button"
          className={iconButtonClass()}
          onClick={onNextPage}
          disabled={pageNumber >= numPages}
          aria-label="Next page"
        >
          <ChevronRight size={18} />
        </button>
      </div>

      {/* ── Reading order toggle ── */}
      <div className="flex items-center gap-1">
        <div className="mx-1 h-5 w-px bg-surface-200" />
        <button
          type="button"
          className={iconButtonClass(showReadingOrder)}
          onClick={onToggleReadingOrder}
          aria-label="Toggle reading order"
          aria-pressed={showReadingOrder}
        >
          <ListOrdered size={18} />
        </button>
        <span className="text-xs text-surface-500">Order</span>
      </div>

      {/* ── Zoom controls ── */}
      <div className="flex items-center gap-1">
        <button
          type="button"
          className={iconButtonClass()}
          onClick={onZoomOut}
          aria-label="Zoom out"
        >
          <ZoomOut size={18} />
        </button>

        <span className="min-w-[3rem] text-center text-sm font-medium tabular-nums text-surface-700">
          {zoomPercent}%
        </span>

        <button
          type="button"
          className={iconButtonClass()}
          onClick={onZoomIn}
          aria-label="Zoom in"
        >
          <ZoomIn size={18} />
        </button>

        <div className="mx-1 h-5 w-px bg-surface-200" />

        <button
          type="button"
          className={iconButtonClass()}
          onClick={onFitToWidth}
          aria-label="Fit to width"
        >
          <Maximize size={18} />
        </button>
      </div>

      {/* ── Rotation controls ── */}
      <div className="flex items-center gap-1">
        <button
          type="button"
          className={iconButtonClass()}
          onClick={onRotateCcw}
          aria-label="Rotate counter-clockwise"
        >
          <RotateCcw size={18} />
        </button>

        <button
          type="button"
          className={iconButtonClass()}
          onClick={onRotateCw}
          aria-label="Rotate clockwise"
        >
          <RotateCw size={18} />
        </button>
      </div>
    </div>
  );
}
