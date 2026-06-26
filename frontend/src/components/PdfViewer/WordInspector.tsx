import { useEffect, useRef, useState, useCallback, type KeyboardEvent } from "react";
import { X, Loader2 } from "lucide-react";
import type { OverlayWord, EngineComparison, OverlayWordStatus } from "@/lib/types";

/* ── Status style palette (matches WordOverlay) ─────────────────────────── */

const STATUS_STYLE: Record<
  OverlayWordStatus,
  { bg: string; text: string; bar: string; label: string }
> = {
  correct: {
    bg: "bg-green-50 text-green-800 border-green-300",
    text: "text-green-700",
    bar: "bg-green-500",
    label: "Correct",
  },
  wrong: {
    bg: "bg-red-50 text-red-800 border-red-300",
    text: "text-red-700",
    bar: "bg-red-500",
    label: "Wrong",
  },
  missing: {
    bg: "bg-blue-50 text-blue-800 border-blue-300",
    text: "text-blue-700",
    bar: "bg-blue-500",
    label: "Missing",
  },
  extra: {
    bg: "bg-orange-50 text-orange-800 border-orange-300",
    text: "text-orange-700",
    bar: "bg-orange-500",
    label: "Extra",
  },
};

/* ── Props ───────────────────────────────────────────────────────────────── */

export interface WordInspectorProps {
  /** The clicked word's overlay data. */
  word: OverlayWord;
  /** Index of the clicked word in the overlay words array. */
  wordIndex: number;
  /** Screen-space coordinates of the click event. */
  position: { x: number; y: number };
  /** Per-engine comparisons (null = loading). */
  comparisons: EngineComparison[] | null;
  /** Whether comparison data is being fetched. */
  loading: boolean;
  /** Called to close the inspector. */
  onClose: () => void;
}

/* ── Confidence bar ─────────────────────────────────────────────────────── */

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(Math.min(value, 1) * 100);
  const barColor =
    value >= 0.9
      ? "bg-green-500"
      : value >= 0.5
        ? "bg-amber-500"
        : "bg-red-500";

  return (
    <div className="flex items-center gap-1.5" aria-label={`${pct}% confidence`}>
      <div className="h-1.5 w-12 overflow-hidden rounded-full bg-surface-200" role="meter" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div
          className={`h-full rounded-full transition-all ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="min-w-[2.5rem] text-right text-[11px] tabular-nums text-surface-500">
        {pct}%
      </span>
    </div>
  );
}

/* ── Status badge ───────────────────────────────────────────────────────── */

function StatusBadge({ status }: { status: OverlayWordStatus }) {
  const cfg = STATUS_STYLE[status];
  return (
    <span
      className={`inline-block rounded-sm border px-1.5 py-0.5 text-[10px] font-medium uppercase leading-tight ${cfg.bg}`}
    >
      {cfg.label}
    </span>
  );
}

/* ── Constants ──────────────────────────────────────────────────────────── */

const POPUP_WIDTH = 300;
const POPUP_GAP = 12;
const MIN_VIEWPORT_MARGIN = 8;

/* ── Component ──────────────────────────────────────────────────────────── */

export function WordInspector({
  word,
  wordIndex: _wordIndex,
  position,
  comparisons,
  loading,
  onClose,
}: WordInspectorProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const [adjustedPos, setAdjustedPos] = useState<{
    left: number;
    top: number;
    /** Which edge the arrow sits on. */
    arrowEdge: "left" | "right";
    /** Arrow vertical offset from popup top (px). */
    arrowTop: number;
  }>({ left: 0, top: 0, arrowEdge: "left", arrowTop: 16 });

  /* ── Measure popup and adjust position to stay in viewport ── */
  useEffect(() => {
    const panel = panelRef.current;
    if (!panel) return;

    const pw = panel.offsetWidth || POPUP_WIDTH;
    const ph = panel.offsetHeight || 240;
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    /* Try right first, then left */
    let left = position.x + POPUP_GAP;
    let arrowEdge: "left" | "right" = "left";

    if (left + pw > vw - MIN_VIEWPORT_MARGIN) {
      left = position.x - pw - POPUP_GAP;
      arrowEdge = "right";
    }

    /* Bounce off left edge */
    if (left < MIN_VIEWPORT_MARGIN) {
      left = MIN_VIEWPORT_MARGIN;
    }

    /* Vertical: center on click Y */
    let top = position.y - ph / 2;
    top = Math.max(MIN_VIEWPORT_MARGIN, Math.min(top, vh - ph - MIN_VIEWPORT_MARGIN));

    /* Arrow vertically aligns with click Y relative to popup */
    const arrowTop = Math.max(8, Math.min(position.y - top - 6, ph - 20));

    setAdjustedPos({ left, top, arrowEdge, arrowTop });
  }, [position.x, position.y]);

  /* ── Escape key closes ── */
  useEffect(() => {
    const handleKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onClose]);

  /* ── Keyboard row navigation ── */
  const handleRowKeyDown = useCallback(
    (e: KeyboardEvent<HTMLElement>, index: number) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        const next = panelRef.current?.querySelector<HTMLButtonElement>(
          `[data-row-index="${index + 1}"]`,
        );
        next?.focus();
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        const prev = panelRef.current?.querySelector<HTMLButtonElement>(
          `[data-row-index="${index - 1}"]`,
        );
        prev?.focus();
      }
    },
    [],
  );

  return (
    <>
      {/* ── Backdrop ── */}
      <div
        className="fixed inset-0 z-40"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* ── Popup panel ── */}
      <div
        ref={panelRef}
        role="dialog"
        aria-label={`Inspector for "${word.text}"`}
        aria-modal="true"
        className="fixed z-50 max-h-80 w-[300px] overflow-hidden rounded-lg border border-surface-200 bg-white shadow-xl"
        style={{ left: adjustedPos.left, top: adjustedPos.top }}
      >
        {/* ── Arrow ── */}
        <div
          className="pointer-events-none absolute h-2.5 w-2.5 rotate-45 border-surface-200 bg-white"
          style={{
            [adjustedPos.arrowEdge === "left" ? "left" : "right"]: -5,
            top: adjustedPos.arrowTop,
            ...(adjustedPos.arrowEdge === "left"
              ? { borderLeft: "1px solid", borderBottom: "1px solid" }
              : { borderRight: "1px solid", borderTop: "1px solid" }),
          }}
        />

        {/* ── Header ── */}
        <div className="flex items-center justify-between border-b border-surface-100 px-3 py-2">
          <h3 className="text-xs font-semibold text-surface-700">
            Word Inspector
          </h3>
          <button
            type="button"
            onClick={onClose}
            className="rounded p-0.5 text-surface-400 hover:bg-surface-100 hover:text-surface-600"
            aria-label="Close inspector"
          >
            <X size={14} />
          </button>
        </div>

        {/* ── Word info ── */}
        <div className="border-b border-surface-100 px-3 py-2">
          <p className="text-xs text-surface-500">
            Ground truth:{" "}
            <span className="font-medium text-surface-800">
              &ldquo;{word.text}&rdquo;
            </span>
          </p>
          <p className="text-[10px] text-surface-400">
            Word index: {_wordIndex}
            {word.engineText && (
              <>
                {" · "}OCR: &ldquo;{word.engineText}&rdquo;
              </>
            )}
          </p>
        </div>

        {/* ── Body — loading, error, empty, or table ── */}
        <div className="overflow-y-auto">
          {loading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 size={18} className="animate-spin text-surface-400" />
              <span className="ml-2 text-xs text-surface-500">
                Loading comparison data&hellip;
              </span>
            </div>
          )}

          {!loading && comparisons && comparisons.length === 0 && (
            <div className="flex items-center justify-center py-8">
              <p className="text-xs text-surface-400">
                No engine data available for this word.
              </p>
            </div>
          )}

          {!loading && !comparisons && (
            <div className="flex items-center justify-center py-8">
              <p className="text-xs text-red-500">
                Failed to load comparison data.
              </p>
            </div>
          )}

          {!loading && comparisons && comparisons.length > 0 && (
            <table className="w-full table-auto text-xs">
              <thead>
                <tr className="border-b border-surface-100 text-[10px] font-medium uppercase tracking-wider text-surface-400">
                  <th className="px-3 py-1.5 text-left">Engine</th>
                  <th className="px-2 py-1.5 text-left">Text</th>
                  <th className="px-2 py-1.5 text-right">Confidence</th>
                  <th className="px-2 py-1.5 text-right">Status</th>
                </tr>
              </thead>
              <tbody>
                {comparisons.map((comp, idx) => (
                  <tr
                    key={comp.engineSlug}
                    data-row-index={idx}
                    tabIndex={0}
                    role="button"
                    aria-label={`${comp.engineName}: "${comp.text}", confidence ${(comp.confidence * 100).toFixed(0)}%, ${STATUS_STYLE[comp.status].label}`}
                    className={`cursor-default border-b border-surface-50 text-left transition-colors last:border-0 focus:bg-surface-50 focus:outline-none ${
                      idx % 2 === 0 ? "bg-white" : "bg-surface-50/50"
                    }`}
                    onKeyDown={(e) => handleRowKeyDown(e, idx)}
                  >
                    <td className="px-3 py-1.5 font-medium text-surface-700">
                      {comp.engineName}
                    </td>
                    <td className="max-w-[80px] truncate px-2 py-1.5 font-mono text-surface-800">
                      &ldquo;{comp.text}&rdquo;
                    </td>
                    <td className="px-2 py-1.5 text-right">
                      <ConfidenceBar value={comp.confidence} />
                    </td>
                    <td className="px-2 py-1.5 text-right">
                      <StatusBadge status={comp.status} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </>
  );
}
