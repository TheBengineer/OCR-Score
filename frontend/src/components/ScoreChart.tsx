import { useCallback, useRef, useState } from "react";

/* ── Types ─────────────────────────────────────────────────────────────── */

export interface PageScore {
  page: number;
  cer: number;
  wer: number;
  char_f1: number;
  word_f1: number;
}

interface ScoreChartProps {
  pages: PageScore[];
  /** Called when a bar is clicked — receives the page number. */
  onPageClick?: (page: number) => void;
  /** Height in pixels (default 240). */
  height?: number;
}

/* ── Helpers ───────────────────────────────────────────────────────────── */

const BAR_MIN_HEIGHT = 2; // px — so zero-CER bars are still visible
const TOOLTIP_WIDTH = 140;

function cerColor(cer: number): string {
  if (cer < 0.05) return "#22c55e"; // green-500
  if (cer < 0.15) return "#eab308"; // yellow-500
  return "#ef4444"; // red-500
}

function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/* ── Component ─────────────────────────────────────────────────────────── */

export default function ScoreChart({
  pages,
  onPageClick,
  height = 240,
}: ScoreChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{
    page: number;
    cer: number;
    wer: number;
    x: number;
    y: number;
  } | null>(null);

  if (!pages.length) {
    return (
      <div
        className="flex items-center justify-center rounded-lg border border-dashed border-surface-300 bg-surface-50 text-sm text-surface-400"
        style={{ height }}
      >
        No page scores available
      </div>
    );
  }

  const maxCer = Math.max(...pages.map((p) => p.cer), 0.01);
  const chartBottom = 24;
  const chartTop = 8;
  const chartHeight = height - chartBottom - chartTop;
  const barWidth = Math.max(8, Math.min(40, (containerRef.current?.clientWidth ?? 600) / pages.length - 4));
  const gap = 4;

  return (
    <div
      ref={containerRef}
      className="relative w-full select-none"
      style={{ height }}
      onMouseLeave={() => setTooltip(null)}
    >
      <svg
        width="100%"
        height={height}
        viewBox={`0 0 ${pages.length * (barWidth + gap) + gap * 2} ${height}`}
        className="overflow-visible"
      >
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((line) => (
          <g key={line}>
            <line
              x1={0}
              y1={chartTop + chartHeight * (1 - line / maxCer)}
              x2={pages.length * (barWidth + gap) + gap * 2}
              y2={chartTop + chartHeight * (1 - line / maxCer)}
              stroke="#e2e8f0"
              strokeWidth={1}
            />
            <text
              x={0}
              y={chartTop + chartHeight * (1 - line / maxCer) - 2}
              fill="#94a3b8"
              fontSize={10}
              textAnchor="start"
            >
              {(line * 100).toFixed(0)}%
            </text>
          </g>
        ))}

        {/* Bars */}
        {pages.map((ps, idx) => {
          const x = gap * 2 + idx * (barWidth + gap);
          const barH = Math.max(
            BAR_MIN_HEIGHT,
            (ps.cer / maxCer) * chartHeight,
          );
          const y = chartTop + chartHeight - barH;

          return (
            <g key={ps.page}>
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={barH}
                rx={3}
                ry={3}
                fill={cerColor(ps.cer)}
                opacity={0.85}
                className="cursor-pointer transition-opacity hover:opacity-100"
                onClick={() => onPageClick?.(ps.page)}
                onMouseEnter={(e) => {
                  const rect = (
                    e.currentTarget as SVGRectElement
                  ).getBoundingClientRect();
                  const parentRect = containerRef.current?.getBoundingClientRect();
                  setTooltip({
                    page: ps.page,
                    cer: ps.cer,
                    wer: ps.wer,
                    x: rect.left - (parentRect?.left ?? 0) + barWidth / 2,
                    y: rect.top - (parentRect?.top ?? 0) - 4,
                  });
                }}
              />
              {/* Page number label */}
              <text
                x={x + barWidth / 2}
                y={height - 4}
                fill="#64748b"
                fontSize={9}
                textAnchor="middle"
              >
                {ps.page}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Tooltip */}
      {tooltip && (
        <div
          className="pointer-events-none absolute z-10 rounded-lg border border-surface-200 bg-white px-3 py-2 text-xs shadow-lg"
          style={{
            left: Math.min(
              tooltip.x - TOOLTIP_WIDTH / 2,
              (containerRef.current?.clientWidth ?? 600) - TOOLTIP_WIDTH - 8,
            ),
            top: Math.max(tooltip.y - 80, 0),
            width: TOOLTIP_WIDTH,
          }}
        >
          <p className="mb-1 font-semibold text-surface-700">
            Page {tooltip.page}
          </p>
          <div className="flex items-center justify-between gap-2 text-surface-600">
            <span>CER</span>
            <span className="font-mono font-medium">
              {formatPct(tooltip.cer)}
            </span>
          </div>
          <div className="flex items-center justify-between gap-2 text-surface-600">
            <span>WER</span>
            <span className="font-mono font-medium">
              {formatPct(tooltip.wer)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
