import { useRef, useCallback, useState, useEffect } from "react";
import type { GeometricHeatmapData, HeatmapBin } from "../lib/types.ts";

/* ── Props ──────────────────────────────────────────────────────────────── */

export interface GeometricHeatmapProps {
  data: GeometricHeatmapData;
  width?: number;
  height?: number;
}

/* ── Constants ──────────────────────────────────────────────────────────── */

const PAGE_OUTLINE_COLOR = "#cbd5e1";
const PAGE_OUTLINE_WIDTH = 2;
const GRID_LINE_COLOR = "rgba(148, 163, 184, 0.25)";
const GRID_LINE_WIDTH = 0.5;
const MARGIN = 16;

/* ── Color helpers ──────────────────────────────────────────────────────── */

function errorColor(errorRate: number): string {
  if (errorRate < 0.05) return "#22c55e"; // green-500
  if (errorRate < 0.15) return "#eab308"; // yellow-500
  return "#ef4444"; // red-500
}

function errorLabel(errorRate: number): string {
  if (errorRate < 0.05) return "low";
  if (errorRate < 0.15) return "medium";
  return "high";
}

/* ── Hatching for colour-blind accessibility ─────────────────────────────── */

function drawBinHatching(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  errorRate: number,
) {
  /* Only hatch medium and high bins. */
  if (errorRate < 0.05) return;

  ctx.save();
  ctx.strokeStyle = `rgba(0, 0, 0, ${Math.min(0.15 + errorRate * 0.5, 0.55)})`;
  ctx.lineWidth = 1;

  if (errorRate < 0.15) {
    /* Diagonal lines for medium. */
    const step = Math.max(3, Math.round(10 - errorRate * 20));
    ctx.beginPath();
    for (let i = -w; i < w + h; i += step) {
      ctx.moveTo(x + i, y);
      ctx.lineTo(x + i + h, y + h);
    }
    ctx.stroke();
  } else {
    /* Cross-hatch for high. */
    const step = Math.max(3, Math.round(10 - errorRate * 10));
    ctx.beginPath();
    for (let i = -w; i < w + h; i += step) {
      ctx.moveTo(x + i, y);
      ctx.lineTo(x + i + h, y + h);
      ctx.moveTo(x + i + h, y);
      ctx.lineTo(x + i, y + h);
    }
    ctx.stroke();
  }

  ctx.restore();
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */

/** Build a grid array from sparse bins. */
function buildGrid(data: GeometricHeatmapData): (HeatmapBin | null)[][] {
  const grid: (HeatmapBin | null)[][] = [];
  for (let r = 0; r < data.gridRows; r++) {
    grid.push(new Array(data.gridCols).fill(null));
  }
  for (const bin of data.bins) {
    if (bin.row >= 0 && bin.row < data.gridRows && bin.col >= 0 && bin.col < data.gridCols) {
      grid[bin.row]![bin.col] = bin;
    }
  }
  return grid;
}

/* ── Component ───────────────────────────────────────────────────────────── */

export default function GeometricHeatmap({
  data,
  width = 500,
  height = 650,
}: GeometricHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{
    errorRate: number;
    sampleCount: number;
    row: number;
    col: number;
    x: number;
    y: number;
  } | null>(null);

  const grid = buildGrid(data);

  /* Figure out the rendering area so the page aspect ratio is preserved. */
  const pageAspect = data.pageHeight > 0 && data.pageWidth > 0
    ? data.pageHeight / data.pageWidth
    : 1.4; /* default letter-ish */

  const availW = width - MARGIN * 2;
  const availH = height - MARGIN * 2;
  let renderW: number;
  let renderH: number;
  if (availW / availH > 1 / pageAspect) {
    renderH = availH;
    renderW = renderH / pageAspect;
  } else {
    renderW = availW;
    renderH = renderW * pageAspect;
  }

  const offX = (width - renderW) / 2;
  const offY = (height - renderH) / 2;

  const cellW = renderW / data.gridCols;
  const cellH = renderH / data.gridRows;

  /* ── Render ────────────────────────────────────────────────────────── */
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const bw = Math.ceil(width * dpr);
    const bh = Math.ceil(height * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }

    ctx.save();
    ctx.scale(dpr, dpr);

    /* Background. */
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    if (data.gridRows === 0 || data.gridCols === 0) {
      ctx.fillStyle = "#94a3b8";
      ctx.font = "14px ui-sans-serif, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("No spatial error data available", width / 2, height / 2);
      ctx.restore();
      return;
    }

    /* ── Draw page outline ──────────────────────────────────────────── */
    ctx.save();
    ctx.strokeStyle = PAGE_OUTLINE_COLOR;
    ctx.lineWidth = PAGE_OUTLINE_WIDTH;
    ctx.strokeRect(offX, offY, renderW, renderH);

    /* Subtle page shadow. */
    ctx.shadowColor = "rgba(0,0,0,0.06)";
    ctx.shadowBlur = 8;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 2;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(offX, offY, renderW, renderH);
    ctx.restore();

    /* ── Draw grid bins ─────────────────────────────────────────────── */
    for (let r = 0; r < data.gridRows; r++) {
      for (let c = 0; c < data.gridCols; c++) {
        const bx = offX + c * cellW;
        const by = offY + r * cellH;
        const bin = grid[r]?.[c];

        if (bin && bin.sampleCount > 0) {
          /* Fill with colour. */
          ctx.fillStyle = errorColor(bin.errorRate);
          ctx.globalAlpha = 0.75;
          ctx.fillRect(bx, by, cellW, cellH);
          ctx.globalAlpha = 1;

          /* Colour-blind hatching. */
          drawBinHatching(ctx, bx, by, cellW, cellH, bin.errorRate);

          /* Label for large bins. */
          if (cellW >= 18 && cellH >= 14) {
            ctx.fillStyle = bin.errorRate > 0.15 ? "rgba(255,255,255,0.9)" : "rgba(0,0,0,0.7)";
            ctx.font = `bold ${Math.max(8, Math.round(cellH * 0.3))}px ui-monospace, monospace`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(
              `${(bin.errorRate * 100).toFixed(0)}%`,
              bx + cellW / 2,
              by + cellH / 2,
            );
          }
        } else {
          /* Empty bin — fill with a faint grid background. */
          ctx.fillStyle = "#f8fafc";
          ctx.fillRect(bx, by, cellW, cellH);
        }

        /* Grid lines. */
        ctx.strokeStyle = GRID_LINE_COLOR;
        ctx.lineWidth = GRID_LINE_WIDTH;
        ctx.strokeRect(bx, by, cellW, cellH);
      }
    }

    ctx.restore();
  }, [data, grid, width, height, renderW, renderH, offX, offY, cellW, cellH]);

  useEffect(() => {
    render();
  }, [render]);

  /* ── Mouse tracking ────────────────────────────────────────────────── */
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) {
        setTooltip(null);
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      if (mx < offX || mx >= offX + renderW || my < offY || my >= offY + renderH) {
        setTooltip(null);
        return;
      }

      const col = Math.floor((mx - offX) / cellW);
      const row = Math.floor((my - offY) / cellH);

      if (row < 0 || row >= data.gridRows || col < 0 || col >= data.gridCols) {
        setTooltip(null);
        return;
      }

      const bin = grid[row]?.[col];
      if (!bin || bin.sampleCount <= 0) {
        setTooltip(null);
        return;
      }

      setTooltip({
        errorRate: bin.errorRate,
        sampleCount: bin.sampleCount,
        row,
        col,
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      });
    },
    [data, grid, renderW, renderH, offX, offY, cellW, cellH],
  );

  const handleMouseLeave = useCallback(() => setTooltip(null), []);

  /* ── Render ────────────────────────────────────────────────────────── */
  return (
    <div className="select-none">
      <div ref={containerRef} className="relative inline-block">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          style={{ width, height }}
          className="rounded-lg border border-surface-200"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        />

        {/* Tooltip */}
        {tooltip && (
          <div
            className="pointer-events-none absolute z-10 rounded-lg border border-surface-200 bg-white px-3 py-2 text-xs shadow-lg"
            style={{
              left: Math.min(tooltip.x + 12, width - 200),
              top: Math.max(tooltip.y - 50, 0),
            }}
          >
            <p className="mb-1 font-medium text-surface-700">
              Zone ({tooltip.row}, {tooltip.col})
            </p>
            <div className="flex items-center gap-2 text-surface-600">
              <span
                className="inline-block h-2.5 w-2.5 rounded"
                style={{ background: errorColor(tooltip.errorRate) }}
              />
              <span>
                Error rate: <span className="font-semibold">{(tooltip.errorRate * 100).toFixed(1)}%</span>
              </span>
            </div>
            <p className="text-surface-400">
              Samples: {tooltip.sampleCount} &middot;{" "}
              <span className="capitalize">{errorLabel(tooltip.errorRate)}</span>
            </p>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="mt-2 flex flex-wrap items-center gap-4 text-xs text-surface-500">
        <span className="font-medium text-surface-600">Error density:</span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-3 w-3 rounded" style={{ background: "#22c55e" }} />
          &lt;5%
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-3 w-3 rounded" style={{ background: "#eab308" }} />
          5–15%
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-3 w-3 rounded" style={{ background: "#ef4444" }} />
          &gt;15%
        </span>
        <span className="ml-2 text-surface-400">
          Hatching pattern indicates severity for colour-blind accessibility.
        </span>
      </div>
    </div>
  );
}
