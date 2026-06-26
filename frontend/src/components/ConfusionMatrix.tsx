import { useRef, useCallback, useState, useEffect } from "react";
import type { ConfusionMatrixData } from "../lib/types.ts";

/* ── Constants ──────────────────────────────────────────────────────────── */

const ROW_LABEL_W = 26;
const COL_LABEL_H = 18;
const MARGIN_TOP = 8;
const MARGIN_RIGHT = 8;
const GAP = 4;
const MIN_CELL = 14;

/* ── Props ──────────────────────────────────────────────────────────────── */

export interface ConfusionMatrixProps {
  data: ConfusionMatrixData;
  width?: number;
  height?: number;
  /** Show only the top N most-confused characters (by off-diagonal sum). */
  maxLabels?: number;
  /** Minimum count a cell must have to be visually filled. */
  minCount?: number;
}

/* ── Color helpers ──────────────────────────────────────────────────────── */

/** White (0) → red (max). Clamped to [0, 1]. */
function heatColor(t: number): string {
  const v = Math.min(t, 1);
  const gb = Math.round(255 * (1 - v));
  return `rgb(255, ${gb}, ${gb})`;
}

/** Draw color-blind accessible hatching over a cell region. */
function drawHatching(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  intensity: number,
) {
  if (intensity < 0.25) return;
  ctx.save();
  ctx.strokeStyle = `rgba(60, 60, 60, ${Math.min(0.35 + intensity * 0.3, 0.6)})`;
  ctx.lineWidth = 1;
  const step = Math.max(4, Math.round(8 - intensity * 4));

  if (intensity < 0.5) {
    /* Diagonal lines (medium). */
    ctx.beginPath();
    for (let i = -w; i < w + h; i += step) {
      ctx.moveTo(x + i, y);
      ctx.lineTo(x + i + h, y + h);
    }
    ctx.stroke();
  } else if (intensity < 0.75) {
    /* Cross-hatch (high). */
    ctx.beginPath();
    for (let i = -w; i < w + h; i += step) {
      ctx.moveTo(x + i, y);
      ctx.lineTo(x + i + h, y + h);
    }
    for (let i = -w; i < w + h; i += step) {
      ctx.moveTo(x + i + h, y);
      ctx.lineTo(x + i, y + h);
    }
    ctx.stroke();
  } else {
    /* Dense cross-hatch (very high). */
    ctx.beginPath();
    const dense = Math.max(3, step - 2);
    for (let i = -w; i < w + h; i += dense) {
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

function computeTopLabels(
  labels: string[],
  matrix: number[][],
  maxLabels: number,
): string[] {
  if (labels.length <= maxLabels) return labels;
  const scores: { label: string; idx: number; score: number }[] = [];
  for (let i = 0; i < labels.length; i++) {
    const row = matrix[i]!;
    let offDiagSum = 0;
    for (let j = 0; j < row.length; j++) {
      if (i !== j) offDiagSum += row[j]!;
    }
    scores.push({ label: labels[i]!, idx: i, score: offDiagSum });
  }
  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, maxLabels).map((s) => s.label);
}

/* ── Component ───────────────────────────────────────────────────────────── */

export default function ConfusionMatrix({
  data,
  width = 600,
  height = 500,
  maxLabels = 26,
  minCount = 0,
}: ConfusionMatrixProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{
    refChar: string;
    predChar: string;
    count: number;
    x: number;
    y: number;
  } | null>(null);
  const [filteredLabels, setFilteredLabels] = useState<string[]>([]);
  const [filterN, setFilterN] = useState(maxLabels);

  /* ── Pick labels respecting the top-N filter ─────────────────────────── */
  useEffect(() => {
    if (data.labels.length <= filterN) {
      setFilteredLabels(data.labels);
    } else {
      setFilteredLabels(computeTopLabels(data.labels, data.matrix, filterN));
    }
  }, [data, filterN]);

  /* ── Pick a sub-matrix for the filtered labels ───────────────────────── */
  const n = filteredLabels.length;
  const labelToIdx = new Map<string, number>();
  for (let i = 0; i < data.labels.length; i++) {
    labelToIdx.set(data.labels[i]!, i);
  }

  const subMatrix: number[][] = [];
  for (const rl of filteredLabels) {
    const ri = labelToIdx.get(rl)!;
    const row: number[] = [];
    for (const cl of filteredLabels) {
      const ci = labelToIdx.get(cl)!;
      row.push(data.matrix[ri]?.[ci] ?? 0);
    }
    subMatrix.push(row);
  }

  const maxVal = subMatrix.reduce(
    (m, r) => Math.max(m, ...r),
    0,
  );
  const scale = maxVal > 0 ? 1 / maxVal : 1;

  /* ── Layout ──────────────────────────────────────────────────────────── */
  const gridLeft = ROW_LABEL_W + GAP;
  const gridTop = MARGIN_TOP;
  const gridRight = width - MARGIN_RIGHT;
  const gridBottom = height - COL_LABEL_H - GAP;
  const gridW = gridRight - gridLeft;
  const gridH = gridBottom - gridTop;
  const cellSize = Math.max(
    MIN_CELL,
    Math.min(gridW / Math.max(n, 1), gridH / Math.max(n, 1)),
  );
  const actualW = cellSize * n;
  const actualH = cellSize * n;
  const offX = Math.max(0, (gridW - actualW) / 2);
  const offY = Math.max(0, (gridH - actualH) / 2);

  /* ── Render ──────────────────────────────────────────────────────────── */
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

    if (n === 0) {
      ctx.fillStyle = "#94a3b8";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No confusion data available", width / 2, height / 2);
      ctx.restore();
      return;
    }

    const fontSize = Math.max(9, Math.round(cellSize * 0.35));

    /* ── Draw cells ──────────────────────────────────────────────────── */
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const cx = gridLeft + offX + j * cellSize;
        const cy = gridTop + offY + i * cellSize;
        const val = subMatrix[i]![j]!;
        const t = val * scale;

        /* Fill. */
        ctx.fillStyle = heatColor(t);
        ctx.fillRect(cx, cy, cellSize, cellSize);

        /* Border. */
        ctx.strokeStyle = "#e2e8f0";
        ctx.lineWidth = 0.5;
        ctx.strokeRect(cx, cy, cellSize, cellSize);

        /* Pattern for colour-blind accessibility. */
        drawHatching(ctx, cx, cy, cellSize, cellSize, t);

        /* Value text for large-enough cells. */
        if (val > 0 && val >= minCount && cellSize >= 22) {
          ctx.fillStyle = t > 0.6 ? "rgba(255,255,255,0.9)" : "rgba(0,0,0,0.65)";
          ctx.font = `bold ${fontSize}px ui-monospace, monospace`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(String(val), cx + cellSize / 2, cy + cellSize / 2);
        }
      }
    }

    /* ── Row labels (left) ───────────────────────────────────────────── */
    ctx.fillStyle = "#334155";
    ctx.font = `${fontSize}px ui-sans-serif, sans-serif`;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let i = 0; i < n; i++) {
      const cx = gridLeft + offX;
      const cy = gridTop + offY + i * cellSize + cellSize / 2;
      ctx.fillText(filteredLabels[i]!, cx - GAP, cy);
    }

    /* ── Column labels (bottom) ──────────────────────────────────────── */
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let j = 0; j < n; j++) {
      const cx = gridLeft + offX + j * cellSize + cellSize / 2;
      const cy = gridBottom + GAP;
      ctx.fillText(filteredLabels[j]!, cx, cy);
    }

    ctx.restore();
  }, [n, subMatrix, scale, cellSize, width, height, gridLeft, gridTop, gridRight, gridBottom, offX, offY, filteredLabels, minCount]);

  useEffect(() => {
    render();
  }, [render]);

  /* ── Mouse tracking ─────────────────────────────────────────────────── */
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas || n === 0) {
        setTooltip(null);
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      const innerLeft = gridLeft + offX;
      const innerTop = gridTop + offY;

      if (
        mx < innerLeft ||
        mx >= innerLeft + actualW ||
        my < innerTop ||
        my >= innerTop + actualH
      ) {
        setTooltip(null);
        return;
      }

      const col = Math.floor((mx - innerLeft) / cellSize);
      const row = Math.floor((my - innerTop) / cellSize);
      if (row < 0 || row >= n || col < 0 || col >= n) {
        setTooltip(null);
        return;
      }

      const count = subMatrix[row]![col]!;
      if (count <= 0) {
        setTooltip(null);
        return;
      }

      setTooltip({
        refChar: filteredLabels[row]!,
        predChar: filteredLabels[col]!,
        count,
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      });
    },
    [n, subMatrix, filteredLabels, cellSize, gridLeft, gridTop, offX, offY, actualW, actualH],
  );

  const handleMouseLeave = useCallback(() => setTooltip(null), []);

  /* ── Render ─────────────────────────────────────────────────────────── */
  return (
    <div className="select-none">
      {/* Toolbar */}
      <div className="mb-3 flex flex-wrap items-center gap-4">
        <label className="flex items-center gap-2 text-sm text-surface-600">
          <span>Top characters:</span>
          <input
            type="number"
            min={2}
            max={data.labels.length}
            value={filterN}
            onChange={(e) =>
              setFilterN(
                Math.min(
                  data.labels.length,
                  Math.max(2, Number(e.target.value) || 2),
                ),
              )
            }
            className="w-16 rounded border border-surface-300 px-2 py-1 text-sm text-surface-700"
          />
        </label>
        <div className="flex items-center gap-2 text-xs text-surface-400">
          <span
            className="inline-block h-3 w-3 rounded"
            style={{ background: "rgb(255,255,255)", border: "1px solid #e2e8f0" }}
          />
          <span>0</span>
          <span className="inline-block h-3 w-3 rounded" style={{ background: "rgb(255,100,100)" }} />
          <span>max</span>
        </div>
      </div>

      {/* Canvas container */}
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
              left: Math.min(tooltip.x + 12, width - 180),
              top: Math.max(tooltip.y - 40, 0),
            }}
          >
            <p className="whitespace-nowrap text-surface-700">
              Reference <span className="font-bold text-surface-900">'{tooltip.refChar}'</span>
              {" → "}
              predicted <span className="font-bold text-surface-900">'{tooltip.predChar}'</span>
              :{" "}
              <span className="font-semibold text-primary-600">{tooltip.count}</span> times
            </p>
          </div>
        )}
      </div>

      {/* Axis labels */}
      <div className="mt-1 flex items-center gap-2 text-xs text-surface-400">
        <span className="inline-block w-[26px] text-right" />
        <span>Reference (rows) →</span>
        <span className="ml-auto">Predicted (columns) ↓</span>
      </div>
    </div>
  );
}
