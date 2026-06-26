import { useRef, useEffect, useCallback } from "react";
import { usePdfViewerState } from "./PdfViewer";
import type { EngineLayerConfig, OverlayChar } from "@/lib/types";

/* ── Props ───────────────────────────────────────────────────────────────── */

export interface CanvasOverlayProps {
  engineLayers: EngineLayerConfig[];
  engineData: Map<string, OverlayChar[]>;
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */

function findScrollParent(el: HTMLElement): HTMLElement | null {
  let cur: HTMLElement | null = el;
  while ((cur = cur.parentElement)) {
    const overflow = window.getComputedStyle(cur).overflow;
    if (overflow === "auto" || overflow === "scroll") return cur;
  }
  return null;
}

/**
 * Returns the visible rectangle of container within its scroll parent,
 * in CSS-pixel coordinates relative to the container origin.
 * Returns null when the container is entirely off-screen.
 */
function computeVisibleCanvasRect(
  container: HTMLElement | null,
  _canvasW: number,
  _canvasH: number,
): [number, number, number, number] | null {
  if (!container) return null;

  const cr = container.getBoundingClientRect();
  const scrollParent = findScrollParent(container);

  if (!scrollParent) {
    return [0, 0, _canvasW, _canvasH];
  }

  const sr = scrollParent.getBoundingClientRect();

  const left = Math.max(cr.left, sr.left);
  const top = Math.max(cr.top, sr.top);
  const right = Math.min(cr.right, sr.right);
  const bottom = Math.min(cr.bottom, sr.bottom);

  if (right <= left || bottom <= top) return null;

  return [
    left - cr.left,
    top - cr.top,
    right - cr.left,
    bottom - cr.top,
  ];
}

/* ── Component ───────────────────────────────────────────────────────────── */

/**
 * High-performance canvas-based overlay for rendering multi-engine OCR
 * character outlines. Uses HTML5 Canvas2D with viewport culling and
 * requestAnimationFrame-driven rendering.
 */
export function CanvasOverlay({ engineLayers, engineData }: CanvasOverlayProps) {
  const { pageWidthPts, pageHeightPts, scale, rotation } = usePdfViewerState();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number>(0);

  const isSideways = rotation % 180 === 90;
  const cssWidth = Math.ceil((isSideways ? pageHeightPts : pageWidthPts) * scale);
  const cssHeight = Math.ceil((isSideways ? pageWidthPts : pageHeightPts) * scale);

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cw = cssWidth;
    const ch = cssHeight;

    /* Resize backing store for high-DPI displays */
    const bw = Math.ceil(cw * dpr);
    const bh = Math.ceil(ch * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cw, ch);

    /* Viewport culling: only render characters in visible region */
    const vp = computeVisibleCanvasRect(containerRef.current, cw, ch);
    if (!vp) {
      ctx.restore();
      return;
    }

    const [vpLeft, vpTop, vpRight, vpBottom] = vp;

    const fontSize = Math.max(6, Math.round(9 * scale));
    ctx.font = `${fontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.textBaseline = "bottom";

    /* Render each active engine layer in z-order */
    for (const layer of engineLayers) {
      if (!layer.visible) continue;

      const chars = engineData.get(layer.id);
      if (!chars || chars.length === 0) continue;

      ctx.save();
      ctx.globalAlpha = layer.opacity;
      ctx.strokeStyle = layer.color;
      ctx.fillStyle = layer.color;
      ctx.lineWidth = Math.max(0.5, 1 * scale);

      for (const ch of chars) {
        /* Convert PDF point bbox to canvas pixel coords, handling rotation */
        const [cx0, cy0, cx1, cy1] = isSideways
          ? [
              ch.bbox[1] * scale,
              ch.bbox[0] * scale,
              ch.bbox[3] * scale,
              ch.bbox[2] * scale,
            ]
          : [
              ch.bbox[0] * scale,
              ch.bbox[1] * scale,
              ch.bbox[2] * scale,
              ch.bbox[3] * scale,
            ];

        /* Viewport culling — skip chars outside visible region */
        if (cx1 <= vpLeft || cx0 >= vpRight || cy1 <= vpTop || cy0 >= vpBottom) {
          continue;
        }

        /* Stroke bounding box */
        ctx.strokeRect(cx0, cy0, cx1 - cx0, cy1 - cy0);

        /* Fill character text at bottom-left inside bbox */
        ctx.fillText(ch.char, cx0 + 1, cy1 - 1);
      }

      ctx.restore();
    }

    ctx.restore();
  }, [engineLayers, engineData, scale, rotation, cssWidth, cssHeight, isSideways]);

  /* requestAnimationFrame render loop */
  useEffect(() => {
    rafRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(rafRef.current);
  }, [render]);

  if (cssWidth <= 0 || cssHeight <= 0) return null;

  return (
    <div
      ref={containerRef}
      className="absolute left-0 top-0 overflow-hidden"
      style={{ width: cssWidth, height: cssHeight }}
    >
      <canvas
        ref={canvasRef}
        className="pointer-events-none"
        style={{ width: cssWidth, height: cssHeight }}
      />
    </div>
  );
}
