import { useMemo } from "react";
import type { OverlayWord } from "@/lib/types";
import { usePdfViewerState } from "./PdfViewer";

export interface ReadingOrderOverlayProps {
  words: OverlayWord[];
}

export function ReadingOrderOverlay({ words }: ReadingOrderOverlayProps) {
  const { scale, rotation } = usePdfViewerState();
  const isSideways = rotation % 180 === 90;

  const hasOrderData = useMemo(
    () => words.some((w) => (w.order ?? 0) > 0),
    [words],
  );

  if (!words.length) return null;

  /* ── No reading order data ── */
  if (!hasOrderData) {
    return (
      <div className="pointer-events-none absolute left-2 top-2 z-30 rounded bg-amber-50/90 px-2 py-1 text-xs text-amber-700 shadow-sm">
        No reading order data
      </div>
    );
  }

  return (
    <>
      {words.map((word, idx) => {
        const order = word.order ?? 0;
        if (order <= 0) return null;

        const [x0, y0, x1, _y1] = word.bbox;
        const orderW = 24;
        const orderH = 24;

        let cssLeft: number;
        let cssTop: number;

        if (isSideways) {
          cssLeft = y0 * scale;
          cssTop = x0 * scale;
        } else {
          cssLeft = x0 * scale;
          cssTop = y0 * scale;
        }

        /* Clamp badge so it doesn't overflow the word's left edge */
        const wordW = (isSideways ? _y1 - y0 : x1 - x0) * scale;
        const maxLeft = cssLeft + wordW - orderW;
        const clampedLeft = Math.min(cssLeft, maxLeft);

        return (
          <div
            key={idx}
            className="pointer-events-none absolute z-20 flex items-center justify-center rounded-full"
            style={{
              left: clampedLeft,
              top: cssTop,
              width: orderW,
              height: orderH,
              backgroundColor: "rgba(255, 255, 255, 0.8)",
              color: "#1e293b",
              fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
              fontSize: 11,
              fontWeight: 700,
              lineHeight: `${orderH}px`,
              textAlign: "center",
              boxShadow: "0 1px 2px rgba(0,0,0,0.15)",
              border: "1px solid rgba(0,0,0,0.1)",
            }}
          >
            {order}
          </div>
        );
      })}
    </>
  );
}
