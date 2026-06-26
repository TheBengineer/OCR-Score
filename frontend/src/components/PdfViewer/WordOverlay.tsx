import { useState, useCallback } from "react";
import type { OverlayWord } from "@/lib/types";
import { usePdfViewerState } from "./PdfViewer";

/* ── Status style configuration ─────────────────────────────────────────── */

interface StatusStyle {
  /** Fill colour (Tailwind hex). */
  fill: string;
  /** Border CSS class. */
  border: string;
  /** Additional background pattern CSS for color-blind accessibility. */
  pattern: string;
  /** Human-readable label. */
  label: string;
}

const STATUS_STYLES: Record<OverlayWord["status"], StatusStyle> = {
  correct: {
    fill: "#16a34a",
    border: "border-2 border-solid border-green-600",
    pattern: "",
    label: "Correct",
  },
  wrong: {
    fill: "#dc2626",
    border: "border-2 border-dashed border-red-600",
    pattern: "",
    label: "Wrong",
  },
  missing: {
    fill: "#2563eb",
    border: "border-2 border-dotted border-blue-600",
    pattern: "",
    label: "Missing",
  },
  extra: {
    fill: "#ea580c",
    border: "border-2 border-dashed border-orange-600",
    /* Crosshatch pattern for color-blind accessibility */
    pattern: `repeating-linear-gradient(
      45deg,
      transparent,
      transparent 3px,
      rgba(0,0,0,0.15) 3px,
      rgba(0,0,0,0.15) 6px
    ),
    repeating-linear-gradient(
      -45deg,
      transparent,
      transparent 3px,
      rgba(0,0,0,0.15) 3px,
      rgba(0,0,0,0.15) 6px
    )`,
    label: "Extra",
  },
};

/* ── Tooltip ─────────────────────────────────────────────────────────────── */

interface TooltipData {
  word: OverlayWord;
  x: number;
  y: number;
}

/* ── Props ───────────────────────────────────────────────────────────────── */

export interface WordOverlayProps {
  /** Words to render on the overlay. */
  words: OverlayWord[];
  /** Fill opacity (0–1). Default 0.3. */
  opacity?: number;
}

/* ── Component ───────────────────────────────────────────────────────────── */

export function WordOverlay({ words, opacity = 0.3 }: WordOverlayProps) {
  const { scale, rotation } = usePdfViewerState();
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);

  const isSideways = rotation % 180 === 90;

  const handleMouseEnter = useCallback(
    (word: OverlayWord, e: React.MouseEvent) => {
      const rect = e.currentTarget.getBoundingClientRect();
      setTooltip({ word, x: rect.left, y: rect.top });
    },
    [],
  );

  const handleMouseLeave = useCallback(() => {
    setTooltip(null);
  }, []);

  if (!words.length) return null;

  return (
    <>
      {words.map((word, idx) => {
        const [x0, y0, x1, y1] = word.bbox;
        const style = STATUS_STYLES[word.status];

        /* Swap coordinates for 90°/270° rotation */
        let cssLeft: number;
        let cssTop: number;
        let cssWidth: number;
        let cssHeight: number;

        if (isSideways) {
          cssLeft = y0 * scale;
          cssTop = x0 * scale;
          cssWidth = (y1 - y0) * scale;
          cssHeight = (x1 - x0) * scale;
        } else {
          cssLeft = x0 * scale;
          cssTop = y0 * scale;
          cssWidth = (x1 - x0) * scale;
          cssHeight = (y1 - y0) * scale;
        }

        const ariaDescription =
          `${style.label} word: "${word.text}"` +
          (word.engineText ? `, OCR read: "${word.engineText}"` : "") +
          `, confidence: ${(word.confidence * 100).toFixed(0)}%`;

        return (
          <div
            key={idx}
            role="img"
            aria-label={ariaDescription}
            className="pointer-events-auto absolute cursor-default"
            style={{
              left: cssLeft,
              top: cssTop,
              width: cssWidth,
              height: cssHeight,
              backgroundColor: style.fill,
              opacity,
              backgroundImage: style.pattern || undefined,
              backgroundBlendMode: style.pattern ? "normal" : undefined,
              minWidth: 4,
              minHeight: 4,
            }}
            onMouseEnter={(e) => handleMouseEnter(word, e)}
            onMouseLeave={handleMouseLeave}
          />
        );
      })}

      {/* ── Tooltip ── */}
      {tooltip && (
        <div
          className="pointer-events-none fixed z-50 max-w-xs rounded-md border border-surface-300 bg-white px-3 py-2 text-xs shadow-lg"
          style={{
            left: tooltip.x + 12,
            top: tooltip.y - 8,
          }}
        >
          <div className="flex items-center gap-2">
            <span
              className="inline-block h-2 w-2 rounded-sm"
              style={{
                backgroundColor:
                  STATUS_STYLES[tooltip.word.status].fill,
              }}
            />
            <span className="font-semibold text-surface-800">
              {STATUS_STYLES[tooltip.word.status].label}
            </span>
          </div>
          <p className="mt-1 text-surface-600">
            GT: <span className="font-medium text-surface-800">&quot;{tooltip.word.text}&quot;</span>
          </p>
          {tooltip.word.engineText && (
            <p className="text-surface-600">
              OCR: <span className="font-medium text-surface-800">&quot;{tooltip.word.engineText}&quot;</span>
            </p>
          )}
          <p className="text-surface-500">
            Confidence:{" "}
            {(tooltip.word.confidence * 100).toFixed(1)}%
          </p>
        </div>
      )}
    </>
  );
}
