import { useState, useMemo, useCallback } from "react";
import type { OverlayWord } from "@/lib/types";
import { usePdfViewerState } from "./PdfViewer";

/* ── Agreement colours ─────────────────────────────────────────────────── */

const AGREEMENT_COLORS: Record<AgreementLevel, string> = {
  consensus: "#16a34a",
  partial: "#d97706",
  disagreement: "#dc2626",
  nodata: "#6b7280",
};

const AGREEMENT_LABELS: Record<AgreementLevel, string> = {
  consensus: "All engines agree",
  partial: "Some disagreement",
  disagreement: "Engines diverge",
  nodata: "No OCR data",
};

/* ── Types ─────────────────────────────────────────────────────────────── */

export type AgreementLevel =
  | "consensus"
  | "partial"
  | "disagreement"
  | "nodata";

export interface LineStripe {
  lineIndex: number;
  yStartPts: number;
  yEndPts: number;
  yPixelStart: number;
  yPixelEnd: number;
  pixelHeight: number;
  agreement: AgreementLevel;
  sampleText: string;
  wordCount: number;
  errorCount: number;
  engineCount: number;
}

export interface DiffRibbonProps {
  /** Words on the page with their bbox positions and status. */
  words: OverlayWord[];
}

/* ── Line grouping ───────────────────────────────────────────────────────
 *  Clusters words into text lines based on their vertical position.
 *  Words whose vertical midpoints fall within LINE_THRESHOLD PDF points
 *  of each other are grouped into the same line. */

const LINE_THRESHOLD_PTS = 15;

function groupWordsIntoLines(words: OverlayWord[]): OverlayWord[][] {
  if (words.length === 0) return [];

  const sorted = [...words].sort((a, b) => {
    const aMid = (a.bbox[1] + a.bbox[3]) / 2;
    const bMid = (b.bbox[1] + b.bbox[3]) / 2;
    return aMid - bMid;
  });

  const lines: OverlayWord[][] = [];
  let currentLine: OverlayWord[] = [sorted[0]!];

  for (let i = 1; i < sorted.length; i++) {
    const prevWord = sorted[i - 1]!;
    const currWord = sorted[i]!;
    const prevMid = (prevWord.bbox[1] + prevWord.bbox[3]) / 2;
    const currMid = (currWord.bbox[1] + currWord.bbox[3]) / 2;

    if (currMid - prevMid <= LINE_THRESHOLD_PTS) {
      currentLine.push(currWord);
    } else {
      lines.push(currentLine);
      currentLine = [currWord];
    }
  }

  if (currentLine.length > 0) {
    lines.push(currentLine);
  }

  return lines;
}

/* ── Agreement computation ─────────────────────────────────────────────── */

function computeAgreement(line: OverlayWord[]): {
  agreement: AgreementLevel;
  errorCount: number;
} {
  const total = line.length;
  if (total === 0) return { agreement: "nodata", errorCount: 0 };

  let correctCount = 0;
  let errorCount = 0;

  for (const word of line) {
    if (word.status === "correct") {
      correctCount++;
    } else {
      errorCount++;
    }
  }

  let agreement: AgreementLevel;
  if (errorCount === 0) {
    agreement = "consensus";
  } else if (correctCount === 0) {
    agreement = "disagreement";
  } else {
    agreement = "partial";
  }

  return { agreement, errorCount };
}

/* ── Stripe computation ────────────────────────────────────────────────── */

function computeStripes(
  words: OverlayWord[],
  pageHeightPts: number,
  scale: number,
): LineStripe[] {
  const lines = groupWordsIntoLines(words);
  if (lines.length === 0) return [];

  return lines.map((line, idx) => {
    const yStartPts = Math.min(...line.map((w) => w.bbox[1]));
    const yEndPts = Math.max(...line.map((w) => w.bbox[3]));
    const { agreement, errorCount } = computeAgreement(line);
    const sampleText = line[0]?.text ?? "";

    return {
      lineIndex: idx + 1,
      yStartPts,
      yEndPts,
      yPixelStart: yStartPts * scale,
      yPixelEnd: yEndPts * scale,
      pixelHeight: Math.max(1, (yEndPts - yStartPts) * scale),
      agreement,
      sampleText,
      wordCount: line.length,
      errorCount,
      engineCount: 3,
    };
  });
}

/* ── Tooltip ───────────────────────────────────────────────────────────── */

function Tooltip({
  stripe,
  style,
}: {
  stripe: LineStripe;
  style: React.CSSProperties;
}) {
  const agreementSummary = (() => {
    const total = stripe.wordCount;
    const correct = total - stripe.errorCount;
    if (stripe.agreement === "consensus") {
      return `${stripe.engineCount}/${stripe.engineCount} engines agree`;
    }
    if (stripe.agreement === "disagreement") {
      return `${stripe.errorCount}/${total} words differ across all engines`;
    }
    return `${correct}/${total} words match, ${stripe.errorCount} differ`;
  })();

  return (
    <div
      className="pointer-events-none absolute z-50 rounded-md border border-surface-200 bg-white px-3 py-2 text-xs shadow-lg"
      style={style}
    >
      <div className="flex items-center gap-1.5">
        <span
          className="inline-block h-2 w-2 shrink-0 rounded-sm"
          style={{ backgroundColor: AGREEMENT_COLORS[stripe.agreement] }}
        />
        <span className="font-semibold text-surface-800">
          {AGREEMENT_LABELS[stripe.agreement]}
        </span>
      </div>
      <p className="mt-0.5 text-surface-500">
        Line {stripe.lineIndex}
        {stripe.sampleText ? (
          <>
            {" · "}
            <span className="font-medium text-surface-700">
              &ldquo;{stripe.sampleText}&rdquo;
            </span>
          </>
        ) : null}
      </p>
      <p className="mt-0.5 text-surface-400">{agreementSummary}</p>
    </div>
  );
}

/* ── Component ─────────────────────────────────────────────────────────── */

export function DiffRibbon({ words }: DiffRibbonProps) {
  const { pageHeightPts, scale, rotation, scrollToPagePosition } =
    usePdfViewerState();

  const [hoveredStripe, setHoveredStripe] = useState<LineStripe | null>(null);
  const [tooltipY, setTooltipY] = useState(0);

  const stripes = useMemo(
    () => computeStripes(words, pageHeightPts, scale),
    [words, pageHeightPts, scale],
  );

  const handleMouseEnter = useCallback(
    (stripe: LineStripe, e: React.MouseEvent) => {
      const rect = e.currentTarget.getBoundingClientRect();
      setHoveredStripe(stripe);
      setTooltipY(rect.top);
    },
    [],
  );

  const handleMouseLeave = useCallback(() => {
    setHoveredStripe(null);
  }, []);

  const handleClick = useCallback(
    (stripe: LineStripe) => {
      const centreY = (stripe.yStartPts + stripe.yEndPts) / 2;
      scrollToPagePosition?.(centreY);
    },
    [scrollToPagePosition],
  );

  if (pageHeightPts <= 0 || stripes.length === 0) {
    return null;
  }

  /* Swap height for rotated pages */
  const renderedHeight = pageHeightPts * scale;
  const isSideways = rotation % 180 === 90;

  return (
    <div
      role="img"
      aria-label="OCR diff ribbon showing per-line agreement across OCR engines"
      className="relative h-full"
    >
      <div
        className="relative w-full"
        style={{
          height: isSideways ? renderedHeight : renderedHeight,
        }}
      >
        {stripes.map((stripe) => (
          <div
            key={stripe.lineIndex}
            role="button"
            tabIndex={0}
            aria-label={`Line ${stripe.lineIndex}: ${AGREEMENT_LABELS[stripe.agreement]}`}
            className="absolute left-0 right-0 cursor-pointer transition-opacity hover:opacity-80"
            style={{
              top: stripe.yPixelStart,
              height: stripe.pixelHeight,
              minHeight: stripe.pixelHeight,
              backgroundColor: AGREEMENT_COLORS[stripe.agreement],
            }}
            onClick={() => handleClick(stripe)}
            onMouseEnter={(e) => handleMouseEnter(stripe, e)}
            onMouseLeave={handleMouseLeave}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                handleClick(stripe);
              }
            }}
          />
        ))}
      </div>

      {/* ── Tooltip ── */}
      {hoveredStripe && (
        <Tooltip
          stripe={hoveredStripe}
          style={{
            right: "calc(100% + 8px)",
            top: Math.min(
              tooltipY,
              window.innerHeight - 160,
            ),
          }}
        />
      )}
    </div>
  );
}
