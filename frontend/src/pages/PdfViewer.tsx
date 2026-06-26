import { useState, useEffect, useCallback } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { PdfViewer, WordOverlay, OverlayLegend } from "@/components/PdfViewer";
import type { OverlayWord, PageResult } from "@/lib/types";
import { getPageResult, getGTPageResult } from "@/lib/api";

/**
 * Sample PDF for development/testing when no document ID is available.
 * Mozilla's compressed.tracemonkey-pldi-09.pdf — a standard PDF.js test file.
 */
const DEV_SAMPLE_PDF =
  "https://raw.githubusercontent.com/mozilla/pdf.js/ba2edeae/web/compressed.tracemonkey-pldi-09.pdf";

/* ── Demo overlay words (used when no API data is available) ────────────── */

const DEMO_WORDS: OverlayWord[] = [
  {
    text: "The",
    bbox: [72, 120, 120, 140],
    confidence: 0.98,
    status: "correct",
  },
  {
    text: "quick",
    bbox: [128, 120, 184, 140],
    confidence: 0.95,
    status: "correct",
  },
  {
    text: "brown",
    bbox: [192, 120, 252, 140],
    confidence: 0.72,
    status: "wrong",
    engineText: "browm",
  },
  {
    text: "fox",
    bbox: [260, 120, 296, 140],
    confidence: 0.88,
    status: "correct",
  },
  {
    text: "jumps",
    bbox: [72, 148, 136, 168],
    confidence: 0.45,
    status: "wrong",
    engineText: "jumpes",
  },
  {
    text: "over",
    bbox: [144, 148, 196, 168],
    confidence: 0.91,
    status: "correct",
  },
  {
    text: "the",
    bbox: [204, 148, 244, 168],
    confidence: 0.0,
    status: "missing",
  },
  {
    text: "lazy",
    bbox: [252, 148, 300, 168],
    confidence: 0.0,
    status: "extra",
    engineText: "lazy",
  },
  {
    text: "dog",
    bbox: [308, 148, 352, 168],
    confidence: 0.97,
    status: "correct",
  },
];

/* ── Word comparison helper ──────────────────────────────────────────────── */

function compareWords(
  ocrWords: PageResult["data"],
  _gtWords: PageResult["data"],
): OverlayWord[] {
  /* Simple word-by-word comparison.
   * This is a straightforward alignment; Phase 3 will implement proper
   * sequence-alignment-based comparison. */
  const result: OverlayWord[] = [];

  const ocrList =
    ocrWords.blocks.flatMap((b) => b.lines).flatMap((l) => l.words) ?? [];
  const gtList =
    _gtWords.blocks.flatMap((b) => b.lines).flatMap((l) => l.words) ?? [];

  const maxLen = Math.max(ocrList.length, gtList.length);

  for (let i = 0; i < maxLen; i++) {
    const ocr = ocrList[i];
    const gt = gtList[i];

    if (gt && ocr) {
      const match = ocr.text === gt.text;
      result.push(match
        ? {
            text: gt.text,
            bbox: ocr.bbox,
            confidence: ocr.confidence,
            status: "correct" as const,
          }
        : {
            text: gt.text,
            bbox: ocr.bbox,
            confidence: ocr.confidence,
            status: "wrong" as const,
            engineText: ocr.text,
          },
      );
    } else if (gt && !ocr) {
      result.push({
        text: gt.text,
        bbox: gt.bbox,
        confidence: 0,
        status: "missing",
      });
    } else if (!gt && ocr) {
      result.push({
        text: ocr.text,
        bbox: ocr.bbox,
        confidence: ocr.confidence,
        status: "extra",
        engineText: ocr.text,
      });
    }
  }

  return result;
}

/* ── Page component ──────────────────────────────────────────────────────── */

export default function PdfViewerPage() {
  const { id } = useParams<{ id: string }>();
  const [searchParams] = useSearchParams();
  const runId = searchParams.get("runId");

  const [overlayWords, setOverlayWords] = useState<OverlayWord[]>(DEMO_WORDS);
  const [opacity, setOpacity] = useState(0.3);
  const [overlayVisible, setOverlayVisible] = useState(true);
  const [loading, setLoading] = useState(false);

  const pdfUrl = id
    ? `/api/v1/documents/${id}/file`
    : DEV_SAMPLE_PDF;

  const handlePageChange = useCallback((pageNumber: number) => {
    console.info("Page changed to", pageNumber);
  }, []);

  /* ── Fetch OCR results + GT when runId and page change ── */
  useEffect(() => {
    const activeRunId = runId;
    const activeDocId = id;
    if (!activeRunId || !activeDocId) return;

    let cancelled = false;
    const gtVersionId = `${activeDocId}-gt-v1`; // placeholder — real logic in Phase 3

    setLoading(true);

    async function fetchData(rid: string, gtvId: string) {
      try {
        const [ocrResult, gtResult] = await Promise.all([
          getPageResult(rid, 1),
          getGTPageResult(gtvId, 1),
        ]);

        if (cancelled) return;

        const words = compareWords(ocrResult.data, gtResult.data);
        setOverlayWords(words);
      } catch (err) {
        console.warn("Failed to fetch OCR/GT data, using demo overlay", err);
        if (!cancelled) setOverlayWords(DEMO_WORDS);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchData(activeRunId, gtVersionId);
    return () => {
      cancelled = true;
    };
  }, [runId, id]);

  return (
    <div className="mx-auto max-w-6xl">
      <h1 className="text-3xl font-bold text-surface-900">PDF Viewer</h1>
      <p className="mt-2 text-surface-500">
        View OCR results overlaid on the original document.
      </p>

      <div className="relative mt-6">
        <PdfViewer
          pdfUrl={pdfUrl}
          onPageChange={handlePageChange}
        >
          {overlayVisible && !loading && (
            <WordOverlay words={overlayWords} opacity={opacity} />
          )}
        </PdfViewer>

        <OverlayLegend
          opacity={opacity}
          onOpacityChange={setOpacity}
          visible={overlayVisible}
          onVisibilityChange={setOverlayVisible}
        />
      </div>
    </div>
  );
}
