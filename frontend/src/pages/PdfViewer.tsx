import { useState, useEffect, useCallback } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import {
  PdfViewer,
  WordOverlay,
  OverlayLegend,
  WordInspector,
  ReadingOrderOverlay,
} from "@/components/PdfViewer";
import type {
  OverlayWord,
  PageResult,
  EngineComparison,
} from "@/lib/types";
import { getPageResult, getGTPageResult, getWordComparison } from "@/lib/api";

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
    order: 1,
  },
  {
    text: "quick",
    bbox: [128, 120, 184, 140],
    confidence: 0.95,
    status: "correct",
    order: 2,
  },
  {
    text: "brown",
    bbox: [192, 120, 252, 140],
    confidence: 0.72,
    status: "wrong",
    engineText: "browm",
    order: 3,
  },
  {
    text: "fox",
    bbox: [260, 120, 296, 140],
    confidence: 0.88,
    status: "correct",
    order: 4,
  },
  {
    text: "jumps",
    bbox: [72, 148, 136, 168],
    confidence: 0.45,
    status: "wrong",
    engineText: "jumpes",
    order: 5,
  },
  {
    text: "over",
    bbox: [144, 148, 196, 168],
    confidence: 0.91,
    status: "correct",
    order: 6,
  },
  {
    text: "the",
    bbox: [204, 148, 244, 168],
    confidence: 0.0,
    status: "missing",
    order: 7,
  },
  {
    text: "lazy",
    bbox: [252, 148, 300, 168],
    confidence: 0.0,
    status: "extra",
    engineText: "lazy",
    order: 8,
  },
  {
    text: "dog",
    bbox: [308, 148, 352, 168],
    confidence: 0.97,
    status: "correct",
    order: 9,
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

/* ── Fallback comparison data generator ─────────────────────────────────────
 * Produces simulated multi-engine data when the compare endpoint is unavailable
 * (Phase 3 implements the proper backend endpoint). */

function generateDemoComparison(
  wordIndex: number,
  word: OverlayWord,
): EngineComparison[] {
  const gt = word.text;

  /* Base the demo engines on the existing engineText hint */
  const isWrong = word.status === "wrong";
  const ocrText = word.engineText ?? gt;

  return [
    {
      engineName: "Tesseract",
      engineSlug: "tesseract",
      text: isWrong ? ocrText : gt,
      confidence: isWrong ? word.confidence : Math.min(word.confidence + 0.05, 1),
      status: isWrong ? "wrong" : "correct",
    },
    {
      engineName: "GCP Document AI",
      engineSlug: "gcp-document-ai",
      text: gt,
      confidence: Math.min(word.confidence + 0.08, 1),
      status: "correct",
    },
    {
      engineName: "AWS Textract",
      engineSlug: "aws-textract",
      text: gt,
      confidence: Math.max(word.confidence - 0.1, 0),
      status: word.status === "extra" ? "extra" : "correct",
    },
  ];
}

/* ── Page component ──────────────────────────────────────────────────────── */

export default function PdfViewerPage() {
  const { id } = useParams<{ id: string }>();
  const [searchParams] = useSearchParams();
  const runId = searchParams.get("runId");

  const [overlayWords, setOverlayWords] = useState<OverlayWord[]>(DEMO_WORDS);
  const [opacity, setOpacity] = useState(0.3);
  const [overlayVisible, setOverlayVisible] = useState(true);
  const [showReadingOrder, setShowReadingOrder] = useState(false);
  const [loading, setLoading] = useState(false);

  /* ── Word inspector state ── */
  const [selectedWordIndex, setSelectedWordIndex] = useState<number | null>(
    null,
  );
  const [inspectorPosition, setInspectorPosition] = useState<{
    x: number;
    y: number;
  }>({ x: 0, y: 0 });
  const [comparisons, setComparisons] = useState<EngineComparison[] | null>(
    null,
  );
  const [comparisonsLoading, setComparisonsLoading] = useState(false);

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

  /* ── Word click handler: open inspector ── */
  const handleWordClick = useCallback(
    (wordIndex: number, _word: OverlayWord, event: React.MouseEvent) => {
      setSelectedWordIndex(wordIndex);
      setInspectorPosition({ x: event.clientX, y: event.clientY });
      setComparisonsLoading(true);
      setComparisons(null);

      /* Attempt API fetch with fallback to demo data */
      const activeRunId = runId;
      if (activeRunId) {
        getWordComparison(activeRunId, 1, wordIndex)
          .then((data) => {
            setComparisons(data.engines);
          })
          .catch(() => {
            /* Compare endpoint not implemented yet — use demo fallback */
            const word = overlayWords[wordIndex];
            if (word) {
              setComparisons(generateDemoComparison(wordIndex, word));
            }
          })
          .finally(() => {
            setComparisonsLoading(false);
          });
      } else {
        /* No runId — use demo data */
        const word = overlayWords[wordIndex];
        if (word) {
          setComparisons(generateDemoComparison(wordIndex, word));
        }
        setComparisonsLoading(false);
      }
    },
    [runId, overlayWords],
  );

  /* ── Close inspector ── */
  const handleCloseInspector = useCallback(() => {
    setSelectedWordIndex(null);
    setComparisons(null);
    setComparisonsLoading(false);
  }, []);

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
          showReadingOrder={showReadingOrder}
          onToggleReadingOrder={() => setShowReadingOrder((v) => !v)}
        >
          {overlayVisible && !loading && (
            <WordOverlay
              words={overlayWords}
              opacity={opacity}
              onWordClick={handleWordClick}
              {...(selectedWordIndex !== null
                ? { selectedIndex: selectedWordIndex }
                : {})}
            />
          )}
          {showReadingOrder && !loading && (
            <ReadingOrderOverlay words={overlayWords} />
          )}
        </PdfViewer>

        <OverlayLegend
          opacity={opacity}
          onOpacityChange={setOpacity}
          visible={overlayVisible}
          onVisibilityChange={setOverlayVisible}
        />

        {selectedWordIndex !== null && overlayWords[selectedWordIndex] && (
          <WordInspector
            word={overlayWords[selectedWordIndex]}
            wordIndex={selectedWordIndex}
            position={inspectorPosition}
            comparisons={comparisons}
            loading={comparisonsLoading}
            onClose={handleCloseInspector}
          />
        )}
      </div>
    </div>
  );
}
