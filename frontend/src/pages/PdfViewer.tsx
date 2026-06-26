import { useState, useEffect, useCallback, useMemo } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import {
  PdfViewer,
  WordOverlay,
  OverlayLegend,
  WordInspector,
  ReadingOrderOverlay,
  CanvasOverlay,
  EngineLayerManager,
  DiffRibbon,
} from "@/components/PdfViewer";
import type {
  OverlayWord,
  PageResult,
  EngineComparison,
  EngineLayerConfig,
  OverlayChar,
} from "@/lib/types";
import { getPageResult, getGTPageResult, getWordComparison, getRun, listEngines } from "@/lib/api";

/* ── Constants ───────────────────────────────────────────────────────────── */

/**
 * Sample PDF for development/testing when no document ID is available.
 * Mozilla's compressed.tracemonkey-pldi-09.pdf — a standard PDF.js test file.
 */
const DEV_SAMPLE_PDF =
  "https://raw.githubusercontent.com/mozilla/pdf.js/ba2edeae/web/compressed.tracemonkey-pldi-09.pdf";

/**
 * Default engine layers with accessible, color-blind-friendly colours:
 *   Tesseract — Indigo  (#4f46e5)
 *   GCP        — Emerald (#059669)
 *   Textract   — Amber   (#d97706)
 */
const DEMO_ENGINE_LAYERS: EngineLayerConfig[] = [
  { id: "tesseract", name: "Tesseract", color: "#4f46e5", opacity: 0.5, visible: true },
  { id: "gcp-document-ai", name: "GCP Document AI", color: "#059669", opacity: 0.5, visible: false },
  { id: "aws-textract", name: "AWS Textract", color: "#d97706", opacity: 0.5, visible: false },
];

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

/* ── Real character extractor ─────────────────────────────────────────────
 * Extracts character-level overlay data from a real PageResult response,
 * preserving each character's actual text, bounding box, and confidence. */

function extractCharsFromPageData(
  pageData: PageResult["data"],
  engineSlug: string,
): OverlayChar[] {
  const chars: OverlayChar[] = [];
  for (const block of pageData.blocks) {
    for (const line of block.lines) {
      for (const word of line.words) {
        for (const ch of word.chars) {
          chars.push({
            char: ch.char,
            bbox: ch.bbox,
            confidence: ch.confidence,
            engineId: engineSlug,
          });
        }
      }
    }
  }
  return chars;
}

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

  /* ── Canvas overlay state ── */
  const [useCanvasOverlay, setUseCanvasOverlay] = useState(true);
  const [engineLayers, setEngineLayers] = useState<EngineLayerConfig[]>(
    DEMO_ENGINE_LAYERS,
  );
  const [engineData, setEngineData] = useState<Map<string, OverlayChar[]>>(
    () => new Map(),
  );

  const handleUpdateLayer = useCallback(
    (id: string, partial: Partial<EngineLayerConfig>) => {
      setEngineLayers((prev) =>
        prev.map((l) => (l.id === id ? { ...l, ...partial } : l)),
      );
    },
    [],
  );

  const handleShowAll = useCallback(() => {
    setEngineLayers((prev) => prev.map((l) => ({ ...l, visible: true })));
  }, []);

  const handleHideAll = useCallback(() => {
    setEngineLayers((prev) => prev.map((l) => ({ ...l, visible: false })));
  }, []);

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
        const [runInfo, ocrResult, gtResult] = await Promise.all([
          getRun(rid),
          getPageResult(rid, 1),
          getGTPageResult(gtvId, 1),
        ]);

        if (cancelled) return;

        const words = compareWords(ocrResult.data, gtResult.data);
        setOverlayWords(words);

        // Determine the engine slug for this run
        let engineSlug = "unknown";
        try {
          const engines = await listEngines();
          const match = engines.find((e) => e.id === runInfo.engine_id);
          if (match) engineSlug = match.slug;
        } catch {
          // fall through with "unknown" slug
        }

        // Extract real character data from the OCR output
        const realChars = extractCharsFromPageData(ocrResult.data, engineSlug);
        const charMap = new Map<string, OverlayChar[]>();
        if (realChars.length > 0) {
          charMap.set(engineSlug, realChars);
        }
        setEngineData(charMap);
      } catch (err) {
        console.warn("Failed to fetch OCR/GT data, using demo overlay", err);
        if (!cancelled) {
          setOverlayWords(DEMO_WORDS);
          setEngineData(new Map());
        }
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

  /* ── Memoised engine data Map (avoids recreating on every render) ── */
  const engineDataMemo = useMemo(() => engineData, [engineData]);

  /* ── Sync engine layers to match available engine data ── */
  useEffect(() => {
    if (engineData.size === 0) return;
    setEngineLayers((prev) => {
      const existing = new Set(prev.map((l) => l.id));
      const next = [...prev];
      for (const engineId of engineData.keys()) {
        if (!existing.has(engineId)) {
          next.push({
            id: engineId,
            name: engineId,
            color: "#4f46e5",
            opacity: 0.5,
            visible: true,
          });
        }
      }
      return next;
    });
  }, [engineData]);

  return (
    <div className="mx-auto max-w-6xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-surface-900">PDF Viewer</h1>
          <p className="mt-2 text-surface-500">
            View OCR results overlaid on the original document.
          </p>
        </div>

        {/* ── Canvas / SVG mode toggle ── */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-surface-400">
            {useCanvasOverlay ? "Canvas" : "SVG"} mode
          </span>
          <button
            type="button"
            onClick={() => setUseCanvasOverlay((v) => !v)}
            className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
              useCanvasOverlay ? "bg-primary-500" : "bg-surface-300"
            }`}
            role="switch"
            aria-checked={useCanvasOverlay}
            aria-label="Toggle canvas overlay mode"
          >
            <span
              className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                useCanvasOverlay ? "translate-x-[18px]" : "translate-x-[2px]"
              }`}
            />
          </button>
        </div>
      </div>

      <div className="relative mt-6">
        <PdfViewer
          pdfUrl={pdfUrl}
          onPageChange={handlePageChange}
          showReadingOrder={showReadingOrder}
          onToggleReadingOrder={() => setShowReadingOrder((v) => !v)}
          ribbon={!loading && overlayWords.length > 0 ? (
            <DiffRibbon words={overlayWords} />
          ) : undefined}
        >
          {/* Canvas overlay mode — high-performance multi-engine rendering */}
          {overlayVisible && !loading && useCanvasOverlay && (
            <CanvasOverlay
              engineLayers={engineLayers}
              engineData={engineDataMemo}
            />
          )}

          {/* SVG fallback mode — simple word-level overlay */}
          {overlayVisible && !loading && !useCanvasOverlay && (
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

        {/* Layer management panel (canvas mode only) */}
        {useCanvasOverlay && (
          <EngineLayerManager
            layers={engineLayers}
            onUpdateLayer={handleUpdateLayer}
            onShowAll={handleShowAll}
            onHideAll={handleHideAll}
          />
        )}

        {/* SVG legend (SVG mode only) */}
        {!useCanvasOverlay && (
          <OverlayLegend
            opacity={opacity}
            onOpacityChange={setOpacity}
            visible={overlayVisible}
            onVisibilityChange={setOverlayVisible}
          />
        )}

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
