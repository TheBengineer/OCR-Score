/* ── Enums ─────────────────────────────────────────────────────────────── */

export type PDFStatus =
  | "uploading"
  | "uploaded"
  | "processing"
  | "ready"
  | "error"
  | "deleted";

export type RunStatus =
  | "pending"
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export type ScoreLevel =
  | "character"
  | "word"
  | "line"
  | "paragraph"
  | "block"
  | "table"
  | "page"
  | "document";

export type ScoreMetric =
  | "cer"
  | "wer"
  | "accuracy"
  | "precision"
  | "recall"
  | "f1"
  | "edit_distance"
  | "confidence"
  | "table_structure";

export type GroundTruthSource =
  | "manual"
  | "consensus"
  | "imported";

/* ── Document ──────────────────────────────────────────────────────────── */

export interface Document {
  id: string;
  filename: string;
  original_filename: string;
  file_size_bytes: number;
  page_count: number;
  md5_hash: string;
  sha256_hash: string;
  mime_type: string;
  status: PDFStatus;
  upload_timestamp: string;
  deleted_at: string | null;
}

export interface DocumentListResponse {
  items: Document[];
  next_cursor: string | null;
}

/* ── Engine ────────────────────────────────────────────────────────────── */

export interface Engine {
  id: string;
  slug: string;
  display_name: string;
  version: string;
  enabled: boolean;
  config_schema: Record<string, unknown> | null;
  secret_schema?: { key: string; env_var: string | null; display_name: string; description: string }[] | null;
  description: string | null;
  created_at: string;
  updated_at: string;
}

/* ── Run ───────────────────────────────────────────────────────────────── */

export interface Run {
  id: string;
  pdf_id: string;
  engine_id: string;
  status: RunStatus;
  engine_config: Record<string, unknown> | null;
  engine_version: string | null;
  started_at: string | null;
  completed_at: string | null;
  error_message: string | null;
  raw_output_uri: string | null;
  run_hash: string;
  environment: Record<string, unknown> | null;
  created_at: string;
}

export interface RunCreateRequest {
  pdf_id: string;
  engine_id: string;
  config?: Record<string, unknown> | null;
}

export interface RunCreateResponse {
  id: string;
  status: RunStatus;
  message: string | null;
}

export interface RunListResponse {
  items: Run[];
  total: number;
}

/* ── Page Result ───────────────────────────────────────────────────────── */

export interface PageResultChar {
  char: string;
  bbox: [number, number, number, number];
  confidence: number;
  order: number;
}

export interface PageResultWord {
  text: string;
  bbox: [number, number, number, number];
  confidence: number;
  order: number;
  chars: PageResultChar[];
}

export interface PageResultLine {
  text: string;
  bbox: [number, number, number, number];
  confidence: number;
  order: number;
  words: PageResultWord[];
}

export interface PageTableBlock {
  type: "text" | "table" | "figure" | "math" | "separator";
  bbox: [number, number, number, number];
  confidence: number;
  order: number;
  lines: PageResultLine[];
}

export interface PageTableCell {
  row: number;
  col: number;
  row_span: number;
  col_span: number;
  text: string;
  bbox: [number, number, number, number];
  confidence: number;
}

export interface PageTable {
  bbox: [number, number, number, number];
  num_rows: number;
  num_cols: number;
  caption: string;
  cells: PageTableCell[];
}

export interface PageResultData {
  blocks: PageTableBlock[];
  tables: PageTable[];
}

export interface PageResult {
  id: string;
  run_id: string;
  page_number: number;
  width: number | null;
  height: number | null;
  data: PageResultData;
  confidence: number | null;
}

export interface PageResultListResponse {
  items: PageResult[];
  page: number;
  page_size: number;
  total: number;
}

/* ── Score ─────────────────────────────────────────────────────────────── */

export interface Score {
  id: string;
  run_id: string;
  gt_version_id: string | null;
  level: ScoreLevel;
  page_number: number | null;
  metric: ScoreMetric;
  value: number;
  confidence_weighted: boolean;
  details: Record<string, unknown> | null;
  computed_at: string;
}

export interface ScoreSummary {
  id: string;
  run_id: string;
  gt_version_id: string | null;
  overall_score: number;
  breakdown: Record<string, unknown> | null;
  computed_at: string;
}

/* ── Ground Truth ──────────────────────────────────────────────────────── */

export interface GroundTruthVersion {
  id: string;
  pdf_id: string;
  version_number: number;
  source: GroundTruthSource;
  created_at: string;
  created_by: string | null;
  notes: string | null;
  deleted_at: string | null;
}

/* ── Word Comparison (multi-engine) ─────────────────────────────────────────── */

export interface EngineComparison {
  engineName: string;
  engineSlug: string;
  text: string;
  confidence: number;
  status: OverlayWordStatus;
}

export interface WordComparison {
  wordIndex: number;
  gtText: string;
  gtBbox: [number, number, number, number];
  engines: EngineComparison[];
}

/* ── Word Overlay ─────────────────────────────────────────────────────────── */

export type OverlayWordStatus = "correct" | "wrong" | "missing" | "extra";

export interface OverlayWord {
  text: string;
  bbox: [number, number, number, number]; // [x0, y0, x1, y1] in PDF points
  confidence: number;
  status: OverlayWordStatus;
  engineText?: string;
  /** MST-based reading order position (1-based). 0 or undefined = no order data. */
  order?: number;
}

/* ── PDF Viewer ─────────────────────────────────────────────────────────── */

export interface PdfViewerConfig {
  defaultZoom?: number;
  minZoom?: number;
  maxZoom?: number;
  zoomStep?: number;
}

export interface GTPageResult {
  id: string;
  gt_version_id: string;
  page_number: number;
  width: number | null;
  height: number | null;
  data: PageResultData;
  confidence: number | null;
}

/* ── Score API types ────────────────────────────────────────────────────── */

export interface RunOverallScores {
  cer: number;
  wer: number;
  char_f1: number;
  word_f1: number;
}

export interface BootstrapCI {
  cer_lower: number;
  cer_upper: number;
  ci_level: number;
}

export interface RunScoresResponse {
  run_id: string;
  gt_version_id: string;
  overall: RunOverallScores | null;
  bootstrap_ci: BootstrapCI | null;
  pages: number;
  evaluated_pages: number;
}

export interface PageScoreEntry {
  page: number;
  cer: number;
  wer: number;
  char_f1: number;
  word_f1: number;
}

export interface RunScoresByPageResponse {
  run_id: string;
  gt_version_id: string;
  pages: PageScoreEntry[];
}

export interface EngineScoreEntry {
  engine_id: string;
  run_id: string | null;
  scores: {
    cer: number;
    wer: number;
    char_f1: number;
    word_f1: number;
    pages: number;
  } | null;
  message?: string;
}

export interface EngineComparisonResponse {
  pdf_id: string;
  gt_version_id: string;
  engines: EngineScoreEntry[];
}

/* ── Canvas Overlay (character-level multi-engine) ─────────────────────────── */

/** A single character from an OCR engine with its bounding box. */
export interface OverlayChar {
  /** The character text (may be a single char, space, or punctuation). */
  char: string;
  /** Bounding box in PDF points: [x0, y0, x1, y1]. */
  bbox: [number, number, number, number];
  /** OCR confidence score (0–1). */
  confidence: number;
  /** Which engine produced this character. */
  engineId: string;
}

/** Per-engine layer configuration for the canvas overlay system. */
export interface EngineLayerConfig {
  /** Unique engine identifier (slug). */
  id: string;
  /** Human-readable engine name. */
  name: string;
  /** Display colour (CSS hex e.g. "#4f46e5"). */
  color: string;
  /** Layer opacity (0–1). */
  opacity: number;
  /** Whether this layer is currently visible. */
  visible: boolean;
}

/** Characters grouped by engine, as returned from the multi-engine compare endpoint. */
export interface OverlayEngineData {
  engineId: string;
  engineName: string;
  characters: OverlayChar[];
}

/* ── Report / Dashboard types ──────────────────────────────────────────── */

export interface SummaryStatistics {
  total_pdfs: number;
  total_runs: number;
  completed_runs: number;
  avg_cer: number;
  avg_wer: number;
  best_engine: {
    id: string;
    avg_cer: number;
  } | null;
  pages_evaluated: number;
}

export interface EngineRanking {
  engine: string;
  display_name: string;
  avg_cer: number;
  avg_wer: number;
  avg_f1: number;
  runs: number;
}

export interface ReportRunScores {
  overall_score: number;
  breakdown: Record<string, unknown> | null;
}

export interface ReportRunEntry {
  run_id: string;
  engine: string;
  pdf: string;
  status: string;
  pages: Array<{ page: number }>;
  page_count: number;
  scores: ReportRunScores | null;
  created_at: string | null;
  completed_at: string | null;
}

export interface ReportData {
  generated_at: string;
  runs: ReportRunEntry[];
  engine_summaries: EngineRanking[];
}

/* ── Batch Processing ──────────────────────────────────────────────────── */

export interface BatchCreateRequest {
  pdf_ids: string[];
  engine_slugs: string[];
  config?: Record<string, unknown> | null;
}

export interface BatchCreateResponse {
  id: string;
  status: string;
  total_items: number;
  message: string | null;
}

export interface BatchResponse {
  id: string;
  pdf_ids: string[];
  engine_slugs: string[];
  config: Record<string, unknown> | null;
  status: string;
  created_at: string;
  total_items: number;
  completed: number;
  failed: number;
  error_message: string | null;
}

export interface BatchItemProgress {
  pdf_id: string;
  engine_slug: string;
  run_id: string | null;
  status: string;
  message: string | null;
}

export interface BatchProgressResponse {
  batch_id: string;
  status: string;
  total: number;
  completed: number;
  failed: number;
  pending: number;
  percent: number;
  items: BatchItemProgress[];
}

/* ── Comparison ─────────────────────────────────────────────────────────── */

export interface ComparisonRunEntry {
  run_id: string;
  pdf_id: string;
  engine_slug: string;
  status: string;
  scores: RunOverallScores & { bootstrap_ci?: BootstrapCI | null; pages?: number } | null;
  message?: string;
}

export interface ComparisonRunsResponse {
  run_ids: string[];
  gt_version_id: string | null;
  entries: ComparisonRunEntry[];
}

export interface EnginePdfEntry {
  pdf_id: string;
  run_id: string | null;
  scores: RunOverallScores | null;
  message?: string;
}

export interface EngineComparisonEntry {
  engine_id: string;
  pdfs: EnginePdfEntry[];
}

export interface ComparisonEnginesResponse {
  engine_ids: string[];
  pdf_ids: string[];
  gt_version_id: string | null;
  engines: EngineComparisonEntry[];
}

/* ── Advanced Visualization Types (Phase 5) ──────────────────────────── */

/**
 * Confusion matrix data: counts of how often each reference character
 * was predicted as each output character.
 *
 * `labels[i]` is the character label for row/column i.
 * `matrix[i][j]` = number of times `labels[i]` was predicted as `labels[j]`.
 * Diagonal entries represent correct predictions.
 */
export interface ConfusionMatrixData {
  labels: string[];
  matrix: number[][];
  total: number;
}

/**
 * A single bin in the geometric error heatmap grid.
 * `row`/`col` are grid coordinates (0-indexed).
 * `errorRate` is the fraction of samples in this bin that are errors (0–1).
 * `sampleCount` is the number of OCR samples that fell into this bin.
 */
export interface HeatmapBin {
  row: number;
  col: number;
  errorRate: number;
  sampleCount: number;
}

/**
 * Geometric heatmap data: spatial error density across a page.
 * The page is divided into gridRows × gridCols bins.
 * `bins` contains non-empty bins; empty bins should be rendered as transparent.
 */
export interface GeometricHeatmapData {
  pageWidth: number;
  pageHeight: number;
  gridRows: number;
  gridCols: number;
  bins: HeatmapBin[];
}
