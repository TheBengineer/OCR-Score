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

export interface GTPageResult {
  id: string;
  gt_version_id: string;
  page_number: number;
  width: number | null;
  height: number | null;
  data: PageResultData;
  confidence: number | null;
}
