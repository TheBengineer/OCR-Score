import type {
  Document,
  DocumentListResponse,
  Engine,
  EngineComparisonResponse,
  GTPageResult,
  PageResult,
  PageResultListResponse,
  Run,
  RunCreateRequest,
  RunCreateResponse,
  RunListResponse,
  RunScoresByPageResponse,
  RunScoresResponse,
  WordComparison,
} from "./types.ts";

// ── Configuration ──────────────────────────────────────────────────────────

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api/v1";

// ── Generic fetch wrapper ──────────────────────────────────────────────────

class ApiError extends Error {
  status: number;
  body: unknown;

  constructor(status: number, body: unknown) {
    super(`API error ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  options?: RequestInit,
): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const headers: Record<string, string> = {};
  let requestBody: BodyInit | null = null;

  if (body instanceof FormData) {
    requestBody = body;
    // Let fetch set Content-Type with boundary for FormData
  } else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    requestBody = JSON.stringify(body);
  }

  const fetchInit: RequestInit = {
    method,
    headers,
    ...(requestBody !== null ? { body: requestBody } : {}),
    ...options,
  };

  const response = await fetch(url, fetchInit);

  if (!response.ok) {
    let errorBody: unknown;
    try {
      errorBody = (await response.json()) as unknown;
    } catch {
      errorBody = await response.text().catch(() => null);
    }
    throw new ApiError(response.status, errorBody);
  }

  // 204 No Content
  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

// ── Document endpoints ─────────────────────────────────────────────────────

export async function listDocuments(
  cursor?: string,
  limit?: number,
): Promise<DocumentListResponse> {
  const params = new URLSearchParams();
  if (cursor) params.set("cursor", cursor);
  if (limit) params.set("limit", String(limit));
  const qs = params.toString();
  return request<DocumentListResponse>("GET", `/documents${qs ? `?${qs}` : ""}`);
}

export async function getDocument(id: string): Promise<Document> {
  return request<Document>("GET", `/documents/${id}`);
}

export async function uploadDocument(file: File): Promise<Document> {
  const formData = new FormData();
  formData.append("file", file);
  return request<Document>("POST", "/documents/upload", formData);
}

export async function deleteDocument(id: string): Promise<void> {
  return request<void>("DELETE", `/documents/${id}`);
}

// ── Engine endpoints ───────────────────────────────────────────────────────

// TODO: Engine CRUD endpoints are not yet implemented in the backend.
// When added, they will likely follow the pattern below.

export async function listEngines(): Promise<Engine[]> {
  return request<Engine[]>("GET", "/engines");
}

// ── Run endpoints ──────────────────────────────────────────────────────────

export async function createRun(
  pdfId: string,
  engineId: string,
  config?: Record<string, unknown>,
): Promise<RunCreateResponse> {
  const body: RunCreateRequest = {
    pdf_id: pdfId,
    engine_id: engineId,
    config: config ?? null,
  };
  return request<RunCreateResponse>("POST", "/runs", body);
}

export async function listRuns(
  params?: {
    pdf_id?: string;
    engine_id?: string;
    status?: string;
    limit?: number;
    offset?: number;
  },
): Promise<RunListResponse> {
  const searchParams = new URLSearchParams();
  if (params?.pdf_id) searchParams.set("pdf_id", params.pdf_id);
  if (params?.engine_id) searchParams.set("engine_id", params.engine_id);
  if (params?.status) searchParams.set("status", params.status);
  if (params?.limit) searchParams.set("limit", String(params.limit));
  if (params?.offset) searchParams.set("offset", String(params.offset));
  const qs = searchParams.toString();
  return request<RunListResponse>("GET", `/runs${qs ? `?${qs}` : ""}`);
}

export async function getRun(id: string): Promise<Run> {
  return request<Run>("GET", `/runs/${id}`);
}

export async function getRunResults(
  runId: string,
  page?: number,
  pageSize?: number,
): Promise<PageResultListResponse> {
  const searchParams = new URLSearchParams();
  if (page) searchParams.set("page", String(page));
  if (pageSize) searchParams.set("page_size", String(pageSize));
  const qs = searchParams.toString();
  return request<PageResultListResponse>(
    "GET",
    `/runs/${runId}/results${qs ? `?${qs}` : ""}`,
  );
}

export async function getPageResult(
  runId: string,
  pageNumber: number,
): Promise<PageResult> {
  return request<PageResult>("GET", `/runs/${runId}/results/${pageNumber}`);
}

export async function getRunRawOutput(runId: string): Promise<unknown> {
  return request<unknown>("GET", `/runs/${runId}/raw`);
}

export async function cancelRun(runId: string): Promise<void> {
  return request<void>("DELETE", `/runs/${runId}`);
}

// ── Ground Truth endpoints ─────────────────────────────────────────────────

export async function getGTPageResult(
  gtVersionId: string,
  pageNumber: number,
): Promise<GTPageResult> {
  return request<GTPageResult>(
    "GET",
    `/ground-truth/${gtVersionId}/pages/${pageNumber}`,
  );
}

// ── Score endpoints ────────────────────────────────────────────────────────

export async function getRunScores(
  runId: string,
  gtVersionId: string,
): Promise<RunScoresResponse> {
  return request<RunScoresResponse>(
    "GET",
    `/runs/${runId}/scores?gt_version_id=${gtVersionId}`,
  );
}

export async function getRunScoresByPage(
  runId: string,
  gtVersionId: string,
): Promise<RunScoresByPageResponse> {
  return request<RunScoresByPageResponse>(
    "GET",
    `/runs/${runId}/scores/by-page?gt_version_id=${gtVersionId}`,
  );
}

export async function getEngineComparison(
  pdfId: string,
  engineIds: string[],
  gtVersionId: string,
): Promise<EngineComparisonResponse> {
  return request<EngineComparisonResponse>(
    "GET",
    `/documents/${pdfId}/runs/comparison?engine_ids=${engineIds.join(",")}&gt_version_id=${gtVersionId}`,
  );
}

// ── Multi-engine word comparison ───────────────────────────────────────────

export async function getWordComparison(
  runId: string,
  pageNumber: number,
  wordIndex: number,
): Promise<WordComparison> {
  return request<WordComparison>(
    "GET",
    `/runs/${runId}/results/${pageNumber}/compare?word_index=${wordIndex}`,
  );
}
