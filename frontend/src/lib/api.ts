export type RetrievalMethod = "hybrid" | "vector" | "bm25" | "reranked";
export type EvidenceScope = "all" | "code" | "docs";
export type Provider = "deepseek" | "gemini" | "openai";

export interface QaRequest {
  repository: string;
  question: string;
  retrieval_method: RetrievalMethod;
  evidence_scope: EvidenceScope;
  provider: Provider;
  context_k: number;
  max_output_tokens: number;
}

export interface SourceEvidence {
  id: string;
  file_path: string;
  symbol: string | null;
  parent_symbol: string | null;
  start_line: number;
  end_line: number;
  language: string;
  chunk_type: string;
  retrieval_rank: number;
  content: string;
}

export interface TokenUsage {
  input_tokens?: number;
  output_tokens?: number;
  reasoning_tokens?: number;
  total_tokens?: number;
}

export interface QaResponse {
  question: string;
  repository: string;
  retrieval_method: RetrievalMethod;
  evidence_scope: EvidenceScope;
  provider: Provider;
  model: string;
  answer: string;
  citations_valid: boolean;
  cited_source_ids: string[];
  generation_latency_seconds: number | null;
  usage: TokenUsage | null;
  sources: SourceEvidence[];
}

interface ApiErrorBody {
  detail?: string;
  code?: string;
}

const configuredUrl = process.env.NEXT_PUBLIC_REPORAG_API_URL?.trim();
export const API_BASE_URL = (configuredUrl || "http://127.0.0.1:8000").replace(
  /\/$/,
  "",
);

export class RepoRagApiError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "RepoRagApiError";
  }
}

async function responseJson<T>(response: Response): Promise<T> {
  try {
    return (await response.json()) as T;
  } catch {
    throw new RepoRagApiError(
      "The RepoRAG API returned an unreadable response.",
      "invalid_response",
      response.status,
    );
  }
}

export async function checkHealth(): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/health`, {
    method: "GET",
    cache: "no-store",
  });
  if (!response.ok) {
    throw new RepoRagApiError("The RepoRAG API is unavailable.", "api_offline", response.status);
  }
  const body = await responseJson<{ status?: string }>(response);
  if (body.status !== "ok") {
    throw new RepoRagApiError("The RepoRAG API is unavailable.", "api_offline", response.status);
  }
}

export async function askRepository(request: QaRequest): Promise<QaResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/qa`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
  } catch {
    throw new RepoRagApiError(
      "Could not reach the RepoRAG API.",
      "api_offline",
      0,
    );
  }

  if (!response.ok) {
    let body: ApiErrorBody = {};
    try {
      body = (await response.json()) as ApiErrorBody;
    } catch {
      // Keep the public fallback below instead of surfacing an unreadable body.
    }
    throw new RepoRagApiError(
      body.detail || "RepoRAG could not complete the request.",
      body.code || "request_failed",
      response.status,
    );
  }
  return responseJson<QaResponse>(response);
}
