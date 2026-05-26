/**
 * API client — connects to FastAPI backend via Next.js rewrite proxy (/api/*)
 */

import {
  Document,
  SearchFilters,
  SearchResponse,
  RecommendationResponse,
  Entity,
  SystemStats,
  Metrics,
  KGStats,
  KGGraphResponse,
  KGNeighborsResponse,
} from "./types";

/**
 * In the browser, requests go to /api/* which Next.js rewrites to the FastAPI backend.
 * During SSR (server components), we call the backend directly since the rewrite
 * proxy is not available for server-side fetch().
 */
const BASE_URL =
  typeof window === "undefined"
    ? (process.env.BACKEND_URL || "http://localhost:8000")
    : "/api";

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options?.headers },
    cache: "no-store",
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API ${res.status}: ${err}`);
  }
  return res.json();
}

// ── Search / Recommend ────────────────────────────────────────────────────────

export async function searchDocuments(
  query: string,
  filters?: SearchFilters,
  topK = 10,
  explain = true,
  ensembleMethod?: string
): Promise<SearchResponse> {
  return apiFetch<SearchResponse>("/search", {
    method: "POST",
    body: JSON.stringify({
      query,
      top_k: topK,
      era: filters?.era,
      region: filters?.region,
      type: filters?.type,
      domain: filters?.domain,
      explain,
      ...(ensembleMethod ? { ensemble_method: ensembleMethod } : {}),
    }),
  });
}

export async function searchBaseline(
  query: string,
  filters?: SearchFilters,
  topK = 10
): Promise<SearchResponse> {
  return apiFetch<SearchResponse>("/search/baseline", {
    method: "POST",
    body: JSON.stringify({
      query,
      top_k: topK,
      era: filters?.era,
      region: filters?.region,
      type: filters?.type,
      domain: filters?.domain,
      explain: false,
    }),
  });
}

export async function getRecommendations(
  query: string,
  topK = 10
): Promise<RecommendationResponse> {
  const res = await apiFetch<SearchResponse>(
    `/recommend?q=${encodeURIComponent(query)}&top_k=${topK}&explain=true`
  );
  return { documents: res.documents, method: res.ensemble_method ?? "hybrid" };
}

// ── Documents ─────────────────────────────────────────────────────────────────

export async function listDocuments(
  page = 1,
  pageSize = 20,
  filters?: SearchFilters
): Promise<{ total: number; documents: Document[] }> {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });
  if (filters?.era?.[0]) params.set("era", filters.era[0]);
  if (filters?.region?.[0]) params.set("region", filters.region[0]);
  if (filters?.type?.[0]) params.set("type", filters.type[0]);
  return apiFetch<{ total: number; documents: Document[] }>(`/documents?${params}`);
}

export async function getDocument(id: string): Promise<Document> {
  return apiFetch<Document>(`/documents/${encodeURIComponent(id)}`);
}

// ── Knowledge Graph ───────────────────────────────────────────────────────────

export async function getKGStats(): Promise<KGStats> {
  return apiFetch("/kg/stats");
}

export async function getKGGraph(
  limitEntities = 80,
  includeDocs = false
): Promise<KGGraphResponse> {
  const params = new URLSearchParams({
    limit_entities: String(limitEntities),
    include_documents: String(includeDocs),
  });
  return apiFetch<KGGraphResponse>(`/kg/graph?${params}`);
}

export async function getKGEntityNeighbors(
  entityId: string
): Promise<KGNeighborsResponse> {
  return apiFetch<KGNeighborsResponse>(
    `/kg/entity/${encodeURIComponent(entityId)}/neighbors`
  );
}

export async function getEntities(type?: string, limit = 50): Promise<Entity[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (type) params.set("type", type);
  return apiFetch<Entity[]>(`/kg/entities?${params}`);
}

// ── Dashboard ─────────────────────────────────────────────────────────────────

export async function getSystemStats(): Promise<SystemStats> {
  return apiFetch<SystemStats>("/stats");
}

export async function getMetrics(): Promise<Metrics> {
  return apiFetch<Metrics>("/metrics");
}

export async function checkHealth(): Promise<{ status: string; recommender_loaded: boolean; adaptive_ranker_loaded: boolean; documents_indexed: number }> {
  return apiFetch("/health");
}
