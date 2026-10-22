import { env } from './env.js';

type Json = Record<string, unknown>;

async function post<T = Json>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${env.aiServiceUrl}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`ai-service ${path} -> ${res.status}: ${await res.text()}`);
  }
  return (await res.json()) as T;
}

async function get<T = Json>(path: string): Promise<T> {
  const res = await fetch(`${env.aiServiceUrl}${path}`);
  if (!res.ok) {
    throw new Error(`ai-service ${path} -> ${res.status}: ${await res.text()}`);
  }
  return (await res.json()) as T;
}

export interface RoutingInfo {
  route: 'auto_approve' | 'human_review' | 'reject_escalate';
  reasons: string[];
  document_score: number;
  field_scores: Record<string, number>;
  thresholds: Record<string, number>;
  forced: boolean;
}

export interface Extraction {
  document_id: string;
  extraction: { merchant: string; total_amount: number | null; currency: string | null; [k: string]: unknown };
  prompt_version: string;
  model: string;
  attempts: number;
  repaired: boolean;
  escalated: boolean;
  latency_ms: number;
  cost_usd: number;
  redactions: Record<string, number>;
  document_confidence?: number | null;
  field_confidence?: Record<string, number>;
  routing?: RoutingInfo | null;
  confidence_notes?: string[];
}

export interface Coding {
  category: string;
  confidence: number;
  rationale: string;
  model: string;
  latency_ms: number;
  cost_usd: number;
}

export interface Policy {
  compliant: boolean;
  route: 'auto_approve' | 'manager_review' | 'human_queue';
  violations: { rule: string; detail: string }[];
}

export const ai = {
  async health(): Promise<Json> {
    const res = await fetch(`${env.aiServiceUrl}/health`);
    return (await res.json()) as Json;
  },
  extract: (document_id: string, text: string) =>
    post<Extraction>('/extract', { document_id, text }),
  code: (description: string, merchant?: string, amount?: number) =>
    post<Coding>('/code', { description, merchant, amount }),
  policyCheck: (amount: number, category: string, has_receipt = true) =>
    post<Policy>('/policy-check', { amount, category, has_receipt }),
  searchMerchants: (query: string, k = 5) => post('/search/merchants', { query, k }),
  searchPolicies: (query: string, k = 5) => post('/search/policies', { query, k }),
  correct: (payload: unknown) => post('/corrections', payload),
  forceReview: (enabled: boolean, actor: string) =>
    post('/admin/kill-switch', { enabled, actor }),
  killSwitch: () => get<{ enabled: boolean; actor: string; changed_at: string }>('/admin/kill-switch'),
};
