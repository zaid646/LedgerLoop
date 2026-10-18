import { env } from './env.js';

export interface WebhookEvent {
  event: 'expense.routed' | 'expense.decided';
  document_id: string;
  route: string;
  status: string;
  amount: number | null;
  merchant: string | null;
  occurred_at: string;
}

export interface DeliveryResult {
  delivered: boolean;
  status?: number;
  reason?: string;
}

/**
 * Fire-and-forget outbound webhook. Disabled unless WEBHOOK_URL is set, and it
 * never throws — a delivery failure must not fail the request that triggered it.
 */
export async function notify(event: WebhookEvent): Promise<DeliveryResult> {
  if (!env.webhookUrl) return { delivered: false, reason: 'WEBHOOK_URL not configured' };
  try {
    const res = await fetch(env.webhookUrl, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(event),
    });
    return { delivered: res.ok, status: res.status };
  } catch (e) {
    return { delivered: false, reason: String(e) };
  }
}
