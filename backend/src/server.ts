import Fastify from 'fastify';
import cors from '@fastify/cors';
import { z } from 'zod';
import { env } from './env.js';
import { approvalsCollection, ensureSchema, extractionsCollection, killSwitchEnabled, pool } from './db.js';
import { ai } from './aiService.js';
import { notify, type DeliveryResult } from './webhooks.js';
import { decisionStatus, diffFields, shouldQueue, validateDecision } from './review.js';

const app = Fastify({ logger: { level: env.logLevel } });

await app.register(cors, { origin: true });

const submitSchema = z.object({
  document_id: z.string().min(1),
  text: z.string().min(1),
  amount: z.number().optional(),
  has_receipt: z.boolean().default(true),
});

type Submission = z.infer<typeof submitSchema>;

/** Shared pipeline: extract -> code -> policy -> persist -> (webhook if routed). */
async function ingest(data: Submission) {
  const { document_id, text, has_receipt } = data;
  const extraction = await ai.extract(document_id, text);
  const merchant = extraction.extraction.merchant;
  const amount = data.amount ?? extraction.extraction.total_amount ?? 0;

  const coding = await ai.code(merchant, merchant, amount);
  const policy = await ai.policyCheck(amount, coding.category, has_receipt);

  // Confidence routing is authoritative; the policy route remains an input.
  const confidenceRoute = extraction.routing?.route ?? 'human_review';
  const forced = await killSwitchEnabled();
  const route = forced ? 'human_review' : confidenceRoute;
  const status = route === 'auto_approve' ? 'approved' : 'pending';

  await pool.query(
    `INSERT INTO expenses (id, merchant, total_amount, currency, category, policy_route, status, escalated, payload)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
     ON CONFLICT (id) DO UPDATE SET category=$5, policy_route=$6, status=$7, escalated=$8, payload=$9`,
    [
      document_id,
      merchant,
      extraction.extraction.total_amount,
      extraction.extraction.currency,
      coding.category,
      policy.route,
      status,
      route !== 'auto_approve',
      JSON.stringify({ extraction, coding, policy }),
    ],
  );

  if (shouldQueue(route)) {
    const original = {
      merchant: extraction.extraction.merchant,
      total_amount: extraction.extraction.total_amount,
      transaction_date: extraction.extraction.transaction_date ?? null,
      currency: extraction.extraction.currency ?? null,
      category: coding.category,
    };
    await pool.query(
      `INSERT INTO review_queue
         (document_id, status, route, reason, document_confidence, field_confidence, original_values, sla_due_at)
       VALUES ($1,'pending',$2,$3,$4,$5,$6, now() + interval '1 day')
       ON CONFLICT (document_id) DO UPDATE SET
         route = EXCLUDED.route,
         reason = EXCLUDED.reason,
         document_confidence = EXCLUDED.document_confidence,
         field_confidence = EXCLUDED.field_confidence,
         original_values = EXCLUDED.original_values
       WHERE review_queue.status = 'pending'`,
      [
        document_id,
        route,
        (extraction.routing?.reasons ?? []).join('; ') || null,
        extraction.document_confidence ?? 0,
        JSON.stringify(extraction.field_confidence ?? {}),
        JSON.stringify(original),
      ],
    );
  }

  const coll = await extractionsCollection();
  await coll.insertOne({ document_id, extraction, coding, policy, created_at: new Date() });

  let webhook: DeliveryResult = { delivered: false, reason: 'not routed' };
  if (route !== 'auto_approve') {
    webhook = await notify({
      event: 'expense.routed',
      document_id,
      route,
      status,
      amount: extraction.extraction.total_amount,
      merchant,
      occurred_at: new Date().toISOString(),
    });
  }

  return { document_id, extraction, coding, policy, status, route, webhook };
}

app.get('/health', async () => {
  let aiHealth: unknown = null;
  try {
    aiHealth = await ai.health();
  } catch (e) {
    aiHealth = { error: String(e) };
  }
  return { status: 'ok', ai_service: aiHealth };
});

// Submit a receipt/transaction: extract -> code -> policy -> persist -> route.
app.post('/expenses/submit', async (req, reply) => {
  const parsed = submitSchema.safeParse(req.body);
  if (!parsed.success) return reply.code(400).send({ detail: parsed.error.flatten() });
  return ingest(parsed.data);
});

// Inbound webhook: an email/scan provider pushes a document to us.
app.post('/webhooks/receipt', async (req, reply) => {
  const parsed = submitSchema.safeParse(req.body);
  if (!parsed.success) return reply.code(400).send({ detail: parsed.error.flatten() });
  const result = await ingest(parsed.data);
  return reply.code(202).send({
    accepted: true,
    document_id: result.document_id,
    route: result.policy.route,
    status: result.status,
  });
});

// List expenses (filter by status / route) — the approvals queue.
app.get('/expenses', async (req) => {
  const q = req.query as { status?: string; route?: string; limit?: string };
  const limit = Math.min(Number(q.limit ?? 50) || 50, 200);
  const params: unknown[] = [];
  const where: string[] = [];
  if (q.status) {
    params.push(q.status);
    where.push(`status = $${params.length}`);
  }
  if (q.route) {
    params.push(q.route);
    where.push(`policy_route = $${params.length}`);
  }
  const clause = where.length ? `WHERE ${where.join(' AND ')}` : '';
  const res = await pool.query(
    `SELECT * FROM expenses ${clause} ORDER BY created_at DESC LIMIT ${limit}`,
    params,
  );
  return { count: res.rowCount, items: res.rows };
});

// Hybrid merchant search (proxies the AI service's OpenSearch retrieval).
app.get('/search/merchants', async (req, reply) => {
  const q = (req.query as { q?: string }).q;
  if (!q) return reply.code(400).send({ detail: 'q query param required' });
  try {
    return await ai.searchMerchants(q);
  } catch (e) {
    return reply.code(503).send({ detail: String(e) });
  }
});

app.get('/expenses/:id', async (req, reply) => {
  const { id } = req.params as { id: string };
  const res = await pool.query('SELECT * FROM expenses WHERE id = $1', [id]);
  if (res.rowCount === 0) return reply.code(404).send({ detail: 'not found' });
  return res.rows[0];
});

const decisionSchema = z.object({ operator: z.string().default('api'), note: z.string().optional() });

async function decide(id: string, decision: 'approved' | 'rejected', body: unknown) {
  const parsed = decisionSchema.safeParse(body ?? {});
  if (!parsed.success) return { error: parsed.error.flatten() };
  const { operator, note } = parsed.data;

  const res = await pool.query(
    `UPDATE expenses SET status = $2, approved_by = $3, decided_at = now() WHERE id = $1 RETURNING *`,
    [id, decision, operator],
  );
  if (res.rowCount === 0) return { notFound: true };
  const row = res.rows[0];

  const approvals = await approvalsCollection();
  await approvals.insertOne({
    document_id: id,
    decision,
    operator,
    note: note ?? null,
    created_at: new Date(),
  });

  const webhook = await notify({
    event: 'expense.decided',
    document_id: id,
    route: row.policy_route,
    status: decision,
    amount: row.total_amount !== null ? Number(row.total_amount) : null,
    merchant: row.merchant,
    occurred_at: new Date().toISOString(),
  });
  return { row, webhook };
}

app.post('/expenses/:id/approve', async (req, reply) => {
  const { id } = req.params as { id: string };
  const out = await decide(id, 'approved', req.body);
  if ('error' in out) return reply.code(400).send({ detail: out.error });
  if ('notFound' in out) return reply.code(404).send({ detail: 'not found' });
  return out;
});

app.post('/expenses/:id/reject', async (req, reply) => {
  const { id } = req.params as { id: string };
  const out = await decide(id, 'rejected', req.body);
  if ('error' in out) return reply.code(400).send({ detail: out.error });
  if ('notFound' in out) return reply.code(404).send({ detail: 'not found' });
  return out;
});

// User correction flows back into the golden eval set.
app.post('/expenses/:id/correct', async (req, reply) => {
  const { id } = req.params as { id: string };
  const body = req.body as { corrected: Record<string, unknown>; operator?: string };
  const result = await ai.correct({ document_id: id, corrected: body.corrected, operator: body.operator ?? 'api' });
  return result;
});

// ---------------------------------------------------------------- review queue

app.get('/review', async (req) => {
  const q = req.query as { status?: string; route?: string; limit?: string };
  const limit = Math.min(Number(q.limit ?? 50) || 50, 200);
  const params: unknown[] = [];
  const where: string[] = [];
  if (q.status) {
    params.push(q.status);
    where.push(`status = $${params.length}`);
  }
  if (q.route) {
    params.push(q.route);
    where.push(`route = $${params.length}`);
  }
  const clause = where.length ? `WHERE ${where.join(' AND ')}` : '';
  const res = await pool.query(
    `SELECT * FROM review_queue ${clause} ORDER BY created_at ASC LIMIT ${limit}`,
    params,
  );
  return { count: res.rowCount, items: res.rows };
});

app.get('/review/stats', async () => {
  const byStatus = await pool.query(
    `SELECT status, count(*)::int AS count FROM review_queue GROUP BY status`,
  );
  const byRoute = await pool.query(
    `SELECT route, count(*)::int AS count FROM review_queue WHERE status = 'pending' GROUP BY route`,
  );
  const decisions = await pool.query(
    `SELECT decision, count(*)::int AS count FROM review_decisions GROUP BY decision`,
  );
  const corrections = await pool.query(`SELECT count(*)::int AS count FROM labeled_corrections`);
  return {
    queue_depth: byStatus.rows.reduce((sum, row) => sum + (row.status === 'pending' ? row.count : 0), 0),
    by_status: byStatus.rows,
    pending_by_route: byRoute.rows,
    decisions: decisions.rows,
    labeled_corrections: corrections.rows[0]?.count ?? 0,
  };
});

const reviewDecisionSchema = z.object({
  reviewer: z.string().min(1),
  decision: z.enum(['approve', 'correct', 'reject']),
  corrected: z.record(z.unknown()).optional(),
  note: z.string().optional(),
});

app.post('/review/:id/decision', async (req, reply) => {
  const { id } = req.params as { id: string };
  const parsed = reviewDecisionSchema.safeParse(req.body);
  if (!parsed.success) return reply.code(400).send({ detail: parsed.error.flatten() });
  const { reviewer, decision, corrected, note } = parsed.data;
  const error = validateDecision(decision, corrected);
  if (error) return reply.code(400).send({ detail: error });

  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    const selected = await client.query('SELECT * FROM review_queue WHERE id = $1 FOR UPDATE', [id]);
    if (selected.rowCount === 0) {
      await client.query('ROLLBACK');
      return reply.code(404).send({ detail: 'review item not found' });
    }
    const item = selected.rows[0];
    if (item.status !== 'pending') {
      await client.query('ROLLBACK');
      return reply.code(409).send({ detail: `already decided: ${item.status}` });
    }

    const original = (item.original_values ?? {}) as Record<string, unknown>;
    const correctedValues = decision === 'correct' ? { ...original, ...(corrected ?? {}) } : null;
    const diffs = correctedValues ? diffFields(original, correctedValues) : [];
    const status = decisionStatus(decision);

    await client.query(
      `INSERT INTO review_decisions
         (queue_id, document_id, reviewer, decision, original_value, corrected_value, note)
       VALUES ($1,$2,$3,$4,$5,$6,$7)`,
      [
        item.id,
        item.document_id,
        reviewer,
        decision,
        JSON.stringify(original),
        correctedValues ? JSON.stringify(correctedValues) : null,
        note ?? null,
      ],
    );

    if (decision === 'correct') {
      for (const diff of diffs) {
        await client.query(
          `INSERT INTO labeled_corrections
             (queue_id, document_id, field, original_value, corrected_value, reviewer, source)
           VALUES ($1,$2,$3,$4,$5,$6,'review_queue')`,
          [
            item.id,
            item.document_id,
            diff.field,
            JSON.stringify(diff.original ?? null),
            JSON.stringify(diff.corrected ?? null),
            reviewer,
          ],
        );
      }
    }

    await client.query(
      `UPDATE review_queue
          SET status = $2, reviewer = $3, reviewed_at = now(), corrected_values = $4
        WHERE id = $1`,
      [item.id, status, reviewer, correctedValues ? JSON.stringify(correctedValues) : null],
    );

    await client.query(
      `UPDATE expenses
          SET status = $2,
              approved_by = $3,
              decided_at = now(),
              merchant = COALESCE($4, merchant),
              total_amount = COALESCE($5, total_amount),
              payload = CASE
                WHEN $6::jsonb IS NULL THEN payload
                ELSE jsonb_set(payload, '{extraction}', COALESCE(payload->'extraction','{}'::jsonb) || $6::jsonb)
              END
        WHERE id = $1`,
      [
        item.document_id,
        status === 'rejected' ? 'rejected' : 'approved',
        reviewer,
        correctedValues?.merchant ?? null,
        correctedValues?.total_amount ?? null,
        correctedValues ? JSON.stringify(correctedValues) : null,
      ],
    );

    await client.query('COMMIT');
    return {
      document_id: item.document_id,
      reviewer,
      decision,
      status,
      original,
      corrected: correctedValues,
      corrections: diffs,
      decided_at: new Date().toISOString(),
    };
  } catch (e) {
    await client.query('ROLLBACK').catch(() => undefined);
    throw e;
  } finally {
    client.release();
  }
});

// --------------------------------------------------------------- kill switch

const killSwitchSchema = z.object({
  enabled: z.boolean(),
  actor: z.string().min(1).default('api'),
});

app.post('/admin/kill-switch', async (req, reply) => {
  const parsed = killSwitchSchema.safeParse(req.body);
  if (!parsed.success) return reply.code(400).send({ detail: parsed.error.flatten() });
  const { enabled, actor } = parsed.data;
  await pool.query(
    `INSERT INTO system_flags (key, value, updated_by, updated_at)
     VALUES ('kill_switch', $1, $2, now())
     ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_by = EXCLUDED.updated_by, updated_at = now()`,
    [JSON.stringify({ enabled }), actor],
  );
  let aiSynced = true;
  try {
    await ai.forceReview(enabled, actor);
  } catch {
    aiSynced = false;
  }
  return { enabled, actor, ai_synced: aiSynced };
});

app.get('/admin/kill-switch', async () => {
  const enabled = await killSwitchEnabled();
  return { enabled };
});

// -------------------------------------------------------------- metrics

const requestDurations = new Float64Array(2000);
let durationCursor = 0;
let durationCount = 0;

app.addHook('onRequest', async (req) => {
  (req as unknown as { _start: number })._start = performance.now();
});
app.addHook('onResponse', async (req) => {
  const start = (req as unknown as { _start?: number })._start;
  if (start === undefined) return;
  requestDurations[durationCursor % requestDurations.length] = performance.now() - start;
  durationCursor += 1;
  durationCount = Math.min(durationCount + 1, requestDurations.length);
});

function escapeLabels(labels: Record<string, string>): string {
  return Object.entries(labels)
    .map(([key, value]) => `${key}="${value.replaceAll('"', '\\"')}"`)
    .join(',');
}

function metricLine(name: string, value: number, labels?: Record<string, string>): string {
  const suffix = labels && Object.keys(labels).length ? `{${escapeLabels(labels)}}` : '';
  return `${name}${suffix} ${Number.isFinite(value) ? value : 0}`;
}

app.get('/metrics/prometheus', async (_req, reply) => {
  const sorted = Array.from(requestDurations.slice(0, durationCount)).sort((a, b) => a - b);
  const p95 = sorted.length ? sorted[Math.min(sorted.length - 1, Math.round(0.95 * (sorted.length - 1)))] : 0;
  const expenses = await pool.query(`SELECT status, count(*)::int AS count FROM expenses GROUP BY status`);
  const queue = await pool.query(`SELECT status, count(*)::int AS count FROM review_queue GROUP BY status`);
  const decisions = await pool.query(`SELECT decision, count(*)::int AS count FROM review_decisions GROUP BY decision`);
  const corrections = await pool.query('SELECT count(*)::int AS count FROM labeled_corrections');
  const totalExpenses = expenses.rows.reduce((sum, row) => sum + row.count, 0);
  const approved = expenses.rows.find((row) => row.status === 'approved')?.count ?? 0;
  const escalated = expenses.rows
    .filter((row) => row.status === 'pending' || row.status === 'rejected')
    .reduce((sum, row) => sum + row.count, 0);
  const depth = queue.rows.find((row) => row.status === 'pending')?.count ?? 0;

  const lines = [
    '# HELP expenseiq_review_queue_depth Pending human-review items.',
    '# TYPE expenseiq_review_queue_depth gauge',
    metricLine('expenseiq_review_queue_depth', depth),
    ...queue.rows.map((row) => metricLine('expenseiq_review_queue_items', row.count, { status: row.status })),
    '# HELP expenseiq_review_decisions_total Reviewer decisions.',
    '# TYPE expenseiq_review_decisions_total counter',
    ...decisions.rows.map((row) => metricLine('expenseiq_review_decisions_total', row.count, { decision: row.decision })),
    metricLine('expenseiq_labeled_corrections_total', corrections.rows[0]?.count ?? 0),
    '# HELP expenseiq_auto_approval_rate Share of expenses auto-approved.',
    '# TYPE expenseiq_auto_approval_rate gauge',
    metricLine('expenseiq_auto_approval_rate', totalExpenses ? approved / totalExpenses : 0),
    '# HELP expenseiq_error_rate Share of expenses escalated or rejected.',
    '# TYPE expenseiq_error_rate gauge',
    metricLine('expenseiq_error_rate', totalExpenses ? escalated / totalExpenses : 0),
    '# HELP expenseiq_request_latency_p95_ms Backend request p95 latency (last 2000).',
    '# TYPE expenseiq_request_latency_p95_ms gauge',
    metricLine('expenseiq_request_latency_p95_ms', p95),
    '# HELP expenseiq_requests_total Backend requests observed.',
    '# TYPE expenseiq_requests_total counter',
    metricLine('expenseiq_requests_total', durationCursor),
    '',
  ];
  reply.type('text/plain; version=0.0.4');
  return lines.join('\n');
});

async function main() {
  await ensureSchema();
  await app.listen({ port: env.port, host: '0.0.0.0' });
  app.log.info(`expenseiq backend on :${env.port} -> ai-service ${env.aiServiceUrl}`);
}

main().catch((e) => {
  app.log.error(e);
  process.exit(1);
});
