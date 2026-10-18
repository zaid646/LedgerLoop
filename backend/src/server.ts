import Fastify from 'fastify';
import cors from '@fastify/cors';
import { z } from 'zod';
import { env } from './env.js';
import { approvalsCollection, ensureSchema, extractionsCollection, pool } from './db.js';
import { ai } from './aiService.js';
import { notify, type DeliveryResult } from './webhooks.js';

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
  const status = policy.route === 'auto_approve' ? 'approved' : 'pending';

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
      extraction.escalated,
      JSON.stringify({ extraction, coding, policy }),
    ],
  );

  const coll = await extractionsCollection();
  await coll.insertOne({ document_id, extraction, coding, policy, created_at: new Date() });

  let webhook: DeliveryResult = { delivered: false, reason: 'not routed' };
  if (policy.route !== 'auto_approve') {
    webhook = await notify({
      event: 'expense.routed',
      document_id,
      route: policy.route,
      status,
      amount: extraction.extraction.total_amount,
      merchant,
      occurred_at: new Date().toISOString(),
    });
  }

  return { document_id, extraction, coding, policy, status, webhook };
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

async function main() {
  await ensureSchema();
  await app.listen({ port: env.port, host: '0.0.0.0' });
  app.log.info(`expenseiq backend on :${env.port} -> ai-service ${env.aiServiceUrl}`);
}

main().catch((e) => {
  app.log.error(e);
  process.exit(1);
});
