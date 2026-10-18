import Fastify from 'fastify';
import cors from '@fastify/cors';
import { z } from 'zod';
import { env } from './env.js';
import { ensureSchema, extractionsCollection, pool } from './db.js';
import { ai } from './aiService.js';

const app = Fastify({ logger: { level: env.logLevel } });

await app.register(cors, { origin: true });

const submitSchema = z.object({
  document_id: z.string().min(1),
  text: z.string().min(1),
  amount: z.number().optional(),
  has_receipt: z.boolean().default(true),
});

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
  const { document_id, text, has_receipt } = parsed.data;

  const extraction = await ai.extract(document_id, text);
  const merchant = extraction.extraction.merchant;
  const amount = parsed.data.amount ?? extraction.extraction.total_amount ?? 0;

  const coding = await ai.code(merchant, merchant, amount);
  const policy = await ai.policyCheck(amount, coding.category, has_receipt);

  await pool.query(
    `INSERT INTO expenses (id, merchant, total_amount, currency, category, policy_route, escalated, payload)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
     ON CONFLICT (id) DO UPDATE SET category=$5, policy_route=$6, escalated=$7, payload=$8`,
    [
      document_id,
      merchant,
      extraction.extraction.total_amount,
      extraction.extraction.currency,
      coding.category,
      policy.route,
      extraction.escalated,
      JSON.stringify({ extraction, coding, policy }),
    ],
  );

  const coll = await extractionsCollection();
  await coll.insertOne({ document_id, extraction, coding, policy, created_at: new Date() });

  return { document_id, extraction, coding, policy };
});

app.get('/expenses/:id', async (req, reply) => {
  const { id } = req.params as { id: string };
  const res = await pool.query('SELECT * FROM expenses WHERE id = $1', [id]);
  if (res.rowCount === 0) return reply.code(404).send({ detail: 'not found' });
  return res.rows[0];
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
