import { Pool } from 'pg';
import { MongoClient, type Collection, type Document } from 'mongodb';
import { env } from './env.js';

export const pool = new Pool({ connectionString: env.postgresDsn });

let mongoClient: MongoClient | null = null;

export async function mongo(): Promise<MongoClient> {
  if (!mongoClient) {
    mongoClient = new MongoClient(env.mongoUri);
    await mongoClient.connect();
  }
  return mongoClient;
}

export async function extractionsCollection(): Promise<Collection<Document>> {
  const client = await mongo();
  return client.db(env.mongoDb).collection('extractions');
}

export async function approvalsCollection(): Promise<Collection<Document>> {
  const client = await mongo();
  return client.db(env.mongoDb).collection('approvals');
}

export async function ensureSchema(): Promise<void> {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS expenses (
      id            TEXT PRIMARY KEY,
      merchant      TEXT,
      total_amount  NUMERIC,
      currency      TEXT,
      category      TEXT,
      policy_route  TEXT,
      status        TEXT DEFAULT 'pending',
      escalated     BOOLEAN DEFAULT FALSE,
      approved_by   TEXT,
      decided_at    TIMESTAMPTZ,
      payload       JSONB,
      created_at    TIMESTAMPTZ DEFAULT now()
    );
  `);
  // Additive migrations for databases created by earlier versions.
  await pool.query(`
    ALTER TABLE expenses ADD COLUMN IF NOT EXISTS status      TEXT DEFAULT 'pending';
    ALTER TABLE expenses ADD COLUMN IF NOT EXISTS approved_by TEXT;
    ALTER TABLE expenses ADD COLUMN IF NOT EXISTS decided_at  TIMESTAMPTZ;
    CREATE INDEX IF NOT EXISTS idx_expenses_status ON expenses (status);
  `);

  // Human-in-the-loop review queue + immutable audit trail.
  await pool.query(`
    CREATE TABLE IF NOT EXISTS review_queue (
      id                  BIGSERIAL PRIMARY KEY,
      document_id         TEXT NOT NULL UNIQUE,
      status              TEXT NOT NULL DEFAULT 'pending'
                          CHECK (status IN ('pending','approved','corrected','rejected')),
      route               TEXT NOT NULL
                          CHECK (route IN ('human_review','reject_escalate')),
      reason              TEXT,
      document_confidence NUMERIC(8,6) NOT NULL DEFAULT 0,
      field_confidence    JSONB NOT NULL DEFAULT '{}'::jsonb,
      original_values     JSONB NOT NULL DEFAULT '{}'::jsonb,
      corrected_values    JSONB,
      reviewer            TEXT,
      reviewed_at         TIMESTAMPTZ,
      sla_due_at          TIMESTAMPTZ,
      created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_review_queue_status ON review_queue (status, created_at);
    CREATE INDEX IF NOT EXISTS idx_review_queue_route ON review_queue (route, created_at);

    CREATE TABLE IF NOT EXISTS review_decisions (
      id              BIGSERIAL PRIMARY KEY,
      queue_id        BIGINT NOT NULL REFERENCES review_queue (id),
      document_id     TEXT NOT NULL,
      reviewer        TEXT NOT NULL,
      decision        TEXT NOT NULL CHECK (decision IN ('approve','correct','reject')),
      original_value  JSONB NOT NULL,
      corrected_value JSONB,
      note            TEXT,
      created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_review_decisions_doc ON review_decisions (document_id, created_at);

    CREATE TABLE IF NOT EXISTS labeled_corrections (
      id              BIGSERIAL PRIMARY KEY,
      queue_id        BIGINT REFERENCES review_queue (id),
      document_id     TEXT NOT NULL,
      field           TEXT NOT NULL,
      original_value  JSONB,
      corrected_value JSONB NOT NULL,
      reviewer        TEXT NOT NULL,
      source          TEXT NOT NULL DEFAULT 'review_queue',
      created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_labeled_corrections_doc ON labeled_corrections (document_id, field);

    CREATE TABLE IF NOT EXISTS system_flags (
      key         TEXT PRIMARY KEY,
      value       JSONB NOT NULL,
      updated_by  TEXT,
      updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    );
  `);
}

export async function killSwitchEnabled(): Promise<boolean> {
  const res = await pool.query("SELECT value FROM system_flags WHERE key = 'kill_switch'");
  if (res.rowCount === 0) return false;
  return Boolean((res.rows[0].value as { enabled?: boolean }).enabled);
}
