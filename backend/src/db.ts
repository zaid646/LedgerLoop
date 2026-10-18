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
}
