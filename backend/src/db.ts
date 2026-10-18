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

export async function ensureSchema(): Promise<void> {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS expenses (
      id            TEXT PRIMARY KEY,
      merchant      TEXT,
      total_amount  NUMERIC,
      currency      TEXT,
      category      TEXT,
      policy_route  TEXT,
      escalated     BOOLEAN DEFAULT FALSE,
      payload       JSONB,
      created_at    TIMESTAMPTZ DEFAULT now()
    );
  `);
}
