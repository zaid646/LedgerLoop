-- LedgerLoop human-in-the-loop review queue (Postgres).
-- Mirrored by backend/src/db.ts ensureSchema() so local dev and CI stay in sync.

CREATE TABLE IF NOT EXISTS review_queue (
  id                  BIGSERIAL PRIMARY KEY,
  document_id         TEXT NOT NULL UNIQUE,
  status              TEXT NOT NULL DEFAULT 'pending'
                      CHECK (status IN ('pending', 'approved', 'corrected', 'rejected')),
  route               TEXT NOT NULL
                      CHECK (route IN ('human_review', 'reject_escalate')),
  reason              TEXT,
  document_confidence NUMERIC(8, 6) NOT NULL DEFAULT 0,
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

-- Immutable audit log: every decision keeps the reviewer, timestamp, original
-- value and corrected value (NULL when the reviewer approved as-is).
CREATE TABLE IF NOT EXISTS review_decisions (
  id              BIGSERIAL PRIMARY KEY,
  queue_id        BIGINT NOT NULL REFERENCES review_queue (id),
  document_id     TEXT NOT NULL,
  reviewer        TEXT NOT NULL,
  decision        TEXT NOT NULL CHECK (decision IN ('approve', 'correct', 'reject')),
  original_value  JSONB NOT NULL,
  corrected_value JSONB,
  note            TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_review_decisions_doc ON review_decisions (document_id, created_at);

-- Reviewer corrections are an evaluation / fine-tuning dataset. One row per
-- changed field so downstream jobs can filter by field and value.
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

-- Runtime safety flags (kill switch). Auditable last-writer-wins for local dev.
CREATE TABLE IF NOT EXISTS system_flags (
  key         TEXT PRIMARY KEY,
  value       JSONB NOT NULL,
  updated_by  TEXT,
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
