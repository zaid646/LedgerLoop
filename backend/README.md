# ExpenseIQ backend (Node.js · TypeScript · Fastify)

Production-facing API for submissions, approvals and webhooks. It orchestrates the
Python AI service (`/extract` → `/code` → `/policy-check`), persists results to
Postgres + MongoDB, and routes items per policy.

```
POST /expenses/submit      body {document_id, text, amount?, has_receipt?}
GET  /expenses/:id
POST /expenses/:id/correct body {corrected, operator?}
GET  /health
```

## Run

```bash
cp .env.example .env
npm install
npm run dev          # tsx watch on :8081
npm run typecheck
```

Requires the AI service (`:8080`) and the infra datastores (`infra/docker-compose.yml`).
