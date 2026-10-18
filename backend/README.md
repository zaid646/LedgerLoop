# ExpenseIQ backend (Node.js · TypeScript · Fastify)

Production-facing API for submissions, approvals and webhooks. It orchestrates the
Python AI service (`/extract` → `/code` → `/policy-check`), persists results to
Postgres + MongoDB, and routes items per policy.

```
POST /expenses/submit        body {document_id, text, amount?, has_receipt?}
GET  /expenses               ?status=pending&route=manager_review&limit=50
GET  /expenses/:id
POST /expenses/:id/approve   body {operator, note?}
POST /expenses/:id/reject    body {operator, note?}
POST /expenses/:id/correct   body {corrected, operator?}
POST /webhooks/receipt       inbound: {document_id, text, amount?, has_receipt?}
GET  /search/merchants       ?q=office chair
GET  /health
```

Routes with a non-`auto_approve` decision land in the approvals queue
(`status=pending`); `approve`/`reject` record the decision in Postgres
(`approved_by`, `decided_at`) plus a MongoDB `approvals` event. When `WEBHOOK_URL`
is set, `expense.routed` / `expense.decided` events are POSTed outbound.

## Run

```bash
cp .env.example .env
npm install
npm run dev          # tsx watch on :8081
npm run typecheck
```

Requires the AI service (`:8080`) and the infra datastores (`infra/docker-compose.yml`).
