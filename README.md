# ExpenseIQ

**An AI expense-workflow engine.** It ingests receipts and card transactions, extracts
structured fields, codes each item to a GL account, checks it against company policy, and
routes approvals to the right human. Every AI step is **versioned, evaluated, traced, and
self-correcting** — so quality is a number, not a vibe.

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-beta-blue">
  <img alt="python" src="https://img.shields.io/badge/python-3.12-blue">
  <img alt="node" src="https://img.shields.io/badge/node-22-green">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

---

## Table of contents

- [What it does](#what-it-does)
- [Objectives](#objectives)
- [Architecture](#architecture)
- [Stack & the gap each piece closes](#stack--the-gap-each-piece-closes)
- [Repository layout](#repository-layout)
- [Quickstart](#quickstart)
- [Build order / roadmap](#build-order--roadmap)
- [Measuring quality](#measuring-quality)
- [Graph layer (Neo4j + GraphQL)](#graph-layer-neo4j--graphql)
- [Merchant clustering](#merchant-clustering)
- [Distributed ingestion benchmark](#distributed-ingestion-benchmark)
- [Review UI & accessibility](#review-ui--accessibility)
- [Evidence](#evidence)
- [Security practices](#security-practices)

---

## What it does

Upload a receipt image (or push a card transaction) and ExpenseIQ:

1. **Redacts** PII / card numbers *before* anything leaves the box.
2. **Extracts** structured fields from the receipt with an LLM using schema-guided JSON.
3. **Repairs** invalid or low-confidence output automatically (retry with error feedback →
   fallback model → human-review queue).
4. **Codes** the item to a GL category using merchant/policy retrieval (hybrid BM25 + vector).
5. **Checks** policy (per-item limits, allowed categories, missing receipts) and **routes**
   for approval.
6. **Records** every model + prompt version, its accuracy, p95 latency and cost-per-document.
7. **Learns**: user corrections flow back into the golden eval set; CI blocks any prompt
   change that regresses accuracy.

## Objectives

| # | Objective | How it's met |
|---|-----------|--------------|
| 1 | **Production service** | Node.js (TypeScript) backend for submissions, approvals, webhooks, calling a Python FastAPI AI service |
| 2 | **Measured quality** | Receipt field accuracy, coding accuracy, p95 latency, cost/document tracked **per prompt and model version** |
| 3 | **Reliability** | Failed/invalid outputs repair themselves; only leftovers reach a human queue |
| 4 | **Continuous improvement** | User corrections feed the eval set; CI gate blocks prompt regressions |

## Architecture

```
                        ┌──────────────── LOCAL (Docker Desktop) ────────────────┐
   receipts / txns      │                                                        │
        │               │   ┌───────────────────┐        ┌────────────────────┐  │
        ▼               │   │  backend          │  HTTP  │  ai-service        │  │
   ┌─────────┐  POST    │   │  Node.js + TS     │ ─────▶ │  Python + FastAPI  │  │
   │ client  │ ───────▶ │   │  Fastify          │        │  · redaction       │  │
   └─────────┘          │   │  submissions /    │ ◀───── │  · extraction      │  │
        ▲   approvals   │   │  approvals /      │  JSON  │  · GL coding       │  │
        │   webhooks    │   │  webhooks         │        │  · policy check    │  │
        │               │   └────────┬──────────┘        └─────────┬──────────┘  │
        │               │            │                             │             │
        │               │   ┌────────▼─────────────────────────────▼──────────┐  │
        │               │   │ Postgres · MongoDB · OpenSearch (hybrid search) │  │
        │               │   └────────┬────────────────────────────────────────┘  │
        │               │            │  traces                                   │
        │               │   ┌────────▼─────────┐   ┌───────────────────────────┐ │
        └───────────────┼───│ Langfuse         │   │ W&B (evals, artifacts)    │ │
                        │   └──────────────────┘   └───────────────────────────┘ │
                        └─────────────────────────────┬──────────────────────────┘
                                                      │ OpenAI-compatible API
                                        ┌─────────────▼───────────────┐
                                        │ RENTED GPU (vLLM, on-demand)│
                                        │  Qwen2.5-Instruct · guided  │
                                        │  JSON · QLoRA + vLLM        │
                                        └─────────────────────────────┘
```

The model is **self-hosted** (vLLM on a rented GPU), not a paid API — so the platform
records a real **cost-per-document** derived from GPU `$/hr ÷ docs/hr`.

## Stack & the gap each piece closes

| Component | Stack | Gap closed |
|---|---|---|
| Backend | Node.js (TypeScript), Fastify, FastAPI | Node.js + Python in production |
| Experimentation | Jupyter notebooks for dataset exploration & baselines | Jupyter |
| Experiment tracking | Weights & Biases: models/prompts/retrieval compared on accuracy, latency, cost | W&B, evals, cost tracking |
| Prompt lifecycle | Prompt registry (draft → eval → prod → rollback) per use case; PromptFoo regression suite as CI gate | Prompt version control, PromptFoo |
| Self-healing harness | Schema validation, retry with error feedback, fallback model, human-review queue | Agent loops, self-healing |
| Extraction & coding | LLM structured outputs for receipts (SROIE/CORD); transaction classification; optional QLoRA served with vLLM | Document intelligence, classification, fine-tuning & serving |
| Databases | PostgreSQL (transactions), MongoDB (raw extractions, traces), OpenSearch hybrid BM25 + vector (policies, merchants) | Postgres, MongoDB, vector search |
| Observability & feedback | Langfuse traces; correction endpoint feeding the golden set | Observability, production feedback loop |
| Safeguards | PII + card-number redaction before every LLM call (reuses the **Fortress** middleware) | Sensitive financial data |
| Warehouse | Snowflake for spend analytics, queried by the "ask about my spend" assistant | Data warehouse |
| Assistant | Copilot-style agent patterns over the warehouse | Intelligent assistants |
| Graph API | Neo4j 5 + Strawberry GraphQL at `/graphql` | Entity graph + traversals over expenses, merchants, policies |
| Merchant clustering | scikit-learn (k-means/HDBSCAN), NetworkX (Louvain), fastembed | Unsupervised merchant segments vs GL codes |
| Distributed benchmark | Ray cluster (Docker) + stdlib multiprocessing | Measured parallel ingestion |
| Review UI | Streamlit + axe-core + Lighthouse | Human review + accessibility evidence |

## Repository layout

```
expenseiq/
├── backend/            Node.js (TypeScript) service — submissions/approvals/webhooks
├── ai-service/         Python FastAPI AI service (extraction, coding, policy, graph)
│   └── app/            routers, schemas, graph (Neo4j/GraphQL/clustering), llm, redaction
├── ui/                 Streamlit review console (queue, clusters, graph details)
├── a11y/               Playwright + axe-core + Lighthouse audit harness
├── scripts/            Graph build/cluster/probe + serial/mp/Ray ingestion benchmark
├── notebooks/          Jupyter: dataset exploration + baseline evals (headless-executed)
├── prompt-registry/    Versioned prompts (draft/eval/prod) per use case
├── evals/              Golden eval set + PromptFoo regression suite
├── harness/            Self-healing loop (validate → retry → fallback → human queue)
├── infra/              docker-compose: Postgres, MongoDB, OpenSearch, Neo4j, Langfuse, Ray
├── warehouse/          Snowflake DDL + queries for spend analytics
├── gpu/                Runbook + endpoint checker for the self-hosted vLLM
├── evidence/           Screenshots + logs (SCREENSHOTS.md manifest)
└── .github/workflows/  CI: lint, tests, PromptFoo regression gate
```

## Quickstart

```bash
cp .env.example .env          # fill in W&B + GPU endpoint + Snowflake

# 1) datastores + observability
cd infra && docker compose up -d && cd ..

# 2) python env + AI service
cd ai-service && uv sync && uv run uvicorn app.main:app --reload --port 8080

# 2b) seed OpenSearch hybrid indices (merchants + policies)
cd ai-service && uv sync --extra search && uv run python scripts/index_opensearch.py

# 2c) Node backend (approvals, webhooks)
cd backend && npm install && npm run dev

# 2d) entity graph + merchant clusters (Neo4j is part of infra-up)
cd ai-service && uv sync --dev --extra search
cd ai-service && uv run python scripts/build_graph.py --reset --export ../runs/graph_export.json
cd ai-service && uv run python scripts/cluster_graph.py --write-neo4j
cd ai-service && uv run python scripts/graphql_probe.py --iterations 25

# 2e) Streamlit review console
make ui-install && make ui-run          # http://localhost:8501

# 2f) Ray cluster + canonical ingestion benchmark
make ray-up && make bench-ray

# 3) GPU: follow gpu/RUNBOOK.md, then verify
python gpu/check_gpu.py
```

## Build order / roadmap

1. **Notebooks, dataset, baseline evals in W&B**
2. **FastAPI AI service, prompt registry, PromptFoo gate**
3. **Node.js backend with Postgres, MongoDB, OpenSearch hybrid retrieval, approvals + webhooks**
4. **Self-healing harness, Langfuse, feedback loop**
5. **Snowflake warehouse + spend assistant** (guarded NL→SQL over the views)
6. **QLoRA GL-coding classifier** fine-tuned and served — plus a four-way comparison (base/tuned 0.5B vs prompted 7B ± retrieval) on an **unseen-merchant** split
7. **Graph + clustering + distributed ingestion + accessible review UI** — Neo4j/GraphQL, k-means/HDBSCAN/Louvain, Ray vs multiprocessing, axe/Lighthouse before/after

## Measuring quality

| Metric | Definition | Where tracked |
|---|---|---|
| Field accuracy | per-field exact/soft match vs golden set (merchant, date, total, tax, currency) | W&B |
| Coding accuracy | predicted GL == expected GL | W&B |
| p95 latency | end-to-end per document | W&B + Langfuse |
| Cost / document | GPU `$/hr ÷ docs/hr` (self-hosted) | W&B |
| Self-heal rate | % repaired without human | Langfuse + harness logs |
| Human-queue rate | % escalated | harness logs |
| Graph latency | p50/p95 per GraphQL query | `runs/graphql_probe.json` |
| Cluster agreement | ARI / NMI / purity vs GL codes | `runs/graph_clusters.json` |
| Ingest throughput | docs/s, speedup, efficiency | `runs/ingest_benchmark.json` |
| Accessibility | axe violations + Lighthouse score | `runs/a11y_scores.json` |

## Graph layer (Neo4j + GraphQL)

`ai-service` serves a Strawberry GraphQL schema at `POST /graphql` over Neo4j 5
(`infra/docker-compose.yml`, `eq-neo4j`). The graph is rebuilt deterministically from the
same golden set and merchant catalog used by the evals.

| Query | Use |
|---|---|
| `graphStats` | entity/relationship counts, cluster count |
| `entities(type, limit, offset)` | paginated expense/merchant/category lists |
| `entity(id)` | single node with properties |
| `neighbors(id, depth, relType)` | 1–3 hop traversal (policy, route, merchant, date) |
| `searchEntities(q)` | case-insensitive label search |
| `clusters(algorithm)` | persisted k-means / HDBSCAN / Louvain results |

Shape after `make graph-build`: **175 entities / 462 relationships** — 75 Expense,
63 Merchant, 13 Category, 8 Policy, 11 Date, 2 Source, 3 Route; 4 `VIOLATES` edges.
Writing clusters adds 16 `:Cluster` nodes and 186 `IN_COMMUNITY` edges (648 relationships
visible through GraphQL; `entities` counts only `:Entity` nodes).

| Query | p50 | p95 |
|---|---:|---:|
| `graphStats` | 11.4 ms | 24.2 ms |
| `entity` | 4.1 ms | 7.3 ms |
| `neighbors` | 7.3 ms | 9.1 ms |
| `searchEntities` | 4.5 ms | 5.5 ms |
| `clusters` | 6.1 ms | 7.5 ms |

25 iterations each, local Docker, HTTP. Raw data: [`runs/graphql_probe.json`](runs/graphql_probe.json).

## Merchant clustering

63 merchants, 11 observed GL categories, 384-d fastembed vectors
(`BAAI/bge-small-en-v1.5`), 34 merchants enriched with receipt context. Labels are the
extracted GL category; the question is whether merchant semantics recover that taxonomy
without supervision.

| Algorithm | Clusters | Noise | ARI | NMI | Purity | Modularity |
|---|---:|---:|---:|---:|---:|---:|
| k-means (k=11) | 11 | 0 | 0.208 | 0.494 | **0.619** | — |
| HDBSCAN | 2 | 0 | **0.248** | 0.324 | 0.444 | — |
| Louvain (kNN k=5) | 3 | 0 | 0.116 | 0.294 | 0.444 | **0.509** |

Purity is the share of the dominant GL code in a cluster; ARI/NMI compare cluster
assignments to those codes. The taxonomy is coarse and overlapping (e.g. Amazon can be
office supplies or hardware), so purity ~0.62 is the realistic ceiling for this label set.

![cluster comparison](evidence/png/16-cluster-compare.png)

## Distributed ingestion benchmark

Offline path only: redaction → heuristic extraction → GL coding → policy → 384-d hashed
embedding. 8,000 documents, chunk size 2,000, **8 CPUs** (Ray head + two 4-CPU workers),
10-core host. Throughput excludes one-time pool/cluster setup; wall clock is in the JSON.

| Backend | Workers | docs/s | Speedup | Efficiency | p50 / p95 |
|---|---:|---:|---:|---:|---:|
| serial | 1 | 4,630 | 1.00× | 1.00 | 0.224 / 0.399 ms |
| multiprocessing | 2 | 9,216 | 1.99× | 0.995 | 0.222 / 0.390 ms |
| multiprocessing | 4 | 14,727 | 3.18× | 0.795 | 0.283 / 0.466 ms |
| multiprocessing | 8 | 15,753 | 3.40× | 0.425 | 0.271 / 0.428 ms |
| **Ray cluster** | 8 | **13,460** | **2.91×** | 0.363 | 0.303 / 0.491 ms |

All five runs produced the same output hash (`532b8f45c5a6a556`). Ray setup was 21 ms;
the cluster run is 0.62 s wall clock vs 1.73 s serial. For sub-millisecond documents,
local multiprocessing wins; Ray's value is scheduling the same unit across separate
containers and nodes, which matters once a document includes an LLM/GPU call. The figure
uses steady-state throughput; the JSON includes both wall-clock and processing-only
numbers.

![ingestion throughput](evidence/png/17-ingest-throughput.png)

Reproduce: `make ray-up && make bench-ray` (cluster) or `make bench-local` (serial + mp).

## Review UI & accessibility

Streamlit console on `:8501` over the GraphQL API: a filterable review queue with policy
route and violations, merchant clusters, graph counts, and a neighbor inspector.
`make ui-install && make ui-run`.

The initial build was audited, remediated, and re-audited:

| Audit | Before | After |
|---|---|---|
| axe-core violations | 2 (1 serious `color-contrast`, 1 moderate `heading-order`) | **0** |
| Lighthouse accessibility | 95 | **100** |

Fixes: removed caption opacity, darkened helper text, exposed the search label, changed
section headings from h3 to h2, added a visible focus outline, and added a text summary
next to the category chart. The keyboard pass confirms Streamlit's skip link is the first
tab stop, the search field has an accessible name, and focus rings are visible.

[before](evidence/a11y/before.png) · [after](evidence/a11y/after.png) ·
[scores](runs/a11y_scores.json) · [keyboard trace](evidence/a11y/keyboard.json)

## Evidence

Screenshots and captured logs live under [`evidence/`](evidence/) with a capture
manifest in [`evidence/SCREENSHOTS.md`](evidence/SCREENSHOTS.md).

| Infra & search | Self-hosted GPU inference | Tracing |
|---|---|---|
| ![infra](evidence/png/01-infra-up.png) | ![gpu](evidence/png/03-gpu-vllm.png) | ![langfuse](evidence/png/07-langfuse-traces.png) |
| **Quality (W&B)** | **PromptFoo** | **CI** |
| ![wandb](evidence/png/06-wandb-run.png) | ![promptfoo](evidence/png/08-promptfoo.png) | ![ci](evidence/png/09-ci-green.png) |
| **API smoke** | **Snowflake views** | **End-to-end submit** |
| ![api](evidence/png/10-api-smoke.png) | ![snowflake](evidence/png/11-snowflake-views.png) | ![submit](evidence/png/12-backend-submit.png) |
| **QLoRA fine-tune** | **QLoRA served** | **Coding compare** |
| ![qlora-eval](evidence/png/13-qlora-eval.png) | ![qlora-serve](evidence/png/14-qlora-serve.png) | ![coding-compare](evidence/png/15-coding-compare.png) |
| **Merchant clustering** | **Ingestion throughput** | **UI accessibility** |
| ![clusters](evidence/png/16-cluster-compare.png) | ![ingest](evidence/png/17-ingest-throughput.png) | [before](evidence/a11y/before.png) → [after](evidence/a11y/after.png) |

Notebooks: [`01_dataset_exploration`](evidence/png/04-notebook-dataset.png) ·
[`02_baseline_extraction`](evidence/png/05-notebook-baseline.png).

## Security practices

- PII and card numbers are redacted **before** every LLM call.
- Secrets live only in `.env` (gitignored); `.env.example` documents required keys.
- Self-hosted inference keeps document content off third-party model APIs.

## License

MIT — see [LICENSE](LICENSE).
