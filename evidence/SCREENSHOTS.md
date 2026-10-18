# Evidence manifest

Screenshots backing the README. Drop PNGs into `evidence/png/` using the exact
names below, then tick the box.

| # | File | What it shows | Captured |
|---|------|---------------|----------|
| 1 | `png/01-infra-up.png` | `docker compose ps` — Postgres/Mongo/OpenSearch healthy | [x] |
| 2 | `png/02-opensearch-knn.png` | `GET _cat/plugins` — knn + neural-search present | [x] |
| 3 | `png/03-gpu-vllm.png` | vLLM endpoint serving `/v1/models` (RTX 4090) | [x] |
| 4 | `png/04-notebook-dataset.png` | `01_dataset_exploration.ipynb` outputs (SROIE/CORD) | [x] |
| 5 | `png/05-notebook-baseline.png` | `02_baseline_extraction.ipynb` metrics + chart | [x] |
| 6 | `png/06-wandb-run.png` | W&B run with heuristic vs LLM curves | [x] |
| 7 | `png/07-langfuse-traces.png` | Langfuse trace tree incl. repair attempt | [x] |
| 8 | `png/08-promptfoo.png` | PromptFoo pass/fail table | [x] |
| 9 | `png/09-ci-green.png` | GitHub Actions CI passing (lint+test+gate) | [x] |
| 10 | `png/10-api-smoke.png` | `/extract` + `/code` + `/policy-check` + `/assistant/ask` (grounded answer + guarded SQL) | [x] |
| 11 | `png/11-snowflake-views.png` | Snowflake `V_SPEND_BY_CATEGORY` query result | [x] |
| 12 | `png/12-backend-submit.png` | `POST /expenses/submit` routed response | [x] |
| 13 | `png/13-qlora-eval.png` | QLoRA base (10.9%) vs adapter (98.4%) + train loss | [x] |
| 14 | `png/14-qlora-serve.png` | `/classify` served adapter predictions | [x] |

Auto-captured text artefacts (no screenshot needed) live in `evidence/logs/`.
