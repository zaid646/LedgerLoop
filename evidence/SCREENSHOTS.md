# Evidence manifest

Screenshots backing the README. Drop PNGs into `evidence/png/` using the exact
names below, then tick the box.

| # | File | What it shows | Captured |
|---|------|---------------|----------|
| 1 | `png/01-infra-up.png` | `docker compose ps` — Postgres/Mongo/OpenSearch healthy | [ ] |
| 2 | `png/02-opensearch-knn.png` | `GET _cat/plugins` — knn + neural-search present | [ ] |
| 3 | `png/03-gpu-vllm.png` | vLLM endpoint serving `/v1/models` (RTX 4090) | [ ] |
| 4 | `png/04-notebook-dataset.png` | `01_dataset_exploration.ipynb` outputs (SROIE/CORD) | [ ] |
| 5 | `png/05-notebook-baseline.png` | `02_baseline_extraction.ipynb` metrics + chart | [ ] |
| 6 | `png/06-wandb-run.png` | W&B run with heuristic vs LLM curves | [ ] |
| 7 | `png/07-langfuse-traces.png` | Langfuse trace tree incl. repair attempt | [ ] |
| 8 | `png/08-promptfoo.png` | PromptFoo pass/fail table | [ ] |
| 9 | `png/09-ci-green.png` | GitHub Actions CI passing (lint+test+gate) | [ ] |
| 10 | `png/10-api-smoke.png` | `/extract` + `/code` + `/policy-check` responses | [ ] |
| 11 | `png/11-snowflake-views.png` | Snowflake `V_SPEND_BY_CATEGORY` query result | [ ] |
| 12 | `png/12-backend-submit.png` | `POST /expenses/submit` routed response | [ ] |

Auto-captured text artefacts (no screenshot needed) live in `evidence/logs/`.
