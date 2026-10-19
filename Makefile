SHELL := /bin/bash
.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# ExpenseIQ — developer entrypoints
# -----------------------------------------------------------------------------
.PHONY: help infra-up infra-down ai-install ai-run ai-test gpu-check lint fmt evals index graph-build graph-cluster graphql-probe ui-install ui-run a11y a11y-install a11y-lighthouse ray-up ray-down bench-local bench-ray backend-install backend-run backend-check assistant-eval ask

help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

infra-up: ## Start Postgres, MongoDB, OpenSearch, Neo4j, Langfuse
	cd infra && docker compose up -d

infra-down: ## Stop infra
	cd infra && docker compose down

ai-install: ## Sync the ai-service python environment (uv, py3.12)
	cd ai-service && uv sync --extra dev

ai-run: ## Run the FastAPI AI service
	cd ai-service && uv run uvicorn app.main:app --reload --port $${AI_SERVICE_PORT:-8080}

ai-test: ## Run ai-service tests
	cd ai-service && uv run pytest -q

gpu-check: ## Verify the self-hosted vLLM endpoint in .env
	cd ai-service && uv run python ../gpu/check_gpu.py

evals: ## Run the golden-set evaluation
	cd ai-service && uv run python ../evals/run_evals.py

index: ## Seed the OpenSearch hybrid indices (merchants + policies)
	cd ai-service && uv run python scripts/index_opensearch.py

graph-build: ## Rebuild the Neo4j entity graph and export runs/graph_export.json
	cd ai-service && uv run python scripts/build_graph.py --reset --export ../runs/graph_export.json

graph-cluster: ## Cluster merchants (k-means/HDBSCAN/Louvain) and persist to Neo4j
	cd ai-service && uv run python scripts/cluster_graph.py --write-neo4j

graphql-probe: ## Measure GraphQL latency into runs/graphql_probe.json
	cd ai-service && uv run python scripts/graphql_probe.py --iterations 25

ui-install: ## Create the Streamlit venv and install UI deps
	uv venv ui/.venv --python 3.12
	uv pip install --python ui/.venv/bin/python -r ui/requirements.txt

ui-run: ## Run the Streamlit review console on :8501
	cd ui && .venv/bin/streamlit run app.py --server.port 8501

a11y-install: ## Install Playwright + axe for the accessibility audit
	cd a11y && npm install && npx playwright install chromium

a11y: ## Run the axe-core audit (before/after via A11Y_LABEL)
	cd a11y && A11Y_URL=http://localhost:8501 A11Y_LABEL=$${A11Y_LABEL:-after} npm run audit

a11y-lighthouse: ## Run Lighthouse accessibility (set CHROME_PATH if needed)
	cd a11y && npx --yes lighthouse http://localhost:8501 --only-categories=accessibility --output=json --output-path=../evidence/a11y/$${A11Y_LABEL:-after}-lighthouse.json --chrome-flags="--headless=new --no-sandbox" --quiet

ray-up: ## Build + start the Ray head and two workers (profile: ray)
	docker compose -f infra/docker-compose.yml --profile ray up -d --build

ray-down: ## Stop the Ray cluster
	docker compose -f infra/docker-compose.yml --profile ray down

bench-local: ## Serial + multiprocessing ingestion benchmark (no Docker)
	cd ai-service && .venv/bin/python ../scripts/ingest_benchmark.py --docs 8000 --chunk-size 2000 --backends serial,mp --workers 2,4,8 --out ../runs/ingest_benchmark_local.json

bench-ray: ## Canonical benchmark on the Ray cluster (uses .venv python to avoid the uv hook)
	docker compose -f infra/docker-compose.yml exec -T ray-head python /workspace/scripts/ingest_benchmark.py --docs 8000 --chunk-size 2000 --backends serial,mp,ray --workers 2,4,8 --ray-address auto --no-figure --out /workspace/runs/ingest_benchmark.json
	cd ai-service && .venv/bin/python ../scripts/ingest_benchmark.py --from-json ../runs/ingest_benchmark.json --figure ../evidence/png/17-ingest-throughput.png

backend-install: ## Install backend deps
	cd backend && npm install

backend-run: ## Run the Node backend
	cd backend && npm run dev

backend-check: ## Typecheck the backend
	cd backend && npm run typecheck

lint: ## Ruff lint (python)
	cd ai-service && uv run ruff check .

fmt: ## Ruff format (python)
	cd ai-service && uv run ruff format .

ask: ## Ask the spend assistant (usage: make ask Q="how much on travel?")
	cd ai-service && uv run python -m app.assistant "$(Q)"

assistant-eval: ## Answer-correctness eval for the spend assistant (needs ai-service up)
	cd ai-service && uv run python ../evals/run_assistant_eval.py --url http://localhost:$${AI_SERVICE_PORT:-8080}
