SHELL := /bin/bash
.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# ExpenseIQ — developer entrypoints
# -----------------------------------------------------------------------------
.PHONY: help infra-up infra-down ai-install ai-run ai-test gpu-check lint fmt evals index backend-install backend-run backend-check

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

infra-up: ## Start Postgres, MongoDB, OpenSearch, Langfuse
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
