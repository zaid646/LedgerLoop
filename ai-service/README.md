# ExpenseIQ AI service (Python · FastAPI)

FastAPI service that performs **receipt extraction**, **GL coding** and **policy checks**
for the ExpenseIQ backend. Inference is delegated to a self-hosted **vLLM** endpoint
(OpenAI-compatible); PII/card numbers are redacted before every call.

```
app/
├── main.py            # FastAPI app + lifespan
├── config.py          # pydantic-settings from ../../.env
├── schemas.py         # pydantic request/response + extraction models
├── llm.py             # async OpenAI-compatible client (vLLM)
├── redaction.py       # Fortress PII / card redaction middleware
├── registry.py        # prompt registry (draft/eval/prod/rollback)
├── self_heal.py       # validate → retry → fallback → human queue
├── tracing.py         # Langfuse wrapper (no-op if unconfigured)
├── metrics.py         # per-request latency / cost accounting
└── routers/
    ├── extract.py     # POST /extract   receipt → structured fields
    ├── code.py        # POST /code      transaction → GL category
    └── policy.py      # POST /policy-check
```

## Run

```bash
uv sync --dev
uv run uvicorn app.main:app --reload --port 8080
```

## Test

```bash
uv run pytest -q
```
