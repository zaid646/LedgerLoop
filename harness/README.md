# Self-healing harness

The repair loop lives in `ai-service/app/self_heal.py` (`extract_with_healing`)
and is invoked by the `/extract` route. It is deliberately small and explicit so
the control flow is auditable.

```
input
  │
  ├─ redact()        strip PII before the text ever leaves the box (redaction.py)
  │
  ▼
attempt 1  ── schema-valid? ──yes──► return (attempts=1, repaired=False)
  │ no
  ▼
attempt 2  feed the validation error back as a system hint  ──yes──► return (repaired=True)
  │ no
  ▼
fallback model (if configured)                              ──yes──► return
  │ no
  ▼
escalate to human queue (escalated=True, low confidence)
```

Guarantees:

- **Schema-first**: output must satisfy the pydantic `ReceiptExtraction` schema
  (guided JSON via vLLM when available).
- **Bounded**: at most `MAX_ATTEMPTS` model calls per document.
- **Observable**: every attempt, repair, fallback and escalation is counted in
  `metrics.py` and emitted as a Langfuse span.
- **Safe**: if all attempts fail the document is escalated, never silently
  accepted.

Every automated repair and every human correction is appended to the golden set
(`evals/golden/`) so regressions become test cases.

## Exercising it

With the GPU endpoint up:

```bash
cd ai-service
uv run python -m app.self_heal --text "TOTAL 12.50  ACME COFFEE  03/14/2024"
```

Without a GPU the loop short-circuits to `escalated=True`, which is also asserted
in `ai-service/tests/test_harness.py`.
