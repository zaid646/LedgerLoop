"""Self-healing extraction loop.

Strategy (bounded, cheapest-first):

    validate -> retry with error feedback -> fallback model -> human queue

Every attempt is recorded so the service can report ``attempts``, ``repaired``
and ``escalated`` for the reliability metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from .config import get_settings
from .llm import LLMClient
from .registry import PromptRegistry
from .schemas import ReceiptExtraction


@dataclass
class HealOutcome:
    extraction: ReceiptExtraction | None
    prompt_version: str
    model: str
    attempts: int
    repaired: bool
    escalated: bool
    latency_ms: float
    cost_usd: float
    errors: list[str] = field(default_factory=list)
    low_confidence_fields: list[str] = field(default_factory=list)


def validate_extraction(data: dict[str, Any]) -> list[str]:
    """Return a list of human-readable schema errors (empty == valid)."""
    try:
        ReceiptExtraction.model_validate(data)
        return []
    except ValidationError as e:
        return [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]


def _low_confidence(extraction: ReceiptExtraction, threshold: float) -> list[str]:
    return [f for f, c in extraction.confidence.items() if isinstance(c, (int, float)) and c < threshold]


async def extract_with_healing(
    text: str,
    *,
    client: LLMClient | None = None,
    registry: PromptRegistry | None = None,
    stage: str = "prod",
) -> HealOutcome:
    settings = get_settings()
    client = client or LLMClient()
    registry = registry or PromptRegistry()
    prompt = registry.resolve("extraction", stage)
    schema = ReceiptExtraction.model_json_schema()

    messages: list[dict[str, str]] = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.render(text=text)},
    ]

    total_latency = 0.0
    total_cost = 0.0
    errors: list[str] = []
    model_used = prompt.params.get("model") or settings.vllm_model
    max_attempts = max(1, settings.max_extract_attempts)

    for attempt in range(1, max_attempts + 1):
        # last attempt uses the fallback model
        model = settings.vllm_fallback_model if attempt == max_attempts and max_attempts > 1 else model_used
        data, result = await client.complete_json(
            messages,
            schema=schema,
            model=model,
            temperature=float(prompt.params.get("temperature", 0.0)),
            max_tokens=int(prompt.params.get("max_tokens", 700)),
        )
        total_latency += result.latency_ms
        total_cost += result.cost_usd

        schema_errors = validate_extraction(data)
        if not schema_errors:
            extraction = ReceiptExtraction.model_validate(data)
            low = _low_confidence(extraction, settings.min_field_confidence)
            return HealOutcome(
                extraction=extraction,
                prompt_version=prompt.version,
                model=model,
                attempts=attempt,
                repaired=attempt > 1,
                escalated=bool(low),
                latency_ms=total_latency,
                cost_usd=total_cost,
                low_confidence_fields=low,
            )

        errors = schema_errors
        # feed the error back for the next attempt
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.render(text=text)},
            {"role": "assistant", "content": result.content or ""},
            {
                "role": "user",
                "content": (
                    "Your previous output was invalid: "
                    + "; ".join(schema_errors)
                    + ". Return ONLY a corrected JSON object matching the schema."
                ),
            },
        ]

    # exhausted retries -> human queue
    return HealOutcome(
        extraction=None,
        prompt_version=prompt.version,
        model=model_used,
        attempts=max_attempts,
        repaired=False,
        escalated=True,
        latency_ms=total_latency,
        cost_usd=total_cost,
        errors=errors,
        low_confidence_fields=[],
    )


async def _amain(text: str) -> int:
    from .redaction import redact

    outcome = await extract_with_healing(redact(text).text)
    print(f"prompt_version={outcome.prompt_version} model={outcome.model}")
    print(f"attempts={outcome.attempts} repaired={outcome.repaired} escalated={outcome.escalated}")
    print(f"latency_ms={outcome.latency_ms:.1f} cost_usd={outcome.cost_usd:.6f}")
    if outcome.extraction is not None:
        print(outcome.extraction.model_dump_json(indent=2))
    if outcome.errors:
        print("errors:", "; ".join(outcome.errors))
    return 0 if outcome.extraction is not None else 1


if __name__ == "__main__":
    import argparse
    import asyncio

    ap = argparse.ArgumentParser(description="Run the self-healing extraction loop once.")
    ap.add_argument("--text", required=True, help="raw receipt/transaction text")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(_amain(args.text)))
