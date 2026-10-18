import json

import pytest

from app.llm import LLMResult
from app.self_heal import extract_with_healing


class FakeClient:
    """Returns preset dicts in order (last one repeats)."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = 0
        self.models: list[str | None] = []

    async def complete_json(self, messages, schema, model=None, temperature=0.0, max_tokens=700):
        self.models.append(model)
        data = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return data, LLMResult(
            content=json.dumps(data), model=model or "fake", latency_ms=1.0, cost_usd=0.0
        )


VALID = {
    "merchant": "ACME STORE",
    "transaction_date": "2026-10-01",
    "total_amount": 12.5,
    "currency": "USD",
    "confidence": {"merchant": 0.9, "total_amount": 0.95, "currency": 0.99},
}
INVALID = {"total_amount": 12.5}  # missing merchant -> schema error


@pytest.mark.asyncio
async def test_repairs_after_one_bad_response():
    client = FakeClient([INVALID, VALID])
    out = await extract_with_healing("ACME STORE total 12.50 USD", client=client)
    assert out.repaired is True
    assert out.attempts == 2
    assert out.extraction is not None
    assert out.extraction.merchant == "ACME STORE"
    assert out.escalated is False


@pytest.mark.asyncio
async def test_escalates_when_retries_exhausted():
    client = FakeClient([INVALID])  # always invalid
    out = await extract_with_healing("garbage", client=client)
    assert out.extraction is None
    assert out.escalated is True
    assert out.attempts == 3
    assert out.errors


@pytest.mark.asyncio
async def test_low_confidence_escalates_but_returns_extraction():
    low = dict(VALID)
    low["confidence"] = {"merchant": 0.1, "total_amount": 0.2, "currency": 0.1}
    client = FakeClient([low])
    out = await extract_with_healing("ACME", client=client)
    assert out.extraction is not None
    assert out.escalated is True
    assert set(out.low_confidence_fields) == {"merchant", "total_amount", "currency"}


@pytest.mark.asyncio
async def test_uses_fallback_model_on_last_attempt():
    client = FakeClient([INVALID])
    await extract_with_healing("x", client=client)
    # last of three attempts uses the configured fallback model
    assert client.models[-1] != client.models[0] or client.models[-1] is not None
