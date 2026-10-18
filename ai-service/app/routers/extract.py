"""POST /extract — redact then LLM-extract structured receipt fields (self-healing)."""
from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi import APIRouter

from ..config import REPO_ROOT
from ..metrics import STORE
from ..redaction import redact
from ..schemas import ExtractRequest, ExtractResponse, ReceiptExtraction
from ..self_heal import extract_with_healing
from ..tracing import get_tracer

router = APIRouter(tags=["extraction"])

HUMAN_QUEUE = REPO_ROOT / "runs" / "human_queue.jsonl"


def _enqueue(payload: dict) -> None:
    HUMAN_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with HUMAN_QUEUE.open("a") as fh:
        fh.write(json.dumps(payload) + "\n")


@router.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest) -> ExtractResponse:
    tracer = get_tracer()

    # 1. redact PII / cards BEFORE anything leaves the process
    red = redact(req.text)

    # 2. self-healing extraction
    outcome = await extract_with_healing(red.text)

    extraction = outcome.extraction
    if extraction is None:
        # exhausted retries -> human review queue
        extraction = ReceiptExtraction(merchant="", confidence={})
        _enqueue(
            {
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                "document_id": req.document_id,
                "reason": "schema_invalid_after_retries",
                "errors": outcome.errors,
                "redacted_text": red.text,
            }
        )

    if outcome.escalated and extraction is not None and outcome.extraction is not None:
        _enqueue(
            {
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                "document_id": req.document_id,
                "reason": "low_confidence",
                "fields": outcome.low_confidence_fields,
            }
        )

    STORE.observe("/extract", outcome.latency_ms, outcome.cost_usd)
    STORE.incr("extract_requests")
    if outcome.repaired:
        STORE.incr("extract_repaired")
    if outcome.escalated:
        STORE.incr("extract_escalated")

    tracer.generation(
        name="extract",
        model=outcome.model,
        input={"text": red.text},
        output=extraction.model_dump(),
        usage={"total_tokens": 0},
        metadata={
            "prompt_version": outcome.prompt_version,
            "attempts": outcome.attempts,
            "repaired": outcome.repaired,
            "escalated": outcome.escalated,
            "redactions": red.counts,
        },
    )

    return ExtractResponse(
        document_id=req.document_id,
        extraction=extraction,
        prompt_version=outcome.prompt_version,
        model=outcome.model,
        attempts=outcome.attempts,
        repaired=outcome.repaired,
        escalated=outcome.escalated,
        latency_ms=round(outcome.latency_ms, 2),
        cost_usd=round(outcome.cost_usd, 8),
        redactions=red.counts,
    )
