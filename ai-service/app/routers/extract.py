"""POST /extract — redact, extract, score confidence and route for human review."""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from fastapi import APIRouter

from ..confidence import get_scorer
from ..confidence.signals import infer_policy_route
from ..config import REPO_ROOT
from ..metrics import STORE
from ..redaction import redact
from ..routing import ROUTE_AUTO, route_document
from ..schemas import ExtractRequest, ExtractResponse, ReceiptExtraction, RoutingInfo
from ..self_heal import extract_with_healing
from ..tracing import get_tracer

log = logging.getLogger("expenseiq.extract")
router = APIRouter(tags=["extraction"])

HUMAN_QUEUE = REPO_ROOT / "runs" / "human_queue.jsonl"


def _enqueue(payload: dict) -> None:
    HUMAN_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with HUMAN_QUEUE.open("a") as fh:
        fh.write(json.dumps(payload) + "\n")


def _score_and_route(
    extraction: ReceiptExtraction,
    text: str,
    *,
    attempts: int,
    repaired: bool,
    schema_ok: bool,
    source: str,
):
    scorer = get_scorer()
    policy_route, _ = infer_policy_route(extraction, source=source)
    result = scorer.score(
        extraction,
        text,
        attempts=attempts,
        repaired=repaired,
        schema_ok=schema_ok,
        policy_route=policy_route,
        source=source,
    )
    missing = [
        name
        for name, spec in scorer.config.fields.items()
        if spec.required and not result.field_signals[name].present
    ]
    decision = route_document(
        result,
        config=scorer.config,
        policy_route=policy_route,
        schema_ok=schema_ok,
        missing_required=missing,
    )
    return result, decision


@router.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest) -> ExtractResponse:
    tracer = get_tracer()

    # 1. redact PII / cards BEFORE anything leaves the process
    red = redact(req.text)

    # 2. self-healing extraction
    outcome = await extract_with_healing(red.text)

    extraction = outcome.extraction
    if extraction is None:
        # exhausted retries -> never auto-approve
        extraction = ReceiptExtraction(merchant="", confidence={})

    # 3. calibrated confidence + threshold routing (fail-closed on errors)
    try:
        result, decision = _score_and_route(
            extraction,
            red.text,
            attempts=outcome.attempts,
            repaired=outcome.repaired,
            schema_ok=outcome.extraction is not None,
            source=req.source,
        )
        routing = RoutingInfo(**decision.to_dict())
        document_confidence = result.document_score
        field_confidence = result.field_scores
        confidence_notes = result.notes
    except Exception as exc:  # noqa: BLE001 - confidence must never break extraction
        log.exception("confidence scoring failed; forcing human review")
        routing = RoutingInfo(
            route="human_review",
            reasons=[f"confidence_error:{type(exc).__name__}"],
            document_score=0.0,
        )
        document_confidence = 0.0
        field_confidence = {}
        confidence_notes = ["confidence_scoring_error"]

    if routing.route != ROUTE_AUTO:
        _enqueue(
            {
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                "document_id": req.document_id,
                "reason": routing.reasons,
                "route": routing.route,
                "document_confidence": document_confidence,
                "field_confidence": field_confidence,
                "escalated": outcome.escalated,
                "errors": outcome.errors,
                "redacted_text": red.text,
            }
        )

    STORE.observe("/extract", outcome.latency_ms, outcome.cost_usd)
    STORE.incr("extract_requests")
    if outcome.repaired:
        STORE.incr("extract_repaired")
    if routing.route != ROUTE_AUTO:
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
            "escalated": routing.route != ROUTE_AUTO,
            "routing": routing.model_dump(),
            "document_confidence": document_confidence,
            "field_confidence": field_confidence,
            "redactions": red.counts,
        },
    )
    if hasattr(tracer, "score"):
        tracer.score(
            name="document_confidence",
            value=float(document_confidence or 0.0),
            comment=f"route={routing.route}",
            metadata={"document_id": req.document_id},
        )

    return ExtractResponse(
        document_id=req.document_id,
        extraction=extraction,
        prompt_version=outcome.prompt_version,
        model=outcome.model,
        attempts=outcome.attempts,
        repaired=outcome.repaired,
        escalated=routing.route != ROUTE_AUTO,
        latency_ms=round(outcome.latency_ms, 2),
        cost_usd=round(outcome.cost_usd, 8),
        redactions=red.counts,
        document_confidence=round(float(document_confidence or 0.0), 6),
        field_confidence={k: round(v, 6) for k, v in field_confidence.items()},
        routing=routing,
        confidence_notes=confidence_notes,
    )
