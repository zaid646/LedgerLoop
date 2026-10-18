"""POST /assistant/ask — natural-language questions over the analytics warehouse."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import assistant
from ..metrics import STORE
from ..schemas import AssistantRequest, AssistantResponse
from ..sql_guard import SqlRejected
from ..tracing import get_tracer
from ..warehouse import WarehouseError

router = APIRouter(tags=["assistant"])


@router.post("/assistant/ask", response_model=AssistantResponse)
async def ask(req: AssistantRequest) -> AssistantResponse:
    tracer = get_tracer()
    try:
        result = await assistant.ask(req.question)
    except SqlRejected as exc:
        raise HTTPException(status_code=400, detail=f"generated SQL rejected: {exc}") from exc
    except WarehouseError as exc:
        raise HTTPException(status_code=503, detail=f"warehouse unavailable: {exc}") from exc

    STORE.observe("/assistant/ask", result["latency_ms"], result["cost_usd"])
    STORE.incr("assistant_questions")

    tracer.generation(
        name="assistant",
        model=result["model"],
        input={"question": req.question},
        output={"sql": result["sql"], "answer": result["answer"], "row_count": result["row_count"]},
        metadata={"prompt_version": "v1"},
    )
    return AssistantResponse(**result)
