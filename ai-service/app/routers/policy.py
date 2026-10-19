"""POST /policy-check — deterministic company-policy evaluation and routing."""
from __future__ import annotations

from fastapi import APIRouter

from ..policy import evaluate
from ..schemas import PolicyCheckRequest, PolicyCheckResponse

router = APIRouter(tags=["policy"])


@router.post("/policy-check", response_model=PolicyCheckResponse)
async def policy_check(req: PolicyCheckRequest) -> PolicyCheckResponse:
    return evaluate(req)
