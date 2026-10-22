"""Admin safety controls (kill switch)."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..routing import KILL_SWITCH

router = APIRouter(tags=["admin"])


class KillSwitchRequest(BaseModel):
    enabled: bool
    actor: str = Field(default="api", min_length=1)


@router.get("/admin/kill-switch")
async def get_kill_switch() -> dict:
    return KILL_SWITCH.status()


@router.post("/admin/kill-switch")
async def set_kill_switch(req: KillSwitchRequest) -> dict:
    return KILL_SWITCH.set(req.enabled, actor=req.actor)
