"""Operational endpoints: Prometheus metrics and online drift checks."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from ..drift_monitor import MONITOR
from ..metrics import render

router = APIRouter(tags=["ops"])


class DriftSample(BaseModel):
    text: str = ""
    fields: dict[str, Any] = Field(default_factory=dict)


class DriftRequest(BaseModel):
    samples: list[DriftSample] = Field(min_length=1, max_length=500)


@router.get("/metrics/prometheus")
async def prometheus_metrics() -> Response:
    return Response(content=render(), media_type=CONTENT_TYPE_LATEST)


@router.post("/drift/check")
async def drift_check(req: DriftRequest) -> dict:
    return {
        "status": "ok",
        "report": MONITOR.observe([sample.model_dump() for sample in req.samples]),
        "monitor": MONITOR.status(),
    }


@router.get("/drift/status")
async def drift_status() -> dict:
    return MONITOR.status()
