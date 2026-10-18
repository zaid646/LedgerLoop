"""Search endpoints — hybrid BM25 + kNN retrieval over merchants and policies."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import search
from ..embeddings import get_embedder
from ..schemas import (
    SearchHealthResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)

router = APIRouter(tags=["search"])


def _run(index: str, req: SearchRequest) -> SearchResponse:
    hits = search.hybrid_search(index, req.query, req.k)
    return SearchResponse(
        query=req.query,
        index=index,
        count=len(hits),
        backend=get_embedder().backend,
        hits=[SearchHit(**h) for h in hits],
    )


@router.post("/search/merchants", response_model=SearchResponse)
async def search_merchants(req: SearchRequest) -> SearchResponse:
    try:
        return _run(search.MERCHANT_INDEX, req)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"search unavailable: {exc}") from exc


@router.post("/search/policies", response_model=SearchResponse)
async def search_policies(req: SearchRequest) -> SearchResponse:
    try:
        return _run(search.POLICY_INDEX, req)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"search unavailable: {exc}") from exc


@router.get("/search/health", response_model=SearchHealthResponse)
async def search_health() -> SearchHealthResponse:
    indices: list[str] = []
    up = search.available()
    if up:
        try:
            indices = sorted(_all_indices())
        except Exception:  # noqa: BLE001
            indices = []
    return SearchHealthResponse(opensearch=up, embedder=get_embedder().backend, indices=indices)


def _all_indices() -> list[str]:
    client = search.get_client()
    return [n for n in client.indices.get(index="expenseiq-*") if not n.startswith(".")]
