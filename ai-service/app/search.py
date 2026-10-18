"""OpenSearch hybrid (BM25 + kNN) retrieval over merchants and policies.

Closes the "hybrid BM25 + vector" gap: documents are indexed with a Lucene
``knn_vector`` field and a full-text field; queries run through a
**normalization-processor** search pipeline that min-max normalises and combines
the lexical and vector scores.

Everything degrades gracefully: if OpenSearch is unreachable the AI service
falls back to its built-in hints (see ``routers/code.py``) and CI stays green.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Any

from .config import get_settings
from .embeddings import get_embedder

log = logging.getLogger("expenseiq.search")

MERCHANT_INDEX = "expenseiq-merchants"
POLICY_INDEX = "expenseiq-policies"
PIPELINE = "expenseiq-hybrid"
VECTOR_FIELD = "embedding"
TEXT_FIELD = "text"

_client: Any = None


# --------------------------------------------------------------------- client


def get_client() -> Any:
    global _client
    if _client is None:
        from opensearchpy import OpenSearch

        s = get_settings()
        kwargs: dict[str, Any] = {"hosts": [s.opensearch_url], "timeout": 5, "max_retries": 1}
        if s.opensearch_use_security:
            kwargs["http_auth"] = (s.opensearch_user, s.opensearch_password)
        _client = OpenSearch(**kwargs)
    return _client


def available() -> bool:
    try:
        return bool(get_client().ping())
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------ mappings


def _mapping() -> dict:
    dim = get_embedder().dim
    return {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "name": {"type": "text", "fields": {"kw": {"type": "keyword"}}},
                "category": {"type": "keyword"},
                TEXT_FIELD: {"type": "text"},
                VECTOR_FIELD: {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"},
                },
            }
        },
    }


def _pipeline_body() -> dict:
    return {
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": [0.5, 0.5]},
                    },
                }
            }
        ]
    }


def ensure_indices(client: Any | None = None) -> None:
    """Create the indices + search pipeline if missing (idempotent)."""
    client = client or get_client()
    from opensearchpy.exceptions import RequestError

    for index in (MERCHANT_INDEX, POLICY_INDEX):
        if not client.indices.exists(index=index):
            client.indices.create(index=index, body=_mapping())
            log.info("created index %s", index)
    with contextlib.suppress(RequestError):
        client.transport.perform_request("PUT", f"/_search/pipeline/{PIPELINE}", body=_pipeline_body())


# ------------------------------------------------------------------- indexing


def index_docs(index: str, docs: list[dict]) -> int:
    """Bulk-index ``docs`` (each needs ``id``, ``name``, ``category``)."""
    if not docs:
        return 0
    embedder = get_embedder()
    bodies: list[dict] = []
    for d in docs:
        text = d.get(TEXT_FIELD) or f"{d.get('name', '')} {d.get('category', '')}".strip()
        doc = {
            "id": str(d["id"]),
            "name": d.get("name", ""),
            "category": d.get("category", "OTHER"),
            TEXT_FIELD: text,
            VECTOR_FIELD: embedder.embed_one(text),
        }
        bodies.append({"index": {"_index": index, "_id": doc["id"]}})
        bodies.append(doc)
    client = get_client()
    resp = client.bulk(body=bodies, refresh=True)
    if resp.get("errors"):
        log.warning("bulk index had errors for %s", index)
    return len(docs)


# -------------------------------------------------------------------- search


def hybrid_search(index: str, query: str, k: int = 5) -> list[dict]:
    """Hybrid BM25 + kNN search; returns [{id,name,category,score}]."""
    vec = get_embedder().embed_one(query)
    body = {
        "size": k,
        "query": {
            "hybrid": {
                "queries": [
                    {"match": {TEXT_FIELD: {"query": query}}},
                    {"knn": {VECTOR_FIELD: {"vector": vec, "k": max(k, 10)}}},
                ]
            }
        },
        "_source": ["id", "name", "category"],
    }
    client = get_client()
    resp = client.search(index=index, body=body, params={"search_pipeline": PIPELINE})
    return [
        {
            "id": h["_source"].get("id"),
            "name": h["_source"].get("name", ""),
            "category": h["_source"].get("category", "OTHER"),
            "score": round(float(h["_score"] or 0.0), 4),
        }
        for h in resp["hits"]["hits"]
    ]


def search_with_fallback(index: str, query: str, k: int = 5) -> list[dict]:
    """Hybrid search that never raises — returns [] if unavailable."""
    try:
        return hybrid_search(index, query, k)
    except Exception as exc:  # noqa: BLE001
        log.debug("search unavailable (%s): %s", index, exc)
        return []


def search_merchants(query: str, k: int = 5) -> list[dict]:
    return search_with_fallback(MERCHANT_INDEX, query, k)


def search_policies(query: str, k: int = 5) -> list[dict]:
    return search_with_fallback(POLICY_INDEX, query, k)
