import math

import pytest

from app import search
from app.embeddings import Embedder
from app.routers import code

# ------------------------------------------------------------------ embedder


def test_hashing_embedder_deterministic_and_normalised():
    emb = Embedder(model_name="definitely-not-a-real-model")
    assert emb.backend == "hashing"
    v1 = emb.embed_one("office chair")
    v2 = emb.embed_one("office chair")
    assert v1 == v2
    assert len(v1) == emb.dim
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0, rel_tol=1e-6)


# ------------------------------------------------------------- search degrade


def test_search_with_fallback_returns_empty_when_unavailable(monkeypatch):
    def boom():
        raise RuntimeError("no opensearch")

    monkeypatch.setattr(search, "get_client", boom)
    assert search.search_with_fallback(search.MERCHANT_INDEX, "anything") == []


# ------------------------------------------------------------ code retrieval


def test_retrieve_uses_retrieval_hits(monkeypatch):
    hits = [
        {"id": "m-1", "name": "GitHub", "category": "SOFTWARE", "score": 0.9},
        {"id": "m-2", "name": "Amazon Web Services", "category": "SOFTWARE", "score": 0.7},
    ]
    monkeypatch.setattr(search, "search_merchants", lambda q, k=5: hits)
    names, cats = code._retrieve("GitHub", "team seats")
    assert "GitHub" in names
    assert cats[0].value == "SOFTWARE"


def test_retrieve_falls_back_to_seed_hints(monkeypatch):
    monkeypatch.setattr(search, "search_merchants", lambda q, k=5: [])
    names, cats = code._retrieve("Uber", "ride to airport")
    assert names == ["uber"]
    assert cats and cats[0].value == "TRAVEL"


# ---------------------------------------------------------------- live (opt-in)


@pytest.mark.skipif(not search.available(), reason="OpenSearch not reachable")
def test_live_hybrid_query_roundtrip():
    search.ensure_indices()
    assert isinstance(search.hybrid_search(search.MERCHANT_INDEX, "coffee", k=3), list)
