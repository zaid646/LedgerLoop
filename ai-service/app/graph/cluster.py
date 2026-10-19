"""Cluster merchant entities and compare the result with extracted GL types.

Three approaches:
* k-means over merchant embeddings (k = number of observed GL categories)
* HDBSCAN (density-based, can leave noise)
* Louvain community detection on an embedding k-NN graph

The category label is **not** part of the embedding text, so agreement (ARI/NMI/
purity) is a genuine comparison rather than a leak.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import networkx as nx
import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ..catalog import merchant_docs
from ..config import REPO_ROOT
from ..embeddings import get_embedder
from .model import merchant_id
from .repo import GraphRepo

_PUNCT_RE = re.compile(r"[^A-Za-z0-9& ]+")
_SUFFIX_RE = re.compile(r"\b(?:SDN\.?|BHD\.?|S/B|SB)\b", re.IGNORECASE)


def _clean_name(name: str) -> str:
    text = _PUNCT_RE.sub(" ", name.upper())
    text = _SUFFIX_RE.sub(" ", text)
    return " ".join(text.split())


def _merchant_context(records: list[dict] | None = None) -> dict[str, str]:
    """Map lowercase merchant name -> receipt text snippet (if present)."""
    if records is None:
        path = REPO_ROOT / "evals" / "golden" / "v0.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    snippets: dict[str, list[str]] = {}
    for rec in records:
        name = str((rec.get("fields") or {}).get("merchant") or "").strip().lower()
        if name:
            snippets.setdefault(name, []).append(str(rec.get("text", ""))[:400])
    return {key: " ".join(values)[:600] for key, values in snippets.items()}


def embed_merchants(
    merchants: list[dict] | None = None,
    records: list[dict] | None = None,
) -> tuple[list[str], np.ndarray, list[str], dict[str, str]]:
    merchants = merchant_docs() if merchants is None else merchants
    context = _merchant_context(records)
    texts = []
    for merchant in merchants:
        name = str(merchant["name"])
        snippet = context.get(name.strip().lower(), "")
        texts.append(f"{_clean_name(name)}. {snippet}".strip())

    embedder = get_embedder()
    vectors = np.asarray(embedder.embed(texts), dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.clip(norms, 1e-12, None)
    ids = [merchant_id(str(m["name"])) for m in merchants]
    labels = [str(m["category"]) for m in merchants]
    with_context = sum(
        1 for merchant in merchants if context.get(str(merchant["name"]).strip().lower())
    )
    return (
        ids,
        vectors,
        labels,
        {
            "backend": embedder.backend,
            "dim": embedder.dim,
            "with_context": with_context,
        },
    )


def _purity(y_true: list[str], y_pred: list[int]) -> float:
    kept = 0
    correct = 0
    for cluster in set(y_pred):
        if cluster == -1:
            continue
        members = [i for i, value in enumerate(y_pred) if value == cluster]
        kept += len(members)
        correct += Counter(y_true[i] for i in members).most_common(1)[0][1]
    return correct / kept if kept else 0.0


def _score(y_true: list[str], y_pred: list[int]) -> dict[str, Any]:
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "purity": float(_purity(y_true, y_pred)),
        "n_clusters": len(set(y_pred)) - (1 if -1 in y_pred else 0),
        "n_noise": int(sum(1 for value in y_pred if value == -1)),
    }


def _knn_graph(vectors: np.ndarray, k: int = 5) -> nx.Graph:
    n = len(vectors)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    if n < 2:
        return graph
    k = max(1, min(k, n - 1))
    similarity = vectors @ vectors.T
    for i in range(n):
        order = np.argsort(-similarity[i])
        for j in order[1 : k + 1]:
            graph.add_edge(i, int(j), weight=float(max(similarity[i, int(j)], 0.0)))
    return graph


def run_clustering(
    merchants: list[dict] | None = None,
    records: list[dict] | None = None,
) -> dict[str, Any]:
    ids, vectors, labels, embedding = embed_merchants(merchants, records)
    k = min(len(set(labels)), len(ids))
    algorithms: dict[str, Any] = {}

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(vectors)
    algorithms["kmeans"] = {
        "k": k,
        **_score(labels, [int(v) for v in kmeans.labels_]),
        "assignments": {eid: int(c) for eid, c in zip(ids, kmeans.labels_, strict=False)},
    }

    hdbscan = HDBSCAN(min_cluster_size=3, min_samples=1, copy=True).fit(vectors)
    algorithms["hdbscan"] = {
        "k": None,
        **_score(labels, [int(v) for v in hdbscan.labels_]),
        "assignments": {eid: int(c) for eid, c in zip(ids, hdbscan.labels_, strict=False)},
    }

    graph = _knn_graph(vectors, k=min(5, len(ids) - 1))
    communities = sorted(
        nx.algorithms.community.louvain_communities(graph, weight="weight", seed=0),
        key=lambda members: min(members) if members else 0,
    )
    louvain_assignments: dict[str, int] = {}
    for index, members in enumerate(communities):
        for node in members:
            louvain_assignments[ids[int(node)]] = index
    algorithms["louvain"] = {
        "k": None,
        **_score(labels, [louvain_assignments[eid] for eid in ids]),
        "modularity": float(
            nx.algorithms.community.modularity(graph, communities, weight="weight")
        ),
        "knn_k": min(5, len(ids) - 1),
        "assignments": louvain_assignments,
    }

    return {
        "n_entities": len(ids),
        "n_types": k,
        "types": sorted(set(labels)),
        "embedding": embedding,
        "context_entities": int(embedding.get("with_context", 0)),
        "algorithms": algorithms,
    }


def write_clusters(repo: GraphRepo, result: dict[str, Any]) -> None:
    repo.upsert_clusters(result)
