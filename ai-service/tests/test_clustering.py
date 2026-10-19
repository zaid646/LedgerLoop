import numpy as np

from app.graph import cluster as cluster_mod
from app.graph.cluster import _knn_graph, _purity, _score, run_clustering


def test_purity_ignores_noise():
    assert _purity(["A", "A", "B"], [0, 0, -1]) == 1.0
    assert _purity(["A", "B"], [0, 0]) == 0.5
    assert _purity(["A"], [-1]) == 0.0


def test_score_perfect_and_random_labels():
    truth = ["A"] * 5 + ["B"] * 5
    perfect = _score(truth, [0] * 5 + [1] * 5)
    assert perfect["ari"] == 1.0
    assert perfect["nmi"] == 1.0
    assert perfect["purity"] == 1.0

    mixed = _score(truth, [0] * 5 + [0] * 5)
    assert mixed["ari"] == 0.0
    assert mixed["n_clusters"] == 1


def test_knn_graph_is_symmetric_with_k_neighbours():
    vectors = np.eye(6)
    graph = _knn_graph(vectors, k=2)
    assert graph.number_of_nodes() == 6
    assert 0 < graph.number_of_edges() <= 12
    assert all(degree >= 1 for _, degree in graph.degree())


def test_run_clustering_recovers_separated_blobs(monkeypatch):
    centers = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    vectors = np.vstack(
        [center + 0.01 * np.array([index, index]) for index, center in enumerate(centers) for _ in range(4)]
    )
    labels = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
    ids = [f"merchant:m{i}" for i in range(12)]

    def fake_embed(merchants=None, records=None):
        return ids, vectors, labels, {"backend": "fake", "dim": 2, "with_context": 0}

    monkeypatch.setattr(cluster_mod, "embed_merchants", fake_embed)
    result = run_clustering()

    assert result["n_entities"] == 12
    assert result["n_types"] == 3
    assert result["algorithms"]["kmeans"]["n_clusters"] == 3
    assert result["algorithms"]["kmeans"]["ari"] > 0.95
    assert result["algorithms"]["louvain"]["n_clusters"] == 3
    assert result["algorithms"]["louvain"]["ari"] > 0.95
    assert result["algorithms"]["louvain"]["modularity"] > 0.5
    assert set(result["algorithms"]["kmeans"]["assignments"]) == set(ids)
