#!/usr/bin/env python3
"""Cluster merchant entities and compare clusters with extracted GL types.

    cd ai-service && uv run python scripts/cluster_graph.py --write-neo4j
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import REPO_ROOT  # noqa: E402
from app.graph.cluster import run_clustering, write_clusters  # noqa: E402
from app.graph.repo import Neo4jRepo  # noqa: E402

ALGORITHMS = ("kmeans", "hdbscan", "louvain")
METRICS = ("ari", "nmi", "purity")


def render(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(ALGORITHMS))
    width = 0.26
    colors = {"ari": "#0b5cad", "nmi": "#3f8f6b", "purity": "#c47f17"}

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for offset, metric in enumerate(METRICS):
        values = [result["algorithms"][algo].get(metric, 0.0) for algo in ALGORITHMS]
        bars = ax.bar(x + (offset - 1) * width, values, width, label=metric.upper(), color=colors[metric])
        for bar, value in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    labels = []
    for algo in ALGORITHMS:
        run = result["algorithms"][algo]
        noise = f", noise {run['n_noise']}" if run.get("n_noise") else ""
        labels.append(f"{algo}\n{run['n_clusters']} clusters{noise}")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score vs extracted GL types")
    ax.set_title(
        "Merchant entity clustering vs GL types "
        f"(n={result['n_entities']}, {result['n_types']} types)"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, frameon=False, loc="upper left")
    ax.text(
        0.0,
        -0.24,
        f"embeddings: {result['embedding']['backend']} (dim {result['embedding']['dim']}) · "
        f"Louvain on k-NN graph k={result['algorithms']['louvain']['knn_k']} · "
        f"context receipts: {result['context_entities']}/{result['n_entities']}",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", default=str(REPO_ROOT / "runs" / "graph_clusters.json")
    )
    parser.add_argument(
        "--figure", default=str(REPO_ROOT / "evidence" / "png" / "16-cluster-compare.png")
    )
    parser.add_argument("--write-neo4j", action="store_true")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    result = run_clustering()
    result["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(
        f"{result['n_entities']} merchants · {result['n_types']} types · "
        f"embedding={result['embedding']['backend']} dim={result['embedding']['dim']}"
    )
    print(f"{'algorithm':10} {'clusters':>8} {'noise':>6} {'ARI':>7} {'NMI':>7} {'purity':>7}")
    for algo in ALGORITHMS:
        run = result["algorithms"][algo]
        print(
            f"{algo:10} {run['n_clusters']:>8} {run['n_noise']:>6} "
            f"{run['ari']:>7.3f} {run['nmi']:>7.3f} {run['purity']:>7.3f}"
        )
    print(f"Louvain modularity: {result['algorithms']['louvain']['modularity']:.3f}")

    if not args.no_figure:
        render(result, Path(args.figure))
        print(f"figure -> {args.figure}")

    if args.write_neo4j:
        repo = Neo4jRepo()
        try:
            write_clusters(repo, result)
            print("clusters written to Neo4j")
        finally:
            repo.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
