#!/usr/bin/env python3
"""Exercise the GraphQL API and record p50/p95 latency per query.

    cd ai-service && uv run python scripts/graphql_probe.py --iterations 20
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

QUERIES: dict[str, str] = {
    "graphStats": """
        query GraphStats {
          graphStats {
            entities relationships clusters
            byType { key count }
            byRelationship { key count }
          }
        }
    """,
    "entities": """
        query Entities {
          entities(type: MERCHANT, limit: 5) { id type label }
        }
    """,
    "entity": """
        query Entity {
          entity(id: "expense:sroie-0") { id type label properties }
        }
    """,
    "neighbors": """
        query Neighbors {
          neighbors(id: "expense:sroie-0", depth: 1) {
            relationship direction depth
            entity { id type label }
          }
        }
    """,
    "searchEntities": """
        query Search {
          searchEntities(q: "coffee", limit: 5) { id type label }
        }
    """,
    "clusters": """
        query Clusters {
          clusters(algorithm: KMEANS) { id label algorithm size metrics }
        }
    """,
}


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8080/graphql")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--out", default=str(Path(__file__).resolve().parents[2] / "runs" / "graphql_probe.json")
    )
    args = parser.parse_args()

    stats: dict[str, dict[str, float | int]] = {}
    samples: dict[str, dict] = {}
    with httpx.Client(timeout=15.0) as client:
        for name, query in QUERIES.items():
            # warmup
            response = client.post(args.url, json={"query": query})
            response.raise_for_status()
            payload = response.json()
            if payload.get("errors"):
                print(f"[warn] {name}: {payload['errors']}")
            samples[name] = payload

            latencies: list[float] = []
            for _ in range(args.iterations):
                started = time.perf_counter()
                response = client.post(args.url, json={"query": query})
                response.raise_for_status()
                latencies.append((time.perf_counter() - started) * 1000)
            stats[name] = {
                "n": len(latencies),
                "mean_ms": round(statistics.fmean(latencies), 2),
                "p50_ms": round(statistics.median(latencies), 2),
                "p95_ms": round(_p95(latencies), 2),
                "min_ms": round(min(latencies), 2),
                "max_ms": round(max(latencies), 2),
            }

    graph_stats = samples.get("graphStats", {}).get("data", {}).get("graphStats", {})
    result = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "url": args.url,
        "iterations": args.iterations,
        "graph": graph_stats,
        "queries": stats,
        "samples": samples,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
