#!/usr/bin/env python3
"""Build the entity graph in Neo4j from the frozen golden set + catalog.

    cd ai-service && uv run python scripts/build_graph.py --reset --export ../runs/graph_export.json

``--dry-run`` builds in memory and prints counts without touching Neo4j.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import REPO_ROOT  # noqa: E402
from app.graph.build import build_graph, write_graph  # noqa: E402
from app.graph.repo import Neo4jRepo  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v0", help="golden set version")
    parser.add_argument("--reset", action="store_true", help="delete all nodes first")
    parser.add_argument("--dry-run", action="store_true", help="skip Neo4j writes")
    parser.add_argument(
        "--export",
        default=str(REPO_ROOT / "runs" / "graph_export.json"),
        help="write a JSON snapshot (used by the offline UI fixture)",
    )
    args = parser.parse_args()

    build = build_graph(version=args.version)
    print(json.dumps({"build": build.stats}, indent=2))

    if args.export:
        path = Path(args.export)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(build.to_dict(), indent=2))
        print(f"exported {len(build.entities)} entities -> {path}")

    if args.dry_run:
        return 0

    repo = Neo4jRepo()
    try:
        stats = write_graph(build, repo, reset=args.reset)
    finally:
        repo.close()
    print(json.dumps({"neo4j": stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
