#!/usr/bin/env python3
"""Seed the OpenSearch hybrid indices (merchants + policies).

    cd ai-service && uv run python scripts/index_opensearch.py

Builds merchant docs from the shared catalog (seeded hints + golden set) and
policy docs from the default rules; then indexes them and runs a sample query.
"""
from __future__ import annotations

import sys

from app import search
from app.catalog import merchant_docs, policy_docs


def main() -> int:
    if not search.available():
        print("OpenSearch not reachable", file=sys.stderr)
        return 2

    search.ensure_indices()
    merchants, policies = merchant_docs(), policy_docs()
    n_m = search.index_docs(search.MERCHANT_INDEX, merchants)
    n_p = search.index_docs(search.POLICY_INDEX, policies)
    print(f"indexed {n_m} merchants -> {search.MERCHANT_INDEX}")
    print(f"indexed {n_p} policies  -> {search.POLICY_INDEX}")

    for q in ("rideshare", "office chair", "team chat software", "coffee"):
        hits = search.search_merchants(q, k=3)
        pretty = ", ".join(f"{h['name']} ({h['category']}, {h['score']})" for h in hits)
        print(f"  merchants[{q!r}] -> {pretty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
