#!/usr/bin/env python3
"""Seed the OpenSearch hybrid indices (merchants + policies).

    cd ai-service && uv run python scripts/index_opensearch.py

Builds merchant docs from the seeded hints + the golden set, and policy docs from
the default rules; then indexes them and runs a sample hybrid query.
"""
from __future__ import annotations

import json
import sys

from app import search
from app.config import REPO_ROOT
from app.routers.policy import DEFAULT_RULES

# Curated merchant -> GL category seeds (the retrieval corpus).
SEED_MERCHANTS: dict[str, str] = {
    "Uber": "TRAVEL",
    "Lyft": "TRAVEL",
    "Uber Eats": "MEALS",
    "Delta Air Lines": "TRAVEL",
    "United Airlines": "TRAVEL",
    "Airbnb": "TRAVEL",
    "Marriott Hotels": "TRAVEL",
    "Shell": "TRAVEL",
    "Starbucks": "MEALS",
    "DoorDash": "MEALS",
    "McDonald's": "MEALS",
    "GitHub": "SOFTWARE",
    "Amazon Web Services": "SOFTWARE",
    "Slack": "SOFTWARE",
    "Microsoft": "SOFTWARE",
    "Adobe": "SOFTWARE",
    "Salesforce": "SOFTWARE",
    "Staples": "OFFICE_SUPPLIES",
    "IKEA": "HARDWARE",
    "MR D.I.Y.": "OFFICE_SUPPLIES",
    "Apple": "HARDWARE",
    "Dell": "HARDWARE",
    "Google Ads": "MARKETING",
    "Meta Ads": "MARKETING",
    "Deloitte": "PROFESSIONAL_SERVICES",
    "FedEx": "PROFESSIONAL_SERVICES",
    "WeWork": "RENT",
    "Aetna": "INSURANCE",
    "Coursera": "TRAINING",
}

# Keyword -> category inference for merchants discovered in the golden set.
_KEYWORDS: list[tuple[str, str]] = [
    ("RESTORAN", "MEALS"),
    ("RESTAURANT", "MEALS"),
    ("CAFE", "MEALS"),
    ("COFFEE", "MEALS"),
    ("BISTRO", "MEALS"),
    ("MART", "MEALS"),
    ("PETRO", "TRAVEL"),
    ("SHELL", "TRAVEL"),
    ("HARDWARE", "HARDWARE"),
    ("STATIONERY", "OFFICE_SUPPLIES"),
    ("BOOKS", "OFFICE_SUPPLIES"),
    ("ENTERPRISE", "OFFICE_SUPPLIES"),
]


def infer_category(name: str) -> str:
    up = name.upper()
    for kw, cat in _KEYWORDS:
        if kw in up:
            return cat
    return "OTHER"


def merchant_docs() -> list[dict]:
    docs: list[dict] = []
    seen: set[str] = set()

    def add(name: str, category: str) -> None:
        key = name.strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        docs.append({"id": f"m-{len(docs)}", "name": name.strip(), "category": category})

    for name, cat in SEED_MERCHANTS.items():
        add(name, cat)

    golden = REPO_ROOT / "evals" / "golden" / "v0.jsonl"
    if golden.exists():
        for line in golden.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            name = (rec.get("fields") or {}).get("merchant")
            if name:
                add(str(name), infer_category(str(name)))
    return docs


def policy_docs() -> list[dict]:
    docs: list[dict] = []
    for i, r in enumerate(DEFAULT_RULES):
        cat = r.applies_to.value if r.applies_to else "ALL"
        bits = [r.name, f"category {cat}"]
        if r.max_amount is not None:
            bits.append(f"max amount {r.max_amount}")
        if r.requires_receipt:
            bits.append("receipt required")
        if r.requires_manager_above is not None:
            bits.append(f"manager approval above {r.requires_manager_above}")
        docs.append({"id": f"p-{i}", "name": r.name, "category": cat, "text": "; ".join(bits)})
    return docs


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
