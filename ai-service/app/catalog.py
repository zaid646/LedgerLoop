"""Shared merchant / policy catalog used by retrieval and the entity graph.

Keeping the seed corpus here (instead of in an indexing script) means the
OpenSearch indexer, the Neo4j graph builder and the clustering job all see the
same merchants and inferred GL categories.
"""
from __future__ import annotations

import json

from .config import REPO_ROOT
from .policy import DEFAULT_RULES

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
KEYWORDS: list[tuple[str, str]] = [
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
    for kw, cat in KEYWORDS:
        if kw in up:
            return cat
    return "OTHER"


def merchant_docs() -> list[dict]:
    """Seed merchants plus merchants discovered in the frozen golden set."""
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
