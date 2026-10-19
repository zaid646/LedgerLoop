"""One ingestion unit: redact -> extract -> code -> policy -> embed -> record.

Deliberately LLM-free so the benchmark is GPU-free, deterministic and runs in CI,
multiprocessing children and Ray containers identically.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "ai-service", REPO_ROOT / "evals", REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Force the deterministic feature-hashing embedder so serial, multiprocessing and
# Ray workers do the same amount of work (fastembed is optional and not in the
# Ray image). Override with BENCH_EMBED_BACKEND only if you know what you're doing.
os.environ["EMBED_BACKEND"] = os.environ.get("BENCH_EMBED_BACKEND", "hashing")

from baselines import heuristic_extract  # noqa: E402

from app.catalog import infer_category  # noqa: E402
from app.embeddings import get_embedder  # noqa: E402
from app.policy import evaluate  # noqa: E402
from app.redaction import redact  # noqa: E402
from app.schemas import GLCategory, PolicyCheckRequest  # noqa: E402


def process_doc(record: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    redacted = redact(str(record.get("text", "")))
    extraction = heuristic_extract(redacted.text)
    merchant = str(extraction.get("merchant") or "unknown").strip()
    category = infer_category(merchant)
    amount = extraction.get("total_amount") or 0.0
    policy = evaluate(
        PolicyCheckRequest(
            amount=float(amount),
            category=GLCategory(category),
            merchant=merchant or None,
            has_receipt=True,
        )
    )
    vector = get_embedder().embed_one(redacted.text[:512])
    norm = sum(value * value for value in vector) ** 0.5
    return {
        "id": record["id"],
        "merchant": merchant,
        "category": category,
        "route": policy.route,
        "violations": len(policy.violations),
        "redactions": redacted.total,
        "vec_norm": round(norm, 6),
        "latency_ms": (time.perf_counter() - started) * 1000.0,
    }


def process_chunk(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [process_doc(record) for record in records]
