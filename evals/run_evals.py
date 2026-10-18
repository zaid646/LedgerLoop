#!/usr/bin/env python3
"""Evaluate receipt extraction on the frozen golden set.

    cd ai-service && uv run python ../evals/run_evals.py --engine heuristic
    cd ai-service && uv run python ../evals/run_evals.py --engine llm --golden v0

Metrics: per-field accuracy, exact-document match, p95 latency, mean cost/doc,
self-heal rate and human-escalation rate. Optionally logged to W&B.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai-service"))
sys.path.insert(0, str(REPO_ROOT / "evals"))

from metrics import field_scores  # noqa: E402

GOLDEN_DIR = REPO_ROOT / "evals" / "golden"
RESULTS_DIR = REPO_ROOT / "evals" / "results"
FIELDS = ("merchant", "total_amount", "transaction_date")


def load_golden(version: str) -> list[dict]:
    path = GOLDEN_DIR / f"{version}.jsonl"
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def run_heuristic(records: list[dict]) -> list[dict]:
    from baselines import heuristic_extract
    from app.redaction import redact

    preds = []
    for r in records:
        text = redact(r["text"]).text
        pred = heuristic_extract(text)
        preds.append(
            {"id": r["id"], "pred": pred, "attempts": 1, "repaired": False,
             "escalated": False, "latency_ms": 0.0, "cost_usd": 0.0}
        )
    return preds


async def run_llm(records: list[dict], stage: str, limit: int | None) -> list[dict]:
    from app.redaction import redact
    from app.self_heal import extract_with_healing

    preds = []
    for i, r in enumerate(records):
        if limit is not None and i >= limit:
            break
        text = redact(r["text"]).text
        outcome = await extract_with_healing(text, stage=stage)
        pred = outcome.extraction.model_dump() if outcome.extraction else {}
        preds.append(
            {
                "id": r["id"],
                "pred": pred,
                "attempts": outcome.attempts,
                "repaired": outcome.repaired,
                "escalated": outcome.escalated,
                "latency_ms": outcome.latency_ms,
                "cost_usd": outcome.cost_usd,
            }
        )
    return preds


def score(records: list[dict], preds: list[dict]) -> dict:
    by_id = {p["id"]: p for p in preds}
    field_pass = {f: 0 for f in FIELDS}
    field_present = {f: 0 for f in FIELDS}
    exact = 0
    latencies, costs = [], []
    repaired = escalated = 0

    for r in records:
        p = by_id.get(r["id"])
        if p is None:
            continue
        scores = field_scores(p["pred"], r["fields"])
        for f in FIELDS:
            if r["fields"].get(f) not in (None, "", []):
                field_present[f] += 1
                field_pass[f] += int(scores[f])
        if all(scores[f] for f in FIELDS if r["fields"].get(f) not in (None, "", [])):
            exact += 1
        latencies.append(p["latency_ms"])
        costs.append(p["cost_usd"])
        repaired += int(p["repaired"])
        escalated += int(p["escalated"])

    n = len(records)
    return {
        "n": n,
        "field_accuracy": {
            f: round(field_pass[f] / field_present[f], 4) if field_present[f] else None for f in FIELDS
        },
        "field_support": field_present,
        "exact_doc_match": round(exact / n, 4) if n else 0.0,
        "p50_latency_ms": round(percentile(latencies, 0.5), 2),
        "p95_latency_ms": round(percentile(latencies, 0.95), 2),
        "mean_cost_usd": round(sum(costs) / n, 8) if n else 0.0,
        "total_cost_usd": round(sum(costs), 6),
        "repaired_rate": round(repaired / n, 4) if n else 0.0,
        "escalated_rate": round(escalated / n, 4) if n else 0.0,
    }


def log_wandb(name: str, summary: dict, preds: list[dict], config: dict, disabled: bool) -> str | None:
    if disabled:
        return None
    try:
        import wandb

        run = wandb.init(project=config.pop("project", "expenseiq"), name=name, config=config)
        wandb.log({k: v for k, v in summary.items() if not isinstance(v, dict)})
        wandb.log({f"acc/{f}": (v or 0.0) for f, v in summary["field_accuracy"].items()})
        table = wandb.Table(columns=["id", "merchant", "total", "date", "repaired", "escalated", "cost"])
        for p in preds[:200]:
            pr = p["pred"]
            table.add_data(
                p["id"], pr.get("merchant"), pr.get("total_amount"),
                pr.get("transaction_date"), p["repaired"], p["escalated"], p["cost_usd"],
            )
        wandb.log({"predictions": table})
        url = run.url
        run.finish()
        print(f"logged to W&B -> {url}")
        return url
    except Exception as e:  # noqa: BLE001
        print(f"[warn] W&B logging skipped: {e}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["heuristic", "llm"], default="heuristic")
    ap.add_argument("--golden", default="v0")
    ap.add_argument("--stage", default="prod")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    records = load_golden(args.golden)
    if args.limit:
        records = records[: args.limit]

    import os

    from app.config import get_settings

    s = get_settings()
    if s.wandb_api_key:
        os.environ["WANDB_API_KEY"] = s.wandb_api_key
    os.environ.setdefault("WANDB_PROJECT", s.wandb_project)
    if s.wandb_entity:
        os.environ["WANDB_ENTITY"] = s.wandb_entity

    if args.engine == "heuristic":
        preds = run_heuristic(records)
        config = {"engine": "heuristic", "golden": args.golden}
    else:
        preds = asyncio.run(run_llm(records, args.stage, args.limit))
        config = {"engine": "llm", "golden": args.golden, "model": s.vllm_model, "stage": args.stage}

    summary = score(records, preds)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = args.name or f"{args.engine}-{args.golden}-{ts}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{name}.json"

    print(json.dumps(summary, indent=2))

    run_url = log_wandb(name, summary, preds, dict(config), args.no_wandb)
    payload = {"name": name, "config": config, "summary": summary, "predictions": preds}
    if run_url:
        payload["wandb_url"] = run_url
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nresults -> {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
