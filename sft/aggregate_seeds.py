#!/usr/bin/env python3
"""Aggregate per-seed SFT evaluation JSON files into mean ± std.

Expected layout::

    sft/results/seeds/<seed>/eval_adapter.json

Usage:
    python sft/aggregate_seeds.py
    python sft/aggregate_seeds.py --seeds-dir sft/results/seeds
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS_DIR = ROOT / "results" / "seeds"

FIELDS = (
    "domain_exact_match",
    "domain_token_f1",
    "format_adherence",
    "completion_loss",
    "perplexity",
)


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds-dir", type=Path, default=DEFAULT_SEEDS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    runs: list[dict[str, Any]] = []
    for path in sorted(args.seeds_dir.glob("*/eval_adapter.json")):
        data = json.loads(path.read_text())
        seed = path.parent.name
        runs.append(
            {
                "seed": seed,
                "path": f"{seed}/eval_adapter.json",
                "domain_exact_match": data["domain_exact_match"],
                "domain_token_f1": data["domain_token_f1"],
                "format_adherence": data["format_adherence"],
                "completion_loss": data["completion_loss"],
                "perplexity": data["perplexity"],
                "domain_completion_loss": data.get("loss_by_source", {}).get("ledgerloop"),
                "dolly_completion_loss": data.get("loss_by_source", {}).get("dolly"),
                "n_domain": data.get("n_domain"),
                "n_eval": data.get("n_eval"),
                "model": data.get("model"),
            }
        )
    if not runs:
        raise SystemExit(f"no */eval_adapter.json files under {args.seeds_dir}")

    summary: dict[str, Any] = {
        "model": runs[0]["model"],
        "n_seeds": len(runs),
        "seeds": [r["seed"] for r in runs],
        "n_domain": runs[0]["n_domain"],
        "n_eval": runs[0]["n_eval"],
        "runs": runs,
        "aggregate": {},
    }
    for field in FIELDS:
        summary["aggregate"][field] = summarize([float(r[field]) for r in runs])
    for field in ("domain_completion_loss", "dolly_completion_loss"):
        vals = [r[field] for r in runs if r[field] is not None]
        summary["aggregate"][field] = summarize([float(v) for v in vals])

    output = args.output or (args.seeds_dir / "multiseed_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"seeds: {summary['seeds']} (n={summary['n_seeds']})")
    for field, stats in summary["aggregate"].items():
        if stats["mean"] is None:
            continue
        print(f"{field:24} {stats['mean']:.6f} ± {stats['std']:.6f}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
