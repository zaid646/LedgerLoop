#!/usr/bin/env python3
"""CI regression gate: fail if a prompt/model change degrades quality.

Compares a candidate results file (from ``evals/run_evals.py``) against a stored
baseline. Blocks on:
  * overall field accuracy dropping by more than ``--tol``
  * p95 latency rising by more than ``--latency-tol`` (relative)

    uv run python ../evals/gate.py --results evals/results/<candidate>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPO_ROOT / "evals" / "baseline.json"


def overall_accuracy(summary: dict) -> float:
    accs = [v for v in summary.get("field_accuracy", {}).values() if v is not None]
    return sum(accs) / len(accs) if accs else 0.0


def load(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    return data.get("summary", data)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--tol", type=float, default=0.02, help="max absolute accuracy drop")
    ap.add_argument("--latency-tol", type=float, default=0.5, help="max relative p95 increase")
    args = ap.parse_args()

    cand = load(Path(args.results))
    base = load(Path(args.baseline))

    c_acc, b_acc = overall_accuracy(cand), overall_accuracy(base)
    c_p95, b_p95 = cand.get("p95_latency_ms", 0.0), base.get("p95_latency_ms", 0.0)

    print(f"baseline  overall_acc={b_acc:.4f}  p95={b_p95} ms")
    print(f"candidate overall_acc={c_acc:.4f}  p95={c_p95} ms")

    failures = []
    if c_acc < b_acc - args.tol:
        failures.append(f"accuracy regressed: {b_acc:.4f} -> {c_acc:.4f} (tol {args.tol})")
    if b_p95 > 0 and c_p95 > b_p95 * (1 + args.latency_tol):
        failures.append(f"p95 latency regressed: {b_p95} -> {c_p95} ms (tol {args.latency_tol:.0%})")

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print("  -", f)
        return 1

    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
