#!/usr/bin/env python3
"""Build the frozen golden eval set from public receipt datasets.

    cd ai-service && uv run python ../evals/build_golden.py --sroie 50 --cord 25

Writes ``evals/golden/v0.jsonl`` (+ ``v0.meta.json``). Keep this set frozen;
new labelled examples arrive via ``evals/golden/corrections.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai-service"))

from app.data import load  # noqa: E402

GOLDEN_DIR = REPO_ROOT / "evals" / "golden"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sroie", type=int, default=50)
    ap.add_argument("--cord", type=int, default=25)
    ap.add_argument("--out", default="v0")
    args = ap.parse_args()

    records = load(limit=max(args.sroie, args.cord))
    sroie = [r for r in records if r.source == "sroie"][: args.sroie]
    cord = [r for r in records if r.source == "cord"][: args.cord]
    selected = sroie + cord

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GOLDEN_DIR / f"{args.out}.jsonl"
    with out_path.open("w") as fh:
        for r in selected:
            fh.write(json.dumps(r.to_golden()) + "\n")

    meta = {
        "version": args.out,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(selected),
        "by_source": {"sroie": len(sroie), "cord": len(cord)},
    }
    (GOLDEN_DIR / f"{args.out}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {len(selected)} examples -> {out_path.relative_to(REPO_ROOT)}")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
