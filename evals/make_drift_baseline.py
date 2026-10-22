#!/usr/bin/env python3
"""Write the serving drift baseline from cached predictions.

The monitor must compare incoming predictions with the prediction distribution
that the model produced at baseline time, not with golden labels.
"""
from __future__ import annotations

import json
from pathlib import Path

from drift_sim import load_samples

OUT = Path(__file__).resolve().parents[1] / "ai-service" / "config" / "drift_baseline.jsonl"


def main() -> int:
    samples = load_samples()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as handle:
        for sample in samples:
            handle.write(
                json.dumps(
                    {"id": sample["id"], "text": sample["text"], "fields": sample["fields"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(samples)} baseline documents to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
