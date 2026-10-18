#!/usr/bin/env python3
"""Export the prompt registry + golden set into PromptFoo artefacts.

Keeps a single source of truth: the registry prompt is the one under test.

    cd ai-service && uv run python ../evals/export_promptfoo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai-service"))
sys.path.insert(0, str(REPO_ROOT / "evals"))

from run_evals import load_golden  # noqa: E402

from app.registry import PromptRegistry  # noqa: E402

OUT = REPO_ROOT / "evals" / "promptfoo"
LIMIT = 40


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    reg = PromptRegistry()
    prompt = reg.resolve("extraction", "prod")

    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user_template.replace("{text}", "{{text}}")},
    ]
    (OUT / "prompt.json").write_text(json.dumps(messages, indent=2) + "\n")

    records = load_golden("v0")[:LIMIT]
    with (OUT / "tests.jsonl").open("w") as fh:
        for r in records:
            gold = r["fields"]
            fh.write(
                json.dumps(
                    {
                        "vars": {
                            "text": r["text"],
                            "merchant": gold.get("merchant") or "",
                            "total": "" if gold.get("total_amount") is None else str(gold["total_amount"]),
                            "date": gold.get("transaction_date") or "",
                        }
                    }
                )
                + "\n"
            )

    print(f"wrote {OUT/'prompt.json'} (prompt {prompt.version}) and {OUT/'tests.jsonl'} ({len(records)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
