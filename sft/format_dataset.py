#!/usr/bin/env python3
"""Build prompt-completion data for LedgerLoop instruction tuning.

Sources
-------
1. A deterministic synthetic LedgerLoop corpus (182 GL-coding pairs).
2. An optional public instruction corpus: Databricks Dolly 15k. A small,
   license-compatible sample is checked in at ``data/public_dolly_sample.jsonl``;
   if it is missing the script downloads it from the Hugging Face Hub.

Outputs
-------
``data/train.jsonl``, ``data/val.jsonl`` and ``data/summary.json``. Every record
contains the raw ``instruction``/``response`` plus the rendered Qwen ChatML
``prompt``, ``completion`` and full ``text`` fields consumed by ``SFTTrainer``.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

try:
    from .domain_data import ALLOWED, build_domain_examples, split_domain_examples
except ImportError:  # direct script execution
    from domain_data import ALLOWED, build_domain_examples, split_domain_examples

DATA_DIR = Path(__file__).resolve().parent / "data"
PUBLIC_FILE = DATA_DIR / "public_dolly_sample.jsonl"
DOLLY_ID = "databricks/databricks-dolly-15k"

DOMAIN_SYSTEM = (
    "You are LedgerLoop's expense-coding assistant. You map business transactions to the "
    "company's general-ledger chart of accounts. When asked for a GL category, answer with "
    "exactly one allowed code and no extra text."
)
GENERAL_SYSTEM = "You are a helpful, accurate assistant. Follow the user's instruction exactly."


def render_prompt(instruction: str, source: str) -> str:
    system = DOMAIN_SYSTEM if source == "ledgerloop" else GENERAL_SYSTEM
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def make_record(
    *,
    record_id: str,
    source: str,
    instruction: str,
    response: str,
    category: str = "GENERAL",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = render_prompt(instruction, source)
    completion = f"{response}<|im_end|>\n"
    record: dict[str, Any] = {
        "id": record_id,
        "source": source,
        "category": category,
        "instruction": instruction,
        "response": response,
        "prompt": prompt,
        "completion": completion,
        "text": prompt + completion,
    }
    if metadata:
        record["metadata"] = metadata
    return record


def domain_records(train_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    examples = build_domain_examples()
    train, val = split_domain_examples(examples, train_size=train_size)
    records = []
    for split_name, rows in (("train", train), ("val", val)):
        for ex in rows:
            records.append(
                (
                    split_name,
                    make_record(
                        record_id=ex.id,
                        source="ledgerloop",
                        instruction=ex.instruction,
                        response=ex.response,
                        category=ex.category,
                        metadata={
                            "merchant": ex.merchant,
                            "description": ex.description,
                            "amount": ex.amount,
                            "currency": ex.currency,
                            "date": ex.date,
                        },
                    ),
                )
            )
    return (
        [r for split, r in records if split == "train"],
        [r for split, r in records if split == "val"],
    )


def read_public_sample(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def download_public_sample(limit: int, seed: int) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The 'datasets' package is required to download Dolly. Install "
            "sft/requirements.txt or use --no-public."
        ) from exc

    print(f"downloading {DOLLY_ID} ...", flush=True)
    ds = load_dataset(DOLLY_ID, split="train")
    rows: list[dict[str, Any]] = []
    for i, ex in enumerate(ds):
        context = (ex.get("context") or "").strip()
        instruction = (ex.get("instruction") or "").strip()
        response = (ex.get("response") or "").strip()
        category = (ex.get("category") or "").strip()
        if context or not instruction or not response:
            continue
        if len(response) > 700 or len(instruction) > 700:
            continue
        if category in {"closed_qa", "summarization"}:
            continue
        rows.append(
            {
                "id": f"dolly-{i:05d}",
                "instruction": instruction,
                "response": response,
                "category": category or "general",
            }
        )
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def public_records(
    samples: int, public_file: Path, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if public_file.exists():
        source_rows = read_public_sample(public_file)
        origin = f"checked-in sample ({public_file.name})"
    else:
        source_rows = download_public_sample(samples, seed)
        public_file.parent.mkdir(parents=True, exist_ok=True)
        public_file.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in source_rows) + "\n",
            encoding="utf-8",
        )
        origin = f"downloaded from {DOLLY_ID}"
    if len(source_rows) < 2:
        raise SystemExit("public sample is empty; use --no-public or check the download")

    source_rows = source_rows[:samples]
    eval_size = max(1, min(40, len(source_rows) // 8))
    public_train = source_rows[:-eval_size]
    public_val = source_rows[-eval_size:]
    make = lambda r: make_record(  # noqa: E731
        record_id=r["id"],
        source="dolly",
        instruction=r["instruction"],
        response=r["response"],
        category=r.get("category", "GENERAL"),
        metadata={"dataset": "databricks-dolly-15k"},
    )
    return [make(r) for r in public_train], [make(r) for r in public_val], origin


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--public-file", type=Path, default=PUBLIC_FILE)
    parser.add_argument("--public-samples", type=int, default=200)
    parser.add_argument("--no-public", action="store_true", help="domain-only smoke/test mode")
    parser.add_argument("--domain-train", type=int, default=152)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="cap train/val rows (debug only)")
    args = parser.parse_args(argv)

    domain_train, domain_val = domain_records(args.domain_train)
    train = list(domain_train)
    val = list(domain_val)
    public_origin = "disabled"
    if not args.no_public:
        p_train, p_val, public_origin = public_records(
            args.public_samples, args.public_file, args.seed
        )
        train.extend(p_train)
        val.extend(p_val)

    if args.limit:
        train = train[: args.limit]
        val = val[: max(1, args.limit // 4)]

    random.Random(args.seed).shuffle(train)
    random.Random(args.seed + 1).shuffle(val)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "val.jsonl", val)

    summary = {
        "train": len(train),
        "val": len(val),
        "train_by_source": _counts(train),
        "val_by_source": _counts(val),
        "domain_categories": len({r["category"] for r in train + val if r["source"] == "ledgerloop"}),
        "allowed_categories": ALLOWED.split(", "),
        "public_source": public_origin,
        "seed": args.seed,
        "format": "qwen-chatml prompt/completion",
        "chat_template": "ChatML with <|im_start|>/<|im_end|>",
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out_dir / 'train.jsonl'}")
    print(f"wrote {args.out_dir / 'val.jsonl'}")
    if train:
        print("\n--- sample prompt ---")
        print(train[0]["prompt"])
        print("--- sample completion ---")
        print(train[0]["completion"])
    return 0


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        out[row["source"]] = out.get(row["source"], 0) + 1
    return out


if __name__ == "__main__":
    sys.exit(main())
