#!/usr/bin/env python3
"""Evaluate a fine-tuned BERT/RoBERTa GL-coding classifier on held-out data.

Reports overall accuracy, macro F1, per-category metrics, and domain-only
exact match / F1 on the 30 unseen-merchant LedgerLoop rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

# Transformers imported lazily so the module loads without it installed.
def _import_transformers():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    return AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "nlp"))
sys.path.insert(0, str(REPO_ROOT / "sft"))

from train_bert import LABEL2ID, ID2LABEL, NUM_LABELS, build_dataset, evaluate, load_jsonl  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="directory with fine-tuned model + tokenizer")
    parser.add_argument("--val-file", type=Path, default=REPO_ROOT / "sft" / "data" / "val.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="output JSON path")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args(argv)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    AutoModelForSequenceClassification, AutoTokenizer = _import_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    val_records = load_jsonl(args.val_file)
    val_examples = build_dataset(val_records)

    results = evaluate(model, tokenizer, val_examples, device, args.max_length)

    output = args.output or args.model_dir / "eval_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(
        {k: results[k] for k in ("accuracy", "macro_f1", "domain_exact_match", "domain_macro_f1", "n_domain") if k in results},
        indent=2,
    ))
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
