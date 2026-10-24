#!/usr/bin/env python3
"""Encoder-only GL-coding fine-tuning for LedgerLoop.

Fine-tunes BERT or RoBERTa as a text classifier on LedgerLoop's GL-coding
corpus (merchant + amount + date + description → one of 13 GL categories).
This is an alternative to the decoder-only QLoRA Qwen2.5 path in sft/.

Trains on 327 examples (152 domain + 175 Dolly) and evaluates on 55 held-out
examples (30 domain + 25 Dolly). Reports exact match and macro F1 on the 30
unseen-merchant domain rows — directly comparable to the QLoRA Qwen2.5
baseline (86.7% exact match, seed 42).

Usage::

    # CPU smoke test (tiny model, few steps)
    python nlp/train_bert.py --model prajjwal1/bert-tiny --epochs 1 --max-steps 10 --device cpu

    # Full CUDA run (remote GPU via SSH)
    python nlp/train_bert.py --model bert-base-uncased --epochs 3 --device cuda

    # RoBERTa variant
    python nlp/train_bert.py --model roberta-base --epochs 3 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

# Transformers imported lazily so the module loads without it installed.
def _import_transformers():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    return AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sft"))

from domain_data import GL_CATEGORIES  # noqa: E402

LABEL2ID = {cat: i for i, cat in enumerate(GL_CATEGORIES)}
ID2LABEL = {i: cat for cat, i in LABEL2ID.items()}
NUM_LABELS = len(GL_CATEGORIES)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def extract_transaction(record: dict[str, Any]) -> str:
    """Extract the transaction text from a Qwen-formatted record.

    The instruction contains 'Merchant: ...\nAmount: ...\nDate: ...\nDescription: ...'
    between 'Transaction:\n' and the end (or the response marker). Falls back to
    the raw instruction text if the transaction block can't be parsed.
    """
    instruction = record.get("instruction", "")
    marker = "Transaction:\n"
    if marker in instruction:
        after = instruction.split(marker, 1)[1]
        # Strip everything from 'Allowed categories' onwards if present
        for stop in ("\nAllowed categories", "Return only", "Answer with", "Use exactly"):
            if stop in after:
                after = after.split(stop, 1)[0]
        return after.strip()
    return instruction.strip()


def build_dataset(records: list[dict[str, Any]], source_filter: str | None = None) -> list[dict[str, Any]]:
    """Convert Qwen-formatted records into (text, label) classification examples."""
    examples = []
    for record in records:
        if source_filter and record.get("source") != source_filter:
            continue
        text = extract_transaction(record)
        category = record.get("category", "OTHER")
        if category not in LABEL2ID:
            category = "OTHER"
        examples.append({"text": text, "label": LABEL2ID[category], "category": category, "id": record.get("id", "")})
    return examples


# ---------------------------------------------------------------------------
# Tokenized dataset
# ---------------------------------------------------------------------------


class GLDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[dict[str, Any]], tokenizer, max_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.examples[idx]["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.examples[idx]["label"], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    model_id: str,
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    device: str,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 128,
    max_steps: int = -1,
    seed: int = 42,
    output_dir: Path | None = None,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup = _import_transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID
    )
    model.to(device)

    train_ds = GLDataset(train_examples, tokenizer, max_length)
    val_ds = GLDataset(val_examples, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = max_steps if max_steps > 0 else len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.03 * total_steps), num_training_steps=total_steps
    )

    model.train()
    global_step = 0
    train_losses: list[float] = []
    started = datetime.now(UTC)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            train_losses.append(loss.item())
            global_step += 1

            if max_steps > 0 and global_step >= max_steps:
                break

        if max_steps > 0 and global_step >= max_steps:
            break

    finished = datetime.now(UTC)

    # Save model
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    # Evaluate
    metrics = evaluate(model, tokenizer, val_examples, device, max_length, results_dir)
    metrics.update(
        {
            "model": model_id,
            "device": device,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "max_length": max_length,
            "seed": seed,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "global_step": global_step,
            "train_runtime_seconds": (finished - started).total_seconds(),
            "train_loss_mean": sum(train_losses) / len(train_losses) if train_losses else 0.0,
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
        }
    )
    return metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model,
    tokenizer,
    val_examples: list[dict[str, Any]],
    device: str,
    max_length: int = 128,
    results_dir: Path | None = None,
    split_name: str = "val",
) -> dict[str, Any]:
    model.eval()
    predictions: list[int] = []
    gold: list[int] = []
    domain_preds: list[int] = []
    domain_gold: list[int] = []
    domain_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for example in val_examples:
            enc = tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = int(outputs.logits.argmax(dim=-1).item())
            label = example["label"]

            predictions.append(pred)
            gold.append(label)

            # Track domain (LedgerLoop) rows separately for unseen-merchant eval
            if example.get("category") and example["category"] != "general_qa":
                domain_preds.append(pred)
                domain_gold.append(label)
                domain_rows.append(
                    {
                        "id": example["id"],
                        "text": example["text"],
                        "gold": example["category"],
                        "prediction": ID2LABEL[pred],
                        "correct": pred == label,
                    }
                )

    model.train()

    # Overall metrics
    correct = sum(1 for p, g in zip(predictions, gold) if p == g)
    accuracy = correct / len(predictions) if predictions else 0.0

    # Macro F1
    f1 = _macro_f1(predictions, gold, NUM_LABELS)

    # Domain-only exact match (unseen merchants)
    domain_correct = sum(1 for p, g in zip(domain_preds, domain_gold) if p == g)
    domain_accuracy = domain_correct / len(domain_preds) if domain_preds else 0.0
    domain_f1 = _macro_f1(domain_preds, domain_gold, NUM_LABELS)

    metrics: dict[str, Any] = {
        "split": split_name,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(f1, 4),
        "domain_exact_match": round(domain_accuracy, 4),
        "domain_macro_f1": round(domain_f1, 4),
        "n_total": len(predictions),
        "n_domain": len(domain_preds),
        "per_category": _per_category_metrics(predictions, gold, NUM_LABELS, ID2LABEL),
    }

    if domain_rows:
        metrics["domain_predictions"] = domain_rows

    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        out = results_dir / f"eval_{split_name}.json"
        out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    return metrics


def _macro_f1(preds: list[int], golds: list[int], num_labels: int) -> float:
    f1s = []
    for label in range(num_labels):
        tp = sum(1 for p, g in zip(preds, golds) if p == label and g == label)
        fp = sum(1 for p, g in zip(preds, golds) if p == label and g != label)
        fn = sum(1 for p, g in zip(preds, golds) if p != label and g == label)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp) / denom if denom > 0 else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def _per_category_metrics(
    preds: list[int], golds: list[int], num_labels: int, id2label: dict[int, str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in range(num_labels):
        tp = sum(1 for p, g in zip(preds, golds) if p == label and g == label)
        fp = sum(1 for p, g in zip(preds, golds) if p == label and g != label)
        fn = sum(1 for p, g in zip(preds, golds) if p != label and g == label)
        support = sum(1 for g in golds if g == label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        result[id2label[label]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="bert-base-uncased", help="HF model id (bert-* or roberta-*)")
    parser.add_argument("--train-file", type=Path, default=REPO_ROOT / "sft" / "data" / "train.jsonl")
    parser.add_argument("--val-file", type=Path, default=REPO_ROOT / "sft" / "data" / "val.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "nlp" / "outputs")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "nlp" / "results")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=-1, help="override total training steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-filter", choices=("ledgerloop", "dolly", None), default=None)
    parser.add_argument("--eval-only", action="store_true", help="only run evaluation (needs --output-dir with model)")
    args = parser.parse_args(argv)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_records = load_jsonl(args.train_file)
    val_records = load_jsonl(args.val_file)
    train_examples = build_dataset(train_records, args.source_filter)
    val_examples = build_dataset(val_records, args.source_filter)

    print(f"device={device} model={args.model}")
    print(f"train={len(train_examples)} val={len(val_examples)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_id=args.model,
        train_examples=train_examples,
        val_examples=val_examples,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        max_steps=args.max_steps,
        seed=args.seed,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
    )

    print(json.dumps({k: metrics[k] for k in ("domain_exact_match", "domain_macro_f1", "accuracy", "macro_f1") if k in metrics}, indent=2))
    print(f"wrote {args.results_dir / 'eval_val.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
