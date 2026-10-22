#!/usr/bin/env python3
"""Evaluate a base model or a LedgerLoop LoRA adapter on held-out instructions.

Reports completion-only cross-entropy, perplexity, exact-match accuracy,
token-level F1 and valid-code adherence for the synthetic GL-coding set, plus
loss on the held-out general-instruction sample.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from .domain_data import GL_CATEGORIES
except ImportError:  # direct script execution
    from domain_data import GL_CATEGORIES

ROOT = Path(__file__).resolve().parent
ALLOWED = set(GL_CATEGORIES)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", type=Path, default=None, help="PEFT adapter directory")
    p.add_argument("--val-file", type=Path, default=ROOT / "data" / "val.jsonl")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-4bit", action="store_true")
    return p.parse_args(argv)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(model_id: str, adapter: Path | None) -> AutoTokenizer:
    source = str(adapter) if adapter and (adapter / "tokenizer_config.json").exists() else model_id
    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model(model_id: str, adapter: Path | None, device: str, use_4bit: bool) -> tuple[Any, bool]:
    quantized = use_4bit and device == "cuda"
    kwargs: dict[str, Any] = {"trust_remote_code": False}
    if quantized:
        kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
            device_map={"": 0},
            dtype=torch.float16,
            attn_implementation="sdpa",
        )
    elif device == "cuda":
        kwargs.update(dtype=torch.float16, attn_implementation="sdpa")
    elif device == "mps":
        kwargs.update(torch_dtype=torch.float16)
    else:
        kwargs.update(torch_dtype=torch.float32)

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter:
        if not (adapter / "adapter_config.json").exists():
            raise SystemExit(f"{adapter} is not a PEFT adapter directory")
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    return model, quantized


def completion_loss(
    model: Any,
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
    device: str,
    max_length: int,
) -> tuple[float, int] | None:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        prompt + completion,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"]
    if len(input_ids) <= len(prompt_ids):
        return None
    labels = list(input_ids)
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100
    n_tokens = sum(1 for x in labels if x != -100)
    if n_tokens == 0:
        return None
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    label_t = torch.tensor([labels], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=mask, labels=label_t)
    return float(out.loss.item()) * n_tokens, n_tokens


def normalize_prediction(raw: str) -> str:
    clean = re.sub(r"[^A-Z0-9_]+", " ", raw.upper())
    for category in sorted(ALLOWED, key=len, reverse=True):
        if re.search(rf"(?<![A-Z0-9_]){re.escape(category)}(?![A-Z0-9_])", clean):
            return category
    return clean.strip() or "INVALID"


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = prediction.split()
    gold_tokens = gold.split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in set(pred_tokens))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def generate(model: Any, tokenizer: AutoTokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, encoded["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.split("<|im_end|>")[0].strip()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    use_4bit = not args.no_4bit and device == "cuda"
    if device != "cuda" and not args.no_4bit:
        print(f"device={device}: 4-bit needs CUDA; evaluating in full precision")

    if not args.val_file.exists():
        raise SystemExit(f"missing {args.val_file}; run `python sft/format_dataset.py` first")
    rows = load_dataset("json", data_files=str(args.val_file), split="train")
    if args.limit:
        rows = rows.select(range(min(args.limit, len(rows))))

    tokenizer = load_tokenizer(args.model, args.adapter)
    model, quantized = load_model(args.model, args.adapter, device, use_4bit)

    started = time.time()
    loss_num = 0.0
    loss_den = 0
    source_loss: dict[str, list[float]] = defaultdict(list)
    source_tokens: dict[str, int] = defaultdict(int)
    per_category: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    predictions: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        result = completion_loss(
            model,
            tokenizer,
            row["prompt"],
            row["completion"],
            device,
            args.max_seq_length,
        )
        if result is not None:
            total_loss, n_tokens = result
            loss_num += total_loss
            loss_den += n_tokens
            source_loss[row["source"]].append(total_loss / n_tokens)
            source_tokens[row["source"]] += n_tokens

        if row["source"] != "ledgerloop":
            continue
        gold = row["category"]
        raw = generate(model, tokenizer, row["prompt"], device, args.max_new_tokens)
        prediction = normalize_prediction(raw)
        exact = prediction == gold
        f1 = token_f1(prediction, gold)
        format_ok = prediction in ALLOWED
        per_category[gold]["n"] += 1
        per_category[gold]["correct"] += int(exact)
        predictions.append(
            {
                "id": row["id"],
                "instruction": row["instruction"].split("Transaction:\n")[-1],
                "gold": gold,
                "prediction": prediction,
                "raw_output": raw,
                "exact_match": exact,
                "token_f1": f1,
                "format_ok": format_ok,
            }
        )
        if (i + 1) % 10 == 0:
            print(f"evaluated {i + 1}/{len(rows)}", flush=True)

    n_domain = len(predictions)
    accuracy = (
        sum(p["exact_match"] for p in predictions) / n_domain if n_domain else 0.0
    )
    mean_f1 = sum(p["token_f1"] for p in predictions) / n_domain if n_domain else 0.0
    format_rate = sum(p["format_ok"] for p in predictions) / n_domain if n_domain else 0.0
    avg_loss = loss_num / loss_den if loss_den else float("nan")

    metrics: dict[str, Any] = {
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "device": device,
        "quantized_4bit": quantized,
        "n_eval": len(rows),
        "n_domain": n_domain,
        "completion_loss": avg_loss,
        "perplexity": float(torch.exp(torch.tensor(avg_loss)).item()) if loss_den else None,
        "loss_by_source": {
            source: sum(vals) / len(vals) for source, vals in source_loss.items()
        },
        "domain_exact_match": accuracy,
        "domain_token_f1": mean_f1,
        "format_adherence": format_rate,
        "per_category": {
            cat: {
                "n": v["n"],
                "correct": v["correct"],
                "accuracy": v["correct"] / v["n"] if v["n"] else 0.0,
            }
            for cat, v in sorted(per_category.items())
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "predictions": predictions,
    }
    output = args.output
    if output is None:
        name = "eval_adapter.json" if args.adapter else "eval_base.json"
        output = args.results_dir / name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    headline = {k: metrics[k] for k in (
        "completion_loss",
        "perplexity",
        "domain_exact_match",
        "domain_token_f1",
        "format_adherence",
        "loss_by_source",
        "n_domain",
        "n_eval",
    )}
    print(json.dumps(headline, indent=2))
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
