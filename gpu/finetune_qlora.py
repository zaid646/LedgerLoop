"""Phase 6 — QLoRA fine-tune a small GL-coding classifier.

Trains a LoRA adapter on top of a 0.5–1.5B instruct model (default
``Qwen/Qwen2.5-0.5B-Instruct``) to map ``<description> - <merchant>`` -> GL
category using the pairs from ``gpu/coding_data.py``. 4-bit NF4 quantisation
(QLoRA) keeps it comfortably inside a single 4090 alongside the served 7B.

Run on the GPU box:

    python gpu/finetune_qlora.py --epochs 3 --out /workspace/expenseiq-qlora
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from coding_data import build


def _encode(tok, ex: dict) -> dict:
    p = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
    t = tok(ex["target"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = p + t
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(p) + t}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--out", default="/workspace/expenseiq-qlora")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16
    )
    model = prepare_model_for_kbit_training(model)
    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)

    train_rows, val_rows = build(seed=args.seed)
    ds = {
        "train": [_encode(tok, r) for r in train_rows],
        "validation": [_encode(tok, r) for r in val_rows],
    }
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)

    out = Path(args.out)
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="epoch",
        report_to=[],
        seed=args.seed,
    )

    class _DS(torch.utils.data.Dataset):
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int) -> dict:
            return self.rows[i]

    start = time.time()
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=_DS(ds["train"]),
        eval_dataset=_DS(ds["validation"]),
        data_collator=collator,
    )
    result = trainer.train()
    minutes = (time.time() - start) / 60

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)

    meta = {
        "base_model": args.model,
        "epochs": args.epochs,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_loss": result.training_loss,
        "minutes": round(minutes, 2),
        "adapter_dir": str(out),
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
