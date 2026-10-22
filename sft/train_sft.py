#!/usr/bin/env python3
"""QLoRA supervised fine-tuning for LedgerLoop's expense-coding assistant.

The script is tuned for a free Colab T4 (16 GB): Qwen2.5-1.5B-Instruct is loaded
in 4-bit NF4, LoRA adapters are trained on prompt/completion JSONL, and the
result is saved as a PEFT adapter. It also supports CPU and Apple MPS dry runs
(without 4-bit quantization) so the pipeline can be smoke-tested locally.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "qwen2.5-1.5b-ledgerloop-lora"
DEFAULT_RESULTS_DIR = ROOT / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train-file", type=Path, default=DEFAULT_DATA_DIR / "train.jsonl")
    p.add_argument("--val-file", type=Path, default=DEFAULT_DATA_DIR / "val.jsonl")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-4bit", action="store_true", help="disable QLoRA (CPU/MPS/debug)")
    p.add_argument("--dry-run", action="store_true", help="10 optimizer steps, tiny sequence length")
    return p.parse_args(argv)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(model_id: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model(model_id: str, device: str, use_4bit: bool) -> tuple[Any, bool]:
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
    model.config.use_cache = False
    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model, quantized


def gpu_name() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "cuda"


def write_loss_curve(log_history: list[dict[str, Any]], path: Path) -> None:
    rows = [r for r in log_history if "loss" in r and "step" in r]
    if not rows:
        return
    (path.parent).mkdir(parents=True, exist_ok=True)
    with (path.parent / "loss_log.csv").open("w", encoding="utf-8") as fh:
        fh.write("step,loss,learning_rate,epoch\n")
        for r in rows:
            fh.write(
                f"{r.get('step')},{r.get('loss')},{r.get('learning_rate', '')},{r.get('epoch', '')}\n"
            )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; wrote loss_log.csv only")
        return
    steps = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, losses, marker="o", linewidth=1.5)
    plt.xlabel("optimizer step")
    plt.ylabel("training loss")
    plt.title("LedgerLoop SFT (Qwen2.5-1.5B, QLoRA)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    if args.dry_run:
        args.max_steps = args.max_steps if args.max_steps > 0 else 10
        args.max_seq_length = min(args.max_seq_length, 256)
        args.epochs = 1.0
        args.output_dir = args.output_dir.with_name(args.output_dir.name + "-dry-run")
    if device != "cuda" and not args.no_4bit:
        print(f"device={device}: 4-bit QLoRA needs CUDA; continuing with full precision")
    use_4bit = not args.no_4bit and device == "cuda"

    if not args.train_file.exists():
        raise SystemExit(f"missing {args.train_file}; run `python sft/format_dataset.py` first")
    if not args.val_file.exists():
        raise SystemExit(f"missing {args.val_file}; run `python sft/format_dataset.py` first")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device} model={args.model} 4bit={use_4bit}")
    tokenizer = load_tokenizer(args.model)
    train_ds = load_dataset("json", data_files=str(args.train_file), split="train")
    val_ds = load_dataset("json", data_files=str(args.val_file), split="train")
    print(f"train={len(train_ds)} val={len(val_ds)}")

    model, quantized = load_model(args.model, device, use_4bit)
    if device == "cuda" and not quantized:
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    config = SFTConfig(
        output_dir=str(args.output_dir),
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
        completion_only_loss=False,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=_warmup_steps(len(train_ds), args),
        weight_decay=args.weight_decay,
        optim="paged_adamw_8bit" if quantized else "adamw_torch",
        max_grad_norm=0.3,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="epoch" if args.max_steps < 0 else "steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="epoch" if args.max_steps < 0 else "steps",
        eval_steps=args.save_steps,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        fp16=False,
        bf16=False,
        gradient_checkpointing=device == "cuda",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        dataloader_pin_memory=device == "cuda",
        remove_unused_columns=True,
        use_cpu=device == "cpu",
        run_name="ledgerloop-sft",
        disable_tqdm=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=lora,
    )

    started = datetime.now(UTC)
    train_result = trainer.train()
    finished = datetime.now(UTC)

    final_dir = args.output_dir / "final"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"saved adapter to {final_dir}")

    metrics = dict(train_result.metrics)
    eval_entries = [row for row in trainer.state.log_history if "eval_loss" in row]
    if eval_entries:
        metrics["final_eval_loss"] = eval_entries[-1]["eval_loss"]
    loss_entries = [row for row in trainer.state.log_history if "loss" in row]
    if loss_entries:
        metrics["last_step_loss"] = loss_entries[-1]["loss"]
        metrics["last_step"] = loss_entries[-1].get("step")
    metrics["train_runtime_seconds"] = (finished - started).total_seconds()
    metrics["train_samples"] = len(train_ds)
    metrics["eval_samples"] = len(val_ds)
    metrics["device"] = device
    metrics["gpu"] = gpu_name()
    metrics["quantized_4bit"] = quantized
    metrics["model"] = args.model
    metrics["hyperparameters"] = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "optim": config.optim,
        "scheduler": config.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": _warmup_steps(len(train_ds), args),
        "seed": args.seed,
    }
    metrics["versions"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": _version("transformers"),
        "trl": _version("trl"),
        "peft": _version("peft"),
        "datasets": _version("datasets"),
        "bitsandbytes": _version("bitsandbytes") if quantized else None,
    }
    metrics["started_utc"] = started.isoformat()
    metrics["finished_utc"] = finished.isoformat()

    (args.results_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (args.results_dir / "trainer_log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2) + "\n", encoding="utf-8"
    )
    write_loss_curve(trainer.state.log_history, args.results_dir / "loss_curve.png")

    print(json.dumps({k: metrics[k] for k in ("train_loss", "train_runtime_seconds") if k in metrics}, indent=2))
    print(f"wrote {args.results_dir / 'train_metrics.json'}")
    return 0


def _warmup_steps(train_size: int, args: argparse.Namespace) -> int:
    if args.max_steps > 0:
        total = args.max_steps
    else:
        per_epoch = max(1, math.ceil(train_size / (args.batch_size * args.grad_accum)))
        total = max(1, math.ceil(per_epoch * args.epochs))
    return max(1, round(total * args.warmup_ratio))


def _version(package: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
