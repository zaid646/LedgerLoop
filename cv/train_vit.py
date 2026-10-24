#!/usr/bin/env python3
"""Vision Transformer fine-tuning for LedgerLoop receipt GL classification.

Fine-tunes a ViT (google/vit-base-patch16-224 by default) on the synthetic
receipt dataset (cv/data/) for GL-category classification from the full
receipt image. Complements the existing YOLOv8n object-detection baseline
(cv/runs/receipt_fields_yolov8n/) which detects merchant/date/total fields.

Trains on 150 synthetic receipt images (50 docs × 3 variants) and evaluates on
50 held-out images (25 docs × 2 variants). Reports accuracy and per-category
metrics. Results are comparable in spirit to YOLOv8n's mAP (different task:
classification vs detection) but measure the same underlying receipt
understanding capability.

Usage::

    # CPU smoke test (tiny model, few steps)
    python cv/train_vit.py --model google/vit-base-patch16-224 --epochs 1 --max-steps 10 --device cpu

    # Full CUDA run (remote GPU via SSH)
    python cv/train_vit.py --model google/vit-base-patch16-224 --epochs 10 --device cuda

Usage with comparison to YOLOv8n baseline::

    python cv/train_vit.py --model google/vit-base-patch16-224 --epochs 10 --device cuda --output-dir cv/outputs/vit-base
    python cv/compare_vit_yolo.py --vit-results cv/results/eval_val.json --yolo-results cv/runs/receipt_fields_yolov8n/results.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

# torchvision imported lazily so the module loads without it installed.
def _import_torchvision():
    from torchvision import datasets, transforms
    return datasets, transforms

# Transformers imported lazily so the module loads without it installed.
def _import_transformers():
    from transformers import AutoModelForImageClassification, AutoImageProcessor, get_linear_schedule_with_warmup
    return AutoModelForImageClassification, AutoImageProcessor, get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "cv"))

# 13 GL categories — same taxonomy as sft/domain_data.py
GL_CATEGORIES: tuple[str, ...] = (
    "TRAVEL", "MEALS", "OFFICE_SUPPLIES", "SOFTWARE", "HARDWARE",
    "MARKETING", "PROFESSIONAL_SERVICES", "UTILITIES", "RENT",
    "INSURANCE", "TRAINING", "ENTERTAINMENT", "OTHER",
)
LABEL2ID = {cat: i for i, cat in enumerate(GL_CATEGORIES)}
ID2LABEL = {i: cat for cat, i in LABEL2ID.items()}
NUM_LABELS = len(GL_CATEGORIES)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ReceiptGLDataset(Dataset):
    """Wraps an ImageFolder dataset and maps folder names to GL category labels.

    Expects directory layout::

        cv/data/vit_train/
            TRAVEL/  *.jpg
            MEALS/   *.jpg
            ...
        cv/data/vit_val/
            ...

    Folder names must match GL_CATEGORIES exactly.
    """

    def __init__(self, root: str, processor, image_size: int = 224):
        _, transforms = _import_torchvision()
        self.processor = processor
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ])
        self.samples: list[tuple[str, int]] = []
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"dataset root not found: {root}")
        for cat_dir in sorted(root_path.iterdir()):
            if not cat_dir.is_dir():
                continue
            cat_name = cat_dir.name.upper()
            if cat_name not in LABEL2ID:
                continue
            label = LABEL2ID[cat_name]
            for img_path in cat_dir.glob("*"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((str(img_path), label))
        if not self.samples:
            raise ValueError(f"no images found under {root} with GL category folders")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path, label = self.samples[idx]
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    model_id: str,
    train_root: str,
    val_root: str,
    device: str,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_steps: int = -1,
    seed: int = 42,
    output_dir: Path | None = None,
    results_dir: Path | None = None,
    image_size: int = 224,
) -> dict[str, Any]:
    AutoModelForImageClassification, AutoImageProcessor, get_linear_schedule_with_warmup = _import_transformers()

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(
        model_id, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID
    )
    model.to(device)

    train_ds = ReceiptGLDataset(train_root, processor, image_size)
    val_ds = ReceiptGLDataset(val_root, processor, image_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

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
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
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
        processor.save_pretrained(output_dir)

    # Evaluate
    metrics = evaluate(model, processor, val_ds, device, image_size, results_dir)
    metrics.update(
        {
            "model": model_id,
            "device": device,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "image_size": image_size,
            "seed": seed,
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
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
    processor,
    val_dataset: Dataset,
    device: str,
    image_size: int = 224,
    results_dir: Path | None = None,
    split_name: str = "val",
) -> dict[str, Any]:
    model.eval()
    predictions: list[int] = []
    gold: list[int] = []

    with torch.inference_mode():
        for i in range(len(val_dataset)):
            sample = val_dataset[i]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            label = sample["labels"].item()

            outputs = model(pixel_values=pixel_values)
            pred = int(outputs.logits.argmax(dim=-1).item())

            predictions.append(pred)
            gold.append(label)

    model.train()

    correct = sum(1 for p, g in zip(predictions, gold) if p == g)
    accuracy = correct / len(predictions) if predictions else 0.0
    f1 = _macro_f1(predictions, gold, NUM_LABELS)

    per_category = _per_category_metrics(predictions, gold, NUM_LABELS, ID2LABEL)

    metrics: dict[str, Any] = {
        "split": split_name,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(f1, 4),
        "n_total": len(predictions),
        "per_category": per_category,
    }

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
    parser.add_argument("--model", default="google/vit-base-patch16-224", help="HF ViT model id")
    parser.add_argument("--train-root", type=Path, default=REPO_ROOT / "cv" / "data" / "vit_train")
    parser.add_argument("--val-root", type=Path, default=REPO_ROOT / "cv" / "data" / "vit_val")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "cv" / "outputs")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "cv" / "results")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-steps", type=int, default=-1, help="override total training steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.train_root.exists():
        raise SystemExit(
            f"training data not found at {args.train_root}. "
            "Prepare it by sorting synthetic receipt images into GL category folders."
        )
    if not args.val_root.exists():
        raise SystemExit(
            f"validation data not found at {args.val_root}. "
            "Prepare it by sorting synthetic receipt images into GL category folders."
        )

    print(f"device={device} model={args.model}")
    print(f"train_root={args.train_root} val_root={args.val_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_id=args.model,
        train_root=str(args.train_root),
        val_root=str(args.val_root),
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        seed=args.seed,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        image_size=args.image_size,
    )

    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1") if k in metrics}, indent=2))
    print(f"wrote {args.results_dir / 'eval_val.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
