#!/usr/bin/env python3
"""ViT-based receipt field detector.

Fine-tunes a Vision Transformer (google/vit-base-patch16-224) as an object
detector for receipt fields (merchant, date, total) on the synthetic SROIE→CORD
split (150 train / 50 val images). Reports mAP@0.5 comparable to the existing
YOLOv8n baseline.

Architecture
------------
ViT backbone (patch 16 → 14×14 grid of 768-d tokens) + a per-patch detection
head predicting class logits (background / merchant / date / total) and box
deltas (cx, cy, w, h) relative to the patch grid cell. Boxes are decoded,
filtered, NMS-suppressed and evaluated with COCO-style mAP@0.5.

Usage::

    # train on Colab T4
    python cv/train_vit_detect.py \
        --data-root cv/data \
        --model google/vit-base-patch16-224 \
        --epochs 10 --batch-size 8 --lr 1e-4 --device cuda \
        --output-dir cv/outputs/vit-detect \
        --results-dir cv/results

    # evaluate mAP
    python cv/train_vit_detect.py --mode eval \
        --data-root cv/data \
        --model-dir cv/outputs/vit-detect \
        --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "cv"))

# ---------------------------------------------------------------------------
# Lazy transformers import
# ---------------------------------------------------------------------------

def _import_transformers():
    from transformers import AutoImageProcessor, AutoModel
    return AutoImageProcessor, AutoModel


# ---------------------------------------------------------------------------
# Label parsing (YOLO format: class cx cy w h, normalized 0-1)
# ---------------------------------------------------------------------------

CLASS_NAMES = ["merchant", "date", "total"]
CLASS2ID = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)  # 3, background handled implicitly


def _read_labels(label_path: Path) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = (float(p) for p in parts[1:])
        boxes.append({"class_id": cls_id, "cx": cx, "cy": cy, "w": w, "h": h})
    return boxes


# ---------------------------------------------------------------------------
# IoU & mAP
# ---------------------------------------------------------------------------


def _box_xyxy(box: dict[str, Any]) -> tuple[float, float, float, float]:
    cx, cy, w, h = box["cx"], box["cy"], box["w"], box["h"]
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_map(
    predictions: list[dict[str, Any]],
    ground_truths: dict[str, list[dict[str, Any]]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """COCO-style mAP@iou_threshold across classes."""
    # Group predictions by class
    per_class_preds: dict[int, list[dict[str, Any]]] = {c: [] for c in range(NUM_CLASSES)}
    per_class_gts: dict[int, list[dict[str, Any]]] = {c: [] for c in range(NUM_CLASSES)}
    gt_seen: dict[tuple[str, int], bool] = {}

    for p in predictions:
        per_class_preds[p["class_id"]].append(p)

    for img_id, boxes in ground_truths.items():
        for b in boxes:
            per_class_gts[b["class_id"]].append({**b, "image_id": img_id})

    results: dict[str, Any] = {}
    total_ap = 0.0
    for cls in range(NUM_CLASSES):
        preds = sorted(per_class_preds[cls], key=lambda x: x["score"], reverse=True)
        gts = per_class_gts[cls]
        n_gts = len(gts)
        matched: dict[tuple[str, int], bool] = {}
        tp = [0] * len(preds)
        fp = [0] * len(preds)

        for i, p in enumerate(preds):
            best_iou, best_gt = 0.0, None
            for g in gts:
                key = (g["image_id"], id(g))
                if matched.get(key, False):
                    continue
                iou_val = _iou(_box_xyxy(p), _box_xyxy(g))
                if iou_val > best_iou:
                    best_iou, best_gt = iou_val, g
            if best_gt is not None and best_iou >= iou_threshold:
                key = (best_gt["image_id"], id(best_gt))
                matched[key] = True
                tp[i] = 1
            else:
                fp[i] = 1

        tp_cum = [sum(tp[: i + 1]) for i in range(len(tp))]
        fp_cum = [sum(fp[: i + 1]) for i in range(len(fp))]
        recalls = [t / max(1, n_gts) for t in tp_cum]
        precisions = [t / max(1, t + f) for t, f in zip(tp_cum, fp_cum)]

        # VOC-style AP (11-point interpolation)
        ap = 0.0
        for t in range(0, 11):
            r = t / 10.0
            p_max = max([p for p, rec in zip(precisions, recalls) if rec >= r], default=0.0)
            ap += p_max / 11.0
        results[CLASS_NAMES[cls]] = {
            "ap": round(ap, 4),
            "tp": sum(tp),
            "fp": sum(fp),
            "n_gts": n_gts,
        }
        total_ap += ap

    m_ap = total_ap / NUM_CLASSES
    return {"mAP": round(m_ap, 4), "per_class": results}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ReceiptFieldDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        processor,
        image_size: int = 224,
    ):
        self.processor = processor
        self.image_size = image_size
        self.samples: list[dict[str, Any]] = []
        img_dir = data_root / "images" / split
        lbl_dir = data_root / "labels" / split
        for img_path in sorted(img_dir.glob("*.jpg")):
            label_path = lbl_dir / (img_path.stem + ".txt")
            boxes = _read_labels(label_path)
            self.samples.append(
                {"image_path": str(img_path), "image_id": img_path.stem, "boxes": boxes}
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        boxes = sample["boxes"]
        # Pad/truncate boxes to fixed max
        max_boxes = 30
        box_tensor = torch.zeros(max_boxes, 5)  # class_id, cx, cy, w, h
        mask = torch.zeros(max_boxes, dtype=torch.bool)
        for i, b in enumerate(boxes[:max_boxes]):
            box_tensor[i] = torch.tensor(
                [b["class_id"], b["cx"], b["cy"], b["w"], b["h"]], dtype=torch.float32
            )
            mask[i] = True
        return {
            "pixel_values": pixel_values,
            "boxes": box_tensor,
            "mask": mask,
            "image_id": sample["image_id"],
            "raw_boxes": boxes,
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "boxes": torch.stack([b["boxes"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "image_id": [b["image_id"] for b in batch],
        "raw_boxes": [b["raw_boxes"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Model: ViT backbone + per-patch detection head
# ---------------------------------------------------------------------------

class ViTFieldDetector(nn.Module):
    def __init__(self, model_id: str, num_classes: int = NUM_CLASSES):
        super().__init__()
        AutoImageProcessor, AutoModel = _import_transformers()
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.backbone = AutoModel.from_pretrained(model_id)
        hidden = self.backbone.config.hidden_size  # 768 for vit-base
        self.num_patches = (224 // 16) ** 2  # 196
        # Detection head: per patch -> class logits (bg + classes) + box delta
        self.cls_head = nn.Linear(hidden, num_classes + 1)
        self.box_head = nn.Linear(hidden, 4)
        self.num_classes = num_classes

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        # last_hidden_state: [B, 1+num_patches, hidden]; token 0 is CLS
        tokens = outputs.last_hidden_state[:, 1:, :]  # [B, 196, hidden]
        cls_logits = self.cls_head(tokens)   # [B, 196, num_classes+1]
        box_deltas = self.box_head(tokens)   # [B, 196, 4]
        return cls_logits, box_deltas


# ---------------------------------------------------------------------------
# Patch grid anchor computation
# ---------------------------------------------------------------------------

def _patch_centers(image_size: int = 224, patch: int = 16) -> torch.Tensor:
    grid = image_size // patch  # 14
    centers = []
    for i in range(grid):
        for j in range(grid):
            cx = (j + 0.5) * patch / image_size
            cy = (i + 0.5) * patch / image_size
            centers.append((cx, cy))
    return torch.tensor(centers, dtype=torch.float32)  # [196, 2]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model_id: str,
    data_root: Path,
    device: str,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-4,
    image_size: int = 224,
    output_dir: Path | None = None,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    AutoImageProcessor, _ = _import_transformers()
    processor = AutoImageProcessor.from_pretrained(model_id)

    train_ds = ReceiptFieldDataset(data_root, "train", processor, image_size)
    val_ds = ReceiptFieldDataset(data_root, "val", processor, image_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = ViTFieldDetector(model_id).to(device)
    patch_centers = _patch_centers(image_size).to(device)  # [196, 2]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    cls_criterion = nn.CrossEntropyLoss(reduction="none")
    box_criterion = nn.SmoothL1Loss(reduction="none")

    model.train()
    started = datetime.now(UTC)
    train_losses: list[float] = []
    grid = image_size // 16
    cell = 1.0 / grid

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            pv = batch["pixel_values"].to(device)
            gt_boxes = batch["boxes"].to(device)   # [B, 30, 5]
            gt_mask = batch["mask"].to(device)     # [B, 30]

            cls_logits, box_deltas = model(pv)     # [B, 196, C+1], [B, 196, 4]
            B, P, _ = cls_logits.shape

            # Build targets per patch: assign each GT box to the patch whose center
            # falls inside the box (simplified assignment).
            cls_target = torch.zeros(B, P, dtype=torch.long, device=device)  # background=0
            box_target = torch.zeros(B, P, 4, device=device)
            box_mask = torch.zeros(B, P, dtype=torch.bool, device=device)

            for b in range(B):
                centers = patch_centers  # [P, 2]
                for gi in range(gt_mask[b].sum().item()):
                    cls_id = int(gt_boxes[b, gi, 0].item())
                    cx, cy, w, h = gt_boxes[b, gi, 1:].tolist()
                    # patches whose center is inside the GT box
                    inside = (
                        (centers[:, 0] >= cx - w / 2)
                        & (centers[:, 0] <= cx + w / 2)
                        & (centers[:, 1] >= cy - h / 2)
                        & (centers[:, 1] <= cy + h / 2)
                    )
                    if inside.sum() == 0:
                        continue
                    cls_target[b, inside] = cls_id + 1  # +1 for background offset
                    pcx = centers[inside, 0].mean().item()
                    pcy = centers[inside, 1].mean().item()
                    # YOLO-style encoded targets relative to patch cell
                    box_target[b, inside, 0] = (cx - pcx) / cell
                    box_target[b, inside, 1] = (cy - pcy) / cell
                    box_target[b, inside, 2] = math.log(max(w, 1e-6) / cell)
                    box_target[b, inside, 3] = math.log(max(h, 1e-6) / cell)
                    box_mask[b, inside] = True

            # Classification loss (focal-ish weighting via CE)
            cls_loss = cls_criterion(cls_logits.transpose(1, 2), cls_target)  # [B, P]
            # balance positives/negatives
            pos_mask = cls_target > 0
            neg_mask = cls_target == 0
            pos_count = pos_mask.sum().clamp_min(1)
            neg_count = neg_mask.sum().clamp_min(1)
            cls_loss = (cls_loss * pos_mask).sum() / pos_count + 0.25 * (
                cls_loss * neg_mask
            ).sum() / neg_count

            # Box loss on positive patches
            box_loss = box_criterion(
                box_deltas, box_target
            ).sum(dim=-1)  # [B, P]
            box_loss = (box_loss * box_mask).sum() / box_mask.sum().clamp_min(1)

            loss = cls_loss + box_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_losses.append(loss.item())

        print(f"epoch {epoch+1}/{epochs} train_loss={epoch_loss/max(1,len(train_loader)):.4f}")

    finished = datetime.now(UTC)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / "vit_detect.pt")
        # Save config
        (output_dir / "config.json").write_text(
            json.dumps({"model": model_id, "image_size": image_size, "num_classes": NUM_CLASSES}, indent=2)
        )

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
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
            "train_runtime_seconds": (finished - started).total_seconds(),
            "train_loss_mean": sum(train_losses) / len(train_losses) if train_losses else 0.0,
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
        }
    )
    return metrics


# ---------------------------------------------------------------------------
# Evaluation (mAP)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: ViTFieldDetector,
    processor,
    val_dataset: Dataset,
    device: str,
    image_size: int = 224,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    patch_centers = _patch_centers(image_size).to(device)  # [196, 2]
    grid = image_size // 16

    predictions: list[dict[str, Any]] = []
    ground_truths: dict[str, list[dict[str, Any]]] = {}

    loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=2
    )

    for batch in loader:
        pv = batch["pixel_values"].to(device)
        cls_logits, box_deltas = model(pv)  # [B,196,C+1], [B,196,4]
        B, P, _ = cls_logits.shape
        probs = torch.softmax(cls_logits, dim=-1)  # [B,196,C+1]

        for b in range(B):
            image_id = batch["image_id"][b]
            gt_boxes = batch["raw_boxes"][b]
            ground_truths[image_id] = [
                {"class_id": int(g["class_id"]), "cx": g["cx"], "cy": g["cy"], "w": g["w"], "h": g["h"]}
                for g in gt_boxes
            ]

            # Get top-K patches by max class confidence (excluding background)
            class_probs = probs[b, :, 1:]  # [196, C]
            max_class_prob, max_class = class_probs.max(dim=-1)  # [196]
            topk = torch.topk(max_class_prob, k=min(200, P))
            for idx in topk.indices.tolist():
                score = float(max_class_prob[idx])
                if score < 0.05:
                    continue
                cls_id = int(max_class[idx].item())
                cx0, cy0 = patch_centers[idx].tolist()
                delta = box_deltas[b, idx].tolist()
                # decode box relative to patch cell (YOLO-style bounded offsets)
                cell = 1.0 / grid
                cx = cx0 + (1.0 / (1.0 + math.exp(-delta[0]))) * cell
                cy = cy0 + (1.0 / (1.0 + math.exp(-delta[1]))) * cell
                w = math.exp(delta[2]) * cell
                h = math.exp(delta[3]) * cell
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                w = min(max(w, 0.0), 1.0)
                h = min(max(h, 0.0), 1.0)
                predictions.append(
                    {
                        "image_id": image_id,
                        "class_id": cls_id,
                        "score": score,
                        "cx": cx,
                        "cy": cy,
                        "w": w,
                        "h": h,
                    }
                )

    model.train()

    # NMS per image per class
    def nms(boxes: list[dict[str, Any]], iou_thr: float = 0.5) -> list[dict[str, Any]]:
        kept = []
        boxes = sorted(boxes, key=lambda x: x["score"], reverse=True)
        suppressed = [False] * len(boxes)
        for i in range(len(boxes)):
            if suppressed[i]:
                continue
            kept.append(boxes[i])
            bi = _box_xyxy(boxes[i])
            for j in range(i + 1, len(boxes)):
                if suppressed[j]:
                    continue
                if boxes[j]["class_id"] != boxes[i]["class_id"]:
                    continue
                if _iou(bi, _box_xyxy(boxes[j])) >= iou_thr:
                    suppressed[j] = True
        return kept

    per_image: dict[str, list[dict[str, Any]]] = {}
    for p in predictions:
        per_image.setdefault(p["image_id"], []).append(p)
    final_preds: list[dict[str, Any]] = []
    for img_id, boxes in per_image.items():
        final_preds.extend(nms(boxes))

    map_results = compute_map(final_preds, ground_truths, iou_threshold=0.5)

    metrics = {
        "split": "val",
        "mAP50": map_results["mAP"],
        "per_class": map_results["per_class"],
        "n_predictions": len(final_preds),
        "n_images": len(ground_truths),
    }

    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "eval_detect.json").write_text(json.dumps(metrics, indent=2) + "\n")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "cv" / "data")
    parser.add_argument("--model", default="google/vit-base-patch16-224")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "cv" / "outputs" / "vit-detect")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "cv" / "results")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args(argv)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"device={device}")

    if args.mode == "eval":
        model_dir = args.model_dir or args.output_dir
        config = json.loads((model_dir / "config.json").read_text())
        model = ViTFieldDetector(config["model"]).to(device)
        state = torch.load(model_dir / "vit_detect.pt", map_location=device)
        model.load_state_dict(state)
        processor = AutoImageProcessor = _import_transformers()[0]
        processor = processor.from_pretrained(config["model"])
        val_ds = ReceiptFieldDataset(args.data_root, "val", processor, args.image_size)
        metrics = evaluate(model, processor, val_ds, device, args.image_size, args.results_dir)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.results_dir.mkdir(parents=True, exist_ok=True)
        metrics = train(
            model_id=args.model,
            data_root=args.data_root,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            image_size=args.image_size,
            output_dir=args.output_dir,
            results_dir=args.results_dir,
        )

    print(json.dumps({k: metrics[k] for k in ("mAP50", "per_class", "n_predictions", "train_runtime_seconds") if k in metrics}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
