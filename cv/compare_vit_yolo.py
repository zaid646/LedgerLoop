#!/usr/bin/env python3
"""Compare ViT GL-classification results with YOLOv8n field-detection baseline.

Reads ViT evaluation JSON (from cv/train_bert.py → cv/train_vit.py) and YOLOv8n
results CSV (from cv/runs/receipt_fields_yolov8n/results.csv) and prints a
side-by-side comparison table.

Usage::

    python cv/compare_vit_yolo.py \
        --vit-results cv/results/eval_val.json \
        --yolo-results cv/runs/receipt_fields_yolov8n/results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def load_vit_results(path: Path) -> dict:
    return json.loads(path.read_text())


def load_yolo_results(path: Path) -> dict:
    """Extract the best (last) epoch metrics from YOLOv8n results.csv."""
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return {}
    last = rows[-1]
    return {
        "epoch": int(last["epoch"]),
        "mAP50(B)": float(last["metrics/mAP50(B)"]),
        "mAP50-95(B)": float(last["metrics/mAP50-95(B)"]),
        "precision(B)": float(last["metrics/precision(B)"]),
        "recall(B)": float(last["metrics/recall(B)"]),
        "train_time_s": float(last["time"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit-results", type=Path, required=True, help="ViT eval JSON")
    parser.add_argument("--yolo-results", type=Path, required=True, help="YOLOv8n results CSV")
    parser.add_argument("--output", type=Path, default=None, help="output comparison JSON")
    args = parser.parse_args(argv)

    vit = load_vit_results(args.vit_results)
    yolo = load_yolo_results(args.yolo_results)

    print("\nViT vs YOLOv8n — Receipt Understanding Comparison")
    print("=" * 60)
    print(f"{'Metric':<35} {'ViT (classification)':>22} {'YOLOv8n (detection)':>22}")
    print("-" * 60)

    vit_acc = vit.get("accuracy", 0.0)
    vit_f1 = vit.get("macro_f1", 0.0)
    yolo_map50 = yolo.get("mAP50(B)", 0.0)
    yolo_map5095 = yolo.get("mAP50-95(B)", 0.0)
    yolo_prec = yolo.get("precision(B)", 0.0)
    yolo_rec = yolo.get("recall(B)", 0.0)

    print(f"{'Overall accuracy / mAP50':<35} {vit_acc:>22.4f} {yolo_map50:>22.4f}")
    print(f"{'Macro F1 / mAP50-95':<35} {vit_f1:>22.4f} {yolo_map5095:>22.4f}")
    print(f"{'Precision':<35} {'N/A (classification)':>22} {yolo_prec:>22.4f}")
    print(f"{'Recall':<35} {'N/A (classification)':>22} {yolo_rec:>22.4f}")
    print(f"{'Task':<35} {'GL category classification':>22} {'Field bounding-box detection':>22}")
    print(f"{'Train epochs':<35} {vit.get('epochs', 'N/A'):>22} {yolo.get('epoch', 'N/A'):>22}")
    print(f"{'Train runtime (s)':<35} {str(vit.get('train_runtime_seconds', 'N/A')):>22} {str(yolo.get('train_time_s', 'N/A')):>22}")
    print(f"{'Model':<35} {vit.get('model', 'N/A'):>22} {'yolov8n.pt':>22}")
    print(f"{'Device':<35} {vit.get('device', 'N/A'):>22} {'cpu':>22}")
    print()

    comparison = {
        "vit": {
            "model": vit.get("model"),
            "accuracy": vit_acc,
            "macro_f1": vit_f1,
            "epochs": vit.get("epochs"),
            "train_runtime_seconds": vit.get("train_runtime_seconds"),
            "device": vit.get("device"),
            "train_examples": vit.get("train_examples"),
            "val_examples": vit.get("val_examples"),
        },
        "yolov8n": {
            "model": "yolov8n.pt",
            "mAP50": yolo_map50,
            "mAP50-95": yolo_map5095,
            "precision": yolo_prec,
            "recall": yolo_rec,
            "epochs": yolo.get("epoch"),
            "train_time_s": yolo.get("train_time_s"),
            "device": "cpu",
        },
        "note": "ViT performs GL-category classification (13 classes); YOLOv8n performs field bounding-box detection (3 classes). Different tasks — metrics not directly comparable.",
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
