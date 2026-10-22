"""Fine-tune YOLOv8n on the deterministic synthetic receipt dataset.

CPU only; ~200 images at 416 px for 25 epochs.  The held-out split is by
document, so validation metrics are not measured on augmented copies of
training documents.

Usage
-----
    python -m cv.synthetic                 # writes cv/data/{images,labels}
    python -m cv.train_yolo                # writes cv/weights + cv/results
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from cv.synthetic import CLASS_NAMES, DATA_DIR, build_dataset

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RUNS_DIR = Path(__file__).resolve().parent / "runs"


def _box_metric(metrics, name: str, default=0.0):
    box = getattr(metrics, "box", None)
    value = getattr(box, name, default) if box is not None else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_list(value) -> list[float]:
    if value is None:
        return []
    return [float(item) for item in np.asarray(value).reshape(-1)]


def _per_class(metrics) -> dict[str, dict[str, float]]:
    names = getattr(metrics, "names", {}) or {}
    box = getattr(metrics, "box", None)
    maps = _as_list(getattr(box, "maps", None))
    ap50 = _as_list(getattr(box, "ap50", None))
    precision = _as_list(getattr(box, "p", None))
    recall = _as_list(getattr(box, "r", None))
    out: dict[str, dict[str, float]] = {}
    for index, class_name in enumerate(CLASS_NAMES):
        key = index if index in names else str(index)
        label = str(names.get(key, class_name))
        out[label] = {
            "map50_95": round(float(maps[index]), 4) if index < len(maps) else 0.0,
            "map50": round(float(ap50[index]), 4) if index < len(ap50) else 0.0,
            "precision": round(float(precision[index]), 4) if index < len(precision) else 0.0,
            "recall": round(float(recall[index]), 4) if index < len(recall) else 0.0,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n for receipt fields")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base", default="yolov8n.pt")
    args = parser.parse_args()

    if not (args.data / "data.yaml").exists():
        manifest = build_dataset(args.data, seed=args.seed)
        print(f"built dataset: {manifest['train_images']} train / {manifest['val_images']} val")
    else:
        manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))

    from ultralytics import YOLO

    model = YOLO(args.base)
    train_result = model.train(
        data=str(args.data / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        seed=args.seed,
        deterministic=True,
        patience=10,
        project=str(RUNS_DIR),
        name="receipt_fields_yolov8n",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    del train_result

    metrics = model.val(
        data=str(args.data / "data.yaml"),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(RUNS_DIR),
        name="receipt_fields_yolov8n_val",
        exist_ok=True,
        plots=True,
    )

    run_dir = RUNS_DIR / "receipt_fields_yolov8n"
    best = run_dir / "weights" / "best.pt"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy2(best, WEIGHTS_DIR / "receipt_fields_yolov8n.pt")

    report = {
        "model": "yolov8n",
        "training": "fine-tuned",
        "dataset": "synthetic receipts rendered from evals/golden/v0.jsonl",
        "dataset_manifest": manifest,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "device": args.device,
        "split": "document-disjoint val",
        "overall": {
            "map50": round(_box_metric(metrics, "map50"), 4),
            "map50_95": round(_box_metric(metrics, "map"), 4),
            "precision": round(_box_metric(metrics, "mp"), 4),
            "recall": round(_box_metric(metrics, "mr"), 4),
        },
        "per_class": _per_class(metrics),
        "weights": str((WEIGHTS_DIR / "receipt_fields_yolov8n.pt").relative_to(ROOT)),
    }
    (RESULTS_DIR / "yolo_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
