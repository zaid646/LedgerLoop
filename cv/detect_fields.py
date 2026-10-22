"""YOLOv8 receipt-field detector.

Weights are produced by ``python -m cv.train_yolo``.  The model is a
**fine-tuned** YOLOv8n (three classes: merchant, date, total), not zero-shot:
COCO has no receipt-field classes, so a stock checkpoint cannot detect them.

Ultralytics is imported lazily so preprocessing tests run without Torch.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "receipt_fields_yolov8n.pt"
CLASS_NAMES = ("merchant", "date", "total")


@dataclass
class Detection:
    field: str
    confidence: float
    box: tuple[int, int, int, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "confidence": round(float(self.confidence), 4),
            "box": [int(v) for v in self.box],
        }


def _clip_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = (round(v) for v in box)
    left = max(0, min(width - 1, left))
    top = max(0, min(height - 1, top))
    right = max(left + 1, min(width - 1, right))
    bottom = max(top + 1, min(height - 1, bottom))
    return left, top, right, bottom


class FieldDetector:
    """Thin wrapper around an Ultralytics YOLO checkpoint."""

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device: str = "cpu"):
        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {weights_path}. Run "
                "`python -m cv.train_yolo` (CPU, ~minutes) or pass --weights."
            )
        from ultralytics import YOLO

        self.weights = weights_path
        self.device = device
        self.model = YOLO(str(weights_path))
        names = self.model.names
        self.names = names if isinstance(names, dict) else dict(enumerate(names))

    def detect(self, image: np.ndarray, conf: float = 0.25) -> list[Detection]:
        height, width = image.shape[:2]
        results = self.model.predict(
            source=image,
            conf=conf,
            device=self.device,
            verbose=False,
            imgsz=416,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls.item())
                field = str(self.names.get(class_id, class_id))
                if field not in CLASS_NAMES:
                    continue
                coords = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        field=field,
                        confidence=float(box.conf.item()),
                        box=_clip_box(coords, width, height),
                    )
                )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def best_fields(self, image: np.ndarray, conf: float = 0.25) -> dict[str, Detection]:
        best: dict[str, Detection] = {}
        for detection in self.detect(image, conf=conf):
            best.setdefault(detection.field, detection)
        return best

    def crop(self, image: np.ndarray, detection: Detection, pad: int = 4) -> np.ndarray:
        height, width = image.shape[:2]
        left, top, right, bottom = detection.box
        left, top = max(0, left - pad), max(0, top - pad)
        right, bottom = min(width, right + pad), min(height, bottom + pad)
        return image[top:bottom, left:right].copy()


def detect_fields(
    image: np.ndarray,
    *,
    weights: str | Path = DEFAULT_WEIGHTS,
    conf: float = 0.25,
    device: str = "cpu",
) -> dict[str, Any]:
    detector = FieldDetector(weights=weights, device=device)
    fields = detector.best_fields(image, conf=conf)
    return {
        "detections": [d.as_dict() for d in fields.values()],
        "fields": {name: d.as_dict() for name, d in fields.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect receipt fields with YOLOv8n")
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=None, help="optional annotated image")
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    result = detect_fields(image, weights=args.weights, conf=args.conf)
    print(json.dumps(result, indent=2))

    if args.out:
        annotated = image.copy()
        for name, detection in result["fields"].items():
            left, top, right, bottom = detection["box"]
            cv2.rectangle(annotated, (left, top), (right, bottom), (0, 160, 0), 2)
            cv2.putText(
                annotated,
                name,
                (left, max(14, top - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 120, 0),
                2,
                cv2.LINE_AA,
            )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out), annotated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
