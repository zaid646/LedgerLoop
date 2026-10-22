"""Compare OCR-only, LLM-only and OCR+LLM field extraction on 75 receipts.

Inputs
------
* ``evals/golden/v0.jsonl`` -- frozen labelled receipt texts (50 SROIE + 25 CORD)
* ``evals/results/llm-qwen7b-v0.json`` -- cached Qwen2.5-7B predictions on the
  same 75 documents (the repo's existing LLM extractor)
* ``cv/data/ocr/*.png`` -- deterministic renders of the golden texts, because
  the repository does not ship receipt photographs

The OCR path never sees the golden text; it only sees pixels.  The LLM path is
a replay of committed predictions, not a new LLM call.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cv.detect_fields import DEFAULT_WEIGHTS, FieldDetector
from cv.ocr import extract_fields, fuse_ocr_llm, ocr_regions
from cv.synthetic import DATA_DIR, render_ocr_set
from evals.metrics import field_scores

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "evals" / "golden" / "v0.jsonl"
LLM_PREDICTIONS = ROOT / "evals" / "results" / "llm-qwen7b-v0.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIELDS = ("merchant", "total_amount", "transaction_date")


def _display_path(path: Path) -> str:
    """Keep committed artifacts free of absolute home-directory paths."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


def load_golden() -> list[dict]:
    with GOLDEN.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_llm_predictions(path: Path = LLM_PREDICTIONS) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["id"]: row.get("pred", {}) for row in payload.get("predictions", [])}


def _score(records: list[dict], predictions: dict[str, dict]) -> dict[str, Any]:
    passed = {name: 0 for name in FIELDS}
    support = {name: 0 for name in FIELDS}
    exact = 0
    for record in records:
        pred = predictions.get(record["id"], {})
        gold = record.get("fields", {})
        scores = field_scores(pred, gold)
        for name in FIELDS:
            if gold.get(name) not in (None, "", []):
                support[name] += 1
                passed[name] += int(scores[name])
        if all(
            scores[name]
            for name in FIELDS
            if gold.get(name) not in (None, "", [])
        ):
            exact += 1
    n = len(records)
    return {
        "n": n,
        "field_accuracy": {
            name: round(passed[name] / support[name], 4) if support[name] else None
            for name in FIELDS
        },
        "field_support": support,
        "exact_doc_match": round(exact / n, 4) if n else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR vs LLM receipt field comparison")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "ocr_vs_llm.json")
    args = parser.parse_args()

    records = load_golden()
    if args.limit:
        records = records[: args.limit]
    render_ocr_set(DATA_DIR / "ocr")
    llm_cache = load_llm_predictions()

    detector = FieldDetector(args.weights)
    ocr_only: dict[str, dict] = {}
    ocr_regions_only: dict[str, dict] = {}
    fused: dict[str, dict] = {}
    llm_only: dict[str, dict] = {}
    details: list[dict] = []

    for index, record in enumerate(records, start=1):
        image_path = DATA_DIR / "ocr" / f"{record['id']}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)

        full = extract_fields(image)
        region_fields, region_debug = ocr_regions(image, detector, conf=args.conf)
        llm = llm_cache.get(record["id"], {})
        fusion = fuse_ocr_llm(full.fields, region_fields, llm)

        ocr_only[record["id"]] = full.fields
        ocr_regions_only[record["id"]] = region_fields
        llm_only[record["id"]] = llm
        fused[record["id"]] = fusion
        details.append(
            {
                "id": record["id"],
                "ocr_full": full.fields,
                "ocr_regions": region_fields,
                "llm": {
                    "merchant": llm.get("merchant"),
                    "transaction_date": llm.get("transaction_date"),
                    "total_amount": llm.get("total_amount"),
                },
                "ocr_llm": fusion,
                "ocr_confidence": round(full.confidence, 4),
                "detections": region_debug["detections"],
            }
        )
        if index % 10 == 0 or index == len(records):
            print(f"processed {index}/{len(records)}", flush=True)

    metrics = {
        "benchmark": "synthetic-render-ocr-vs-cached-llm",
        "caveat": (
            "Images are deterministic renders of the golden receipt TEXT, not "
            "photographs. OCR numbers reflect clean synthetic pages and are an "
            "upper bound vs real phone photos."
        ),
        "n_documents": len(records),
        "source_golden": str(GOLDEN.relative_to(ROOT)),
        "llm_replay": str(LLM_PREDICTIONS.relative_to(ROOT)),
        "llm_model": "Qwen2.5-7B-Instruct (cached)",
        "yolo_weights": _display_path(args.weights),
        "yolo_conf": args.conf,
        "environment": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "methods": {
            "ocr_only": "Tesseract on preprocessed full page + regex parser",
            "ocr_yolo": "YOLOv8n field crops + Tesseract per crop",
            "llm_only": "Cached Qwen2.5-7B extraction",
            "ocr_llm": (
                "Fusion: merchant from LLM; total/date from YOLO+OCR when present, "
                "else full-page OCR, else LLM"
            ),
        },
        "ocr_only": _score(records, ocr_only),
        "ocr_yolo": _score(records, ocr_regions_only),
        "llm_only": _score(records, llm_only),
        "ocr_llm": _score(records, fused),
        "details": details,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("ocr_only", "ocr_yolo", "llm_only", "ocr_llm")},
                     indent=2))
    print(f"results -> {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
