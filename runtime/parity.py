"""Cross-runtime parity check: PyTorch vs ONNX vs TFLite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .backends import OnnxEmbedder, TFLiteEmbedder, TorchEmbedder
from .common import (
    ARTIFACTS_DIR,
    labels,
    load_prototypes,
    predict_labels,
)

TEXTS = [
    "team lunch - Starbucks",
    "flight booking - Uber",
    "printer paper - Staples",
    "monthly subscription - GitHub",
    "office chair - IKEA",
    "search ads - Google Ads",
    "client dinner - DoorDash",
    "cloud compute - Amazon Web Services",
    "consulting fees - Deloitte",
    "electricity bill - Shell",
    "online course - Coursera",
    "health insurance premium - Aetna",
    "laptop - Apple",
    "ad campaign - Meta Ads",
    "fuel purchase - Shell",
]

EXPECTED = [
    "MEALS",
    "TRAVEL",
    "OFFICE_SUPPLIES",
    "SOFTWARE",
    "HARDWARE",
    "MARKETING",
    "MEALS",
    "SOFTWARE",
    "PROFESSIONAL_SERVICES",
    "UTILITIES",
    "TRAINING",
    "INSURANCE",
    "HARDWARE",
    "MARKETING",
    "TRAVEL",
]


def _stats(reference: np.ndarray, candidate: np.ndarray) -> dict:
    cosine = np.sum(reference * candidate, axis=1)
    absolute = np.abs(reference - candidate)
    return {
        "cosine_min": round(float(cosine.min()), 6),
        "cosine_mean": round(float(cosine.mean()), 6),
        "max_abs_diff": round(float(absolute.max()), 6),
        "mean_abs_diff": round(float(absolute.mean()), 6),
    }


def compare(artifacts_dir: Path | str | None = None, threads: int = 4) -> dict:
    artifacts = Path(artifacts_dir or ARTIFACTS_DIR)
    prototypes = load_prototypes(artifacts)
    label_names = labels()

    torch_backend = TorchEmbedder(threads=threads)
    reference = torch_backend.embed(TEXTS)
    reference_labels = predict_labels(reference, prototypes, label_names)

    candidates = {
        "onnx_fp32": OnnxEmbedder(artifacts, int8=False, threads=threads),
        "onnx_int8": OnnxEmbedder(artifacts, int8=True, threads=threads),
        "tflite_fp32": TFLiteEmbedder(artifacts, threads=threads),
    }
    result = {
        "model": "BAAI/bge-small-en-v1.5",
        "texts": len(TEXTS),
        "reference": "PyTorch fp32",
        "labels": label_names,
        "expected_accuracy": round(
            float(np.mean(reference_labels == np.asarray(EXPECTED, dtype=object))), 4
        ),
        "runtimes": {},
    }
    for name, backend in candidates.items():
        embeddings = backend.embed(TEXTS)
        candidate_labels = predict_labels(embeddings, prototypes, label_names)
        result["runtimes"][name] = {
            **_stats(reference, embeddings),
            "label_agreement": round(
                float(np.mean(candidate_labels == reference_labels)), 4
            ),
            "accuracy": round(
                float(np.mean(candidate_labels == np.asarray(EXPECTED, dtype=object))), 4
            ),
            "predicted_labels": candidate_labels.tolist(),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default=str(ARTIFACTS_DIR))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    result = compare(args.artifacts, args.threads)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
