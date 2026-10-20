"""Shared preprocessing, label mapping and artifact paths.

All runtimes must use the same tokenizer configuration, pooling rule and label
prototypes. Keeping those here makes parity tests assert one implementation
rather than three subtly different ones.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "BAAI/bge-small-en-v1.5"
MAX_LENGTH = 64
ARTIFACTS_DIR = Path(
    os.environ.get("LEDGERLOOP_RUNTIME_ARTIFACTS", REPO_ROOT / "artifacts" / "runtime")
)
TOKEN_FIELDS = ("input_ids", "attention_mask", "token_type_ids")
FP32_ONNX = "bge-small-en-v1.5.fp32.onnx"
INT8_ONNX = "bge-small-en-v1.5.int8.onnx"
FIXED_ONNX = "bge-small-en-v1.5.fixed.onnx"
TFLITE = "bge-small-en-v1.5.fp32.tflite"
PROTOTYPES = "prototypes.npy"
METADATA = "metadata.json"

_FALLBACK_LABELS = [
    "TRAVEL",
    "MEALS",
    "OFFICE_SUPPLIES",
    "SOFTWARE",
    "HARDWARE",
    "MARKETING",
    "PROFESSIONAL_SERVICES",
    "UTILITIES",
    "RENT",
    "INSURANCE",
    "TRAINING",
    "ENTERTAINMENT",
    "OTHER",
]


def labels() -> list[str]:
    """Return the GL label order shared by every runtime."""
    try:
        from gpu.coding_data import CATEGORIES

        return list(CATEGORIES)
    except Exception:
        return list(_FALLBACK_LABELS)


def prototype_texts() -> list[str]:
    """Label prototypes used for the cross-runtime nearest-neighbour mapping."""
    try:
        from gpu.coding_data import CATEGORIES, DESCRIPTIONS
    except Exception:
        return [f"expense category: {name.replace('_', ' ').lower()}" for name in labels()]
    texts = []
    for category in CATEGORIES:
        words = category.replace("_", " ").lower()
        examples = ", ".join(DESCRIPTIONS.get(category, [])[:6])
        texts.append(f"{words}: {examples}" if examples else words)
    return texts


@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID)


def preprocess(texts: list[str], tokenizer=None) -> dict[str, np.ndarray]:
    """Tokenize with the exact configuration used by every runtime adapter."""
    tokenizer = tokenizer or get_tokenizer()
    encoded = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    arrays = {
        "input_ids": np.asarray(encoded["input_ids"], dtype=np.int64),
        "attention_mask": np.asarray(encoded["attention_mask"], dtype=np.int64),
    }
    token_types = encoded.get("token_type_ids")
    if token_types is None:
        token_types = np.zeros_like(arrays["input_ids"])
    arrays["token_type_ids"] = np.asarray(token_types, dtype=np.int64)
    return {name: arrays[name] for name in TOKEN_FIELDS}


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def predict_labels(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
    label_names: list[str] | None = None,
) -> np.ndarray:
    """Cosine nearest-prototype label for each embedding."""
    names = label_names or labels()
    scores = np.einsum(
        "ij,kj->ik",
        normalize(embeddings).astype(np.float64),
        normalize(prototypes).astype(np.float64),
    )
    indices = scores.argmax(axis=1)
    return np.asarray([names[int(index)] for index in indices], dtype=object)


def artifact(name: str, artifacts_dir: Path | str | None = None) -> Path:
    return Path(artifacts_dir or ARTIFACTS_DIR) / name


def load_metadata(artifacts_dir: Path | str | None = None) -> dict:
    path = artifact(METADATA, artifacts_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_prototypes(artifacts_dir: Path | str | None = None) -> np.ndarray:
    path = artifact(PROTOTYPES, artifacts_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run `make runtime-export` before parity tests"
        )
    return np.load(path)


def artifacts_ready(artifacts_dir: Path | str | None = None) -> bool:
    root = Path(artifacts_dir or ARTIFACTS_DIR)
    return all((root / name).exists() for name in (FP32_ONNX, INT8_ONNX, TFLITE, PROTOTYPES, METADATA))
