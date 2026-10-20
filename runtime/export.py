"""Export BAAI/bge-small-en-v1.5 to ONNX (fp32 + int8) and TFLite.

The QLoRA Qwen2.5-0.5B classifier is intentionally not the export target: no
adapter checkpoint is checked into the repository (``gpu/finetune_qlora.py``
writes to ``/workspace/expenseiq-qlora``), and a 0.5B decoder graph is too heavy
for a reliable ONNX -> TF -> TFLite conversion. The 33M-parameter BGE embedder
is the smallest model the service actually consumes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .common import (
    ARTIFACTS_DIR,
    FIXED_ONNX,
    FP32_ONNX,
    INT8_ONNX,
    MAX_LENGTH,
    METADATA,
    MODEL_ID,
    PROTOTYPES,
    TFLITE,
    labels,
    normalize,
    prototype_texts,
)


class _PooledEmbedding:
    def __new__(cls, model):
        import torch

        class _Module(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, input_ids, attention_mask, token_type_ids):
                hidden = self.inner(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                ).last_hidden_state
                return torch.nn.functional.normalize(hidden[:, 0], p=2, dim=-1)

        return _Module(model)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict:
    return {
        "bytes": path.stat().st_size,
        "mb": round(path.stat().st_size / 1_000_000, 2),
        "sha256": _sha256(path),
    }


def _load():
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = _PooledEmbedding(AutoModel.from_pretrained(MODEL_ID).eval()).eval()
    return torch, tokenizer, model


def _export_onnx(model, sample: dict, path: Path, dynamic: bool) -> None:
    import torch

    names = ["input_ids", "attention_mask", "token_type_ids"]
    args = tuple(sample[name] for name in names)
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in names}
        dynamic_axes["embedding"] = {0: "batch"}
    torch.onnx.export(
        model,
        args,
        str(path),
        input_names=names,
        output_names=["embedding"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def _export_tflite(fixed_onnx: Path, output_dir: Path) -> Path:
    if importlib.util.find_spec("onnx2tf") is None:
        raise RuntimeError("onnx2tf is not installed; install runtime/requirements.txt")
    if importlib.util.find_spec("tensorflow") is None:
        raise RuntimeError("tensorflow is not installed; install runtime/requirements.txt")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    command = [
        sys.executable,
        "-m",
        "onnx2tf",
        "-i",
        str(fixed_onnx),
        "-o",
        str(output_dir),
        "-v",
        "error",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-25:])
        raise RuntimeError(f"onnx2tf failed:\n{tail}")
    candidates = sorted(output_dir.glob("*_float32.tflite"))
    if not candidates:
        raise RuntimeError(f"onnx2tf produced no float32 TFLite model in {output_dir}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default=str(ARTIFACTS_DIR))
    parser.add_argument("--skip-tflite", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    torch, tokenizer, model = _load()
    torch.set_num_threads(args.threads)
    sample_text = ["team lunch - Starbucks"]
    encoded = tokenizer(
        sample_text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    sample = {name: encoded[name] for name in ("input_ids", "attention_mask", "token_type_ids")}

    fp32_path = artifacts / FP32_ONNX
    fixed_path = artifacts / FIXED_ONNX
    int8_path = artifacts / INT8_ONNX
    _export_onnx(model, sample, fp32_path, dynamic=True)
    _export_onnx(model, sample, fixed_path, dynamic=False)

    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    tflite_path = None
    tflite_error = None
    if not args.skip_tflite:
        try:
            raw = _export_tflite(fixed_path, artifacts / "tflite")
            tflite_path = artifacts / TFLITE
            shutil.copyfile(raw, tflite_path)
        except Exception as exc:
            tflite_error = str(exc)

    prototype_inputs = tokenizer(
        prototype_texts(),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        hidden = model(
            input_ids=prototype_inputs["input_ids"],
            attention_mask=prototype_inputs["attention_mask"],
            token_type_ids=prototype_inputs["token_type_ids"],
        )
    np.save(artifacts / PROTOTYPES, normalize(hidden.numpy()).astype(np.float32))

    files = {}
    for path in (fp32_path, fixed_path, int8_path, tflite_path):
        if path is not None and path.exists():
            files[path.name] = _record(path)

    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_id": MODEL_ID,
        "model_kind": "embedding",
        "task": "GL-category nearest-prototype mapping",
        "pooling": "CLS + L2 normalize",
        "max_length": MAX_LENGTH,
        "preprocessing": {
            "padding": "max_length",
            "truncation": True,
            "max_length": MAX_LENGTH,
            "input_fields": ["input_ids", "attention_mask", "token_type_ids"],
        },
        "labels": labels(),
        "prototype_texts": prototype_texts(),
        "files": files,
        "classifier_export": {
            "model": "Qwen/Qwen2.5-0.5B-Instruct + QLoRA",
            "attempted": False,
            "reason": (
                "No adapter checkpoint is checked in (gpu/finetune_qlora.py writes to "
                "/workspace/expenseiq-qlora), and a 0.5B decoder graph is too heavy for "
                "a reliable ONNX -> TF -> TFLite path. The service already ships "
                "bge-small-en-v1.5 for embeddings, so that is the smallest real model "
                "artifact available for cross-runtime parity."
            ),
        },
        "tflite_error": tflite_error,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    (artifacts / METADATA).write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({**metadata, "prototype_texts": f"{len(metadata['prototype_texts'])} texts"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
