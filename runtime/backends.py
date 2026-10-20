"""Thin CPU adapters with identical preprocessing and output semantics."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import (
    ARTIFACTS_DIR,
    FP32_ONNX,
    INT8_ONNX,
    MODEL_ID,
    TFLITE,
    TOKEN_FIELDS,
    preprocess,
)


class TorchEmbedder:
    name = "PyTorch"
    precision = "fp32"

    def __init__(self, model_id: str = MODEL_ID, threads: int = 4) -> None:
        import torch
        from transformers import AutoModel

        torch.set_num_threads(threads)
        self._torch = torch
        base = AutoModel.from_pretrained(model_id).eval()

        class _Pooled(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids, attention_mask, token_type_ids):
                hidden = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                ).last_hidden_state
                return torch.nn.functional.normalize(hidden[:, 0], p=2, dim=-1)

        self.model = _Pooled(base).eval()

    def embed_preprocessed(self, encoded: dict[str, np.ndarray]) -> np.ndarray:
        tensors = {
            name: self._torch.from_numpy(encoded[name])
            for name in TOKEN_FIELDS
        }
        with self._torch.no_grad():
            return self.model(**tensors).cpu().numpy().astype(np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.embed_preprocessed(preprocess(texts))

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def model_size_mb(self) -> float:
        params = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return round(params / 1_000_000, 2)


class OnnxEmbedder:
    name = "ONNX Runtime"
    precision = "fp32"

    def __init__(
        self,
        artifacts_dir: Path | str | None = None,
        int8: bool = False,
        threads: int = 4,
    ) -> None:
        import onnxruntime as ort

        self.path = Path(artifacts_dir or ARTIFACTS_DIR) / (INT8_ONNX if int8 else FP32_ONNX)
        self.precision = "int8" if int8 else "fp32"
        self.name = f"ONNX Runtime {self.precision}"
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name

    def embed_preprocessed(self, encoded: dict[str, np.ndarray]) -> np.ndarray:
        rows = []
        batch = encoded[TOKEN_FIELDS[0]].shape[0]
        dynamic_batch = any(
            dim is None or str(dim).lower() in {"batch", "?"}
            for dim in self.session.get_inputs()[0].shape[:1]
        )
        if dynamic_batch and batch > 1:
            return self.session.run(
                [self.output_name], {name: encoded[name] for name in TOKEN_FIELDS}
            )[0].astype(np.float32)
        for index in range(batch):
            feed = {name: encoded[name][index : index + 1] for name in TOKEN_FIELDS}
            rows.append(self.session.run([self.output_name], feed)[0][0])
        return np.asarray(rows, dtype=np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.embed_preprocessed(preprocess(texts))

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def model_size_mb(self) -> float:
        return round(self.path.stat().st_size / 1_000_000, 2)


class TFLiteEmbedder:
    name = "TFLite"
    precision = "fp32"

    def __init__(
        self,
        artifacts_dir: Path | str | None = None,
        threads: int = 4,
    ) -> None:
        import tensorflow as tf

        self.path = Path(artifacts_dir or ARTIFACTS_DIR) / TFLITE
        self.interpreter = tf.lite.Interpreter(
            model_path=str(self.path),
            num_threads=threads,
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        self.input_names = [detail["name"] for detail in self.input_details]

    def embed_preprocessed(self, encoded: dict[str, np.ndarray]) -> np.ndarray:
        rows = []
        for index in range(encoded[TOKEN_FIELDS[0]].shape[0]):
            for detail in self.input_details:
                name = detail["name"]
                if name not in encoded:
                    raise KeyError(f"TFLite input {name!r} is not produced by preprocessing")
                self.interpreter.set_tensor(detail["index"], encoded[name][index : index + 1])
            self.interpreter.invoke()
            rows.append(self.interpreter.get_tensor(self.output_index)[0])
        return np.asarray(rows, dtype=np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.embed_preprocessed(preprocess(texts))

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def model_size_mb(self) -> float:
        return round(self.path.stat().st_size / 1_000_000, 2)


def make_backend(name: str, artifacts_dir: Path | str | None = None, threads: int = 4):
    key = name.lower()
    if key in {"torch", "pytorch"}:
        return TorchEmbedder(threads=threads)
    if key in {"onnx", "onnx-fp32", "onnx_fp32"}:
        return OnnxEmbedder(artifacts_dir=artifacts_dir, int8=False, threads=threads)
    if key in {"onnx-int8", "onnx_int8", "int8"}:
        return OnnxEmbedder(artifacts_dir=artifacts_dir, int8=True, threads=threads)
    if key in {"tflite", "tf"}:
        return TFLiteEmbedder(artifacts_dir=artifacts_dir, threads=threads)
    raise ValueError(f"unknown backend: {name}")
