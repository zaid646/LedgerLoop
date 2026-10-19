"""Embedding provider for hybrid retrieval.

Prefers **fastembed** (ONNX, no torch) for real semantic vectors; falls back to a
deterministic feature-hashing embedding so retrieval still works offline and in
CI without downloading a model or adding a heavy dependency.
"""
from __future__ import annotations

import hashlib
import math
from functools import lru_cache

EMBED_DIM = 384  # bge-small-en-v1.5 dimensionality


class Embedder:
    """Text -> unit vector. ``backend`` is ``fastembed`` or ``hashing``."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", backend: str = "auto") -> None:
        self.model_name = model_name
        self.dim = EMBED_DIM
        self.backend = "hashing"
        self._model = None
        if backend == "hashing":
            return
        try:  # pragma: no cover - depends on optional extra
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=model_name)
            probe = next(iter(self._model.embed(["probe"])))
            self.dim = len(probe)
            self.backend = "fastembed"
        except Exception:  # noqa: BLE001 - fall back silently
            self._model = None
            if backend == "fastembed":
                raise

    def embed(self, texts: list[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        if self._model is not None:  # pragma: no cover - requires fastembed
            return [[float(x) for x in v] for v in self._model.embed(texts)]
        return [self._hash(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _hash(self, text: str) -> list[float]:
        """Feature hashing (bag of tokens), L2-normalised. Deterministic."""
        vec = [0.0] * self.dim
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


@lru_cache
def get_embedder() -> Embedder:
    from .config import get_settings

    settings = get_settings()
    return Embedder(settings.embed_model, backend=settings.embed_backend)
