"""Online drift monitor: buffers extracted documents and scores 50-doc windows."""
from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

from .config import REPO_ROOT
from .drift import (
    DriftThresholds,
    detect,
    embed_texts,
    extract_features,
)
from .metrics import observe_drift

BASELINE_PATH = Path(
    os.getenv("DRIFT_BASELINE_PATH", REPO_ROOT / "ai-service" / "config" / "drift_baseline.jsonl")
)


class DriftMonitor:
    def __init__(self, *, baseline_path: Path | None = None, window: int = 50) -> None:
        self.baseline_path = Path(baseline_path or BASELINE_PATH)
        self.window = window
        self._lock = threading.Lock()
        self._buffer: list[dict] = []
        self._baseline_features = None
        self._baseline_embeddings: list[list[float]] | None = None
        self._latest: dict | None = None
        self._error: str | None = None

    def _load_baseline(self) -> None:
        if self._baseline_features is not None:
            return
        rows = []
        if self.baseline_path.exists():
            for line in self.baseline_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows.append({"text": row.get("text") or "", "fields": row.get("fields") or {}})
        rows = rows[: self.window] if rows else []
        self._baseline_features = extract_features(rows)
        self._baseline_embeddings = embed_texts(self._baseline_features.texts)

    def observe(self, samples: list[dict]) -> dict | None:
        with self._lock:
            self._buffer.extend(samples)
            self._buffer = self._buffer[-self.window :]
            if len(self._buffer) < self.window:
                return self._latest
            return self._evaluate_locked()

    def _evaluate_locked(self) -> dict:
        try:
            self._load_baseline()
            current_features = extract_features(self._buffer)
            current_embeddings = embed_texts(current_features.texts)
            report = detect(
                self._baseline_features,
                current_features,
                baseline_embeddings=self._baseline_embeddings,
                current_embeddings=current_embeddings,
                thresholds=DriftThresholds.load(),
            ).to_dict()
            report["window_docs"] = len(self._buffer)
            report["checked_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            observe_drift(report)
            self._latest = report
            self._error = None
        except Exception as exc:  # noqa: BLE001 - monitoring must not break extraction
            self._error = f"{type(exc).__name__}: {exc}"
            self._latest = {"error": self._error}
        return self._latest or {"error": self._error or "drift unavailable"}

    def status(self) -> dict:
        return {
            "baseline_docs": len(getattr(self._baseline_features, "texts", []) or []),
            "buffered_docs": len(self._buffer),
            "window_docs": self.window,
            "latest": self._latest,
            "error": self._error,
        }


MONITOR = DriftMonitor()
