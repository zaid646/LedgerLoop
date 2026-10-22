"""Configuration for ARS-style confidence scoring (YAML, validated by dataclasses)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..config import REPO_ROOT

DEFAULT_CONFIG_PATH = REPO_ROOT / "ai-service" / "config" / "confidence.yaml"


@dataclass(frozen=True)
class FieldConfig:
    weight: float
    required: bool
    auto_threshold: float


@dataclass(frozen=True)
class DocumentConfig:
    auto_threshold: float
    reject_threshold: float


@dataclass(frozen=True)
class SignalWeights:
    llm_confidence: float = 0.42
    retrieval_agreement: float = 0.33
    schema_ok: float = 0.10
    repair: float = 0.10
    policy: float = 0.05
    logprob: float = 0.0


@dataclass(frozen=True)
class CalibrationConfig:
    file: str = "calibration.json"
    bins: int = 10


@dataclass(frozen=True)
class RetrievalConfig:
    merchant_token_overlap: float = 0.60


@dataclass(frozen=True)
class ConfidenceConfig:
    version: str
    fields: dict[str, FieldConfig]
    document: DocumentConfig
    signal_weights: SignalWeights
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    config_path: Path | None = None

    @property
    def calibration_path(self) -> Path:
        base = self.config_path.parent if self.config_path else DEFAULT_CONFIG_PATH.parent
        return (base / self.calibration.file).resolve()

    @classmethod
    def load(cls, path: str | Path | None = None) -> ConfidenceConfig:
        config_path = Path(path).resolve() if path else DEFAULT_CONFIG_PATH
        raw = yaml.safe_load(config_path.read_text()) or {}
        fields = {
            name: FieldConfig(
                weight=float(spec["weight"]),
                required=bool(spec.get("required", False)),
                auto_threshold=float(spec["auto_threshold"]),
            )
            for name, spec in (raw.get("fields") or {}).items()
        }
        if not fields:
            raise ValueError(f"confidence config {config_path} defines no fields")
        doc = raw.get("document") or {}
        weights = raw.get("signal_weights") or {}
        calibration = raw.get("calibration") or {}
        retrieval = raw.get("retrieval") or {}
        return cls(
            version=str(raw.get("version", "unversioned")),
            fields=fields,
            document=DocumentConfig(
                auto_threshold=float(doc["auto_threshold"]),
                reject_threshold=float(doc["reject_threshold"]),
            ),
            signal_weights=SignalWeights(**{k: float(v) for k, v in weights.items()}),
            calibration=CalibrationConfig(
                file=str(calibration.get("file", "calibration.json")),
                bins=int(calibration.get("bins", 10)),
            ),
            retrieval=RetrievalConfig(
                merchant_token_overlap=float(retrieval.get("merchant_token_overlap", 0.60))
            ),
            config_path=config_path,
        )
