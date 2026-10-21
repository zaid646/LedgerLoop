"""ARS-style per-field + per-document confidence scoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from ..schemas import ReceiptExtraction
from .calibration import IsotonicCalibrator, load_calibrators
from .config import ConfidenceConfig
from .signals import (
    FieldSignals,
    build_field_signals,
    infer_policy_route,
)


@dataclass(frozen=True)
class ConfidenceResult:
    document_score: float
    raw_document_score: float
    field_scores: dict[str, float]
    raw_field_scores: dict[str, float]
    field_signals: dict[str, FieldSignals]
    calibration_version: str
    llm_logprobs_available: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "document_score": round(self.document_score, 6),
            "raw_document_score": round(self.raw_document_score, 6),
            "field_scores": {k: round(v, 6) for k, v in self.field_scores.items()},
            "calibration_version": self.calibration_version,
            "llm_logprobs_available": self.llm_logprobs_available,
            "notes": self.notes,
        }


class ConfidenceScorer(Protocol):
    def score(
        self,
        extraction: ReceiptExtraction,
        text: str,
        *,
        attempts: int = 1,
        repaired: bool = False,
        schema_ok: bool = True,
        policy_route: str | None = None,
        policy_score: float | None = None,
        source: str = "receipt",
        logprobs: dict[str, float] | None = None,
    ) -> ConfidenceResult: ...


class ARSConfidenceScorer:
    """Weighted signal blend followed by per-field and per-document calibration."""

    def __init__(
        self,
        config: ConfidenceConfig | None = None,
        calibrators: dict[str, IsotonicCalibrator] | None = None,
        *,
        calibration_version: str = "uncalibrated",
    ) -> None:
        self.config = config or ConfidenceConfig.load()
        self.calibrators = calibrators if calibrators is not None else {}
        self.calibration_version = calibration_version

    @classmethod
    def from_default(cls) -> ARSConfidenceScorer:
        return _default_scorer()

    def _raw_field_score(self, signals: FieldSignals) -> float:
        if not signals.present:
            return 0.0
        weights = self.config.signal_weights
        terms: list[tuple[float, float]] = [
            (weights.llm_confidence, signals.llm_confidence),
            (weights.retrieval_agreement, signals.retrieval_agreement),
            (weights.schema_ok, signals.schema_ok),
            (weights.repair, signals.repair),
            (weights.policy, signals.policy),
        ]
        if signals.logprob_confidence is not None:
            terms.append((weights.logprob, signals.logprob_confidence))
        total_weight = sum(weight for weight, _ in terms if weight > 0)
        if total_weight <= 0:
            return 0.0
        return max(
            0.0,
            min(1.0, sum(weight * value for weight, value in terms if weight > 0) / total_weight),
        )

    def score(
        self,
        extraction: ReceiptExtraction,
        text: str,
        *,
        attempts: int = 1,
        repaired: bool = False,
        schema_ok: bool = True,
        policy_route: str | None = None,
        policy_score: float | None = None,
        source: str = "receipt",
        logprobs: dict[str, float] | None = None,
    ) -> ConfidenceResult:
        if policy_route is None:
            policy_route, measured_score = infer_policy_route(extraction, source=source)
            if policy_score is None:
                policy_score = measured_score
        if policy_score is None:
            from .signals import policy_signal

            policy_score = policy_signal(policy_route)

        notes: list[str] = []
        if logprobs is None:
            notes.append("llm_logprobs_unavailable: using model self-reported confidence")
        if repaired:
            notes.append("self_heal_repaired")

        signals = build_field_signals(
            extraction,
            text,
            attempts=attempts,
            schema_ok=schema_ok,
            policy_route=policy_route,
            policy_score=policy_score,
            config=self.config,
            logprobs=logprobs,
        )

        raw_fields = {name: self._raw_field_score(sig) for name, sig in signals.items()}
        calibrated_fields: dict[str, float] = {}
        for name, raw in raw_fields.items():
            calibrator = self.calibrators.get(name)
            calibrated_fields[name] = (
                max(0.0, min(1.0, calibrator.predict(raw))) if calibrator else raw
            )

        total_weight = sum(field.weight for field in self.config.fields.values()) or 1.0
        raw_document = (
            sum(self.config.fields[name].weight * raw for name, raw in raw_fields.items())
            / total_weight
        )
        document_calibrator = self.calibrators.get("document")
        document = (
            max(0.0, min(1.0, document_calibrator.predict(raw_document)))
            if document_calibrator
            else raw_document
        )

        missing_required = [
            name for name, field in self.config.fields.items() if field.required and not signals[name].present
        ]
        if missing_required:
            notes.append("missing_required:" + ",".join(missing_required))

        return ConfidenceResult(
            document_score=document,
            raw_document_score=raw_document,
            field_scores=calibrated_fields,
            raw_field_scores=raw_fields,
            field_signals=signals,
            calibration_version=self.calibration_version,
            llm_logprobs_available=bool(logprobs),
            notes=notes,
        )


@lru_cache(maxsize=1)
def _default_scorer() -> ARSConfidenceScorer:
    config = ConfidenceConfig.load()
    path = Path(config.calibration_path)
    calibrators = load_calibrators(path)
    version = "identity"
    if path.exists():
        import json

        version = str(json.loads(path.read_text()).get("version", "unknown"))
    return ARSConfidenceScorer(config, calibrators, calibration_version=version)


def get_scorer() -> ARSConfidenceScorer:
    return _default_scorer()
