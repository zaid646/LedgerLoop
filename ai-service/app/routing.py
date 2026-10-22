"""Threshold routing: auto-approve, human review, or reject/escalate.

Fail-closed by design: invalid schemas, missing required fields and documents
below the reject threshold never auto-approve. The kill switch forces every
document into human review regardless of score.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .confidence.config import ConfidenceConfig
from .confidence.scorer import ConfidenceResult

ROUTE_AUTO = "auto_approve"
ROUTE_REVIEW = "human_review"
ROUTE_REJECT = "reject_escalate"


@dataclass(frozen=True)
class RoutingDecision:
    route: str
    reasons: list[str] = field(default_factory=list)
    document_score: float = 0.0
    field_scores: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    forced: bool = False

    @property
    def needs_human(self) -> bool:
        return self.route != ROUTE_AUTO

    def to_dict(self) -> dict:
        return {
            "route": self.route,
            "reasons": self.reasons,
            "document_score": round(self.document_score, 6),
            "field_scores": {k: round(v, 6) for k, v in self.field_scores.items()},
            "thresholds": self.thresholds,
            "forced": self.forced,
        }


class KillSwitch:
    """Process-local safety flag; disable auto-approval immediately."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._enabled = os.getenv("FORCE_HUMAN_REVIEW", "0").lower() in {"1", "true", "yes", "on"}
        self._actor = "env" if self._enabled else ""
        self._changed_at = datetime.now(UTC).isoformat(timespec="seconds") if self._enabled else ""

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set(self, enabled: bool, *, actor: str = "api") -> dict:
        with self._lock:
            self._enabled = bool(enabled)
            self._actor = actor
            self._changed_at = datetime.now(UTC).isoformat(timespec="seconds")
            return self._status_unlocked()

    def reset(self) -> None:
        self.set(False, actor="reset")

    def status(self) -> dict:
        with self._lock:
            return self._status_unlocked()

    def _status_unlocked(self) -> dict:
        return {
            "enabled": self._enabled,
            "actor": self._actor,
            "changed_at": self._changed_at,
        }


KILL_SWITCH = KillSwitch()


def route_document(
    result: ConfidenceResult,
    *,
    config: ConfidenceConfig | None = None,
    policy_route: str = "auto_approve",
    schema_ok: bool = True,
    missing_required: list[str] | None = None,
    force_review: bool | None = None,
) -> RoutingDecision:
    config = config or ConfidenceConfig.load()
    forced = KILL_SWITCH.enabled if force_review is None else bool(force_review)
    reasons: list[str] = []
    missing = missing_required or []
    thresholds = {
        "document_auto": config.document.auto_threshold,
        "document_reject": config.document.reject_threshold,
        **{f"field_auto:{name}": spec.auto_threshold for name, spec in config.fields.items()},
    }

    if forced:
        reasons.append("kill_switch_force_human_review")
        return RoutingDecision(
            route=ROUTE_REVIEW,
            reasons=reasons,
            document_score=result.document_score,
            field_scores=result.field_scores,
            thresholds=thresholds,
            forced=True,
        )

    if not schema_ok:
        reasons.append("schema_invalid")
        return RoutingDecision(
            route=ROUTE_REJECT,
            reasons=reasons,
            document_score=result.document_score,
            field_scores=result.field_scores,
            thresholds=thresholds,
        )
    if missing:
        reasons.append("missing_required:" + ",".join(sorted(missing)))
        return RoutingDecision(
            route=ROUTE_REJECT,
            reasons=reasons,
            document_score=result.document_score,
            field_scores=result.field_scores,
            thresholds=thresholds,
        )
    if result.document_score < config.document.reject_threshold:
        reasons.append("document_below_reject_threshold")
        return RoutingDecision(
            route=ROUTE_REJECT,
            reasons=reasons,
            document_score=result.document_score,
            field_scores=result.field_scores,
            thresholds=thresholds,
        )

    low_fields = [
        name
        for name, spec in config.fields.items()
        if result.field_scores.get(name, 0.0) < spec.auto_threshold
    ]
    reasons.extend(f"field_below_threshold:{name}" for name in low_fields)
    if result.document_score < config.document.auto_threshold:
        reasons.append("document_below_auto_threshold")
    if policy_route in {"manager_review", "human_queue"}:
        reasons.append(f"policy:{policy_route}")

    if low_fields or result.document_score < config.document.auto_threshold or policy_route != "auto_approve":
        return RoutingDecision(
            route=ROUTE_REVIEW,
            reasons=reasons,
            document_score=result.document_score,
            field_scores=result.field_scores,
            thresholds=thresholds,
        )

    return RoutingDecision(
        route=ROUTE_AUTO,
        reasons=["all_thresholds_met"],
        document_score=result.document_score,
        field_scores=result.field_scores,
        thresholds=thresholds,
    )
