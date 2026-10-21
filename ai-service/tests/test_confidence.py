from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.confidence.calibration import (
    IsotonicCalibrator,
    expected_calibration_error,
    fit_isotonic,
)
from app.confidence.config import ConfidenceConfig
from app.confidence.scorer import ARSConfidenceScorer
from app.confidence.signals import infer_policy_route, retrieval_agreement
from app.routing import (
    KILL_SWITCH,
    ROUTE_AUTO,
    ROUTE_REJECT,
    ROUTE_REVIEW,
    route_document,
)
from app.schemas import ReceiptExtraction

CONFIG = ConfidenceConfig.load()


def _extraction(**overrides) -> ReceiptExtraction:
    payload = {
        "merchant": "ACME STORE",
        "transaction_date": "2026-10-01",
        "total_amount": 12.5,
        "currency": "USD",
        "confidence": {"merchant": 0.95, "total_amount": 0.95, "transaction_date": 0.95},
    }
    payload.update(overrides)
    return ReceiptExtraction.model_validate(payload)


def test_isotonic_is_monotone_and_clipped():
    calibrator = fit_isotonic([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0, 1, 0, 1, 1])
    predictions = [calibrator.predict(x) for x in (-1, 0, 0.3, 0.6, 1, 2)]
    assert predictions == sorted(predictions)
    assert predictions[0] == calibrator.ys[0]
    assert predictions[-1] == calibrator.ys[-1]


def test_isotonic_round_trip():
    calibrator = IsotonicCalibrator(xs=[0.0, 0.5, 1.0], ys=[0.1, 0.4, 0.9])
    assert IsotonicCalibrator.from_dict(calibrator.to_dict()).predict(0.5) == 0.4


def test_ece_zero_when_calibrated():
    ece, bins = expected_calibration_error([0, 0, 1, 1], [0.01, 0.02, 0.98, 0.99], n_bins=10)
    assert ece < 0.02
    assert sum(b["count"] for b in bins) == 4


def test_ece_handles_empty():
    assert expected_calibration_error([], []) == (0.0, [])


def test_ece_detects_overconfidence():
    ece, _ = expected_calibration_error([0, 0, 0, 1], [1.0, 1.0, 1.0, 1.0], n_bins=5)
    assert ece == pytest.approx(0.75)


def test_retrieval_agreement_grounds_values():
    text = "ACME STORE\nTOTAL 12.50\nDATE 01/10/2026"
    assert retrieval_agreement("total_amount", 12.5, text) == 1.0
    assert retrieval_agreement("transaction_date", "2026-10-01", text) == 1.0
    assert retrieval_agreement("merchant", "ACME STORE", text) == 1.0
    assert retrieval_agreement("total_amount", 99.0, text) == 0.0
    assert retrieval_agreement("total_amount", None, text) == 0.0


def test_missing_amount_scores_lower_than_grounded_amount():
    scorer = ARSConfidenceScorer(CONFIG, {}, calibration_version="test")
    text = "ACME STORE\nTOTAL 12.50\nDATE 01/10/2026"
    good = scorer.score(_extraction(), text)
    bad = scorer.score(_extraction(total_amount=None), text)
    assert good.document_score > bad.document_score
    assert good.field_scores["total_amount"] > bad.field_scores["total_amount"]
    assert "missing_required:total_amount" in bad.notes


def test_policy_route_is_inferred_from_amount_and_category():
    route, score = infer_policy_route(_extraction(total_amount=5.0, merchant="STARBUCKS"))
    assert route in {"auto_approve", "manager_review", "human_queue"}
    assert 0.0 <= score <= 1.0


def test_custom_calibrator_is_applied():
    calibrator = IsotonicCalibrator(xs=[0.0, 1.0], ys=[0.2, 0.8])
    scorer = ARSConfidenceScorer(CONFIG, {"merchant": calibrator}, calibration_version="t")
    result = scorer.score(_extraction(), "ACME STORE TOTAL 12.50 DATE 01/10/2026")
    assert 0.2 < result.field_scores["merchant"] < 0.8
    assert result.calibration_version == "t"


def test_router_auto_approve_when_all_thresholds_met():
    scorer = ARSConfidenceScorer(CONFIG, {}, calibration_version="test")
    result = scorer.score(
        _extraction(),
        "ACME STORE TOTAL 12.50 DATE 01/10/2026",
        policy_route="auto_approve",
    )
    decision = route_document(result, config=CONFIG, force_review=False)
    assert decision.route == ROUTE_AUTO
    assert decision.needs_human is False


def test_router_reviews_low_confidence_document():
    scorer = ARSConfidenceScorer(CONFIG, {}, calibration_version="test")
    result = scorer.score(
        _extraction(total_amount=None, confidence={"merchant": 0.2}),
        "unreadable",
        policy_route="auto_approve",
    )
    decision = route_document(
        result,
        config=CONFIG,
        missing_required=[],
        force_review=False,
    )
    assert decision.route in {ROUTE_REVIEW, ROUTE_REJECT}
    assert decision.needs_human is True
    assert decision.reasons


def test_router_rejects_missing_required_field():
    scorer = ARSConfidenceScorer(CONFIG, {}, calibration_version="test")
    result = scorer.score(
        _extraction(total_amount=None),
        "ACME STORE",
        policy_route="auto_approve",
    )
    decision = route_document(
        result,
        config=CONFIG,
        missing_required=["total_amount"],
        force_review=False,
    )
    assert decision.route == ROUTE_REJECT
    assert "missing_required:total_amount" in decision.reasons


def test_router_rejects_invalid_schema():
    scorer = ARSConfidenceScorer(CONFIG, {}, calibration_version="test")
    result = scorer.score(_extraction(), "ACME", policy_route="auto_approve")
    decision = route_document(result, config=CONFIG, schema_ok=False, force_review=False)
    assert decision.route == ROUTE_REJECT
    assert "schema_invalid" in decision.reasons


def test_router_reviews_below_reject_floor():
    scorer = ARSConfidenceScorer(CONFIG, {}, calibration_version="test")
    result = scorer.score(
        _extraction(total_amount=None, confidence={}),
        "",
        policy_route="auto_approve",
    )
    decision = route_document(result, config=CONFIG, force_review=False)
    assert decision.route == ROUTE_REJECT


def test_kill_switch_forces_review_even_for_perfect_document():
    scorer = ARSConfidenceScorer(CONFIG, {}, calibration_version="test")
    result = scorer.score(
        _extraction(),
        "ACME STORE TOTAL 12.50 DATE 01/10/2026",
        policy_route="auto_approve",
    )
    status = KILL_SWITCH.set(True, actor="test")
    try:
        assert status["enabled"] is True
        decision = route_document(result, config=CONFIG)
        assert decision.route == ROUTE_REVIEW
        assert decision.forced is True
        assert "kill_switch_force_human_review" in decision.reasons
    finally:
        KILL_SWITCH.reset()


def test_kill_switch_status_has_audit_fields():
    KILL_SWITCH.set(True, actor="operator-1")
    try:
        status = KILL_SWITCH.status()
        assert status["actor"] == "operator-1"
        datetime.fromisoformat(status["changed_at"]).astimezone(UTC)
    finally:
        KILL_SWITCH.reset()
