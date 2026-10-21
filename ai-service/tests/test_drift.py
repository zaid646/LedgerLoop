from __future__ import annotations

import math

from app.drift import (
    DriftThresholds,
    centroid_distance,
    detect,
    extract_features,
    ks_2samp,
    parse_receipt_date,
    psi_categorical,
    psi_numeric,
)


def test_psi_is_zero_for_identical_samples_and_positive_under_shift():
    baseline = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert psi_numeric(baseline, baseline) < 1e-9
    shifted = [value * 3 for value in baseline]
    assert psi_numeric(baseline, shifted) > 0.2


def test_categorical_psi_flags_missing_category():
    baseline = ["A", "A", "B", "B", "C", "C"]
    assert psi_categorical(baseline, list(baseline)) < 1e-9
    assert psi_categorical(baseline, ["C"] * 6) > 0.2


def test_ks_detects_location_shift_and_returns_unit_p_value_for_identical():
    baseline = [float(value) for value in range(50)]
    statistic, p_value = ks_2samp(baseline, baseline)
    assert statistic == 0.0
    assert p_value == 1.0
    shifted = [value + 100 for value in baseline]
    statistic, p_value = ks_2samp(baseline, shifted)
    assert statistic == 1.0
    assert p_value < 0.01


def test_receipt_date_parser_handles_iso_and_dmy():
    assert parse_receipt_date("2026-10-01") == parse_receipt_date("01/10/2026")
    assert parse_receipt_date("01/10/26").year == 2026
    assert parse_receipt_date("not a date") is None
    assert parse_receipt_date("31/02/2026") is None


def test_features_bucket_quarters_and_categories():
    samples = [
        {"text": "a", "fields": {"total_amount": 10, "transaction_date": "01/02/2026", "merchant": "HARDWARE"}},
        {"text": "b", "fields": {"total_amount": 20, "transaction_date": "15/11/2026", "merchant": "UNKNOWN SHOP"}},
    ]
    features = extract_features(samples)
    assert features.quarters == [1, 4]
    assert features.merchant_categories == ["HARDWARE", "OTHER"]
    assert features.amounts == [10.0, 20.0]


def test_detect_fires_amount_psi_when_distribution_scales():
    baseline = [
        {"text": "x", "fields": {"total_amount": float(value), "transaction_date": "01/01/2026", "merchant": "HARDWARE"}}
        for value in range(10, 110)
    ]
    current = [
        {"text": "x", "fields": {"total_amount": float(value) * 3, "transaction_date": "01/01/2026", "merchant": "HARDWARE"}}
        for value in range(10, 110)
    ]
    report = detect(
        extract_features(baseline),
        extract_features(current),
        thresholds=DriftThresholds(psi_amount=0.2, embedding_centroid_distance=0.2),
    )
    assert report.fired is True
    assert report.detectors["psi_amount"]["fired"] is True


def test_centroid_distance_separates_embedding_clusters():
    baseline = [[1.0, 0.0], [1.0, 0.1], [1.0, -0.1]]
    same = [[1.0, 0.0], [1.0, 0.2]]
    other = [[0.0, 1.0], [0.1, 1.0]]
    assert centroid_distance(baseline, same) < 0.01
    assert centroid_distance(baseline, other) > 0.5


def test_report_serialises_measured_values_and_thresholds():
    samples = [{"text": "x", "fields": {"total_amount": 10, "transaction_date": "01/01/2026", "merchant": "HARDWARE"}}]
    report = detect(extract_features(samples), extract_features(samples))
    payload = report.to_dict()
    assert payload["fired"] is False
    assert math.isclose(payload["metrics"]["psi_amount"], 0.0, abs_tol=1e-9)
    assert "embedding_centroid_distance" in payload["detectors"]
    assert payload["thresholds"]["ks_alpha"] > 0


def test_serving_baseline_is_prediction_aligned():
    from app.drift_monitor import BASELINE_PATH

    assert BASELINE_PATH.name == "drift_baseline.jsonl"
    assert BASELINE_PATH.exists()
    lines = [line for line in BASELINE_PATH.read_text().splitlines() if line.strip()]
    assert len(lines) == 75
    row = __import__("json").loads(lines[0])
    assert {"id", "text", "fields"} <= set(row)
