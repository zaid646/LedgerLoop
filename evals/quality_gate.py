#!/usr/bin/env python3
"""CI gate for confidence calibration, routing quality and drift detection.

Reads the deterministic replay artifacts produced by:
  * ``evals/replay_confidence.py``  (calibration + routing + latency)
  * ``evals/drift_sim.py``          (null false alarms + scenario detection)

Usage:
    uv run python evals/quality_gate.py \
      --confidence evals/results/confidence-replay.json \
      --drift evals/results/drift-sim-ci.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MIN_AUTO_MACRO_ACCURACY = 0.90
MIN_AUTO_EXACT_MATCH = 0.80
MIN_ERRORS_CAUGHT = 0.70
MIN_AUTO_RATE = 0.25
MAX_DOCUMENT_ECE = 0.05
MAX_COMBINED_P95_MS = 5.0
MAX_NULL_FALSE_ALARM = 0.10
MIN_SHIFT_DETECTION = 0.80
MAX_TIME_TO_DETECT_WINDOWS = 3


def _check(checks: list[tuple[str, bool, str]]) -> None:
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")


def evaluate_confidence(path: Path) -> list[tuple[str, bool, str]]:
    data = json.loads(path.read_text())
    calibration = data["calibration"]
    routing = data["routing"]
    point = routing["operating_point"]
    latency = data["latency"]
    caught = (point["errors_caught_by_human"] / point["error_count"]) if point["error_count"] else 1.0
    return [
        (
            "document_ece",
            calibration["document_ece_after"] <= MAX_DOCUMENT_ECE,
            f"{calibration['document_ece_after']:.4f} <= {MAX_DOCUMENT_ECE} (was {calibration['document_ece_before']:.4f})",
        ),
        (
            "auto_macro_accuracy",
            point["auto_macro_field_accuracy"] >= MIN_AUTO_MACRO_ACCURACY,
            f"{point['auto_macro_field_accuracy']:.4f} >= {MIN_AUTO_MACRO_ACCURACY}",
        ),
        (
            "auto_exact_match",
            point["auto_exact_match"] >= MIN_AUTO_EXACT_MATCH,
            f"{point['auto_exact_match']:.4f} >= {MIN_AUTO_EXACT_MATCH}",
        ),
        (
            "human_error_catch_rate",
            caught >= MIN_ERRORS_CAUGHT,
            f"{caught:.4f} >= {MIN_ERRORS_CAUGHT}",
        ),
        ("auto_rate", point["auto_rate"] >= MIN_AUTO_RATE, f"{point['auto_rate']:.4f} >= {MIN_AUTO_RATE}"),
        (
            "routing_latency_p95",
            latency["combined_p95_ms"] <= MAX_COMBINED_P95_MS,
            f"{latency['combined_p95_ms']:.3f} ms <= {MAX_COMBINED_P95_MS} ms",
        ),
    ]


def evaluate_drift(path: Path) -> list[tuple[str, bool, str]]:
    data = json.loads(path.read_text())
    checks = [
        (
            "drift_null_false_alarm",
            data["null_false_alarm_rate"] <= MAX_NULL_FALSE_ALARM,
            f"{data['null_false_alarm_rate']:.3f} <= {MAX_NULL_FALSE_ALARM}",
        )
    ]
    for scenario, result in data["scenarios"].items():
        if scenario == "null":
            continue
        checks.append(
            (
                f"drift_detect_{scenario}",
                result["detection_rate"] >= MIN_SHIFT_DETECTION,
                f"{result['detection_rate']:.3f} >= {MIN_SHIFT_DETECTION}",
            )
        )
    for scenario, result in data["time_to_detect"].items():
        median = result.get("median_windows")
        passed = median is not None and median <= MAX_TIME_TO_DETECT_WINDOWS
        checks.append(
            (
                f"drift_ttd_{scenario}",
                passed,
                f"{median} windows <= {MAX_TIME_TO_DETECT_WINDOWS}",
            )
        )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence", type=Path, required=True)
    parser.add_argument("--drift", type=Path, required=True)
    args = parser.parse_args()

    checks = evaluate_confidence(args.confidence) + evaluate_drift(args.drift)
    _check(checks)
    failures = [check for check in checks if not check[1]]
    if failures:
        print(f"\nQUALITY GATE FAILED ({len(failures)} check(s))")
        return 1
    print(f"\nQUALITY GATE PASSED ({len(checks)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
