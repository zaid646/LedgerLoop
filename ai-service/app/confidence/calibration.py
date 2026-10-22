"""Isotonic calibration + expected calibration error (ECE).

Implementation is dependency-free so the same code runs in the request path,
the offline replay and CI. Fitting uses pool-adjacent-violators (PAVA), which is
the same monotone least-squares solution as ``sklearn.isotonic`` for this use.
"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IsotonicCalibrator:
    xs: list[float]
    ys: list[float]

    def predict(self, value: float) -> float:
        if not self.xs:
            return max(0.0, min(1.0, float(value)))
        x = float(value)
        if x <= self.xs[0]:
            return self.ys[0]
        if x >= self.xs[-1]:
            return self.ys[-1]
        idx = bisect.bisect_left(self.xs, x)
        if idx < len(self.xs) and self.xs[idx] == x:
            return self.ys[idx]
        left, right = idx - 1, idx
        span = self.xs[right] - self.xs[left]
        if span <= 0:
            return self.ys[right]
        frac = (x - self.xs[left]) / span
        return self.ys[left] + frac * (self.ys[right] - self.ys[left])

    def to_dict(self) -> dict:
        return {"xs": self.xs, "ys": self.ys}

    @classmethod
    def from_dict(cls, data: dict) -> IsotonicCalibrator:
        xs = [float(v) for v in data.get("xs", [])]
        ys = [float(v) for v in data.get("ys", [])]
        if len(xs) != len(ys):
            raise ValueError("calibrator xs/ys length mismatch")
        return cls(xs=xs, ys=ys)


def fit_isotonic(xs: list[float], ys: list[float]) -> IsotonicCalibrator:
    """Fit a monotone, piecewise-linear map from score -> observed accuracy."""
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if not xs:
        return IsotonicCalibrator(xs=[], ys=[])

    grouped: dict[float, list[float]] = {}
    for x, y in zip(xs, ys, strict=True):
        grouped.setdefault(float(x), []).append(float(y))
    unique_x = sorted(grouped)
    points = [(x, sum(grouped[x]) / len(grouped[x])) for x in unique_x]

    # PAVA: merge adjacent blocks while the previous mean exceeds the next.
    blocks: list[dict[str, float | int]] = []
    for index, (_x, mean) in enumerate(points):
        blocks.append({"start": index, "end": index + 1, "mean": mean})
        while len(blocks) >= 2 and blocks[-2]["mean"] > blocks[-1]["mean"]:  # type: ignore[operator]
            right = blocks.pop()
            left = blocks.pop()
            left_weight = int(left["end"]) - int(left["start"])
            right_weight = int(right["end"]) - int(right["start"])
            weight = left_weight + right_weight
            blocks.append(
                {
                    "start": left["start"],
                    "end": right["end"],
                    "mean": (
                        float(left["mean"]) * left_weight + float(right["mean"]) * right_weight
                    )
                    / weight,
                }
            )

    fitted = [0.0] * len(points)
    for block in blocks:
        for index in range(int(block["start"]), int(block["end"])):
            fitted[index] = float(block["mean"])

    return IsotonicCalibrator(
        xs=[x for x, _ in points],
        ys=[min(1.0, max(0.0, value)) for value in fitted],
    )


def expected_calibration_error(
    y_true: list[int] | list[float],
    y_prob: list[float],
    n_bins: int = 10,
) -> tuple[float, list[dict]]:
    """Return (ECE, reliability bins) using equal-width confidence bins."""
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if not y_true:
        return 0.0, []

    bins: list[dict] = []
    ece = 0.0
    total = len(y_true)
    for index in range(n_bins):
        low = index / n_bins
        high = (index + 1) / n_bins
        if index == n_bins - 1:
            members = [(y, p) for y, p in zip(y_true, y_prob, strict=True) if low <= p <= high]
        else:
            members = [(y, p) for y, p in zip(y_true, y_prob, strict=True) if low <= p < high]
        if not members:
            bins.append(
                {"bin": index, "low": low, "high": high, "count": 0, "confidence": 0.0, "accuracy": 0.0}
            )
            continue
        confidence = sum(p for _, p in members) / len(members)
        accuracy = sum(float(y) for y, _ in members) / len(members)
        ece += (len(members) / total) * abs(accuracy - confidence)
        bins.append(
            {
                "bin": index,
                "low": low,
                "high": high,
                "count": len(members),
                "confidence": round(confidence, 6),
                "accuracy": round(accuracy, 6),
            }
        )
    return round(ece, 6), bins


def save_calibrators(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_calibrators(path: Path) -> dict[str, IsotonicCalibrator]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {
        name: IsotonicCalibrator.from_dict(spec)
        for name, spec in (raw.get("calibrators") or {}).items()
    }
