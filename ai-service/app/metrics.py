"""In-process metrics plus Prometheus export.

``STORE`` is the dependency-free request/latency/cost store used by the API.
The Prometheus collectors below are layered on top for scraping and alerting.
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field

from prometheus_client import Counter, Gauge, Histogram, generate_latest


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


@dataclass
class MetricsStore:
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _latencies: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=2000)))
    _costs: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def observe(self, endpoint: str, latency_ms: float, cost_usd: float) -> None:
        with self._lock:
            self._latencies[endpoint].append(latency_ms)
            self._costs[endpoint].append(cost_usd)

    def incr(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._counters[key] += n

    def snapshot(self) -> dict:
        with self._lock:
            out: dict = {"endpoints": {}, "counters": dict(self._counters)}
            for ep, lat in self._latencies.items():
                vals = list(lat)
                costs = self._costs.get(ep, [])
                out["endpoints"][ep] = {
                    "count": len(vals),
                    "p50_ms": round(_percentile(vals, 0.50), 2),
                    "p95_ms": round(_percentile(vals, 0.95), 2),
                    "total_cost_usd": round(sum(costs), 6),
                    "avg_cost_usd": round(sum(costs) / len(costs), 6) if costs else 0.0,
                }
            return out


STORE = MetricsStore()

EXTRACTIONS = Counter(
    "expenseiq_extractions_total",
    "Extraction requests by routing outcome.",
    ["route"],
)
EXTRACTION_LATENCY = Histogram(
    "expenseiq_extraction_latency_seconds",
    "End-to-end extraction latency.",
    ["route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
DOCUMENT_CONFIDENCE = Histogram(
    "expenseiq_document_confidence",
    "Calibrated document confidence.",
    ["route"],
    buckets=(0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0),
)
DRIFT_SCORE = Gauge(
    "expenseiq_drift_score",
    "Latest drift score as value/threshold ratio, per detector.",
    ["detector"],
)
DRIFT_VALUE = Gauge(
    "expenseiq_drift_metric",
    "Latest raw drift metric value.",
    ["detector"],
)
DRIFT_FIRED = Gauge(
    "expenseiq_drift_fired",
    "1 when the latest drift check fired any detector.",
)
DRIFT_CHECKS = Counter(
    "expenseiq_drift_checks_total",
    "Completed drift checks.",
    ["window"],
)
DRIFT_WINDOW_SIZE = Gauge(
    "expenseiq_drift_window_docs",
    "Documents in the latest drift window.",
)


def observe_extraction(*, route: str, confidence: float, latency_ms: float) -> None:
    EXTRACTIONS.labels(route=route).inc()
    EXTRACTION_LATENCY.labels(route=route).observe(max(0.0, latency_ms) / 1000.0)
    DOCUMENT_CONFIDENCE.labels(route=route).observe(min(1.0, max(0.0, confidence)))


def observe_drift(report: dict) -> None:
    detectors = report.get("detectors") or {}
    for name, detector in detectors.items():
        threshold = float(detector.get("threshold") or 0.0)
        value = float(detector.get("value") or 0.0)
        DRIFT_VALUE.labels(detector=name).set(value)
        DRIFT_SCORE.labels(detector=name).set(value / threshold if threshold else 0.0)
    DRIFT_FIRED.set(1.0 if report.get("fired") else 0.0)
    DRIFT_CHECKS.labels(window=str(report.get("window_docs", "unknown"))).inc()
    if "window_docs" in report:
        DRIFT_WINDOW_SIZE.set(float(report["window_docs"]))


def render() -> bytes:
    return generate_latest()
