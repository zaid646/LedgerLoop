"""In-process metrics: request counts, latency p50/p95, cost, escalation rate.

Deliberately dependency-free so it works in tests and CI. A Prometheus exporter
can be layered on later.
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field


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
