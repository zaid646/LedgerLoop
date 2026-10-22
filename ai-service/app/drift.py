"""Distribution drift detectors for extracted fields and receipt embeddings.

All detectors are CPU-only and dependency-free except for the embedder, which
uses bge-small-en-v1.5 via fastembed when installed and deterministic feature
hashing otherwise.

Detectors:
  * PSI on log(1 + total_amount), five baseline quantile bins
  * two-sample KS on log(1 + total_amount)
  * PSI on transaction quarter
  * two-sample KS on transaction-date ordinal
  * PSI on merchant category (inferred from the extracted merchant)
  * cosine distance between bge-small document-embedding centroids
"""
from __future__ import annotations

import contextlib
import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from .catalog import infer_category
from .config import REPO_ROOT
from .embeddings import get_embedder

DRIFT_CONFIG_PATH = REPO_ROOT / "ai-service" / "config" / "drift.yaml"
_EPS = 1e-6
_DATE_RE = re.compile(r"(?<!\d)(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{1,4})(?!\d)")


@dataclass(frozen=True)
class DriftThresholds:
    psi_amount: float = 0.20
    ks_alpha: float = 0.01
    psi_date_quarter: float = 0.20
    psi_merchant_category: float = 0.20
    embedding_centroid_distance: float = 0.02

    @classmethod
    def load(cls, path: str | Path | None = None) -> DriftThresholds:
        config_path = Path(path).resolve() if path else DRIFT_CONFIG_PATH
        raw = yaml.safe_load(config_path.read_text()) or {}
        return cls(**(raw.get("thresholds") or {}))

    def to_dict(self) -> dict[str, float]:
        return {
            "psi_amount": self.psi_amount,
            "ks_alpha": self.ks_alpha,
            "psi_date_quarter": self.psi_date_quarter,
            "psi_merchant_category": self.psi_merchant_category,
            "embedding_centroid_distance": self.embedding_centroid_distance,
        }


@dataclass
class FieldFeatures:
    amounts: list[float] = field(default_factory=list)
    quarters: list[int] = field(default_factory=list)
    day_ordinals: list[int] = field(default_factory=list)
    merchant_categories: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)


@dataclass
class DriftReport:
    fired: bool
    score: float
    metrics: dict[str, float]
    detectors: dict[str, dict[str, float | bool]]
    thresholds: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "fired": self.fired,
            "score": round(self.score, 6),
            "metrics": {key: round(value, 6) for key, value in self.metrics.items()},
            "detectors": self.detectors,
            "thresholds": self.thresholds,
        }


def parse_receipt_date(value: object) -> date | None:
    if value in (None, ""):
        return None
    match = _DATE_RE.search(str(value))
    if not match:
        return None
    first, middle, last = match.groups()
    if len(first) == 4:
        year, month, day = first, middle, last
    else:
        day, month, year = first, middle, last
    if len(year) == 2:
        year = "20" + year if int(year) < 70 else "19" + year
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def extract_features(samples: list[dict]) -> FieldFeatures:
    features = FieldFeatures()
    for sample in samples:
        fields = sample.get("fields") or {}
        amount = fields.get("total_amount")
        if amount is not None:
            with contextlib.suppress(TypeError, ValueError):
                features.amounts.append(float(amount))
        parsed = parse_receipt_date(fields.get("transaction_date"))
        if parsed is not None:
            features.quarters.append((parsed.month - 1) // 3 + 1)
            features.day_ordinals.append(parsed.toordinal())
        merchant = str(fields.get("merchant") or "")
        features.merchant_categories.append(infer_category(merchant) if merchant else "UNKNOWN")
        features.texts.append(str(sample.get("text") or ""))
    return features


def psi_numeric(baseline: list[float], current: list[float], bins: int = 5) -> float:
    if not baseline or not current:
        return 0.0
    edges = _quantile_edges(baseline, bins)
    base_counts = _bin_counts(baseline, edges)
    curr_counts = _bin_counts(current, edges)
    base_total = sum(base_counts)
    curr_total = sum(curr_counts)
    score = 0.0
    for base_count, curr_count in zip(base_counts, curr_counts, strict=True):
        base_pct = max(base_count / base_total, _EPS)
        curr_pct = max(curr_count / curr_total, _EPS)
        score += (curr_pct - base_pct) * math.log(curr_pct / base_pct)
    return max(0.0, score)


def psi_categorical(baseline: list[str], current: list[str]) -> float:
    if not baseline or not current:
        return 0.0
    baseline_counts: dict[str, int] = {}
    current_counts: dict[str, int] = {}
    for value in baseline:
        baseline_counts[value] = baseline_counts.get(value, 0) + 1
    for value in current:
        current_counts[value] = current_counts.get(value, 0) + 1
    keys = set(baseline_counts) | set(current_counts)
    base_total = len(baseline)
    curr_total = len(current)
    score = 0.0
    for key in keys:
        base_pct = max(baseline_counts.get(key, 0) / base_total, _EPS)
        curr_pct = max(current_counts.get(key, 0) / curr_total, _EPS)
        score += (curr_pct - base_pct) * math.log(curr_pct / base_pct)
    return max(0.0, score)


def ks_2samp(baseline: list[float], current: list[float]) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and asymptotic p-value."""
    if not baseline or not current:
        return 0.0, 1.0
    base = sorted(baseline)
    curr = sorted(current)
    values = sorted(set(base) | set(curr))
    i = j = 0
    max_diff = 0.0
    for value in values:
        while i < len(base) and base[i] <= value:
            i += 1
        while j < len(curr) and curr[j] <= value:
            j += 1
        max_diff = max(max_diff, abs(i / len(base) - j / len(curr)))
    n, m = len(base), len(curr)
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * max_diff
    if lam <= 0:
        p_value = 1.0
    else:
        total = 0.0
        for k in range(1, 101):
            total += ((-1) ** (k - 1)) * math.exp(-2 * k * k * lam * lam)
        p_value = max(0.0, min(1.0, 2 * total))
    return max_diff, p_value


def centroid_distance(
    baseline_embeddings: list[list[float]], current_embeddings: list[list[float]]
) -> float:
    if not baseline_embeddings or not current_embeddings:
        return 0.0
    base = _centroid(baseline_embeddings)
    curr = _centroid(current_embeddings)
    dot = sum(a * b for a, b in zip(base, curr, strict=True))
    base_norm = math.sqrt(sum(a * a for a in base))
    curr_norm = math.sqrt(sum(b * b for b in curr))
    if base_norm == 0 or curr_norm == 0:
        return 0.0
    cosine = dot / (base_norm * curr_norm)
    return max(0.0, min(2.0, 1.0 - cosine))


def embed_texts(texts: list[str], *, backend: str = "auto") -> list[list[float]]:
    if backend == "hashing":
        from .embeddings import Embedder

        return Embedder(backend="hashing").embed(texts)
    return get_embedder().embed(texts)


def detect(
    baseline_features: FieldFeatures,
    current_features: FieldFeatures,
    *,
    baseline_embeddings: list[list[float]] | None = None,
    current_embeddings: list[list[float]] | None = None,
    thresholds: DriftThresholds | None = None,
) -> DriftReport:
    thresholds = thresholds or DriftThresholds()
    base_amounts = [math.log1p(max(0.0, value)) for value in baseline_features.amounts]
    curr_amounts = [math.log1p(max(0.0, value)) for value in current_features.amounts]
    amount_psi = psi_numeric(base_amounts, curr_amounts, bins=5)
    amount_ks, amount_ks_p = ks_2samp(base_amounts, curr_amounts)
    date_psi = psi_categorical(
        [str(quarter) for quarter in baseline_features.quarters],
        [str(quarter) for quarter in current_features.quarters],
    )
    date_ks, date_ks_p = ks_2samp(
        [float(value) for value in baseline_features.day_ordinals],
        [float(value) for value in current_features.day_ordinals],
    )
    merchant_psi = psi_categorical(
        baseline_features.merchant_categories,
        current_features.merchant_categories,
    )
    embedding_distance = centroid_distance(
        baseline_embeddings or [],
        current_embeddings or [],
    )

    detectors = {
        "psi_amount": {
            "value": round(amount_psi, 6),
            "threshold": thresholds.psi_amount,
            "fired": amount_psi > thresholds.psi_amount,
        },
        "ks_amount": {
            "value": round(amount_ks, 6),
            "p_value": round(amount_ks_p, 6),
            "threshold": thresholds.ks_alpha,
            "fired": amount_ks_p < thresholds.ks_alpha,
        },
        "psi_date_quarter": {
            "value": round(date_psi, 6),
            "threshold": thresholds.psi_date_quarter,
            "fired": date_psi > thresholds.psi_date_quarter,
        },
        "ks_date": {
            "value": round(date_ks, 6),
            "p_value": round(date_ks_p, 6),
            "threshold": thresholds.ks_alpha,
            "fired": date_ks_p < thresholds.ks_alpha,
        },
        "psi_merchant_category": {
            "value": round(merchant_psi, 6),
            "threshold": thresholds.psi_merchant_category,
            "fired": merchant_psi > thresholds.psi_merchant_category,
        },
        "embedding_centroid_distance": {
            "value": round(embedding_distance, 6),
            "threshold": thresholds.embedding_centroid_distance,
            "fired": embedding_distance > thresholds.embedding_centroid_distance,
        },
    }
    ratios = [
        float(detector["value"]) / float(detector["threshold"])
        for detector in detectors.values()
        if float(detector["threshold"]) > 0
    ]
    return DriftReport(
        fired=any(bool(detector["fired"]) for detector in detectors.values()),
        score=max(ratios) if ratios else 0.0,
        metrics={
            "psi_amount": amount_psi,
            "ks_amount": amount_ks,
            "ks_amount_p_value": amount_ks_p,
            "psi_date_quarter": date_psi,
            "ks_date": date_ks,
            "ks_date_p_value": date_ks_p,
            "psi_merchant_category": merchant_psi,
            "embedding_centroid_distance": embedding_distance,
        },
        detectors=detectors,
        thresholds=thresholds.to_dict(),
    )


def _quantile_edges(values: list[float], bins: int) -> list[float]:
    ordered = sorted(values)
    edges = [ordered[0]]
    for index in range(1, bins):
        position = (len(ordered) - 1) * index / bins
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            edge = ordered[lower]
        else:
            edge = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
        if edge > edges[-1]:
            edges.append(edge)
    edges.append(ordered[-1])
    if len(edges) < 2:
        return [ordered[0], ordered[0] + _EPS]
    return edges


def _bin_counts(values: list[float], edges: list[float]) -> list[int]:
    counts = [0] * (len(edges) - 1)
    for value in values:
        if value <= edges[0]:
            counts[0] += 1
            continue
        if value >= edges[-1]:
            counts[-1] += 1
            continue
        for index in range(len(edges) - 1):
            if edges[index] <= value < edges[index + 1]:
                counts[index] += 1
                break
    return counts


def _centroid(vectors: list[list[float]]) -> list[float]:
    dim = len(vectors[0])
    out = [0.0] * dim
    for vector in vectors:
        for index, value in enumerate(vector):
            out[index] += value
    return [value / len(vectors) for value in out]
