"""Replay drift scenarios over cached predictions and measure detection.

Usage:
    uv run python evals/drift_sim.py --backend auto --replicates 40
    uv run python evals/drift_sim.py --backend hashing --replicates 20  # CI

The simulator never calls an LLM. It resamples cached predictions, applies a
synthetic distribution shift, and runs the production drift detectors.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai-service"))

from app.drift import (  # noqa: E402
    DriftThresholds,
    detect,
    embed_texts,
    extract_features,
)

GOLDEN = REPO_ROOT / "evals" / "golden" / "v0.jsonl"
PREDICTIONS = REPO_ROOT / "evals" / "results" / "llm-qwen7b-v0.json"
DEFAULT_OUT = REPO_ROOT / "evals" / "results" / "drift-sim.json"
DEFAULT_PLOT = REPO_ROOT / "evidence" / "png" / "drift-detection.png"

NEW_MERCHANTS = [
    "NOVA MART {n} SDN BHD",
    "KEDAI SEGAR {n}",
    "TEGUH TRADING {n}",
    "BINTANG ROTI {n}",
    "SINAR JAYA ENTERPRISE {n}",
]
SCENARIOS = ("null", "new_merchants", "receipt_format", "noisy_ocr", "amount_shift", "date_shift")


def load_samples() -> list[dict]:
    golden = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    payload = json.loads(PREDICTIONS.read_text())
    predictions = {row["id"]: row["pred"] for row in payload["predictions"]}
    samples = []
    for row in golden:
        pred = predictions.get(row["id"])
        if pred is None:
            continue
        samples.append({"id": row["id"], "text": row.get("text") or "", "fields": dict(pred)})
    return samples


def apply_scenario(samples: list[dict], scenario: str, rng: random.Random) -> list[dict]:
    if scenario == "null":
        return [{"id": s["id"], "text": s["text"], "fields": dict(s["fields"])} for s in samples]
    out = []
    for index, sample in enumerate(samples):
        sample = {"id": sample["id"], "text": sample["text"], "fields": dict(sample["fields"])}
        if scenario == "new_merchants":
            if rng.random() < 0.65:
                merchant = rng.choice(NEW_MERCHANTS).format(n=index + 1)
                sample["fields"]["merchant"] = merchant
                first, _, rest = sample["text"].partition("\n")
                sample["text"] = f"{merchant}\n{rest}" if rest else merchant
        elif scenario == "receipt_format":
            amount = sample["fields"].get("total_amount")
            parsed = _date(sample["fields"].get("transaction_date"))
            body = sample["text"]
            header = (
                "E-INVOICE / TAX INVOICE\n"
                f"INVOICE NO: INV-{index + 1:06d}\n"
                "SUBTOTAL TAX TOTAL\n"
            )
            footer = "\nPAYMENT RECEIVED\nTHANK YOU FOR SHOPPING"
            if parsed:
                body = body.replace(str(sample["fields"].get("transaction_date")), parsed.isoformat())
            if amount is not None:
                body = f"{body}\nGRAND TOTAL {float(amount):.2f}"
            sample["text"] = header + body + footer
            sample["fields"]["transaction_date"] = parsed.isoformat() if parsed else None
        elif scenario == "noisy_ocr":
            sample["text"] = _corrupt_text(sample["text"], rng)
            if rng.random() < 0.60 and sample["fields"].get("total_amount") is not None:
                sample["fields"]["total_amount"] = round(float(sample["fields"]["total_amount"]) * rng.uniform(1.15, 1.45), 2)
            if rng.random() < 0.50:
                parsed = _date(sample["fields"].get("transaction_date"))
                if parsed:
                    shifted = parsed + timedelta(days=rng.choice([30, 31, 60, 90]))
                    sample["fields"]["transaction_date"] = shifted.strftime("%d/%m/%Y")
            if rng.random() < 0.40 and sample["fields"].get("merchant"):
                words = str(sample["fields"]["merchant"]).split()
                sample["fields"]["merchant"] = " ".join(words[:-1]) if len(words) > 1 else words[0][:4]
        elif scenario == "amount_shift":
            if rng.random() < 0.85 and sample["fields"].get("total_amount") is not None:
                sample["fields"]["total_amount"] = round(float(sample["fields"]["total_amount"]) * 2.2, 2)
        elif scenario == "date_shift":
            parsed = _date(sample["fields"].get("transaction_date"))
            if parsed and rng.random() < 0.70:
                shifted = parsed + timedelta(days=45)
                sample["fields"]["transaction_date"] = shifted.strftime("%d/%m/%Y")
        else:
            raise ValueError(f"unknown scenario {scenario}")
        out.append(sample)
    return out


def _corrupt_text(text: str, rng: random.Random) -> str:
    replacements = {"O": "0", "0": "O", "1": "l", "5": "S", "S": "5", "B": "8", "E": "3"}
    chars = list(text)
    for index in range(len(chars)):
        if rng.random() < 0.08 and chars[index] in replacements:
            chars[index] = replacements[chars[index]]
    lines = "".join(chars).splitlines()
    rng.shuffle(lines)
    return "\n".join(lines[: max(1, int(len(lines) * 0.8))])


def _date(value: object) -> date | None:
    from app.drift import parse_receipt_date

    return parse_receipt_date(value)


def _window(samples: list[dict], size: int, rng: random.Random) -> list[dict]:
    if len(samples) <= size:
        return list(samples)
    return rng.sample(samples, size)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _indices(count: int, size: int, rng: random.Random) -> list[int]:
    if count <= size:
        return list(range(count))
    return rng.sample(range(count), size)


def _variant(
    baseline: list[dict], scenario: str, backend: str, seed: int
) -> tuple[list[dict], list[list[float]]]:
    samples = apply_scenario(baseline, scenario, random.Random(seed))
    embeddings = embed_texts([str(sample.get("text") or "") for sample in samples], backend=backend)
    return samples, embeddings


def calibrate(
    baseline: list[dict],
    *,
    backend: str,
    windows: int,
    seed: int,
    quantile: float,
) -> tuple[DriftThresholds, dict]:
    rng = random.Random(seed)
    base_features = extract_features(baseline)
    base_embeddings = embed_texts(base_features.texts, backend=backend)
    metric_values: dict[str, list[float]] = {
        "psi_amount": [],
        "psi_date_quarter": [],
        "psi_merchant_category": [],
        "embedding_centroid_distance": [],
        "ks_amount_p_value": [],
        "ks_date_p_value": [],
    }
    for _ in range(windows):
        indices = _indices(len(baseline), 50, rng)
        feats = extract_features([baseline[index] for index in indices])
        embeddings = [base_embeddings[index] for index in indices]
        report = detect(
            base_features,
            feats,
            baseline_embeddings=base_embeddings,
            current_embeddings=embeddings,
            thresholds=DriftThresholds(ks_alpha=0.01),
        )
        for key in metric_values:
            metric_values[key].append(report.metrics[key])
    thresholds = DriftThresholds(
        psi_amount=max(0.2, _percentile(metric_values["psi_amount"], quantile)),
        ks_alpha=0.01,
        psi_date_quarter=max(0.2, _percentile(metric_values["psi_date_quarter"], quantile)),
        psi_merchant_category=max(0.2, _percentile(metric_values["psi_merchant_category"], quantile)),
        embedding_centroid_distance=max(
            0.005, _percentile(metric_values["embedding_centroid_distance"], quantile)
        ),
    )
    diagnostics = {
        "windows": windows,
        "quantile": quantile,
        "null_metric_median": {key: statistics.median(values) for key, values in metric_values.items()},
        "null_metric_p99": {key: _percentile(values, 0.99) for key, values in metric_values.items()},
        "calibrated_thresholds": thresholds.to_dict(),
    }
    return thresholds, diagnostics


def evaluate(
    baseline: list[dict],
    *,
    scenario: str,
    backend: str,
    replicates: int,
    seed: int,
    thresholds: DriftThresholds,
    window_size: int = 50,
    variants: int = 4,
) -> dict:
    rng = random.Random(seed)
    base_features = extract_features(baseline)
    base_embeddings = embed_texts(base_features.texts, backend=backend)
    precomputed = [
        _variant(baseline, scenario, backend, seed + 1000 + index) for index in range(variants)
    ]
    fired = 0
    scores: list[float] = []
    detector_fires: dict[str, int] = {}
    for replicate in range(replicates):
        shifted, shifted_embeddings = precomputed[replicate % len(precomputed)]
        indices = _indices(len(shifted), window_size, rng)
        feats = extract_features([shifted[index] for index in indices])
        embeddings = [shifted_embeddings[index] for index in indices]
        report = detect(
            base_features,
            feats,
            baseline_embeddings=base_embeddings,
            current_embeddings=embeddings,
            thresholds=thresholds,
        )
        fired += int(report.fired)
        scores.append(report.score)
        for name, detector in report.detectors.items():
            detector_fires[name] = detector_fires.get(name, 0) + int(bool(detector["fired"]))
    return {
        "scenario": scenario,
        "replicates": replicates,
        "precomputed_variants": variants,
        "detection_rate": fired / replicates if replicates else 0.0,
        "median_score": statistics.median(scores) if scores else 0.0,
        "per_detector_fire_rate": {
            name: count / replicates if replicates else 0.0 for name, count in detector_fires.items()
        },
    }


def time_to_detect(
    baseline: list[dict],
    *,
    scenario: str,
    backend: str,
    replicates: int,
    seed: int,
    thresholds: DriftThresholds,
    window_size: int,
    variants: int = 2,
) -> dict:
    rng = random.Random(seed)
    base_features = extract_features(baseline)
    base_embeddings = embed_texts(base_features.texts, backend=backend)
    precomputed = [
        _variant(baseline, scenario, backend, seed + 2000 + index) for index in range(variants)
    ]
    first_windows: list[int] = []
    for replicate in range(replicates):
        shifted, shifted_embeddings = precomputed[replicate % len(precomputed)]
        found = None
        for window_index in range(1, 9):
            ramp = min(1.0, 0.125 * window_index)
            indices = _indices(len(baseline), window_size, rng)
            current: list[dict] = []
            embeddings: list[list[float]] = []
            for index in indices:
                if rng.random() <= ramp:
                    current.append(shifted[index])
                    embeddings.append(shifted_embeddings[index])
                else:
                    current.append(baseline[index])
                    embeddings.append(base_embeddings[index])
            feats = extract_features(current)
            report = detect(
                base_features,
                feats,
                baseline_embeddings=base_embeddings,
                current_embeddings=embeddings,
                thresholds=thresholds,
            )
            if report.fired:
                found = window_index
                break
        if found is not None:
            first_windows.append(found)
    return {
        "scenario": scenario,
        "replicates": replicates,
        "detected": len(first_windows),
        "median_windows": statistics.median(first_windows) if first_windows else None,
        "median_docs": (statistics.median(first_windows) * window_size) if first_windows else None,
    }


def measure_latency(
    baseline: list[dict],
    *,
    backend: str,
    thresholds: DriftThresholds,
    iterations: int = 20,
) -> dict:
    base_features = extract_features(baseline)
    base_embeddings = embed_texts(base_features.texts, backend=backend)
    current = baseline[-50:]
    feats = extract_features(current)
    embedding_ms: list[float] = []
    current_embeddings: list[list[float]] = []
    for _ in range(3):
        start = time.perf_counter()
        current_embeddings = embed_texts(feats.texts, backend=backend)
        embedding_ms.append((time.perf_counter() - start) * 1000)
    detector_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        detect(
            base_features,
            feats,
            baseline_embeddings=base_embeddings,
            current_embeddings=current_embeddings,
            thresholds=thresholds,
        )
        detector_ms.append((time.perf_counter() - start) * 1000)
    return {
        "iterations": iterations,
        "embedding_iterations": len(embedding_ms),
        "detector_only_ms": {
            "p50": _percentile(detector_ms, 0.5),
            "p95": _percentile(detector_ms, 0.95),
        },
        "embedding_batch_ms": {
            "p50": _percentile(embedding_ms, 0.5),
            "p95": _percentile(embedding_ms, 0.95),
        },
    }


def plot_results(results: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional
        return
    scenarios = [name for name in SCENARIOS if name in results["scenarios"]]
    rates = [results["scenarios"][name]["detection_rate"] for name in scenarios]
    colors = ["#6b7280" if name == "null" else "#0b5394" for name in scenarios]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(scenarios, rates, color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Detection rate")
    ax.set_title("Drift detection on 50-document windows")
    ax.grid(axis="x", alpha=0.3)
    for bar, rate in zip(bars, rates, strict=True):
        ax.text(rate + 0.01, bar.get_y() + bar.get_height() / 2, f"{rate:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("auto", "fastembed", "hashing"), default="auto")
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--null-windows", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    baseline = load_samples()
    if not baseline:
        raise SystemExit("no cached predictions found")
    thresholds, diagnostics = calibrate(
        baseline,
        backend=args.backend,
        windows=args.null_windows,
        seed=args.seed + 1,
        quantile=0.995,
    )

    scenarios = {}
    for index, scenario in enumerate(SCENARIOS):
        scenarios[scenario] = evaluate(
            baseline,
            scenario=scenario,
            backend=args.backend,
            replicates=args.replicates,
            seed=args.seed + 100 + index,
            thresholds=thresholds,
        )
    ttd = {
        scenario: time_to_detect(
            baseline,
            scenario=scenario,
            backend=args.backend,
            replicates=max(5, args.replicates // 4),
            seed=args.seed + 500 + index,
            thresholds=thresholds,
            window_size=25,
        )
        for index, scenario in enumerate(SCENARIOS)
        if scenario != "null"
    }
    latency = measure_latency(baseline, backend=args.backend, thresholds=thresholds)
    detected_backend = "fastembed" if args.backend in ("auto", "fastembed") else "hashing"
    if args.backend == "auto":
        from app.embeddings import get_embedder

        detected_backend = get_embedder().backend
    results = {
        "name": f"drift-sim-{detected_backend}",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embed_backend_requested": args.backend,
        "embed_backend_used": detected_backend,
        "baseline_docs": len(baseline),
        "calibration": diagnostics,
        "scenarios": scenarios,
        "null_false_alarm_rate": scenarios["null"]["detection_rate"],
        "time_to_detect": ttd,
        "latency": latency,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str) + "\n")
    if not args.no_plot:
        plot_results(results, args.plot)
    print(json.dumps({"out": str(args.out), "backend": detected_backend, "null_far": results["null_false_alarm_rate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
