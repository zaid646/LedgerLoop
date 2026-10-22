#!/usr/bin/env python3
"""Replay cached LLM outputs through calibration + routing (CPU only).

Input is the frozen 75-document golden set and the cached Qwen2.5-7B predictions
(``evals/results/llm-qwen7b-v0.json``). No model or paid API is called.

Outputs:
  * ``ai-service/config/calibration.json``       deployable isotonic curves
  * ``evals/results/confidence-replay.json``     measured metrics + reliability bins
  * ``evidence/png/confidence-reliability.png``  reliability diagram

Calibration is evaluated out-of-fold (5 folds by document) so ECE is not
measured on the same documents used to fit the curve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai-service"))
sys.path.insert(0, str(REPO_ROOT / "evals"))

from metrics import field_scores  # noqa: E402

from app.confidence.calibration import (  # noqa: E402
    expected_calibration_error,
    fit_isotonic,
    save_calibrators,
)
from app.confidence.config import ConfidenceConfig  # noqa: E402
from app.confidence.scorer import ARSConfidenceScorer  # noqa: E402
from app.confidence.signals import infer_policy_route  # noqa: E402
from app.routing import ROUTE_AUTO, ROUTE_REJECT, route_document  # noqa: E402
from app.schemas import ReceiptExtraction  # noqa: E402

GOLDEN_PATH = REPO_ROOT / "evals" / "golden" / "v0.jsonl"
CACHE_PATH = REPO_ROOT / "evals" / "results" / "llm-qwen7b-v0.json"
OUT_PATH = REPO_ROOT / "evals" / "results" / "confidence-replay.json"
CALIBRATION_PATH = REPO_ROOT / "ai-service" / "config" / "calibration.json"
PLOT_PATH = REPO_ROOT / "evidence" / "png" / "confidence-reliability.png"
FIELDS = ("merchant", "total_amount", "transaction_date")
FOLDS = 5
AUTO_ACCURACY_FLOOR = 0.95


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def load_samples() -> list[dict]:
    golden = [json.loads(line) for line in GOLDEN_PATH.read_text().splitlines() if line.strip()]
    cached = json.loads(CACHE_PATH.read_text())
    predictions = {row["id"]: row for row in cached["predictions"]}
    samples: list[dict] = []
    for record in golden:
        row = predictions[record["id"]]
        extraction = ReceiptExtraction.model_validate(row["pred"])
        policy_route, _ = infer_policy_route(extraction, source=record.get("source", "receipt"))
        scores = field_scores(row["pred"], record["fields"])
        present = [field for field in FIELDS if record["fields"].get(field) not in (None, "", [])]
        samples.append(
            {
                "id": record["id"],
                "source": record.get("source", ""),
                "text": record["text"],
                "gold": record["fields"],
                "extraction": extraction,
                "attempts": int(row.get("attempts", 1)),
                "repaired": bool(row.get("repaired", False)),
                "policy_route": policy_route,
                "labels": {field: int(scores[field]) for field in FIELDS},
                "present": present,
                "doc_accuracy": sum(int(scores[f]) for f in present) / len(present),
                "exact": all(scores[f] for f in present),
                "any_error": any(not scores[f] for f in present),
            }
        )
    return samples


def raw_scores(samples: list[dict], config: ConfidenceConfig) -> list[dict]:
    scorer = ARSConfidenceScorer(config, {}, calibration_version="raw")
    output = []
    for sample in samples:
        result = scorer.score(
            sample["extraction"],
            sample["text"],
            attempts=sample["attempts"],
            repaired=sample["repaired"],
            schema_ok=True,
            policy_route=sample["policy_route"],
            source=sample["source"],
        )
        output.append(
            {
                "id": sample["id"],
                "field_raw": result.raw_field_scores,
                "document_raw": result.raw_document_score,
            }
        )
    return output


def cross_fit(samples: list[dict], raws: list[dict], config: ConfidenceConfig) -> dict:
    fold_of = {sample["id"]: index % FOLDS for index, sample in enumerate(samples)}
    field_oof = {field: [None] * len(samples) for field in FIELDS}
    document_oof = [None] * len(samples)

    for fold in range(FOLDS):
        train = [i for i, sample in enumerate(samples) if fold_of[sample["id"]] != fold]
        test = [i for i, sample in enumerate(samples) if fold_of[sample["id"]] == fold]
        for field in FIELDS:
            train_rows = [
                (raws[i]["field_raw"][field], samples[i]["labels"][field])
                for i in train
                if field in samples[i]["present"]
            ]
            calibrator = fit_isotonic(
                [row[0] for row in train_rows],
                [row[1] for row in train_rows],
            )
            for i in test:
                field_oof[field][i] = calibrator.predict(raws[i]["field_raw"][field])
        doc_calibrator = fit_isotonic(
            [raws[i]["document_raw"] for i in train],
            [samples[i]["doc_accuracy"] for i in train],
        )
        for i in test:
            document_oof[i] = doc_calibrator.predict(raws[i]["document_raw"])

    field_before = {
        field: expected_calibration_error(
            [samples[i]["labels"][field] for i in range(len(samples)) if field in samples[i]["present"]],
            [
                raws[i]["field_raw"][field]
                for i in range(len(samples))
                if field in samples[i]["present"]
            ],
            config.calibration.bins,
        )
        for field in FIELDS
    }
    field_after = {
        field: expected_calibration_error(
            [samples[i]["labels"][field] for i in range(len(samples)) if field in samples[i]["present"]],
            [field_oof[field][i] for i in range(len(samples)) if field in samples[i]["present"]],
            config.calibration.bins,
        )
        for field in FIELDS
    }
    doc_before = expected_calibration_error(
        [sample["doc_accuracy"] for sample in samples],
        [raw["document_raw"] for raw in raws],
        config.calibration.bins,
    )
    doc_after = expected_calibration_error(
        [sample["doc_accuracy"] for sample in samples],
        document_oof,
        config.calibration.bins,
    )
    return {
        "field_oof": field_oof,
        "document_oof": document_oof,
        "field_before": field_before,
        "field_after": field_after,
        "doc_before": doc_before,
        "doc_after": doc_after,
    }


def fit_final(samples: list[dict], raws: list[dict], config: ConfidenceConfig) -> dict:
    calibrators = {}
    for field in FIELDS:
        rows = [
            (raws[i]["field_raw"][field], samples[i]["labels"][field])
            for i in range(len(samples))
            if field in samples[i]["present"]
        ]
        calibrators[field] = fit_isotonic([row[0] for row in rows], [row[1] for row in rows])
    calibrators["document"] = fit_isotonic(
        [raw["document_raw"] for raw in raws],
        [sample["doc_accuracy"] for sample in samples],
    )
    digest = hashlib.sha256(GOLDEN_PATH.read_bytes()).hexdigest()[:8]
    payload = {
        "version": f"golden-v0-{digest}",
        "method": "isotonic (PAVA), cross-fitted for ECE; final curve fit on all 75 docs",
        "n_documents": len(samples),
        "n_field_labels": {
            field: sum(1 for sample in samples if field in sample["present"]) for field in FIELDS
        },
        "calibrators": {name: calibrator.to_dict() for name, calibrator in calibrators.items()},
    }
    save_calibrators(CALIBRATION_PATH, payload)
    return calibrators


def evaluate_routes(
    samples: list[dict],
    document_scores: list[float],
    field_scores_by_doc: list[dict],
    config: ConfidenceConfig,
) -> dict:
    sweep = []
    thresholds = [round(0.35 + step * 0.01, 2) for step in range(61)]
    for threshold in thresholds:
        auto, review, reject = [], [], []
        candidate_config = replace(
            config, document=replace(config.document, auto_threshold=threshold)
        )
        for index, sample in enumerate(samples):
            # missing_required is derived from extraction presence, not gold labels.
            missing = [
                name
                for name, spec in config.fields.items()
                if spec.required and getattr(sample["extraction"], name, None) in (None, "", [])
            ]
            result = _Result(document_scores[index], field_scores_by_doc[index])
            decision = route_document(
                result,
                config=candidate_config,
                policy_route=sample["policy_route"],
                schema_ok=True,
                missing_required=missing,
                force_review=False,
            )
            if decision.route == ROUTE_AUTO:
                auto.append(index)
            elif decision.route == ROUTE_REJECT:
                reject.append(index)
            else:
                review.append(index)

        macro = _macro_field_accuracy(samples, auto)
        errors = [i for i, sample in enumerate(samples) if sample["any_error"]]
        caught_review = sum(1 for i in errors if i in set(review))
        caught_human = sum(1 for i in errors if i in set(review) or i in set(reject))
        sweep.append(
            {
                "threshold": threshold,
                "auto_count": len(auto),
                "auto_rate": round(len(auto) / len(samples), 4),
                "auto_macro_field_accuracy": round(macro, 4),
                "auto_exact_match": round(
                    sum(1 for i in auto if samples[i]["exact"]) / len(auto), 4
                )
                if auto
                else 0.0,
                "review_rate": round(len(review) / len(samples), 4),
                "reject_rate": round(len(reject) / len(samples), 4),
                "errors_caught_by_review": caught_review,
                "errors_caught_by_human": caught_human,
                "error_count": len(errors),
            }
        )

    eligible = [row for row in sweep if row["auto_macro_field_accuracy"] >= AUTO_ACCURACY_FLOOR]
    configured = [row for row in eligible if row["threshold"] == config.document.auto_threshold]
    if configured:
        chosen = configured[0]
    else:
        chosen = max(eligible, key=lambda row: (row["auto_rate"], row["auto_macro_field_accuracy"]))
        chosen = min(
            [row for row in eligible if row["auto_rate"] == chosen["auto_rate"]],
            key=lambda row: row["threshold"],
        )

    # Score-only coverage/accuracy curve: rank by document score and auto-approve
    # the top slice. This is the usual threshold sweep; the three-outcome router
    # above additionally enforces missing-field, per-field and policy guards.
    coverage_curve = []
    for threshold in thresholds:
        auto = [i for i, score in enumerate(document_scores) if score >= threshold]
        if not auto:
            continue
        coverage_curve.append(
            {
                "threshold": threshold,
                "coverage": round(len(auto) / len(samples), 4),
                "auto_macro_field_accuracy": round(_macro_field_accuracy(samples, auto), 4),
                "auto_exact_match": round(
                    sum(1 for i in auto if samples[i]["exact"]) / len(auto), 4
                ),
                "errors_routed_away": sum(
                    1
                    for i, sample in enumerate(samples)
                    if sample["any_error"] and i not in set(auto)
                ),
                "error_count": sum(1 for sample in samples if sample["any_error"]),
            }
        )
    return {
        "sweep": sweep,
        "coverage_curve": coverage_curve,
        "chosen": chosen,
        "accuracy_floor": AUTO_ACCURACY_FLOOR,
    }


def _macro_field_accuracy(samples: list[dict], indices: list[int]) -> float:
    numerators = {field: 0 for field in FIELDS}
    denominators = {field: 0 for field in FIELDS}
    for index in indices:
        sample = samples[index]
        for field in FIELDS:
            if field in sample["present"]:
                denominators[field] += 1
                numerators[field] += sample["labels"][field]
    supported = [field for field in FIELDS if denominators[field]]
    if not supported:
        return 0.0
    return sum(numerators[field] / denominators[field] for field in supported) / len(supported)


class _Result:
    """Minimal ConfidenceResult-compatible object for routing."""

    def __init__(self, document_score: float, field_scores: dict[str, float]) -> None:
        self.document_score = document_score
        self.field_scores = field_scores


def measure_latency(
    samples: list[dict],
    config: ConfidenceConfig,
    calibrators: dict,
    iterations: int,
) -> dict:
    scorer = ARSConfidenceScorer(
        config,
        calibrators,
        calibration_version="measured",
    )
    score_times: list[float] = []
    route_times: list[float] = []
    for _ in range(iterations):
        for sample in samples:
            start = time.perf_counter_ns()
            result = scorer.score(
                sample["extraction"],
                sample["text"],
                attempts=sample["attempts"],
                repaired=sample["repaired"],
                schema_ok=True,
                policy_route=sample["policy_route"],
                source=sample["source"],
            )
            score_times.append((time.perf_counter_ns() - start) / 1_000_000)
            start = time.perf_counter_ns()
            route_document(
                result,
                config=config,
                policy_route=sample["policy_route"],
                force_review=False,
            )
            route_times.append((time.perf_counter_ns() - start) / 1_000_000)
    total = [s + r for s, r in zip(score_times, route_times, strict=True)]
    return {
        "iterations": iterations,
        "documents_per_iteration": len(samples),
        "scoring_p50_ms": round(_percentile(score_times, 0.50), 4),
        "scoring_p95_ms": round(_percentile(score_times, 0.95), 4),
        "routing_p50_ms": round(_percentile(route_times, 0.50), 4),
        "routing_p95_ms": round(_percentile(route_times, 0.95), 4),
        "combined_p50_ms": round(_percentile(total, 0.50), 4),
        "combined_p95_ms": round(_percentile(total, 0.95), 4),
        "combined_mean_ms": round(statistics.fmean(total), 4),
    }


def plot_reliability(before: list[dict], after: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for axis, bins, title in (
        (axes[0], before, "Before calibration"),
        (axes[1], after, "After calibration (5-fold OOF)"),
    ):
        populated = [row for row in bins if row["count"]]
        axis.plot([0, 1], [0, 1], linestyle="--", color="#888", linewidth=1)
        axis.bar(
            [row["confidence"] for row in populated],
            [row["accuracy"] for row in populated],
            width=0.08,
            color="#2a6fdb",
            alpha=0.85,
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Predicted confidence")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Observed accuracy")
    fig.suptitle("LedgerLoop document-confidence reliability (n=75)", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    config = ConfidenceConfig.load()
    samples = load_samples()
    raws = raw_scores(samples, config)
    folds = cross_fit(samples, raws, config)
    calibrators = fit_final(samples, raws, config)

    oof_fields = [
        {field: folds["field_oof"][field][i] for field in FIELDS} for i in range(len(samples))
    ]
    routing = evaluate_routes(samples, folds["document_oof"], oof_fields, config)
    latency = measure_latency(samples, config, calibrators, args.iterations)

    overall_macro = _macro_field_accuracy(samples, list(range(len(samples))))
    chosen = routing["chosen"]
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": {
            "golden": str(GOLDEN_PATH.relative_to(REPO_ROOT)),
            "cached_predictions": str(CACHE_PATH.relative_to(REPO_ROOT)),
            "documents": len(samples),
            "model": "Qwen2.5-7B-Instruct (cached)",
            "llm_logprobs_available": False,
        },
        "calibration": {
            "method": "isotonic (PAVA), 5-fold cross-fitted",
            "field_ece_before": {field: folds["field_before"][field][0] for field in FIELDS},
            "field_ece_after": {field: folds["field_after"][field][0] for field in FIELDS},
            "document_ece_before": folds["doc_before"][0],
            "document_ece_after": folds["doc_after"][0],
            "document_reliability_before": folds["doc_before"][1],
            "document_reliability_after": folds["doc_after"][1],
        },
        "routing": {
            "operating_point": chosen,
            "overall_macro_field_accuracy": round(overall_macro, 4),
            "baseline_reported": 0.8089,
            "accuracy_floor": AUTO_ACCURACY_FLOOR,
            "config": {
                "document_auto_threshold": config.document.auto_threshold,
                "document_reject_threshold": config.document.reject_threshold,
                "field_auto_thresholds": {
                    name: spec.auto_threshold for name, spec in config.fields.items()
                },
            },
            "sweep": routing["sweep"],
            "coverage_curve": routing["coverage_curve"],
        },
        "latency": latency,
        "limitations": [
            "Cached outputs do not contain token logprobs; model self-reported confidence is used.",
            "Calibration/sweep use 75 documents from two receipt datasets; treat ECE as an estimate.",
            "Routing metrics are offline replay, not live production traffic.",
        ],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    if not args.no_plot:
        plot_reliability(
            folds["doc_before"][1],
            folds["doc_after"][1],
            PLOT_PATH,
        )

    print(json.dumps(
        {
            "document_ece_before": payload["calibration"]["document_ece_before"],
            "document_ece_after": payload["calibration"]["document_ece_after"],
            "field_ece_before": payload["calibration"]["field_ece_before"],
            "field_ece_after": payload["calibration"]["field_ece_after"],
            "operating_point": chosen,
            "overall_macro_field_accuracy": payload["routing"]["overall_macro_field_accuracy"],
            "latency": latency,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
