"""Benchmark PyTorch, ONNX Runtime and TFLite on CPU only.

Each backend runs in its own process so RSS is measured without the other
runtimes resident. Latency is per single-item inference on a fixed 64-token
input; tokenization is precomputed outside the timed loop.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .backends import make_backend
from .common import ARTIFACTS_DIR, MAX_LENGTH, MODEL_ID, REPO_ROOT, TOKEN_FIELDS, preprocess

TEXTS = [
    "team lunch - Starbucks",
    "flight booking - Uber",
    "printer paper - Staples",
    "monthly subscription - GitHub",
    "office chair - IKEA",
    "search ads - Google Ads",
    "client dinner - DoorDash",
    "cloud compute - Amazon Web Services",
    "consulting fees - Deloitte",
    "electricity bill - Shell",
    "online course - Coursera",
    "health insurance premium - Aetna",
    "laptop - Apple",
    "ad campaign - Meta Ads",
    "fuel purchase - Shell",
]

BACKENDS = {
    "torch": ("PyTorch", "fp32"),
    "onnx": ("ONNX Runtime", "fp32"),
    "onnx-int8": ("ONNX Runtime", "int8"),
    "tflite": ("TFLite", "fp32"),
}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(value / divisor, 1)


def _rss_mb() -> float:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 1_000_000, 1)
    except Exception:
        return _peak_rss_mb()


def run_child(backend_name: str, artifacts: Path, threads: int, runs: int, warmup: int) -> dict:
    encoded = preprocess(TEXTS)
    batches = [
        {name: encoded[name][index : index + 1] for name in TOKEN_FIELDS}
        for index in range(len(TEXTS))
    ]
    rss_before = _rss_mb()
    started = time.perf_counter()
    backend = make_backend(backend_name, artifacts_dir=artifacts, threads=threads)
    backend.embed_preprocessed(batches[0])
    load_s = time.perf_counter() - started
    rss_after = _rss_mb()

    for index in range(warmup):
        backend.embed_preprocessed(batches[index % len(batches)])

    latencies = []
    for index in range(runs):
        batch = batches[index % len(batches)]
        started = time.perf_counter()
        backend.embed_preprocessed(batch)
        latencies.append((time.perf_counter() - started) * 1000.0)

    total_s = sum(latencies) / 1000.0
    display_name, precision = BACKENDS[backend_name]
    return {
        "backend": display_name,
        "backend_key": backend_name,
        "precision": precision,
        "runs": runs,
        "warmup": warmup,
        "seq_len": MAX_LENGTH,
        "batch_size": 1,
        "latency_p50_ms": round(_percentile(latencies, 0.50), 3),
        "latency_p95_ms": round(_percentile(latencies, 0.95), 3),
        "latency_mean_ms": round(statistics.fmean(latencies), 3),
        "throughput_samples_s": round(len(latencies) / total_s, 1),
        "model_size_mb": backend.model_size_mb(),
        "load_s": round(load_s, 3),
        "rss_before_load_mb": rss_before,
        "rss_after_load_mb": rss_after,
        "rss_delta_load_mb": round(rss_after - rss_before, 1),
        "peak_rss_mb": max(_peak_rss_mb(), rss_after, _rss_mb()),
    }


def _machine() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def _markdown(results: list[dict], parity: dict | None) -> str:
    lines = [
        "| Backend | Precision | p50 ms | p95 ms | samples/s | Model MB | RSS after load MB | Peak RSS MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['backend']} | {row['precision']} | {row['latency_p50_ms']:.3f} | "
            f"{row['latency_p95_ms']:.3f} | {row['throughput_samples_s']:.1f} | "
            f"{row['model_size_mb']:.1f} | {row['rss_after_load_mb']:.0f} | "
            f"{row['peak_rss_mb']:.0f} |"
        )
    if parity:
        lines.extend(["", "Parity vs PyTorch:"])
        for name, row in parity["runtimes"].items():
            lines.append(
                f"- `{name}`: cosine min {row['cosine_min']:.6f}, "
                f"max abs diff {row['max_abs_diff']:.6f}, "
                f"label agreement {row['label_agreement']:.2%}"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="", help="internal: run a single backend")
    parser.add_argument("--backends", default="torch,onnx,onnx-int8,tflite")
    parser.add_argument("--artifacts", default=str(ARTIFACTS_DIR))
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default="runs/runtime_benchmark.json")
    parser.add_argument("--parity-out", default="runs/runtime_parity.json")
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args()
    artifacts = Path(args.artifacts)

    if args.backend:
        result = run_child(args.backend, artifacts, args.threads, args.runs, args.warmup)
        print("RESULT_JSON:" + json.dumps(result))
        return 0

    selected = [name.strip() for name in args.backends.split(",") if name.strip()]
    for name in selected:
        if name not in BACKENDS:
            parser.error(f"unknown backend: {name}")
    results = []
    for name in selected:
        env = os.environ.copy()
        env.update(
            {
                "OMP_NUM_THREADS": str(args.threads),
                "MKL_NUM_THREADS": str(args.threads),
                "CUDA_VISIBLE_DEVICES": "",
            }
        )
        command = [
            sys.executable,
            "-m",
            "runtime.benchmark",
            "--backend",
            name,
            "--artifacts",
            str(artifacts),
            "--runs",
            str(args.runs),
            "--warmup",
            str(args.warmup),
            "--threads",
            str(args.threads),
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, env=env, cwd=REPO_ROOT
        )
        if completed.returncode != 0:
            sys.stderr.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            raise SystemExit(f"backend {name} failed")
        payload = [
            line for line in completed.stdout.splitlines() if line.startswith("RESULT_JSON:")
        ]
        if not payload:
            raise SystemExit(f"backend {name} produced no result")
        results.append(json.loads(payload[-1].split("RESULT_JSON:", 1)[1]))

    parity = None
    if not args.skip_parity:
        parity_path = Path(args.parity_out)
        parity_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "runtime.parity",
            "--artifacts",
            str(artifacts),
            "--threads",
            str(args.threads),
            "--out",
            str(parity_path),
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, cwd=REPO_ROOT
        )
        if completed.returncode == 0 and parity_path.exists():
            parity = json.loads(parity_path.read_text())
        else:
            sys.stderr.write(completed.stdout + completed.stderr)

    output = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": MODEL_ID,
        "model_kind": "embedding",
        "device": "cpu",
        "threads": args.threads,
        "iterations": args.runs,
        "warmup": args.warmup,
        "machine": _machine(),
        "results": results,
        "parity": parity,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    table = _markdown(results, parity)
    print(table)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
