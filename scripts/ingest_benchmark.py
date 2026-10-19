#!/usr/bin/env python3
"""Measure ingestion throughput: serial vs multiprocessing vs a Ray cluster.

The worker is the offline pipeline (redaction -> heuristic extraction -> GL
coding -> policy -> hashed embedding). No GPU or LLM endpoint is involved.

    # local baselines
    cd ai-service && uv run python ../scripts/ingest_benchmark.py --docs 2000

    # Ray cluster from infra/docker-compose.yml (1 head + 2 workers)
    docker compose -f infra/docker-compose.yml --profile ray up -d --build
    docker compose -f infra/docker-compose.yml exec ray-head \\
        python /workspace/scripts/ingest_benchmark.py --docs 2000 --ray-address auto

    # CI smoke test (no cluster)
    python scripts/ingest_benchmark.py --docs 24 --chunk-size 8 \\
        --backends serial,ray --ray-local-mode
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "evals"))
os.environ.setdefault("BENCH_EMBED_BACKEND", "hashing")


def load_records(count: int) -> list[dict[str, Any]]:
    path = REPO_ROOT / "evals" / "golden" / "v0.jsonl"
    base = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records = []
    for index in range(count):
        source = base[index % len(base)]
        records.append(
            {
                "id": f"{source['id']}#{index}",
                "text": source["text"],
                "source": source.get("source", ""),
            }
        )
    return records


def _chunks(records: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    size = max(1, size)
    return [records[start : start + size] for start in range(0, len(records), size)]


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))]


def _hash(rows: list[dict[str, Any]]) -> str:
    clean = [
        {key: value for key, value in row.items() if key != "latency_ms"}
        for row in sorted(rows, key=lambda row: row["id"])
    ]
    payload = json.dumps(clean, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _summary(
    name: str,
    rows: list[dict[str, Any]],
    setup_s: float,
    work_s: float,
    serial: dict[str, Any] | None,
    workers: int,
) -> dict[str, Any]:
    wall = setup_s + work_s
    latencies = [float(row["latency_ms"]) for row in rows]
    speedup = (serial["wall_s"] / wall) if serial and wall else 1.0
    speedup_work = (serial["work_s"] / work_s) if serial and work_s else 1.0
    return {
        "backend": name,
        "workers": workers,
        "docs": len(rows),
        "setup_s": round(setup_s, 4),
        "work_s": round(work_s, 4),
        "wall_s": round(wall, 4),
        # wall clock includes one-time pool/cluster setup
        "docs_per_s": round(len(rows) / wall, 2) if wall else 0.0,
        "speedup_vs_serial": round(speedup, 3),
        "efficiency": round(speedup / workers, 3) if workers else 0.0,
        # steady-state: processing only, excluding setup
        "docs_per_s_work": round(len(rows) / work_s, 2) if work_s else 0.0,
        "speedup_work": round(speedup_work, 3),
        "efficiency_work": round(speedup_work / workers, 3) if workers else 0.0,
        "p50_ms": round(statistics.median(latencies), 3) if latencies else 0.0,
        "p95_ms": round(_p95(latencies), 3),
        "output_hash": _hash(rows),
    }


def _serial(records: list[dict[str, Any]], chunk_size: int) -> tuple[list[dict], float, float]:
    from bench.worker import process_chunk

    process_chunk(records[: min(chunk_size, len(records))])  # warmup
    started = time.perf_counter()
    rows: list[dict] = []
    for chunk in _chunks(records, chunk_size):
        rows.extend(process_chunk(chunk))
    return rows, 0.0, time.perf_counter() - started


def _multiprocessing(
    records: list[dict[str, Any]], chunk_size: int, workers: int
) -> tuple[list[dict], float, float]:
    from bench.worker import process_chunk

    context = mp.get_context("spawn")
    started = time.perf_counter()
    pool = context.Pool(workers)
    setup_s = time.perf_counter() - started
    try:
        pool.map(process_chunk, [records[: min(chunk_size, len(records))]] * workers)
        started = time.perf_counter()
        chunks = pool.map(process_chunk, _chunks(records, chunk_size))
        work_s = time.perf_counter() - started
    finally:
        pool.close()
        pool.join()
    rows = [row for chunk in chunks for row in chunk]
    return rows, setup_s, work_s


def _ray(
    records: list[dict[str, Any]],
    chunk_size: int,
    workers: int,
    address: str,
    local_mode: bool,
) -> tuple[list[dict], float, float, int]:
    import ray
    from bench.worker import process_chunk

    started = time.perf_counter()
    if local_mode:
        # A local Ray cluster (no Docker). Ray >= 2.50 removed local_mode,
        # so this exercises real task scheduling on `workers` local CPUs.
        ray.init(
            num_cpus=workers,
            include_dashboard=False,
            logging_level=logging.ERROR,
        )
    else:
        ray.init(address=address, logging_level=logging.ERROR)
    setup_s = time.perf_counter() - started
    cpus = int(ray.cluster_resources().get("CPU", workers))
    remote = ray.remote(process_chunk)
    chunks = _chunks(records, chunk_size)
    try:
        ray.get(
            [
                remote.remote(records[: min(chunk_size, len(records))])
                for _ in range(min(workers, len(chunks)))
            ]
        )
        started = time.perf_counter()
        results = ray.get([remote.remote(chunk) for chunk in chunks])
        work_s = time.perf_counter() - started
    finally:
        ray.shutdown()
    rows = [row for chunk in results for row in chunk]
    return rows, setup_s, work_s, cpus


def render(result: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = result["results"]
    labels = [f"{row['backend']}\n{row['workers']} worker(s)" for row in runs]
    throughput = [row["docs_per_s_work"] for row in runs]
    speedup = [row["speedup_work"] for row in runs]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    bars = axes[0].bar(labels, throughput, color="#0b5cad")
    for bar, value in zip(bars, throughput, strict=False):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[0].set_title("Throughput")
    axes[0].set_ylabel("documents / second (steady state)")
    axes[0].grid(axis="y", alpha=0.25)

    bars = axes[1].bar(labels, speedup, color="#3f8f6b")
    for bar, value in zip(bars, speedup, strict=False):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].axhline(1.0, color="#888888", linewidth=1, linestyle="--")
    axes[1].set_title("Steady-state speedup vs single process")
    axes[1].set_ylabel("×")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Offline ingestion benchmark — redaction → heuristic extract → GL code → policy → hashed embedding",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.01,
        f"{result['docs']:,} docs · chunk {result['chunk_size']} · "
        "processing only; one-time pool/cluster setup excluded (see JSON for wall clock)",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--workers", default="2,4", help="multiprocessing worker counts")
    parser.add_argument("--backends", default="serial,mp,ray")
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--ray-local-mode", action="store_true")
    parser.add_argument("--out", default=str(REPO_ROOT / "runs" / "ingest_benchmark.json"))
    parser.add_argument(
        "--figure", default=str(REPO_ROOT / "evidence" / "png" / "17-ingest-throughput.png")
    )
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--from-json", help="render an existing result JSON and exit")
    args = parser.parse_args()

    if args.from_json:
        result = json.loads(Path(args.from_json).read_text())
        render(result, Path(args.figure))
        print(f"wrote {args.figure}")
        return 0

    records = load_records(max(args.docs, 1))
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    workers = [int(item) for item in args.workers.split(",") if item.strip()]
    results: list[dict[str, Any]] = []
    output_hashes: dict[str, str] = {}
    serial_summary: dict[str, Any] | None = None

    if "serial" in backends:
        rows, setup_s, work_s = _serial(records, args.chunk_size)
        summary = _summary("serial", rows, setup_s, work_s, None, 1)
        results.append(summary)
        output_hashes["serial"] = summary["output_hash"]
        serial_summary = summary
        print(json.dumps(summary))

    if "mp" in backends:
        for worker_count in workers:
            rows, setup_s, work_s = _multiprocessing(records, args.chunk_size, worker_count)
            summary = _summary("mp", rows, setup_s, work_s, serial_summary, worker_count)
            results.append(summary)
            output_hashes[f"mp-{worker_count}"] = summary["output_hash"]
            print(json.dumps(summary))

    if "ray" in backends:
        ray_workers = 2 if args.ray_local_mode else (max(workers) if workers else 2)
        rows, setup_s, work_s, cpus = _ray(
            records,
            args.chunk_size,
            ray_workers,
            args.ray_address,
            args.ray_local_mode,
        )
        summary = _summary("ray", rows, setup_s, work_s, serial_summary, cpus)
        summary["mode"] = "local-cluster" if args.ray_local_mode else f"cluster:{args.ray_address}"
        results.append(summary)
        output_hashes["ray"] = summary["output_hash"]
        print(json.dumps(summary))

    hashes = set(output_hashes.values())
    result = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "docs": len(records),
        "chunk_size": args.chunk_size,
        "engine": "heuristic",
        "embedding_backend": "hashing",
        "ray_local_mode": args.ray_local_mode,
        "outputs_identical": len(hashes) <= 1,
        "output_hashes": output_hashes,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"outputs identical: {result['outputs_identical']} ({len(hashes)} hash(es))")
    print(f"wrote {out}")

    if not args.no_figure:
        render(result, Path(args.figure))
        print(f"figure -> {args.figure}")

    if not result["outputs_identical"]:
        print("ERROR: backend outputs differ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
