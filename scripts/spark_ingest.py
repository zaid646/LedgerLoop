#!/usr/bin/env python3
"""PySpark distributed ingestion pipeline for LedgerLoop receipts.

An alternative to the existing Ray / multiprocessing path in
``scripts/ingest_benchmark.py``. Reads raw receipt JSONL, performs PII
redaction + heuristic field extraction + GL coding + policy check in Spark,
and writes Parquet output (HDFS-local or local FS).

Runs on the local machine (``--master local[*]``) or on a remote Spark
cluster (``--master spark://<host>:7077``).

Usage::

    # local run, 8 000 docs
    python scripts/spark_ingest.py \
        --input evals/golden/v0.jsonl \
        --output runs/spark_ingest_output \
        --docs 8000 \
        --master local[*]

    # remote Spark cluster
    python scripts/spark_ingest.py \
        --input evals/golden/v0.jsonl \
        --output hdfs:///user/root/ledgerloop/spark_output \
        --docs 8000 \
        --master spark://<head-node>:7077
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "evals"))
sys.path.insert(0, str(REPO_ROOT / "ai-service"))
os.environ.setdefault("BENCH_EMBED_BACKEND", "hashing")

# ---------------------------------------------------------------------------
# PII redaction (mirrors app.redaction.redact for Spark UDF compatibility)
# ---------------------------------------------------------------------------

_CARD_RE = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b\+?\d{1,3}[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}\b")


def redact_pii(text: str) -> tuple[str, int]:
    """Redact cards, SSNs, emails, phones. Returns (redacted_text, count)."""
    if not text:
        return "", 0
    count = 0

    def _sub(pattern: re.Pattern[str], repl: str) -> str:
        nonlocal count
        new_text, n = pattern.subn(repl, text)
        count += n
        return new_text

    text = _sub(_CARD_RE, "[CARD_NUMBER_REDACTED]")
    text = _sub(_SSN_RE, "[SSN_REDACTED]")
    text = _sub(_EMAIL_RE, "[EMAIL_REDACTED]")
    text = _sub(_PHONE_RE, "[PHONE_REDACTED]")
    return text, count


# ---------------------------------------------------------------------------
# Per-document processing (called via Spark UDF / mapPartitions)
# ---------------------------------------------------------------------------

def process_doc(record: Any) -> dict[str, Any]:
    """Process one receipt record through the full offline pipeline.

    Accepts either a plain dict or a pyspark.sql.Row (when used as a Spark UDF).
    """
    from baselines import heuristic_extract
    from app.catalog import infer_category
    from app.policy import evaluate
    from app.redaction import redact
    from app.schemas import GLCategory, PolicyCheckRequest

    started = time.perf_counter()
    # Spark Row objects support [] access but not .get()
    text = str(record["text"] if hasattr(record, "__getitem__") else record.get("text", ""))
    record_id = record["id"] if hasattr(record, "__getitem__") else record.get("id", "")
    redacted_text, redactions = redact_pii(text)
    extraction = heuristic_extract(redacted_text)
    merchant = str(extraction.get("merchant") or "unknown").strip()
    category = infer_category(merchant)
    amount = float(extraction.get("total_amount") or 0.0)
    policy = evaluate(
        PolicyCheckRequest(
            amount=amount,
            category=GLCategory(category),
            merchant=merchant or None,
            has_receipt=True,
        )
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "id": record_id,
        "merchant": merchant,
        "category": category,
        "route": policy.route,
        "violations": len(policy.violations),
        "redactions": redactions,
        "latency_ms": round(latency_ms, 4),
    }


def process_partition(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Process a partition of records (used by mapPartitions)."""
    return [process_doc(record) for record in records]


# ---------------------------------------------------------------------------
# Spark job (pyspark imported lazily)
# ---------------------------------------------------------------------------

def _import_pyspark():
    """Import pyspark modules lazily so the module loads without pyspark."""
    from pyspark.sql import SparkSession, functions as F
    from pyspark.sql.types import (
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )
    return SparkSession, F, DoubleType, IntegerType, StringType, StructField, StructType


def _build_output_schema():
    """Build the output StructType using actual PySpark DataType objects."""
    _, _, DoubleType, IntegerType, StringType, StructField, StructType = _import_pyspark()
    return StructType(
        [
            StructField("id", StringType(), False),
            StructField("merchant", StringType(), True),
            StructField("category", StringType(), True),
            StructField("route", StringType(), True),
            StructField("violations", IntegerType(), True),
            StructField("redactions", IntegerType(), True),
            StructField("latency_ms", DoubleType(), True),
        ]
    )


def build_spark(master: str, app_name: str = "ledgerloop-spark-ingest"):
    SparkSession, _, _, _, _, _, _ = _import_pyspark()
    scripts_dir = str(REPO_ROOT / "scripts")
    evals_dir = str(REPO_ROOT / "evals")
    ai_service_dir = str(REPO_ROOT / "ai-service")
    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        # Make modules importable on executors
        .config("spark.executorEnv.PYTHONPATH", f"{scripts_dir}:{evals_dir}:{ai_service_dir}")
    )
    spark = builder.getOrCreate()
    # Distribute the modules that process_doc imports at runtime
    spark.sparkContext.addPyFile(str(REPO_ROOT / "scripts" / "spark_ingest.py"))
    spark.sparkContext.addPyFile(str(REPO_ROOT / "scripts" / "bench" / "__init__.py"))
    spark.sparkContext.addPyFile(str(REPO_ROOT / "scripts" / "bench" / "worker.py"))
    spark.sparkContext.addPyFile(str(REPO_ROOT / "evals" / "baselines.py"))
    return spark


def load_records(spark, input_path: str, docs: int):
    """Load raw receipt records as a Spark DataFrame."""
    raw = spark.read.json(input_path)
    count = raw.count()
    if count == 0:
        raise ValueError(f"no records found at {input_path}")
    repeats = max(1, docs // count)
    remainder = docs % count
    if remainder:
        sampled = raw.limit(remainder)
        df = raw.unionByName(sampled)
        for _ in range(repeats - 1):
            df = df.unionByName(raw)
    else:
        df = raw
        for _ in range(repeats - 1):
            df = df.unionByName(raw)
    return df.limit(docs)


def run_spark_ingest(
    input_path: str,
    output_path: str,
    docs: int,
    master: str,
    partitions: int | None = None,
    warmup: int = 1,
    repeats: int = 3,
) -> dict[str, Any]:
    """Execute the PySpark ingestion pipeline and return a summary dict.

    Benchmark methodology
    ---------------------
    Spark is lazy: building the transformation plan is near-instant, and the
    real work happens when an action forces execution. This function therefore
    measures throughput by forcing an action (``count``) on the fully
    transformed DataFrame:

    * ``load_s``        — read + materialise the input, then cache it.
    * ``warmup_s``      — first execution pass(es). Dominated by JVM startup,
                          codegen, and executor initialisation. Reported
                          separately and *excluded* from steady-state numbers.
    * ``steady_state_s``— subsequent execution passes over the cached input.
                          This is the meaningful throughput figure.
    * ``write_s``       — materialise the result to Parquet.
    """
    SparkSession, F, DoubleType, IntegerType, StringType, StructField, StructType = _import_pyspark()

    output_schema = _build_output_schema()

    spark = build_spark(master)
    try:
        spark.sparkContext.setLogLevel("WARN")

        started = time.perf_counter()
        df = load_records(spark, input_path, docs)
        n_partitions = partitions or spark.sparkContext.defaultParallelism
        df = df.repartition(n_partitions).cache()
        df.count()  # materialise the cache
        load_s = time.perf_counter() - started

        process_udf = F.udf(process_doc, output_schema)
        struct_cols = F.struct([F.col(c) for c in df.columns])
        result = df.withColumn("result", process_udf(struct_cols))
        for field in output_schema.fields:
            result = result.withColumn(field.name, F.col(f"result.{field.name}"))
        result = result.drop("result")
        result = result.withColumn("doc_index", F.monotonically_increasing_id())

        # --- Warmup: first passes include JVM/codegen/executor startup. ------
        # NOTE: count() alone is NOT sufficient — Spark prunes unused columns,
        # which would skip the UDF. foreachPartition materialises every column.
        def _consume(rows):
            for _ in rows:
                pass

        n_docs = result.count()  # untimed; establishes row count only

        warmup_times: list[float] = []
        for _ in range(max(0, warmup)):
            t0 = time.perf_counter()
            result.foreachPartition(_consume)  # forces full UDF execution
            warmup_times.append(time.perf_counter() - t0)

        # --- Steady state: cached input, repeated full pipeline execution. ---
        steady_times: list[float] = []
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            result.foreachPartition(_consume)
            steady_times.append(time.perf_counter() - t0)

        steady_sorted = sorted(steady_times)
        steady_mean = sum(steady_times) / len(steady_times)
        steady_p50 = steady_sorted[len(steady_sorted) // 2]
        steady_min = steady_sorted[0]
        work_s = steady_mean

        started = time.perf_counter()
        result.write.mode("overwrite").parquet(output_path)
        write_s = time.perf_counter() - started

        rows = result.collect()
        n_docs = len(rows)
        total_latency = sum(float(r["latency_ms"]) for r in rows)
        total_redactions = sum(int(r["redactions"]) for r in rows)
        routes = {}
        for r in rows:
            route = str(r["route"])
            routes[route] = routes.get(route, 0) + 1

        wall = load_s + work_s + write_s
        latencies = sorted([float(r["latency_ms"]) for r in rows])
        p50 = latencies[n_docs // 2] if latencies else 0.0

        summary = {
            "engine": "pyspark",
            "docs": n_docs,
            "partitions": n_partitions,
            "master": master,
            "methodology": "cache input; warmup passes excluded; steady-state = mean of repeated full-pipeline foreachPartition actions (forces UDF evaluation, unlike count() which prunes columns)",
            "load_s": round(load_s, 4),
            "warmup_runs": len(warmup_times),
            "warmup_times_s": [round(t, 4) for t in warmup_times],
            "warmup_s": round(sum(warmup_times) / len(warmup_times), 4) if warmup_times else 0.0,
            "steady_state_runs": len(steady_times),
            "steady_state_times_s": [round(t, 4) for t in steady_times],
            "steady_state_mean_s": round(steady_mean, 4),
            "steady_state_p50_s": round(steady_p50, 4),
            "steady_state_min_s": round(steady_min, 4),
            "steady_state_docs_per_s": round(n_docs / steady_mean, 2) if steady_mean else 0.0,
            "work_s": round(work_s, 4),
            "write_s": round(write_s, 4),
            "wall_s": round(wall, 4),
            "docs_per_s": round(n_docs / wall, 2) if wall else 0.0,
            "docs_per_s_work": round(n_docs / work_s, 2) if work_s else 0.0,
            "p50_latency_ms": round(p50, 3),
            "total_redactions": total_redactions,
            "routes": routes,
            "output_path": output_path,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        return summary
    finally:
        spark.stop()


def _hash_rows(rows: list[dict[str, Any]]) -> str:
    clean = [
        {k: v for k, v in row.items() if k != "latency_ms"}
        for row in sorted(rows, key=lambda r: r["id"])
    ]
    payload = json.dumps(clean, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="input JSONL path")
    parser.add_argument("--output", required=True, help="output Parquet path")
    parser.add_argument("--docs", type=int, default=8000, help="number of docs to process")
    parser.add_argument(
        "--master", default="local[*]", help="Spark master URL (default local[*])"
    )
    parser.add_argument(
        "--partitions", type=int, default=None, help="Spark partitions (default: defaultParallelism)"
    )
    parser.add_argument("--warmup", type=int, default=1, help="untimed warmup passes (default 1)")
    parser.add_argument("--repeats", type=int, default=3, help="timed steady-state passes (default 3)")
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "runs" / "spark_ingest.json"),
        help="summary JSON output path",
    )
    args = parser.parse_args(argv)

    summary = run_spark_ingest(
        input_path=args.input,
        output_path=args.output,
        docs=args.docs,
        master=args.master,
        partitions=args.partitions,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    print(json.dumps(summary, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
