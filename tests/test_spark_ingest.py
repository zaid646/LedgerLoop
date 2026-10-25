"""Tests for the PySpark distributed ingestion pipeline (scripts/spark_ingest.py).

These tests exercise the PII redaction UDF and the per-document processing
function without requiring a Spark cluster. A full end-to-end Spark job test
is gated behind the ``spark`` marker and requires ``pyspark`` to be installed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _import_spark_ingest():
    """Import spark_ingest lazily so tests can run without pyspark installed."""
    import importlib

    return importlib.import_module("spark_ingest")


# ---------------------------------------------------------------------------
# PII redaction tests
# ---------------------------------------------------------------------------


def test_redact_pii_card_number():
    spark_ingest = _import_spark_ingest()
    text, count = spark_ingest.redact_pii("Payment 4111-1111-1111-1111 accepted")
    assert "[CARD_NUMBER_REDACTED]" in text
    assert count == 1


def test_redact_pii_email():
    spark_ingest = _import_spark_ingest()
    text, count = spark_ingest.redact_pii("Contact jane.doe@example.com today")
    assert "[EMAIL_REDACTED]" in text
    assert count == 1


def test_redact_pii_phone():
    spark_ingest = _import_spark_ingest()
    text, count = spark_ingest.redact_pii("Call +1-555-123-4567 now")
    assert "[PHONE_REDACTED]" in text
    assert count == 1


def test_redact_pii_ssn():
    spark_ingest = _import_spark_ingest()
    text, count = spark_ingest.redact_pii("SSN 123-45-6789 on file")
    assert "[SSN_REDACTED]" in text
    assert count == 1


def test_redact_pii_multiple():
    spark_ingest = _import_spark_ingest()
    text, count = spark_ingest.redact_pii("4111-1111-1111-1111 and jane@example.com")
    assert count == 2


def test_redact_pii_empty():
    spark_ingest = _import_spark_ingest()
    text, count = spark_ingest.redact_pii("")
    assert text == ""
    assert count == 0


# ---------------------------------------------------------------------------
# Per-document processing tests
# ---------------------------------------------------------------------------


def test_process_doc_returns_expected_keys():
    spark_ingest = _import_spark_ingest()
    record = {
        "id": "test-1",
        "text": "Merchant: Acme Corp\nAmount: USD 42.00\nDate: 2026-01-01\nDescription: office supplies",
        "source": "synthetic",
    }
    result = spark_ingest.process_doc(record)
    expected_keys = {
        "id", "merchant", "category", "route", "violations",
        "redactions", "latency_ms",
    }
    assert set(result.keys()) == expected_keys
    assert result["id"] == "test-1"
    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] >= 0


def test_process_doc_category_infer():
    spark_ingest = _import_spark_ingest()
    record = {
        "id": "test-2",
        "text": "Merchant: Microsoft\nAmount: USD 1312.96\nDate: 2026-04-07\nDescription: Microsoft 365 licenses",
        "source": "synthetic",
    }
    result = spark_ingest.process_doc(record)
    allowed = {
        "TRAVEL", "MEALS", "OFFICE_SUPPLIES", "SOFTWARE", "HARDWARE",
        "MARKETING", "PROFESSIONAL_SERVICES", "UTILITIES", "RENT",
        "INSURANCE", "TRAINING", "ENTERTAINMENT", "OTHER",
    }
    assert result["category"] in allowed


def test_process_doc_redactions_count():
    spark_ingest = _import_spark_ingest()
    record = {
        "id": "test-3",
        "text": "Card 4111-1111-1111-1111 used at Store",
        "source": "synthetic",
    }
    result = spark_ingest.process_doc(record)
    assert result["redactions"] == 1


# ---------------------------------------------------------------------------
# Spark end-to-end test (gated)
# ---------------------------------------------------------------------------


@pytest.mark.spark
def test_spark_job_runs_locally(tmp_path: Path):
    """End-to-end Spark job on local[*]. Requires pyspark installed."""
    spark_ingest = _import_spark_ingest()

    input_path = str(REPO_ROOT / "evals" / "golden" / "v0.jsonl")
    output_path = str(tmp_path / "spark_output")

    summary = spark_ingest.run_spark_ingest(
        input_path=input_path,
        output_path=output_path,
        docs=24,
        master="local[2]",
        partitions=2,
    )

    assert summary["engine"] == "pyspark"
    assert summary["docs"] == 24
    assert summary["docs_per_s"] > 0
    assert summary["output_path"] == output_path


@pytest.mark.spark
def test_spark_output_parquet_readable(tmp_path: Path):
    """Verify the written Parquet can be read back."""
    spark_ingest = _import_spark_ingest()
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F  # noqa: F401

    input_path = str(REPO_ROOT / "evals" / "golden" / "v0.jsonl")
    output_path = str(tmp_path / "spark_output_parquet")

    summary = spark_ingest.run_spark_ingest(
        input_path=input_path,
        output_path=output_path,
        docs=12,
        master="local[2]",
        partitions=2,
    )

    spark = SparkSession.builder.master("local[2]").getOrCreate()
    try:
        df = spark.read.parquet(output_path)
        assert df.count() == 12
        expected_cols = {
            "id", "merchant", "category", "route",
            "violations", "redactions", "latency_ms", "doc_index",
        }
        assert set(df.columns) >= expected_cols
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
