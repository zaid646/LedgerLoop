import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bench.worker import process_chunk  # noqa: E402
from scripts.ingest_benchmark import _summary  # noqa: E402


def test_worker_is_deterministic_and_uses_hashed_vectors():
    records = [
        {"id": "t-1", "text": "Starbucks coffee 12.50 card 4111 1111 1111 1111"},
        {"id": "t-2", "text": "Uber ride 24.00 to airport"},
        {"id": "t-3", "text": "AWS invoice 300.00"},
    ]
    first = process_chunk(records)
    second = process_chunk(records)
    strip = lambda rows: [  # noqa: E731 - tiny test helper
        {key: value for key, value in row.items() if key != "latency_ms"} for row in rows
    ]
    assert strip(first) == strip(second)
    assert [row["id"] for row in first] == ["t-1", "t-2", "t-3"]
    assert first[0]["redactions"] >= 1
    assert all(0.0 < row["vec_norm"] <= 1.0 + 1e-9 for row in first)
    assert {row["route"] for row in first} <= {"auto_approve", "manager_review", "human_queue"}


def test_summary_separates_setup_from_steady_state():
    rows = [{"id": f"d{index}", "latency_ms": 1.0} for index in range(100)]
    serial = _summary("serial", rows, 0.0, 0.10, None, 1)
    assert serial["speedup_work"] == 1.0
    assert serial["docs_per_s_work"] == 1000.0

    parallel = _summary("mp", rows, 0.01, 0.05, serial, 2)
    assert parallel["wall_s"] == 0.06
    assert parallel["docs_per_s_work"] == 2000.0
    assert parallel["speedup_work"] == 2.0
    assert parallel["efficiency_work"] == 1.0
