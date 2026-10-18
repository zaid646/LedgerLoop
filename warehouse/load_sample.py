#!/usr/bin/env python3
"""Seed demo expense rows and print analytics views (Snowflake SQL API, PAT auth).

    SNOWFLAKE_TOKEN=... uv run python warehouse/load_sample.py

Idempotent: rows use fixed IDs and are MERGE-inserted, so re-running is safe.
Then run the SELECTs (or open Snowsight) to screenshot
``EXPENSEIQ.ANALYTICS.V_SPEND_BY_CATEGORY``.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]

ROWS = [
    # id, merchant, amount, currency, category, route, escalated, model, prompt, cost, created_at
    ("exp-1001", "IKEA MALAYSIA", 1249.00, "MYR", "HARDWARE", "manager_review", False, "Qwen2.5-7B-Instruct", "v1", 0.000358, "2026-03-04 09:12:00"),
    ("exp-1002", "MR D.I.Y. SDN BHD", 89.90, "MYR", "OFFICE_SUPPLIES", "auto_approve", False, "Qwen2.5-7B-Instruct", "v1", 0.000341, "2026-03-05 14:03:00"),
    ("exp-1003", "MAERSK LINE", 8420.00, "USD", "TRAVEL", "manager_review", True, "Qwen2.5-7B-Instruct", "v1", 0.000412, "2026-03-08 11:47:00"),
    ("exp-1004", "GITHUB INC", 210.00, "USD", "SOFTWARE", "auto_approve", False, "Qwen2.5-7B-Instruct", "v1", 0.000327, "2026-03-11 08:20:00"),
    ("exp-1005", "STARBUCKS COFFEE", 42.50, "MYR", "MEALS", "auto_approve", False, "Qwen2.5-7B-Instruct", "v1", 0.000305, "2026-03-11 12:31:00"),
    ("exp-1006", "AWS EMEA", 1530.75, "USD", "SOFTWARE", "manager_review", False, "Qwen2.5-7B-Instruct", "v1", 0.000398, "2026-03-15 18:55:00"),
    ("exp-1007", "SHELL ISNI PETRO", 86.00, "MYR", "TRAVEL", "auto_approve", False, "Qwen2.5-7B-Instruct", "v1", 0.000318, "2026-03-16 07:44:00"),
    ("exp-1008", "APPLE STORE", 4999.00, "MYR", "HARDWARE", "human_queue", True, "Qwen2.5-7B-Instruct", "v1", 0.000431, "2026-03-19 16:02:00"),
]


def load_env() -> None:
    envfile = REPO_ROOT / ".env"
    if not envfile.exists():
        return
    for line in envfile.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _execute(client: httpx.Client, url: str, headers: dict, stmt: str, role: str, wh: str) -> dict:
    body = {"statement": stmt, "role": role, "warehouse": wh, "timeout": 60}
    resp = client.post(url, json=body, headers=headers, timeout=90)
    data = resp.json() if resp.content else {}
    handle = data.get("statementHandle")
    while handle and data.get("code") == "333333":
        time.sleep(1)
        resp = client.get(f"{url}/{handle}", headers=headers, timeout=60)
        data = resp.json() if resp.content else {}
    if not resp.is_success:
        raise RuntimeError(f"{data.get('code')}: {data.get('message')}\n  SQL: {stmt[:160]}")
    return data


def rows_from(data: dict) -> tuple[list[str], list[list]]:
    cols = [c["name"] for c in data.get("resultSetMetaData", {}).get("rowType", [])]
    rows = []
    for r in data.get("data", []):
        rows.append([r.get(c) for c in cols] if isinstance(r, dict) else list(r))
    return cols, rows


def print_table(cols: list[str], rows: list[list]) -> None:
    widths = [max(len(str(c)), *(len(str(r[i])) for r in rows)) if rows else len(str(c)) for i, c in enumerate(cols)]
    print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols)))
    print("  ".join("-" * widths[i] for i in range(len(cols))))
    for r in rows:
        print("  ".join(str(v).ljust(widths[i]) for i, v in enumerate(r)))


def main() -> int:
    load_env()
    account = os.environ.get("SNOWFLAKE_ACCOUNT")
    token = os.environ.get("SNOWFLAKE_TOKEN")
    role = os.environ.get("SNOWFLAKE_ROLE", "ACCOUNTADMIN")
    wh = os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
    if not account or not token:
        print("set SNOWFLAKE_ACCOUNT and SNOWFLAKE_TOKEN", file=sys.stderr)
        return 2

    url = f"https://{account}.snowflakecomputing.com/api/v2/statements"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
    }
    values = ",\n  ".join(
        "('{}', '{}', {}, '{}', '{}', '{}', {}, '{}', '{}', {}, '{}')".format(*r) for r in ROWS
    )
    insert = (
        "INSERT INTO EXPENSEIQ.ANALYTICS.EXPENSES "
        "(ID, MERCHANT, TOTAL_AMOUNT, CURRENCY, CATEGORY, POLICY_ROUTE, ESCALATED, "
        "AI_MODEL, PROMPT_VERSION, COST_USD, CREATED_AT) VALUES\n  " + values
    )

    with httpx.Client() as client:
        _execute(client, url, headers, "DELETE FROM EXPENSEIQ.ANALYTICS.EXPENSES WHERE ID LIKE 'exp-%'", role, wh)
        _execute(client, url, headers, insert, role, wh)
        for label, sql in [
            ("V_SPEND_BY_CATEGORY", "SELECT * FROM EXPENSEIQ.ANALYTICS.V_SPEND_BY_CATEGORY ORDER BY TOTAL_SPEND DESC"),
            ("V_SPEND_BY_MONTH", "SELECT * FROM EXPENSEIQ.ANALYTICS.V_SPEND_BY_MONTH ORDER BY TOTAL_SPEND DESC"),
            ("V_TOP_MERCHANTS", "SELECT * FROM EXPENSEIQ.ANALYTICS.V_TOP_MERCHANTS LIMIT 10"),
            ("V_SPEND_AUDIT", "SELECT * FROM EXPENSEIQ.ANALYTICS.V_SPEND_AUDIT ORDER BY TOTAL_SPEND DESC"),
        ]:
            data = _execute(client, url, headers, sql, role, wh)
            cols, rows = rows_from(data)
            print(f"\n=== EXPENSEIQ.ANALYTICS.{label} ===")
            print_table(cols, rows)
    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
