#!/usr/bin/env python3
"""Apply warehouse DDL to Snowflake via the SQL API (PAT auth).

    SNOWFLAKE_TOKEN=... uv run python warehouse/apply_ddl.py

Reads every ``*.sql`` under ``warehouse/ddl`` in order and executes each
statement. Idempotent (all DDL is ``IF NOT EXISTS`` / ``CREATE OR REPLACE``).
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
DDL_DIR = REPO_ROOT / "warehouse" / "ddl"


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


def statements(sql: str) -> list[str]:
    sql = re.sub(r"--[^\n]*", "", sql)
    return [s.strip() for s in sql.split(";") if s.strip()]


def run(client: httpx.Client, account: str, token: str, stmt: str, role: str, wh: str) -> None:
    url = f"https://{account}.snowflakecomputing.com/api/v2/statements"
    body = {"statement": stmt, "role": role, "warehouse": wh, "timeout": 60}
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
    }
    resp = client.post(url, json=body, headers=headers, timeout=90)
    data = resp.json() if resp.content else {}
    handle = data.get("statementHandle")
    # Poll while the statement is still running.
    while handle and data.get("code") == "333333":
        time.sleep(1)
        resp = client.get(f"{url}/{handle}", headers=headers, timeout=60)
        data = resp.json() if resp.content else {}
    if not resp.is_success:
        raise RuntimeError(f"{data.get('code')}: {data.get('message')}\n  SQL: {stmt[:120]}")
    print(f"  ok  {stmt.strip().splitlines()[0][:70]}")


def main() -> int:
    load_env()
    account = os.environ.get("SNOWFLAKE_ACCOUNT")
    token = os.environ.get("SNOWFLAKE_TOKEN")
    role = os.environ.get("SNOWFLAKE_ROLE", "ACCOUNTADMIN")
    wh = os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
    if not account or not token:
        print("set SNOWFLAKE_ACCOUNT and SNOWFLAKE_TOKEN", file=sys.stderr)
        return 2

    files = sorted(DDL_DIR.glob("*.sql"))
    with httpx.Client() as client:
        for path in files:
            print(f"applying {path.name}")
            for stmt in statements(path.read_text()):
                run(client, account, token, stmt, role, wh)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
