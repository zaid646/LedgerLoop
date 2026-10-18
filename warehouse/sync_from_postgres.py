"""Sync expense facts from Postgres into the Snowflake warehouse.

Reads the backend's `expenses` table and replaces the `EXPENSEIQ.ANALYTICS.EXPENSES`
landing table via the SQL API. AI provenance (model, prompt version, cost,
escalation) is lifted out of the stored pipeline `payload`.

    cd ai-service && uv run python ../warehouse/sync_from_postgres.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import psycopg

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:  # noqa: BLE001 - dotenv optional
    pass

DSN = os.environ.get("POSTGRES_DSN", "postgresql://expenseiq:expenseiq@localhost:5432/expenseiq")
ACCOUNT = os.environ.get("SNOWFLAKE_ACCOUNT")
TOKEN = os.environ.get("SNOWFLAKE_TOKEN")
ROLE = os.environ.get("SNOWFLAKE_ROLE", "ACCOUNTADMIN")
WH = os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")


def q(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def fetch_rows() -> list[dict]:
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, merchant, total_amount, currency, category, policy_route, "
            "escalated, payload, created_at FROM expenses ORDER BY created_at"
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def to_values(row: dict) -> str:
    payload = row.get("payload") or {}
    if isinstance(payload, str):
        payload = json.loads(payload)
    coding = payload.get("coding", {})
    extraction = payload.get("extraction", {})
    ai_model = coding.get("model") or extraction.get("model")
    prompt_version = coding.get("prompt_version") or extraction.get("prompt_version")
    cost = (coding.get("cost_usd") or 0) + (extraction.get("cost_usd") or 0)
    escalated = extraction.get("escalated")
    if escalated is None:
        escalated = row.get("escalated")
    created = row.get("created_at")
    created_s = created.strftime("%Y-%m-%d %H:%M:%S") if created else None

    return "(" + ", ".join(
        [
            q(row["id"]),
            q(row["merchant"]),
            q(float(row["total_amount"]) if row["total_amount"] is not None else None),
            q(row["currency"]),
            q(row["category"]),
            q(row["policy_route"]),
            q(bool(escalated) if escalated is not None else False),
            q(ai_model),
            q(prompt_version),
            q(float(cost) if cost else None),
            f"TO_TIMESTAMP_NTZ({q(created_s)})" if created_s else "NULL",
        ]
    ) + ")"


def execute(client: httpx.Client, url: str, headers: dict, statement: str) -> None:
    body = {"statement": statement, "role": ROLE, "warehouse": WH}
    r = client.post(url, json=body, headers=headers, timeout=120)
    data = r.json() if r.content else {}
    if not r.is_success:
        raise SystemExit(f"SQL failed: {data.get('code')}: {data.get('message')}")


def main() -> int:
    if not (ACCOUNT and TOKEN):
        print("set SNOWFLAKE_ACCOUNT and SNOWFLAKE_TOKEN", file=sys.stderr)
        return 2

    rows = fetch_rows()
    if not rows:
        print("no rows in postgres expenses", file=sys.stderr)
        return 1

    url = f"https://{ACCOUNT}.snowflakecomputing.com/api/v2/statements"
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
    }
    insert = (
        "INSERT INTO EXPENSEIQ.ANALYTICS.EXPENSES "
        "(ID, MERCHANT, TOTAL_AMOUNT, CURRENCY, CATEGORY, POLICY_ROUTE, ESCALATED, "
        "AI_MODEL, PROMPT_VERSION, COST_USD, CREATED_AT) VALUES\n"
        + ",\n".join(to_values(r) for r in rows)
    )
    with httpx.Client() as client:
        execute(client, url, headers, "DELETE FROM EXPENSEIQ.ANALYTICS.EXPENSES")
        execute(client, url, headers, insert)
    print(f"synced {len(rows)} expense rows postgres -> snowflake")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
