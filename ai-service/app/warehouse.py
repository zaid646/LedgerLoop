"""Snowflake SQL API client (PAT auth) for the spend assistant.

Executes a single read-only statement and returns columns + rows. Kept tiny and
dependency-light (httpx only) so the AI service can query the warehouse views.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from .config import get_settings


class WarehouseError(RuntimeError):
    pass


def _endpoint() -> str:
    s = get_settings()
    return f"https://{s.snowflake_account}.snowflakecomputing.com/api/v2/statements"


def _headers() -> dict[str, str]:
    s = get_settings()
    return {
        "Authorization": f"Bearer {s.snowflake_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
    }


def _body(statement: str) -> dict[str, Any]:
    s = get_settings()
    return {
        "statement": statement,
        "role": s.snowflake_role,
        "warehouse": s.snowflake_warehouse,
        "database": s.snowflake_database,
        "schema": s.snowflake_schema,
        "timeout": 60,
    }


def run_sql(statement: str, *, client: httpx.Client | None = None) -> tuple[list[str], list[list[Any]]]:
    """Execute ``statement`` and return (columns, rows). Raises WarehouseError."""
    s = get_settings()
    if not (s.snowflake_account and s.snowflake_token):
        raise WarehouseError("SNOWFLAKE_ACCOUNT / SNOWFLAKE_TOKEN not configured")

    url = _endpoint()
    headers = _headers()
    owns = client is None
    client = client or httpx.Client()
    try:
        resp = client.post(url, json=_body(statement), headers=headers, timeout=90)
        data = resp.json() if resp.content else {}
        handle = data.get("statementHandle")
        while handle and data.get("code") == "333333":
            time.sleep(1)
            resp = client.get(f"{url}/{handle}", headers=headers, timeout=60)
            data = resp.json() if resp.content else {}
        if not resp.is_success:
            raise WarehouseError(f"{data.get('code')}: {data.get('message')}")
    finally:
        if owns:
            client.close()

    row_type = data.get("resultSetMetaData", {}).get("rowType", [])
    cols = [c["name"] for c in row_type]
    rows = []
    for r in data.get("data", []):
        rows.append([r.get(c) for c in cols] if isinstance(r, dict) else list(r))
    return cols, rows
