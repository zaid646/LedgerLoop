"""SQL guard for the spend assistant.

The assistant lets an LLM write SQL, so every generated statement is constrained
before it reaches the warehouse:

* single statement, no comments / code fences
* must be a read-only ``SELECT`` / ``WITH`` query
* may only reference the whitelisted analytics relations (or CTEs over them)
* row-capped with ``LIMIT`` when the model forgot one
"""
from __future__ import annotations

import re

MAX_ROWS = 200

ALLOWED_RELATIONS = {
    "expenses",
    "v_spend_by_category",
    "v_spend_by_month",
    "v_top_merchants",
    "v_spend_audit",
}

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|merge|truncate|grant|revoke|"
    r"copy|call|replace|use|execute|put|remove|comment|attach|commit|rollback)\b",
    re.IGNORECASE,
)
_RELATION = re.compile(r"\b(?:from|join)\s+([A-Za-z_][\w.$]*)", re.IGNORECASE)
_CTE = re.compile(r"\b([A-Za-z_]\w*)\s+as\s*\(", re.IGNORECASE)
_LIMIT = re.compile(r"\blimit\b", re.IGNORECASE)


class SqlRejected(ValueError):
    """Raised when a generated statement violates the guard."""


def _strip_fences(sql: str) -> str:
    sql = sql.strip()
    if sql.startswith("```"):
        sql = re.sub(r"^```[a-zA-Z]*\n?", "", sql)
        sql = re.sub(r"\n?```$", "", sql)
    return sql.strip()


def guard_sql(sql: str) -> str:
    """Validate + normalise a generated statement, or raise :class:`SqlRejected`."""
    cleaned = _strip_fences(sql)
    cleaned = cleaned.rstrip().rstrip(";").strip()

    if not cleaned:
        raise SqlRejected("empty statement")
    if "--" in cleaned or "/*" in cleaned:
        raise SqlRejected("comments are not allowed")
    if ";" in cleaned:
        raise SqlRejected("multiple statements are not allowed")
    if not re.match(r"^(select|with)\b", cleaned, re.IGNORECASE):
        raise SqlRejected("only SELECT / WITH queries are allowed")
    if _FORBIDDEN.search(cleaned):
        raise SqlRejected("statement contains a forbidden keyword")

    ctes = {m.group(1).lower() for m in _CTE.finditer(cleaned)}
    for rel in _RELATION.findall(cleaned):
        leaf = rel.split(".")[-1].lower()
        if leaf not in ALLOWED_RELATIONS and leaf not in ctes:
            raise SqlRejected(f"relation '{rel}' is not allowed")

    if not _LIMIT.search(cleaned):
        cleaned = f"{cleaned} LIMIT {MAX_ROWS}"
    return cleaned
