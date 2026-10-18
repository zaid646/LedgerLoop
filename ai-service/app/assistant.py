"""Spend assistant — natural-language question -> guarded SQL -> answer.

The LLM only ever proposes SQL; :mod:`app.sql_guard` constrains it to a single
read-only ``SELECT``/``WITH`` over the analytics relations before
:mod:`app.warehouse` executes it. The result rows are then summarised back into
a sentence. Both LLM calls go through the shared schema-guided client, so this
works against the same self-hosted vLLM endpoint as the rest of the service.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import warehouse
from .config import get_settings
from .llm import LLMClient
from .registry import PromptRegistry
from .sql_guard import MAX_ROWS, guard_sql

Relation = list[list[Any]]
Executor = Callable[[str], "tuple[list[str], Relation]"]

SCHEMA_DOC = """\
EXPENSEIQ.ANALYTICS.EXPENSES(ID, MERCHANT, TOTAL_AMOUNT, CURRENCY, CATEGORY, POLICY_ROUTE, ESCALATED, AI_MODEL, PROMPT_VERSION, COST_USD, CREATED_AT)
EXPENSEIQ.ANALYTICS.V_SPEND_BY_CATEGORY(CATEGORY, CURRENCY, N, TOTAL_SPEND)
EXPENSEIQ.ANALYTICS.V_SPEND_BY_MONTH(MONTH, CATEGORY, TOTAL_SPEND, ESCALATED_COUNT)
EXPENSEIQ.ANALYTICS.V_TOP_MERCHANTS(MERCHANT, TOTAL_SPEND, N)
EXPENSEIQ.ANALYTICS.V_SPEND_AUDIT(CATEGORY, TOTAL_SPEND, AVG_AI_COST, PROMPT_VERSIONS)"""

EXAMPLES = """\
Q: How much did we spend per category?
SQL: SELECT CATEGORY, TOTAL_SPEND FROM EXPENSEIQ.ANALYTICS.V_SPEND_BY_CATEGORY ORDER BY TOTAL_SPEND DESC
Q: Which merchants cost the most?
SQL: SELECT MERCHANT, TOTAL_SPEND FROM EXPENSEIQ.ANALYTICS.V_TOP_MERCHANTS
Q: How much did we spend on travel?
SQL: SELECT TOTAL_SPEND FROM EXPENSEIQ.ANALYTICS.V_SPEND_BY_CATEGORY WHERE CATEGORY = 'TRAVEL'
Q: How many expenses were escalated by month?
SQL: SELECT MONTH, SUM(ESCALATED_COUNT) AS ESCALATED FROM EXPENSEIQ.ANALYTICS.V_SPEND_BY_MONTH GROUP BY 1 ORDER BY 1
Q: What is our average AI cost per category?
SQL: SELECT CATEGORY, AVG_AI_COST FROM EXPENSEIQ.ANALYTICS.V_SPEND_AUDIT"""

_SQL_SCHEMA = {
    "type": "object",
    "properties": {"sql": {"type": "string"}, "reasoning": {"type": "string"}},
    "required": ["sql"],
}
_ANSWER_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}


def build_sql_messages(question: str) -> list[dict[str, str]]:
    prompt = PromptRegistry().resolve("assistant", "prod")
    return [
        {"role": "system", "content": prompt.system},
        {
            "role": "user",
            "content": prompt.render(schema=SCHEMA_DOC, examples=EXAMPLES, question=question),
        },
    ]


def build_answer_messages(
    question: str, sql: str, columns: list[str], rows: Relation
) -> list[dict[str, str]]:
    preview = "\n".join(", ".join(str(v) for v in row) for row in rows[:20])
    return [
        {"role": "system", "content": "You answer a business question in one or two sentences using ONLY the query result. Be concise and include concrete numbers."},
        {
            "role": "user",
            "content": (
                f"Question: {question}\nSQL: {sql}\n"
                f"Columns: {', '.join(columns)}\nRows:\n{preview or '(no rows)'}\n\nAnswer:"
            ),
        },
    ]


async def ask(
    question: str,
    *,
    llm: Any | None = None,
    executor: Executor | None = None,
    max_rows: int = MAX_ROWS,
    summarize: bool | None = None,
) -> dict[str, Any]:
    """Answer ``question`` over the warehouse. Raises SqlRejected / WarehouseError.

    When ``summarize`` is true (default: ``ANSWER_SUMMARY`` env / settings) a
    second LLM pass phrases the rows as a sentence; otherwise the result is the
    guarded SQL plus rows only.
    """
    llm = llm or LLMClient()
    executor = executor or warehouse.run_sql
    if summarize is None:
        summarize = get_settings().answer_summary

    data, sql_result = await llm.complete_json(build_sql_messages(question), schema=_SQL_SCHEMA)
    sql = guard_sql(str(data.get("sql", "")))

    columns, rows = executor(sql)
    rows = rows[:max_rows]

    answer = ""
    latency_ms = sql_result.latency_ms
    cost_usd = sql_result.cost_usd
    model = sql_result.model
    if summarize:
        try:
            adata, a_result = await llm.complete_json(
                build_answer_messages(question, sql, columns, rows),
                schema=_ANSWER_SCHEMA,
                max_tokens=200,
            )
            answer = str(adata.get("answer", "")).strip()
            latency_ms += a_result.latency_ms
            cost_usd += a_result.cost_usd
        except Exception:  # noqa: BLE001 - the SQL + rows are still useful without prose
            answer = ""

    return {
        "question": question,
        "sql": sql,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "answer": answer,
        "model": model,
        "latency_ms": round(latency_ms, 2),
        "cost_usd": round(cost_usd, 8),
    }


def _main() -> int:
    import asyncio
    import json
    import sys

    if len(sys.argv) < 2:
        print('usage: python -m app.assistant "your question"')
        return 2
    result = asyncio.run(ask(" ".join(sys.argv[1:])))
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
