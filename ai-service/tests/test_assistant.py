import pytest
from fastapi.testclient import TestClient

from app import assistant, main
from app.llm import LLMResult
from app.sql_guard import SqlRejected, guard_sql

client = TestClient(main.app)


# --------------------------------------------------------------- sql guard


def test_guard_allows_select_and_adds_limit():
    sql = guard_sql("SELECT CATEGORY FROM EXPENSEIQ.ANALYTICS.V_SPEND_BY_CATEGORY")
    assert sql.lower().endswith("limit 200")


def test_guard_keeps_existing_limit():
    sql = guard_sql("SELECT * FROM EXPENSEIQ.ANALYTICS.V_TOP_MERCHANTS LIMIT 5")
    assert sql.lower().count("limit") == 1


def test_guard_allows_cte_over_allowed_relation():
    sql = "WITH t AS (SELECT CATEGORY FROM expenses) SELECT * FROM t LIMIT 3"
    assert guard_sql(sql).startswith("WITH")


@pytest.mark.parametrize(
    "bad",
    [
        "DROP TABLE EXPENSEIQ.ANALYTICS.EXPENSES",
        "SELECT * FROM EXPENSEIQ.ANALYTICS.EXPENSES; DELETE FROM EXPENSEIQ.ANALYTICS.EXPENSES",
        "SELECT * FROM information_schema.tables",
        "SELECT 1 -- comment",
    ],
)
def test_guard_rejects_bad_sql(bad):
    with pytest.raises(SqlRejected):
        guard_sql(bad)


def test_guard_strips_code_fences():
    sql = guard_sql("```sql\nSELECT 1 AS X FROM expenses\n```")
    assert sql.startswith("SELECT 1")


# --------------------------------------------------------------- orchestration


class _FakeLLM:
    def __init__(self, sql: str) -> None:
        self._sql = sql
        self.calls = 0

    async def complete_json(self, messages, schema, max_tokens=700):  # noqa: ANN001
        self.calls += 1
        res = LLMResult(content="{}", model="fake", latency_ms=1.5, cost_usd=0.0)
        if "sql" in schema.get("properties", {}):
            return {"sql": self._sql, "reasoning": "r"}, res
        return {"answer": "You spent 42.0 on TRAVEL."}, res


def _fake_executor(sql: str):  # noqa: ANN202
    return ["CATEGORY", "TOTAL_SPEND"], [["TRAVEL", 42.0], ["MEALS", 10.0]]


async def test_ask_runs_guarded_sql_and_summarises():
    llm = _FakeLLM("SELECT CATEGORY, TOTAL_SPEND FROM EXPENSEIQ.ANALYTICS.V_SPEND_BY_CATEGORY")
    out = await assistant.ask("how much per category?", llm=llm, executor=_fake_executor)
    assert out["sql"].lower().endswith("limit 200")
    assert out["row_count"] == 2
    assert out["answer"].startswith("You spent")
    assert llm.calls == 2


async def test_ask_summary_disabled_skips_second_call():
    llm = _FakeLLM("SELECT * FROM EXPENSEIQ.ANALYTICS.EXPENSES")
    out = await assistant.ask("list expenses", llm=llm, executor=_fake_executor, summarize=False)
    assert out["answer"] == ""
    assert out["row_count"] == 2
    assert llm.calls == 1


def test_assistant_endpoint(monkeypatch):
    monkeypatch.setattr(
        assistant, "LLMClient", lambda: _FakeLLM("SELECT * FROM EXPENSEIQ.ANALYTICS.EXPENSES")
    )
    monkeypatch.setattr(assistant.warehouse, "run_sql", _fake_executor)
    r = client.post("/assistant/ask", json={"question": "list expenses"})
    assert r.status_code == 200
    body = r.json()
    assert body["row_count"] == 2
    assert body["columns"] == ["CATEGORY", "TOTAL_SPEND"]


def test_assistant_endpoint_rejects_dangerous_sql(monkeypatch):
    monkeypatch.setattr(
        assistant, "LLMClient", lambda: _FakeLLM("DELETE FROM EXPENSEIQ.ANALYTICS.EXPENSES")
    )
    monkeypatch.setattr(assistant.warehouse, "run_sql", _fake_executor)
    r = client.post("/assistant/ask", json={"question": "delete everything"})
    assert r.status_code == 400
