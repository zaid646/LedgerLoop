from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SQL = REPO_ROOT / "warehouse" / "ddl" / "postgres" / "003_review_queue.sql"
DB_TS = REPO_ROOT / "backend" / "src" / "db.ts"


def test_review_schema_defines_required_tables_and_audit_columns():
    ddl = SQL.read_text()
    for table in ("review_queue", "review_decisions", "labeled_corrections", "system_flags"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in ddl
    for column in (
        "reviewer",
        "created_at",
        "original_value",
        "corrected_value",
        "document_confidence",
        "field_confidence",
        "original_values",
        "corrected_values",
    ):
        assert column in ddl
    assert "ON review_decisions (document_id, created_at)" in ddl
    assert "ON labeled_corrections (document_id, field)" in ddl


def test_backend_migration_matches_checked_in_ddl():
    ddl = SQL.read_text()
    migration = DB_TS.read_text()
    for table in ("review_queue", "review_decisions", "labeled_corrections", "system_flags"):
        assert table in ddl
        assert f"CREATE TABLE IF NOT EXISTS {table}" in migration
    for column in ("reviewer", "reviewed_at", "original_values", "corrected_values"):
        assert column in migration


def test_review_ui_page_exists_and_uses_backend_endpoints():
    page = REPO_ROOT / "ui" / "pages" / "2_Review_Queue.py"
    text = page.read_text()
    assert "/review" in text
    assert "/review/{queue_id}/decision" in text.replace("f\"{BACKEND_URL}{path}\"", "")
    assert "admin/kill-switch" in text
    assert "Reviewer id" in text
