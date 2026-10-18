# Warehouse (Snowflake)

Expense facts land from Postgres via a scheduled sync; analytics views back the
"ask about my spend" assistant and the AI spend audit.

- `ddl/001_setup.sql` — database `EXPENSEIQ`, schema `ANALYTICS`, landing table.
- `ddl/002_views.sql` — spend-by-category / by-month / top-merchants / AI-cost audit.
- `apply_ddl.py` — runs the DDL through the Snowflake SQL API (PAT auth).

```bash
SNOWFLAKE_TOKEN=<pat> uv run python warehouse/apply_ddl.py
# or, with snowsql:
snowsql -a ujsbcto-of02014 -u ZAID646 -f warehouse/ddl/001_setup.sql
```

Connection (from `.env`): account `ujsbcto-of02014`, role `ACCOUNTADMIN`,
warehouse `COMPUTE_WH`.
