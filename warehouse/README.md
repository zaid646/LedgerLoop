# Warehouse (Snowflake)

Expense facts land from Postgres via a scheduled sync; analytics views back the
"ask about my spend" assistant and the AI spend audit.

- `ddl/001_setup.sql` — database `EXPENSEIQ`, schema `ANALYTICS`, landing table.
- `ddl/002_views.sql` — spend-by-category / by-month / top-merchants / AI-cost audit.
- `apply_ddl.py` — runs the DDL through the Snowflake SQL API (PAT auth).
- `load_sample.py` — seeds 8 demo expense rows and prints every view.

```bash
SNOWFLAKE_TOKEN=<pat> uv run python warehouse/apply_ddl.py
SNOWFLAKE_TOKEN=<pat> uv run python warehouse/load_sample.py   # seed + print views
# or, with snowsql:
snowsql -a ujsbcto-of02014 -u ZAID646 -f warehouse/ddl/001_setup.sql
```

## Querying the views (for the screenshot)

Views are empty until facts load, so run `load_sample.py` first. Then in
**Snowsight** → *Projects → Worksheets* run:

```sql
SELECT * FROM EXPENSEIQ.ANALYTICS.V_SPEND_BY_CATEGORY ORDER BY TOTAL_SPEND DESC;
SELECT * FROM EXPENSEIQ.ANALYTICS.V_TOP_MERCHANTS;
SELECT * FROM EXPENSEIQ.ANALYTICS.V_SPEND_AUDIT ORDER BY TOTAL_SPEND DESC;
```

Set context first if needed: `USE DATABASE EXPENSEIQ; USE SCHEMA ANALYTICS;`
and `USE WAREHOUSE COMPUTE_WH;`. Browse data without SQL via
*Data → Databases → EXPENSEIQ → ANALYTICS → Views*.

Connection (from `.env`): account `ujsbcto-of02014`, role `ACCOUNTADMIN`,
warehouse `COMPUTE_WH`.
