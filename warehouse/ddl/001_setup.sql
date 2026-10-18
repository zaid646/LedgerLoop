-- ExpenseIQ warehouse setup (Snowflake)
-- Account: ujsbcto-of02014  Warehouse: COMPUTE_WH
-- Apply with: warehouse/apply_ddl.py (SQL API) or `snowsql -f`.

CREATE DATABASE IF NOT EXISTS EXPENSEIQ;
CREATE SCHEMA IF NOT EXISTS EXPENSEIQ.ANALYTICS;

-- Landing table for expense facts synced from Postgres.
CREATE TABLE IF NOT EXISTS EXPENSEIQ.ANALYTICS.EXPENSES (
  ID            STRING,
  MERCHANT      STRING,
  TOTAL_AMOUNT  NUMBER(12, 2),
  CURRENCY      STRING,
  CATEGORY      STRING,
  POLICY_ROUTE  STRING,
  ESCALATED     BOOLEAN,
  AI_MODEL      STRING,
  PROMPT_VERSION STRING,
  COST_USD      NUMBER(12, 6),
  CREATED_AT    TIMESTAMP_NTZ
);
