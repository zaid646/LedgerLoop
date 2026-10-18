#!/bin/bash
# Creates auxiliary databases on first boot of the Postgres container.
set -e

function create_db() {
  local db=$1
  echo "  -> ensuring database '$db'"
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
      SELECT 'CREATE DATABASE $db'
      WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$db')\gexec
EOSQL
}

for db in langfuse; do
  create_db "$db"
done
