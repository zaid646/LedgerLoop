"""Offline ingestion -> eval -> regression gate DAG.

The DAG uses only local commands and no cloud credentials. Run it with
``airflow dags test ledgerloop_offline_pipeline 2026-01-01`` from
``airflow/`` after installing ``airflow/requirements.txt``.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = f"cd {REPO_ROOT} && uv run --project ai-service python"
INGEST = (
    f"{PYTHON} scripts/ingest_benchmark.py "
    "--docs 200 --chunk-size 50 --backends serial "
    "--out runs/airflow_ingest.json --no-figure"
)
EVAL = (
    f"{PYTHON} evals/run_evals.py "
    "--engine heuristic --golden v0 --no-wandb --name airflow-heuristic"
)
GATE = (
    f"{PYTHON} evals/gate.py "
    "--results evals/results/airflow-heuristic.json "
    "--baseline evals/baseline.json"
)

with DAG(
    dag_id="ledgerloop_offline_pipeline",
    description="Offline ingestion, deterministic eval and regression gate",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["ledgerloop", "offline", "ci"],
    doc_md=__doc__,
) as dag:
    ingest = BashOperator(task_id="ingest", bash_command=INGEST)
    evaluate = BashOperator(task_id="eval", bash_command=EVAL)
    gate = BashOperator(task_id="regression_gate", bash_command=GATE)

    ingest >> evaluate >> gate
