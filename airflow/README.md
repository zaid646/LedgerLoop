# Airflow pipeline

`dags/ledgerloop_pipeline.py` wires three offline tasks:

```
ingest (serial ingestion benchmark) -> eval (heuristic golden-set eval) -> regression_gate
```

No GPU, W&B, cloud endpoint or credentials are used. The DAG is deliberately
unscheduled (`schedule=None`); trigger it manually or with `dags test`.

## Local run (offline)

```bash
python3.12 -m venv airflow/.venv
airflow/.venv/bin/pip install -r airflow/requirements.txt

export AIRFLOW_HOME="$PWD/airflow/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/airflow/dags"
airflow/.venv/bin/airflow db migrate
airflow/.venv/bin/airflow dags test ledgerloop_offline_pipeline 2026-01-01
```

`airflow dags test` runs the tasks in order without a scheduler. The repo's
`uv`-managed `ai-service` environment is created on demand by the task
commands, so no second Python environment needs to be activated.

To run a single task:

```bash
airflow/.venv/bin/airflow tasks test ledgerloop_offline_pipeline ingest 2026-01-01
airflow/.venv/bin/airflow tasks test ledgerloop_offline_pipeline eval 2026-01-01
airflow/.venv/bin/airflow tasks test ledgerloop_offline_pipeline regression_gate 2026-01-01
```

Outputs are written to `runs/airflow_ingest.json` and
`evals/results/airflow-heuristic.json`; the gate compares the latter against
`evals/baseline.json`.
