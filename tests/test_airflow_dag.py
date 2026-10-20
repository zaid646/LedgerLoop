from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DAG_PATH = REPO_ROOT / "airflow" / "dags" / "ledgerloop_pipeline.py"


class _FakeDAG:
    instances: list[_FakeDAG] = []

    def __init__(self, dag_id: str, **kwargs):
        self.dag_id = dag_id
        self.kwargs = kwargs
        self.tasks = {}
        _FakeDAG.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeBashOperator:
    def __init__(self, task_id: str, bash_command: str, **kwargs):
        self.task_id = task_id
        self.bash_command = bash_command
        self.kwargs = kwargs
        self.upstream: set[str] = set()
        self.downstream: set[str] = set()

    def __rshift__(self, other: _FakeBashOperator):
        self.downstream.add(other.task_id)
        other.upstream.add(self.task_id)
        return other


def _load_dag(monkeypatch):
    airflow = types.ModuleType("airflow")
    airflow.DAG = _FakeDAG
    airflow.__version__ = "test"
    operators = types.ModuleType("airflow.operators")
    bash = types.ModuleType("airflow.operators.bash")
    bash.BashOperator = _FakeBashOperator
    monkeypatch.setitem(sys.modules, "airflow", airflow)
    monkeypatch.setitem(sys.modules, "airflow.operators", operators)
    monkeypatch.setitem(sys.modules, "airflow.operators.bash", bash)

    spec = importlib.util.spec_from_file_location("ledgerloop_pipeline", DAG_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dag_task_wiring(monkeypatch) -> None:
    module = _load_dag(monkeypatch)
    assert module.dag.dag_id == "ledgerloop_offline_pipeline"
    assert module.dag.kwargs["schedule"] is None
    assert module.dag.kwargs["catchup"] is False

    assert module.ingest.task_id == "ingest"
    assert module.evaluate.task_id == "eval"
    assert module.gate.task_id == "regression_gate"
    assert module.ingest.downstream == {"eval"}
    assert module.evaluate.upstream == {"ingest"}
    assert module.evaluate.downstream == {"regression_gate"}
    assert module.gate.upstream == {"eval"}


def test_dag_commands_are_offline_and_ordered(monkeypatch) -> None:
    module = _load_dag(monkeypatch)
    ingest = module.ingest.bash_command
    evaluation = module.evaluate.bash_command
    gate = module.gate.bash_command
    assert "scripts/ingest_benchmark.py" in ingest
    assert "--backends serial" in ingest
    assert "evals/run_evals.py" in evaluation
    assert "--no-wandb" in evaluation
    assert "evals/gate.py" in gate
    assert "--baseline evals/baseline.json" in gate
    for command in (ingest, evaluation, gate):
        lowered = command.lower()
        assert "aws_" not in lowered
        assert "snowflake" not in lowered
        assert "vllm" not in lowered


def test_dag_source_compiles() -> None:
    compile(DAG_PATH.read_text(), str(DAG_PATH), "exec")
