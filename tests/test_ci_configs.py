from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _scripts(job: dict) -> str:
    script = job.get("script", [])
    if isinstance(script, str):
        return script
    return "\n".join(str(line) for line in script)


def test_gitlab_pipeline_mirrors_github_jobs() -> None:
    gitlab = yaml.safe_load((REPO_ROOT / ".gitlab-ci.yml").read_text())
    github = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())

    assert gitlab["stages"] == ["lint", "test", "eval", "smoke"]
    jobs = {
        name: job
        for name, job in gitlab.items()
        if isinstance(job, dict) and name not in {"variables", "default"}
    }
    assert {"lint", "pytest", "eval-gate", "benchmark-smoke"} <= set(jobs)
    assert "ruff check ." in _scripts(jobs["lint"])
    assert "pytest -q" in _scripts(jobs["pytest"])
    assert "run_evals.py" in _scripts(jobs["eval-gate"])
    assert "gate.py" in _scripts(jobs["eval-gate"])
    assert "ingest_benchmark.py" in _scripts(jobs["benchmark-smoke"])

    assert {"python-service", "benchmark-smoke", "eval-gate", "promptfoo"} <= set(
        github["jobs"]
    )


def test_gitlab_pipeline_needs_no_cloud_credentials() -> None:
    text = (REPO_ROOT / ".gitlab-ci.yml").read_text().lower()
    for forbidden in ("aws_access", "aws_secret", "snowflake_password", "vllm_api_key", "secrets."):
        assert forbidden not in text


def test_jenkinsfile_smoke_stage() -> None:
    text = (REPO_ROOT / "Jenkinsfile").read_text()
    assert "pipeline {" in text
    assert text.count("{") == text.count("}")
    assert "stage('Smoke')" in text
    assert "python3 -m pip install" in text
    assert "uv sync --dev" in text
    assert "ruff check ." in text
    assert "pytest -q tests/test_policy.py" in text
    assert "test_airflow_dag.py" in text
    assert "withCredentials" not in text


def test_github_ci_runs_repo_policy_tests() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "../tests/test_airflow_dag.py" in workflow
    assert "../tests/test_ci_configs.py" in workflow


def test_airflow_local_runbook_is_present() -> None:
    runbook = (REPO_ROOT / "airflow" / "README.md").read_text()
    assert "airflow dags test ledgerloop_offline_pipeline" in runbook
    assert "--no-wandb" in (REPO_ROOT / "airflow" / "dags" / "ledgerloop_pipeline.py").read_text()


def test_quality_gate_and_backend_are_wired_into_ci() -> None:
    github = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    gitlab = (REPO_ROOT / ".gitlab-ci.yml").read_text()
    for config in (github, gitlab):
        assert "quality_gate.py" in config
        assert "drift_sim.py" in config
        assert "npm test" in config
        assert "npm run typecheck" in config
