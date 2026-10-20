pipeline {
  agent any

  options {
    timestamps()
    disableConcurrentBuilds()
  }

  environment {
    UV_CACHE_DIR = "${WORKSPACE}/.uv-cache"
    PYTHONDONTWRITEBYTECODE = "1"
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Smoke') {
      steps {
        sh 'export PATH="$HOME/.local/bin:$PATH"; python3 -m pip install --user --no-cache-dir uv'
        sh 'export PATH="$HOME/.local/bin:$PATH"; cd ai-service && uv sync --dev'
        sh 'export PATH="$HOME/.local/bin:$PATH"; cd ai-service && uv run ruff check .'
        sh 'export PATH="$HOME/.local/bin:$PATH"; cd ai-service && uv run pytest -q tests/test_policy.py tests/test_redaction.py tests/test_search.py'
        sh 'export PATH="$HOME/.local/bin:$PATH"; cd ai-service && uv run pytest -q ../tests/test_airflow_dag.py ../tests/test_ci_configs.py'
      }
    }
  }

  post {
    always {
      cleanWs()
    }
  }
}
