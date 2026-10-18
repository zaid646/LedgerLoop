"""Application configuration (pydantic-settings) loaded from the repo-root .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM backend (self-hosted vLLM) ---
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "changeme"
    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    vllm_fallback_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # --- tracking / observability ---
    wandb_api_key: str = ""
    wandb_project: str = "expenseiq"
    wandb_entity: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # --- datastores ---
    postgres_dsn: str = "postgresql://expenseiq:expenseiq@localhost:5432/expenseiq"
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "expenseiq"
    opensearch_url: str = "http://localhost:9200"
    opensearch_user: str = "admin"
    opensearch_password: str = "ExpenseIQ!123"
    # The bundled OpenSearch runs with the security plugin disabled; flip to true
    # when pointing at a secured cluster.
    opensearch_use_security: bool = False
    # Embedding model for the vector channel (fastembed/ONNX). Falls back to
    # feature hashing when fastembed is not installed.
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # --- service ---
    ai_service_port: int = 8080
    log_level: str = "info"
    # Second LLM pass in /assistant/ask that phrases the query result as a
    # sentence. Adds one extra generation per question; disable with ANSWER_SUMMARY=0.
    answer_summary: bool = True

    # --- warehouse (Snowflake SQL API, PAT auth) ---
    snowflake_account: str = ""
    snowflake_token: str = ""
    snowflake_role: str = "ACCOUNTADMIN"
    snowflake_warehouse: str = "COMPUTE_WH"
    snowflake_database: str = "EXPENSEIQ"
    snowflake_schema: str = "ANALYTICS"

    # --- harness behaviour ---
    max_extract_attempts: int = 3
    min_field_confidence: float = 0.55

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
