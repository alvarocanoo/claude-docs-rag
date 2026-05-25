"""Runtime configuration loaded from environment / .env."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    anthropic_api_key: str = Field(default="")

    # Either provide POSTGRES_DSN (e.g. Neon connection string) — wins if set —
    # or the individual fields (local docker compose default).
    postgres_dsn_override: str = Field(default="", alias="POSTGRES_DSN")
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "cdrag"
    postgres_password: str = "cdrag_dev"
    postgres_db: str = "cdrag"

    redis_url: str = "redis://localhost:6379/0"

    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    model_simple: str = "claude-haiku-4-5-20251001"
    model_complex: str = "claude-sonnet-4-6"
    model_fallback_local: str = "qwen2.5:7b"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_retrieval: int = 20
    top_k_rerank: int = 5

    semantic_cache_threshold: float = 0.93
    cache_ttl_seconds: int = 86400

    @property
    def postgres_dsn(self) -> str:
        if self.postgres_dsn_override:
            return self.postgres_dsn_override
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
