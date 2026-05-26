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

    # LLM provider selector (see ADR-010). Options: "anthropic" | "groq".
    # Both backends are wired in agent/client.py; default is anthropic for
    # backward compatibility, but for free-tier portfolio demos set it to
    # "groq" and provide GROQ_API_KEY instead of ANTHROPIC_API_KEY.
    llm_provider: str = Field(default="anthropic", alias="LLM_PROVIDER")

    anthropic_api_key: str = Field(default="")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")

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

    # Comma-separated list of origins allowed by CORS on /search, /ask, etc.
    # Defaults cover local dev. In prod, set CDRAG_CORS_ORIGINS to the deployed
    # frontend URL (e.g. "https://claude-docs-rag.vercel.app").
    cors_origins_raw: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        alias="CDRAG_CORS_ORIGINS",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins_raw.split(",") if o.strip()]

    @property
    def is_llm_configured(self) -> bool:
        """True iff the selected provider has the credentials it needs."""
        provider = self.llm_provider.lower()
        if provider == "anthropic":
            return bool(self.anthropic_api_key)
        if provider == "groq":
            return bool(self.groq_api_key)
        return False

    @property
    def postgres_dsn(self) -> str:
        if self.postgres_dsn_override:
            return self.postgres_dsn_override
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
