"""Environment-backed settings. Relative data paths resolve from ``data/`` at the repo root."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

KG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = KG_ROOT.parent
# All durable / runtime data lives here — never next to the ``graph/`` code package.
DATA_ROOT = PROJECT_ROOT / "data"


def _env_file() -> str | None:
    candidate = PROJECT_ROOT / ".env"
    return str(candidate) if candidate.exists() else None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FASHIONGPT_",
        env_file=_env_file(),
        extra="ignore",
    )

    env: str = "development"
    # Relative paths are under DATA_ROOT (repo/data/…).
    raw_data_dir: Path = Path("raw")
    generated_dir: Path = Path("generated")
    ontology_path: Path = Path("schemas/ecom-products-ontology.json")
    news_ontology_path: Path = Path("schemas/news-ontology.json")
    review_decisions_path: Path = Path("config/entity_reviews.json")
    graph_backend: str = "memory"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    api_token: str = ""
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    kg_api_host: str = "127.0.0.1"
    kg_api_port: int = 8001
    kg_url: str = "http://127.0.0.1:8001"
    jwt_secret: str = ""
    jwt_access_minutes: int = 60
    jwt_refresh_days: int = 14
    admin_bootstrap_secret: str = ""
    fashion_brain_db: Path = Path("generated/fashion_brain/app.db")
    cors_origins: str = ""
    extraction_mode: str = "deterministic"
    answer_mode: str = "llm"
    # LLM schema→Cypher path for product asks; templated ecom Cypher remains fallback.
    text2cypher_enabled: bool = True
    # Resolve user terms to real Neo4j facet ids before Text-to-Cypher.
    entity_linker_enabled: bool = True
    entity_linker_embeddings: bool = True
    entity_linker_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    entity_linker_top_k: int = 5
    entity_linker_min_score: float = 0.35
    entity_linker_score_delta: float = 0.08
    model_provider: str = "openrouter"
    model_base_url: str = "https://openrouter.ai/api/v1"
    model_api_key: str = ""
    model_name: str = "~openai/gpt-sol-latest"
    image_model: str = "google/gemini-nano-banana-2.1"
    model_app_url: str = "http://127.0.0.1:8000"
    model_app_title: str = "Fashion Brain"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_vision_model: str = ""
    ollama_timeout_seconds: float = 60.0
    embedding_provider: str = "lexical"
    resolution_similarity: str = "off"
    log_level: str = "INFO"
    log_json: bool = True
    request_timeout_seconds: float | None = None
    sql_enabled: bool = False
    sql_dsn: str = ""
    # E-comm scrape Postgres (master_products). Prefer FASHIONGPT_ECOM_DSN, or POSTGRES_* parts.
    ecom_dsn: str = ""
    postgres_host: str = Field(default="127.0.0.1", validation_alias=AliasChoices("FASHIONGPT_POSTGRES_HOST", "POSTGRES_HOST"))
    postgres_port: int = Field(default=5432, validation_alias=AliasChoices("FASHIONGPT_POSTGRES_PORT", "POSTGRES_PORT"))
    postgres_user: str = Field(default="postgres", validation_alias=AliasChoices("FASHIONGPT_POSTGRES_USER", "POSTGRES_USER"))
    postgres_password: str = Field(default="", validation_alias=AliasChoices("FASHIONGPT_POSTGRES_PASSWORD", "POSTGRES_PASSWORD"))
    postgres_db: str = Field(default="fashion_brain", validation_alias=AliasChoices("FASHIONGPT_POSTGRES_DB", "POSTGRES_DB"))
    ecom_sync_batch_size: int = 500
    ecom_sync_limit: int | None = None

    @field_validator("ecom_sync_limit", mode="before")
    @classmethod
    def _empty_sync_limit(cls, value: object) -> object:
        if value is None or value == "":
            return None
        return value

    def resolve_ecom_dsn(self) -> str:
        """Full Postgres URL for e-comm sync."""

        if self.ecom_dsn.strip():
            return self.ecom_dsn.strip()
        user = self.postgres_user or "postgres"
        password = self.postgres_password
        host = self.postgres_host or "127.0.0.1"
        port = self.postgres_port or 5432
        db = self.postgres_db or "fashion_brain"
        if not user and not password:
            return ""
        if password:
            return f"postgresql://{user}:{password}@{host}:{port}/{db}"
        return f"postgresql://{user}@{host}:{port}/{db}"

    @field_validator("request_timeout_seconds", mode="before")
    @classmethod
    def _unlimited_timeout(cls, value: object) -> object:
        """0, a blank value, or none means the request may run until it finishes."""

        if value is None or value == "":
            return None
        if isinstance(value, str) and value.strip().lower() in {"0", "none", "unlimited", "inf", "infinite"}:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0:
            return None
        return value

    def path(self, value: Path) -> Path:
        if value.is_absolute():
            return value
        return DATA_ROOT / value

    @property
    def raw_dir(self) -> Path:
        return self.path(self.raw_data_dir)

    @property
    def generated(self) -> Path:
        return self.path(self.generated_dir)

    @property
    def ontology_file(self) -> Path:
        return self.path(self.ontology_path)

    @property
    def news_ontology_file(self) -> Path:
        return self.path(self.news_ontology_path)

    @property
    def reviews_file(self) -> Path:
        return self.path(self.review_decisions_path)

    @property
    def fashion_brain_db_file(self) -> Path:
        return self.path(self.fashion_brain_db)

    def resolve_jwt_secret(self) -> str:
        """Secret for Fashion Brain access/refresh tokens."""

        if self.jwt_secret:
            return self.jwt_secret
        if self.api_token:
            return f"fb-jwt:{self.api_token}"
        if self.env in {"development", "test"}:
            return "fashion-brain-dev-jwt-secret"
        raise ValueError("FASHIONGPT_JWT_SECRET or FASHIONGPT_API_TOKEN is required outside development.")


def load_settings(**overrides: object) -> Settings:
    """Load settings from the environment, then apply explicit overrides."""

    data = {key: value for key, value in overrides.items() if value is not None}
    if data:
        return Settings(**data)  # type: ignore[arg-type]
    return Settings()


def env_flag(name: str, default: str = "") -> str:
    return os.environ.get(name, default)
