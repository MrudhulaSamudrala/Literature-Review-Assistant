"""Centralized application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings for the Smart Literature Review Assistant."""

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Smart Literature Review Assistant"
    service_name: str = "smart-literature-review-assistant"
    app_env: str = "development"
    evaluation_mode: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173"

    openai_api_key: str = Field(default="")
    openai_embedding_model: str = "text-embedding-ada-002"
    openai_llm_model: str = "gpt-4o-mini"
    llm_provider: str = "local"
    local_llm_model: str = "google/flan-t5-small"
    embedding_provider: str = "local"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    vector_top_k: int = 5
    vector_index_name: str = "literature_vector"
    vector_node_label: str = "LiteratureChunk"
    local_vector_index_name: str = "literature_vector_dev"
    local_vector_node_label: str = "LiteratureChunkDev"
    vector_embedding_dimension: int = 1536

    papers_dir: Path = BACKEND_ROOT / "data" / "papers"
    processed_dir: Path = BACKEND_ROOT / "data" / "processed"
    chunks_dir: Path = BACKEND_ROOT / "data" / "chunks"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def embedding_model(self) -> str:
        if self.embedding_provider.strip().lower() == "local":
            return self.local_embedding_model.strip() or "sentence-transformers/all-MiniLM-L6-v2"
        return self.openai_embedding_model.strip() or "text-embedding-ada-002"

    @property
    def llm_model(self) -> str:
        if self.llm_provider.strip().lower() == "local":
            return self.local_llm_model.strip() or "google/flan-t5-small"
        return self.openai_llm_model.strip() or "gpt-4o-mini"


@lru_cache
def get_settings() -> Settings:
    return Settings()
