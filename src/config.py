"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_model: str = "google-gla:gemini-2.0-flash"
    judge_model: str = "google-gla:gemini-2.0-flash"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "personal_kb"
    qdrant_use_memory: bool = True
    search_score_threshold: float = 0.1

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Data
    notes_dir: str = "data/notes"

    # Input validation
    max_query_length: int = 1000

    # Guardrails
    guardrails_enabled: bool = True

    # Tracing
    tracing_enabled: bool = False
    phoenix_endpoint: str = "http://127.0.0.1:6006/v1/traces"


def get_settings() -> Settings:
    """Factory function for settings (enables test overrides)."""
    return Settings()
