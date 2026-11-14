from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All config comes from environment variables / .env (like @nestjs/config)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://documind:documind@localhost:5433/documind"

    # "fake" lets you run the whole pipeline without an API key.
    llm_provider: Literal["openai_compatible", "fake"] = "openai_compatible"
    llm_api_key: str = ""
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    chat_model: str = "gemini-2.5-flash"
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768  # must match the DB column; changing it needs a fresh DB

    chunk_size: int = 1000  # characters
    chunk_overlap: int = 150
    top_k: int = 5  # chunks sent to the LLM
    candidates: int = 30  # chunks pulled from each search before fusion

    upload_dir: str = "data/uploads"
    max_upload_mb: int = 20


settings = Settings()
