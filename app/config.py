from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All config comes from environment variables / .env (like @nestjs/config)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://documind:documind@localhost:5433/documind"

    # --- LLM (any OpenAI-compatible API). "fake" runs everything without an API key. ---
    llm_provider: Literal["openai_compatible", "fake"] = "openai_compatible"
    llm_api_key: str = ""
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    chat_model: str = "gemini-2.5-flash"
    vision_model: str = ""  # for OCR of scanned pages; empty = use chat_model
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768  # must match the DB column; changing it needs a fresh DB
    llm_max_retries: int = 4  # retries with backoff on 429 / 5xx
    llm_stream_usage: bool = True  # ask for token usage on streamed answers
    # Prices in USD per 1M tokens, used for the cost estimate shown per answer.
    price_input_per_m: float = 0.30
    price_output_per_m: float = 2.50

    # --- Ingestion ---
    chunk_size: int = 1000  # characters
    chunk_overlap: int = 150
    ocr_engine: Literal["auto", "tesseract", "llm", "none"] = "auto"
    ocr_min_chars: int = 25  # pages with less extractable text than this get OCR'd
    ocr_dpi: int = 200

    # --- Retrieval ---
    candidates: int = 30  # chunks pulled from each search (vector, keyword) before fusion
    rerank_candidates: int = 20  # fused chunks sent to the reranker
    top_k: int = 5  # chunks sent to the LLM
    reranker: Literal["local", "none"] = "local"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    model_cache_dir: str = "data/models"

    # --- Conversations ---
    history_turns: int = 6  # previous messages included for follow-up questions

    # --- Background jobs ---
    queue_backend: Literal["rq", "background"] = "rq"  # "background" = in-process, no Redis needed
    redis_url: str = "redis://localhost:6379/0"
    ingest_max_retries: int = 3

    # --- File storage ---
    storage_backend: Literal["local", "s3"] = "local"
    upload_dir: str = "data/uploads"
    s3_bucket: str = ""
    s3_prefix: str = "uploads/"
    max_upload_mb: int = 20

    # --- Auth. Generate a secret with: python -c "import secrets; print(secrets.token_urlsafe(48))" ---
    jwt_secret: str = ""  # required; the app refuses to start without a strong one
    jwt_expire_minutes: int = 60

    # --- Tracing (optional): https://cloud.langfuse.com free tier ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.price_input_per_m + completion_tokens * self.price_output_per_m) / 1_000_000


settings = Settings()
