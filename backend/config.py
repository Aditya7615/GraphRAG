"""Central configuration. Every tunable knob lives here and is env-overridable."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM (Google Gemini) ------------------------------------------------
    google_api_key: str = ""
    llm_model: str = "gemini-2.0-flash"
    # Temperature is pinned to 0.0 for deterministic, extractive answers.
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_seconds: int = 60

    # --- Embeddings ---------------------------------------------------------
    # Local sentence-transformers model -> no API key, no rate limits, low latency.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_device: str = "cpu"
    embedding_batch_size: int = 64
    # bge models want this prefix on the *query* side only.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "

    # --- Qdrant -------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "enterprise_rag"
    qdrant_timeout: int = 60
    # HNSW / quantization tuning for the 100k+ document target.
    qdrant_hnsw_m: int = 32
    qdrant_hnsw_ef_construct: int = 256
    qdrant_search_hnsw_ef: int = 128
    qdrant_use_scalar_quantization: bool = True
    qdrant_on_disk_payload: bool = True

    # --- Chunking -----------------------------------------------------------
    parent_chunk_tokens: int = 2000
    parent_chunk_overlap_tokens: int = 200
    child_chunk_tokens: int = 400
    child_chunk_overlap_tokens: int = 60
    tokenizer_encoding: str = "cl100k_base"

    # --- Retrieval ----------------------------------------------------------
    dense_prefetch_limit: int = 50
    sparse_prefetch_limit: int = 50
    # Number of child hits kept after Reciprocal Rank Fusion.
    retrieval_top_k: int = 12
    # Number of unique parent documents finally sent to the LLM.
    max_parents_in_context: int = 4
    max_context_tokens: int = 6000
    # Fused RRF scores below this are dropped before they can pollute context.
    min_relevance_score: float = 0.0

    # --- Document store -----------------------------------------------------
    # "auto" tries Postgres and transparently falls back to the JSON store.
    docstore_backend: str = "auto"  # auto | postgres | json
    postgres_dsn: str = "postgresql+psycopg2://rag:rag@localhost:5432/ragdb"
    json_docstore_path: Path = ROOT_DIR / "data" / "docstore" / "parents.json"

    # --- Ingestion ----------------------------------------------------------
    synthetic_data_dir: Path = ROOT_DIR / "data" / "synthetic"
    upload_dir: Path = ROOT_DIR / "data" / "uploads"
    bm25_stats_path: Path = ROOT_DIR / "data" / "docstore" / "bm25_stats.json"
    ingest_workers: int = 4

    # --- API / UI -----------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"
    cors_origins: str = "*"
    log_level: str = "INFO"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    for p in (s.json_docstore_path.parent, s.synthetic_data_dir, s.upload_dir):
        p.mkdir(parents=True, exist_ok=True)
    return s


settings = get_settings()
