"""Centralized application configuration.

All settings are read from environment variables (or a local ``.env`` file)
using Pydantic Settings. Credentials are optional so the application can start
without Neo4j or Groq being configured. No connections are made here.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed view over the environment variables listed in ``.env.example``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # treat `KEY=` as "not set" so defaults apply
        extra="ignore",
    )

    # Application
    app_env: str = "development"
    app_name: str = "Graph RAG POC"
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    # ChromaDB (vector store, Story 1.3.2): an existing local database. The judgments
    # collection is read-only; newly ingested documents go to a separate collection.
    chroma_persist_directory: Path = Path("data/chroma")
    chroma_collection_name: str = "judgments"
    chroma_new_documents_collection: str = "new_documents"

    # Neo4j (graph store)
    neo4j_uri: str | None = None
    neo4j_username: str | None = None
    neo4j_password: SecretStr | None = None
    neo4j_database: str = "neo4j"

    # Groq (LLM)
    groq_api_key: SecretStr | None = None
    groq_model_name: str | None = None
    groq_timeout_seconds: float = Field(default=60.0, gt=0)

    # Entity/relationship extraction (Story 1.5): chunks processed per document request.
    extraction_max_chunks_per_document: int = Field(default=20, gt=0)

    # Full-corpus graph build (Story 1.6.2). Sequential; documents are read in batches, and
    # every Groq request waits at least GRAPH_BUILD_DELAY_SECONDS after the previous one.
    graph_build_batch_size: int = Field(default=10, gt=0)
    graph_build_delay_seconds: float = Field(default=2.0, ge=0)
    graph_build_max_retries: int = Field(default=3, ge=0)
    graph_build_retry_delay_seconds: float = Field(default=5.0, ge=0)
    graph_build_max_consecutive_failures: int = Field(default=5, gt=0)
    graph_build_jobs_db_path: Path = Path("data/graph_build_jobs.sqlite3")

    # Graph retrieval (Story 1.7): server-side timeout for each Neo4j read transaction.
    graph_retrieval_query_timeout_seconds: float = Field(default=10.0, gt=0)

    # Hybrid retrieval (Story 1.8): defaults for request values left out. Upper bounds match
    # the request limits (vector top-k <= 20, graph paths <= 100, max hops <= 3).
    vector_retrieval_top_k: int = Field(default=10, ge=1, le=20)
    graph_retrieval_top_k: int = Field(default=10, ge=1, le=100)
    graph_retrieval_max_hops: int = Field(default=2, ge=1, le=3)
    hybrid_final_top_k: int = Field(default=10, ge=1, le=100)
    hybrid_rrf_k: int = Field(default=60, ge=1)
    hybrid_retrieval_timeout_seconds: float = Field(default=15.0, gt=0)

    # Answer generation (Story 1.9): character budget for the evidence block sent to Groq.
    # ~12,000 characters is roughly 3,000 tokens: ten 800-character chunks plus headers fit,
    # far below the context window of current Groq models, leaving room for the answer.
    generation_max_context_chars: int = Field(default=12000, ge=1000, le=100000)

    # Retrieval details in the answer response (Story 1.9.1): each list (vector, graph, hybrid,
    # graph-only chunks, paths, entities, ...) is capped at this many items, and each chunk is
    # shown as a preview of this many characters. Counts are always computed on full sets.
    retrieval_details_max_items: int = Field(default=50, ge=1, le=500)
    retrieval_details_preview_chars: int = Field(default=400, ge=50, le=5000)

    # Embeddings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = Field(default=384, gt=0)
    embedding_batch_size: int = Field(default=32, gt=0)

    # Ingestion (Story 1.2). Chunk values are character counts, not tokens.
    documents_directory: Path = Path("data/documents")
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)

    @model_validator(mode="after")
    def _check_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the single, cached Settings instance used across the application."""
    return Settings()


settings = get_settings()
