"""Centralized frontend configuration.

Only the backend URL and HTTP limits live here: Groq, Neo4j and ChromaDB credentials belong to
the backend and are never read by the frontend.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# The answer API accepts top_k in 1..100 (Story 1.9 AnswerRequest); the UI offers a smaller,
# safe range so a single question cannot request an oversized LLM context.
MAX_UI_TOP_K = 20


class FrontendSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GRAPH_RAG_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    api_base_url: str = "http://localhost:8000"
    health_timeout_seconds: float = 5.0
    # Retrieval (~15 s) plus one Groq call (~60 s) with headroom. Never retried automatically.
    answer_timeout_seconds: float = 120.0
    default_top_k: int = 10  # matches the backend default HYBRID_FINAL_TOP_K


@lru_cache
def get_settings() -> FrontendSettings:
    return FrontendSettings()
