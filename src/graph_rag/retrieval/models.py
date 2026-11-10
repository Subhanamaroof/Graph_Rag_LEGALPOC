"""Request and response schemas for vector retrieval (Story 1.4, ChromaDB since 1.3.2)."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

DEFAULT_TOP_K = 5
MAX_TOP_K = 20  # keeps responses small; enough context for a POC prompt
MAX_QUERY_LENGTH = 2000  # characters; the model only reads the first ~256 tokens anyway


class VectorSearchRequest(BaseModel):
    """A natural-language question and how many chunks to return."""

    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty or whitespace")
        return value


class RetrievedChunk(BaseModel):
    """One chunk found by vector search. The embedding vector is never included.

    ``document_id`` is the stored ``case_id`` (the legal judgment the chunk belongs to);
    it is ``None`` only if a record has no ``case_id``. ``chunk_id`` is the ChromaDB record ID.
    """

    document_id: str | None
    chunk_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float  # cosine similarity = 1 - distance; a relevance score, not a probability
    distance: float  # raw ChromaDB cosine distance (smaller = closer)


class VectorSearchResponse(BaseModel):
    """Results of one vector search, ordered best match first."""

    query: str
    total_results: int
    results: list[RetrievedChunk] = Field(default_factory=list)
