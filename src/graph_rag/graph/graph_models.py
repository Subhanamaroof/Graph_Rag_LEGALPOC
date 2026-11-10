"""Request/response schemas for Neo4j graph construction (Story 1.6)."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from graph_rag.graph.models import MAX_CHUNKS_PER_REQUEST

BuildStatus = Literal["completed", "partial", "failed"]


class GraphInitializeResponse(BaseModel):
    status: Literal["initialized"]
    server_agent: str
    database: str
    constraints: list[str]
    indexes: list[str]


class GraphBuildRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=200)
    max_chunks: int | None = Field(default=None, ge=1, le=MAX_CHUNKS_PER_REQUEST)

    @field_validator("document_id")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or whitespace")
        return value.strip()


class ChunkWriteResult(BaseModel):
    """Counts read from Neo4j's own result counters for one chunk's transaction."""

    nodes_created: int = 0
    nodes_merged: int = 0  # node rows that matched an already existing node
    relationships_created: int = 0
    relationships_merged: int = 0
    relationships_skipped: int = 0  # not written (both endpoints resolved to the same node)


class FailedChunk(BaseModel):
    chunk_id: str
    stage: Literal["extraction", "graph"]
    error: str


class GraphBuildResponse(BaseModel):
    """``completed`` only if every chunk of the document was extracted and written."""

    document_id: str
    status: BuildStatus
    total_chunks: int
    chunks_processed: int
    chunks_succeeded: int
    chunks_failed: int
    chunks_skipped: int  # beyond max_chunks; not processed
    entities_extracted: int
    nodes_created: int
    nodes_merged: int
    relationships_extracted: int
    relationships_created: int
    relationships_merged: int
    relationships_skipped: int
    failed_chunks: list[FailedChunk] = Field(default_factory=list)
