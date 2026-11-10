"""Request and response schemas for hybrid retrieval (Story 1.8).

Hybrid retrieval returns ranked, source-grounded ChromaDB chunks found by vector search, by
graph retrieval, or by both, never a generated answer. Graph data types (nodes, edges,
evidence, matched entities) are the Story 1.7 models, reused unchanged.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from graph_rag.graph.models import EntityType, RelationshipType
from graph_rag.retrieval.graph_retrieval_models import (
    MAX_HOPS,
    MAX_LIMIT,
    MAX_QUESTION_LENGTH,
    AnalysisMethod,
    Evidence,
    GraphEdge,
    GraphNode,
    GraphResult,
    MatchedEntity,
    QueryEntity,
)
from graph_rag.retrieval.graph_retrieval_models import RetrievalStatus as GraphRetrievalStatus
from graph_rag.retrieval.models import MAX_TOP_K as MAX_VECTOR_TOP_K

MAX_FINAL_TOP_K = 100

HybridRetrievalStatus = Literal["success", "partial", "no_results", "failed"]
RetrievalSource = Literal["vector", "graph"]
BranchStatus = Literal["completed", "failed"]
UnresolvedReason = Literal[
    "missing_document_id", "missing_chunk_id", "not_found", "document_mismatch", "lookup_failed"
]


class HybridRetrievalRequest(BaseModel):
    """Limits left out (``null``) use the configured defaults (``VECTOR_RETRIEVAL_TOP_K``,
    ``GRAPH_RETRIEVAL_TOP_K``, ``HYBRID_FINAL_TOP_K``, ``GRAPH_RETRIEVAL_MAX_HOPS``).

    ``graph_top_k`` is the number of ranked graph *paths* (Story 1.7 ``limit``); the number of
    chunks they point to can be larger or smaller. The filters apply to the graph branch only.
    """

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    vector_top_k: int | None = Field(default=None, ge=1, le=MAX_VECTOR_TOP_K)
    graph_top_k: int | None = Field(default=None, ge=1, le=MAX_LIMIT)
    final_top_k: int | None = Field(default=None, ge=1, le=MAX_FINAL_TOP_K)
    max_hops: int | None = Field(default=None, ge=1, le=MAX_HOPS)
    entity_types: list[EntityType] | None = Field(default=None, min_length=1)
    relationship_types: list[RelationshipType] | None = Field(default=None, min_length=1)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty or whitespace")
        return value


class GraphPathReference(BaseModel):
    """A ranked Story 1.7 graph result whose evidence points to this chunk."""

    graph_rank: int  # rank of the path in the graph results (1 = best)
    graph_score: float  # Story 1.7 path score: an ordering signal, not legal relevance
    query_entity: str
    matched_entity: GraphNode
    related_node: GraphNode
    path: str
    hop_distance: int


class GraphContext(BaseModel):
    """The graph knowledge connected to one chunk: the paths that cite it, their nodes, and
    the relationships and evidence that were extracted from this very chunk."""

    paths: list[GraphPathReference] = Field(default_factory=list)
    entities: list[GraphNode] = Field(default_factory=list)
    relationships: list[GraphEdge] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class HybridResult(BaseModel):
    """One original ChromaDB chunk. Fields of a branch that did not find it are ``None``."""

    rank: int  # final position after RRF (1 = best)
    document_id: str | None
    chunk_id: str
    content: str  # original chunk text from ChromaDB
    metadata: dict[str, Any] = Field(default_factory=dict)
    retrieval_sources: list[RetrievalSource]
    rrf_score: float
    vector_rank: int | None = None
    vector_score: float | None = None  # cosine similarity (1 - distance), from Story 1.4
    vector_distance: float | None = None  # raw ChromaDB cosine distance
    graph_rank: int | None = None  # position in the graph-associated chunk list
    graph_context: GraphContext | None = None


class UnresolvedReference(BaseModel):
    """A graph source reference that did not yield an original chunk. Nothing is invented."""

    document_id: str | None
    chunk_id: str | None
    relationship_id: str
    graph_rank: int
    reason: UnresolvedReason


class VectorBranchSummary(BaseModel):
    status: BranchStatus
    results_count: int = 0
    error: str | None = None


class GraphBranchSummary(BaseModel):
    status: BranchStatus
    graph_retrieval_status: GraphRetrievalStatus | None = None  # Story 1.7 status, unchanged
    analysis_method: AnalysisMethod | None = None
    identified_entities: list[QueryEntity] = Field(default_factory=list)
    matched_entities: list[MatchedEntity] = Field(default_factory=list)
    unmatched_entities: list[QueryEntity] = Field(default_factory=list)
    paths_count: int = 0
    source_references_count: int = 0  # distinct (document_id, chunk_id) pairs
    duplicate_references_removed: int = 0
    resolved_chunks_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class HybridRetrievalResponse(BaseModel):
    question: str
    retrieval_status: HybridRetrievalStatus
    total_results: int  # results returned (after final_top_k)
    vector_results_count: int
    graph_results_count: int  # graph-associated chunks resolved in ChromaDB
    merged_results_count: int  # distinct chunks after de-duplication, before final_top_k
    rrf_k: int
    results: list[HybridResult] = Field(default_factory=list)
    vector: VectorBranchSummary
    graph: GraphBranchSummary
    unresolved_references: list[UnresolvedReference] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    # Internal details kept for Story 1.9.1 explainability. Excluded from serialization, so the
    # /v1/retrieval/hybrid JSON is unchanged; Story 1.9 reads them in-process.
    ranked_results: list[HybridResult] = Field(default_factory=list, exclude=True)
    # ^ the full RRF ranking of every merged chunk; ``results`` is its first final_top_k.
    graph_paths: list[GraphResult] = Field(default_factory=list, exclude=True)
    # ^ the ranked Story 1.7 graph results exactly as the graph branch returned them.
