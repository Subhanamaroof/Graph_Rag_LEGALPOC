"""Request and response schemas for answer generation (Story 1.9).

An answer is built only from Story 1.8 hybrid retrieval evidence. Every citation is a
``(document_id, chunk_id)`` pair that was actually retrieved; the response returns the answer,
the validated claims, the evidence (sources) and the diagnostics of both stages.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from graph_rag.retrieval.graph_retrieval_models import (
    MAX_HOPS,
    MAX_QUESTION_LENGTH,
    GraphEdge,
    GraphNode,
    GraphResult,
    MatchedEntity,
)
from graph_rag.retrieval.graph_retrieval_models import RetrievalStatus as GraphRetrievalStatus
from graph_rag.retrieval.hybrid_retrieval_models import (
    MAX_FINAL_TOP_K,
    BranchStatus,
    GraphContext,
    HybridRetrievalStatus,
    RetrievalSource,
    UnresolvedReference,
)

AnswerStatus = Literal[
    "success", "partial", "insufficient_evidence", "retrieval_failed", "generation_failed"
]
InvalidCitationReason = Literal["missing_document_id", "missing_chunk_id", "not_in_evidence"]
RejectedClaimReason = Literal["no_citations", "all_citations_invalid"]
RetrievalOrigin = Literal["vector", "graph", "both"]


class AnswerRequest(BaseModel):
    """``top_k`` and ``max_hops`` are passed to Story 1.8 as ``final_top_k`` and ``max_hops``;
    ``null`` uses the configured hybrid defaults (``HYBRID_FINAL_TOP_K``,
    ``GRAPH_RETRIEVAL_MAX_HOPS``)."""

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    top_k: int | None = Field(default=None, ge=1, le=MAX_FINAL_TOP_K)
    max_hops: int | None = Field(default=None, ge=1, le=MAX_HOPS)
    include_sources: bool = True

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty or whitespace")
        return value


class Citation(BaseModel):
    document_id: str
    chunk_id: str


class Claim(BaseModel):
    """A factual statement from the answer and the retrieved chunks that support it."""

    text: str
    citations: list[Citation] = Field(min_length=1)


class InvalidCitation(BaseModel):
    """A citation the LLM produced that was removed (never replaced)."""

    document_id: str | None
    chunk_id: str | None
    reason: InvalidCitationReason
    location: Literal["claim", "answer_text"]


class RejectedClaim(BaseModel):
    """A claim dropped because no valid citation supports it."""

    text: str
    reason: RejectedClaimReason


class Source(BaseModel):
    """One evidence chunk that was given to the LLM, exactly as Story 1.8 returned it."""

    source_number: int  # [SOURCE n] label used in the prompt
    document_id: str
    chunk_id: str
    rank: int  # Story 1.8 hybrid rank (1 = best)
    retrieval_sources: list[RetrievalSource]
    retrieval_source: Literal["vector", "graph", "vector_and_graph"]
    rrf_score: float
    vector_rank: int | None = None
    graph_rank: int | None = None
    content: str  # original chunk text from ChromaDB, unchanged
    metadata: dict[str, Any] = Field(default_factory=dict)
    cited: bool = False  # referenced by at least one validated claim


class RetrievalSummary(BaseModel):
    """Counts and diagnostics copied from the Story 1.8 response and context preparation."""

    retrieval_status: HybridRetrievalStatus | None = None  # None: retrieval raised
    vector_results: int = 0
    graph_results: int = 0
    found_by_both: int = 0
    merged_results: int = 0  # distinct chunks after the Story 1.8 merge, before its top-k
    final_evidence_count: int = 0  # chunks returned by Story 1.8
    evidence_used: int = 0  # chunks placed in the LLM context
    duplicates_removed: int = 0
    excluded_missing_identifier: int = 0
    excluded_empty_content: int = 0
    excluded_over_budget: int = 0
    unresolved_references: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class GenerationMetadata(BaseModel):
    model: str | None = None
    llm_called: bool = False
    context_characters: int = 0
    citations_validated: bool = False  # True once validation ran on the LLM output
    claims_returned: int = 0  # claims produced by the LLM
    claims_accepted: int = 0
    invalid_citations: list[InvalidCitation] = Field(default_factory=list)
    rejected_claims: list[RejectedClaim] = Field(default_factory=list)


# --- Retrieval details (Story 1.9.1) ---------------------------------------------------------
# Everything below is copied from the Story 1.8 response of the same request: nothing is
# re-retrieved, re-scored or generated. Optional values are ``None`` when the source lacks them.


class RetrievalChunkDetail(BaseModel):
    """One distinct chunk of the Story 1.8 merge, keyed by ``(document_id, chunk_id)``.

    The same chunk can appear in several lists (e.g. ``vector_results`` and
    ``hybrid_results``); its values are identical in each.
    """

    document_id: str | None
    chunk_id: str
    case_id: str | None = None
    title: str | None = None
    court: str | None = None
    date: str | None = None
    text_preview: str  # first ``text_preview_chars`` characters of the original chunk text
    text_length: int  # length of the full chunk text
    text_truncated: bool
    retrieval_origin: RetrievalOrigin
    hybrid_rank: int  # position in the full Story 1.8 RRF ranking (1 = best)
    rrf_score: float
    in_final_results: bool  # within Story 1.8 final_top_k, i.e. offered to the context builder
    vector_rank: int | None = None
    vector_score: float | None = None  # Story 1.4 cosine similarity, unchanged
    vector_distance: float | None = None  # raw ChromaDB cosine distance
    graph_rank: int | None = None  # Story 1.8 position in the graph-associated chunk list
    graph_context: GraphContext | None = None  # only in graph_results and graph_only_chunks


class FusionSummary(BaseModel):
    """Counts over the full Story 1.8 merge (before final_top_k).

    ``hybrid_count = vector_only_count + graph_only_count + both_count``;
    ``overlap_count`` equals ``both_count`` (chunks found by both branches).
    """

    vector_count: int = 0
    graph_count: int = 0  # graph-associated chunks resolved in ChromaDB
    hybrid_count: int = 0
    overlap_count: int = 0
    vector_only_count: int = 0
    graph_only_count: int = 0
    both_count: int = 0
    final_results_count: int = 0  # chunks inside final_top_k
    rrf_k: int | None = None


class ClaimCitationDetail(BaseModel):
    """A validated citation and the evidence source it points to."""

    document_id: str
    chunk_id: str
    source_number: int  # [SOURCE n] label of the chunk in the LLM context
    hybrid_rank: int
    retrieval_origin: RetrievalOrigin


class ClaimEvidence(BaseModel):
    """A claim returned by the LLM and its validated citations. A rejected claim (no valid
    citation) has ``supported=False``, no citations and its rejection reason."""

    claim_text: str
    supported: bool
    citations: list[ClaimCitationDetail] = Field(default_factory=list)
    rejection_reason: RejectedClaimReason | None = None


class RetrievalDetails(BaseModel):
    vector_status: BranchStatus
    graph_status: BranchStatus
    graph_retrieval_status: GraphRetrievalStatus | None = None  # Story 1.7 status, if it ran
    vector_results: list[RetrievalChunkDetail] = Field(default_factory=list)  # vector rank order
    graph_results: list[RetrievalChunkDetail] = Field(default_factory=list)  # graph rank order
    hybrid_results: list[RetrievalChunkDetail] = Field(default_factory=list)  # RRF order
    graph_only_chunks: list[RetrievalChunkDetail] = Field(default_factory=list)
    graph_paths: list[GraphResult] = Field(default_factory=list)  # Story 1.7 ranked paths
    graph_entities: list[GraphNode] = Field(default_factory=list)  # nodes of those paths
    graph_relationships: list[GraphEdge] = Field(default_factory=list)  # edges of those paths
    matched_entities: list[MatchedEntity] = Field(default_factory=list)
    unresolved_references: list[UnresolvedReference] = Field(default_factory=list)
    claim_citations: list[ClaimEvidence] = Field(default_factory=list)
    fusion_summary: FusionSummary = Field(default_factory=FusionSummary)
    max_items_per_list: int  # RETRIEVAL_DETAILS_MAX_ITEMS
    text_preview_chars: int  # RETRIEVAL_DETAILS_PREVIEW_CHARS
    truncated_lists: list[str] = Field(default_factory=list)  # lists cut at max_items_per_list


class AnswerResponse(BaseModel):
    question: str
    status: AnswerStatus
    answer: str | None = None  # None whenever no grounded answer could be produced
    claims: list[Claim] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    retrieval_summary: RetrievalSummary
    generation_metadata: GenerationMetadata
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    # Story 1.9.1. None only when hybrid retrieval raised and produced no response.
    retrieval_details: RetrievalDetails | None = None
