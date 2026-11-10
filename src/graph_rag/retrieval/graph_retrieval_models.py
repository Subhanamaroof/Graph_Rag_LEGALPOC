"""Request and response schemas for graph retrieval (Story 1.7).

Graph retrieval returns graph facts (entities, relationships, paths) with their provenance,
never a generated answer. Entity and relationship types are the Story 1.5 enums, so a type
that does not exist in the graph is rejected at the API boundary.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from graph_rag.graph.models import EntityType, RelationshipType

DEFAULT_MAX_HOPS = 2
MAX_HOPS = 3
DEFAULT_LIMIT = 20
MAX_LIMIT = 100
MAX_QUESTION_LENGTH = 2000

RetrievalStatus = Literal["success", "partial", "no_match", "no_results", "failed"]
AnalysisMethod = Literal["deterministic", "llm"]
MatchType = Literal["exact_normalized", "exact_name", "alias", "partial"]


class GraphRetrievalRequest(BaseModel):
    """``entity_types`` filters the *related* nodes returned; ``relationship_types`` restricts
    which edges traversal may follow (every edge of a path must have one of these types)."""

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    max_hops: int = Field(default=DEFAULT_MAX_HOPS, ge=1, le=MAX_HOPS)
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)
    entity_types: list[EntityType] | None = Field(default=None, min_length=1)
    relationship_types: list[RelationshipType] | None = Field(default=None, min_length=1)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty or whitespace")
        return value


# --- Query understanding ---------------------------------------------------------------


class QueryEntity(BaseModel):
    """An entity named in the question. ``mention`` is the text as written in the question;
    ``name`` is what is searched for (equal to ``mention`` unless a known abbreviation such as
    "IPC" was expanded)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1)
    mention: str = Field(min_length=1)
    entity_type: EntityType | None = None


class QueryAnalysis(BaseModel):
    question: str  # the original question, unchanged
    entities: list[QueryEntity] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    relationship_intent: RelationshipType | None = None
    method: AnalysisMethod
    warnings: list[str] = Field(default_factory=list)


# --- Graph data --------------------------------------------------------------------------


class GraphNode(BaseModel):
    entity_id: str
    name: str
    entity_type: str | None = None
    description: str | None = None


class MatchedEntity(BaseModel):
    """A Neo4j node that matched a query entity. Provenance lists are capped (see counts)."""

    query_name: str
    entity_id: str
    name: str
    normalized_name: str | None = None
    entity_type: str | None = None
    match_type: MatchType
    ambiguous: bool = False  # True when the same query entity matched several nodes
    source_document_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    document_count: int = 0
    chunk_count: int = 0


class GraphEdge(BaseModel):
    """One stored relationship. ``source_entity_id -> target_entity_id`` is the stored
    direction, independent of the direction traversal walked it."""

    relationship_id: str
    relationship_type: str
    source_entity_id: str
    target_entity_id: str
    description: str | None = None
    evidence: str | None = None
    source_document_id: str | None = None
    source_chunk_id: str | None = None

    @property
    def provenance_complete(self) -> bool:
        return bool(self.evidence and self.source_document_id and self.source_chunk_id)


class Evidence(BaseModel):
    relationship_id: str
    relationship_type: str
    evidence: str | None
    document_id: str | None
    chunk_id: str | None


class GraphResult(BaseModel):
    rank: int = 0
    score: float = 0.0
    ranking_signals: dict[str, float] = Field(default_factory=dict)
    query_entity: str
    matched_entity: GraphNode
    related_node: GraphNode
    nodes: list[GraphNode]  # path order, matched entity first
    relationships: list[GraphEdge]  # one edge per hop, path order
    path: str  # readable path with stored edge directions
    hop_distance: int
    supporting_edges: int  # edges behind this path, incl. parallel edges from other chunks
    evidence: list[Evidence] = Field(default_factory=list)
    source_document_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    provenance_complete: bool = True


class GraphRetrievalResponse(BaseModel):
    question: str
    analysis_method: AnalysisMethod
    identified_entities: list[QueryEntity] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    relationship_intent: RelationshipType | None = None
    matched_entities: list[MatchedEntity] = Field(default_factory=list)
    unmatched_entities: list[QueryEntity] = Field(default_factory=list)
    ambiguous_entities: list[str] = Field(default_factory=list)
    results: list[GraphResult] = Field(default_factory=list)
    total_results: int = 0
    truncated: bool = False  # more distinct paths were found than ``limit``
    retrieval_status: RetrievalStatus
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
