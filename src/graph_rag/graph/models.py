"""Schemas for entity and relationship extraction (Story 1.5).

Two layers:

- ``Raw*`` models describe what the LLM is *asked* to return. Types are plain strings so a
  single bad item can be rejected (with a warning) without discarding the whole chunk.
- ``Entity`` / ``Relationship`` / ``ChunkExtraction`` are the validated output. Types are
  enums, provenance is mandatory and relationships must point at entities of the same result.
  This is the shape Story 1.6 will write to Neo4j.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EntityType(StrEnum):
    PERSON = "PERSON"
    COURT = "COURT"
    ORGANIZATION = "ORGANIZATION"
    CASE = "CASE"
    STATUTE = "STATUTE"
    LEGAL_SECTION = "LEGAL_SECTION"
    DATE = "DATE"
    LOCATION = "LOCATION"
    LEGAL_CONCEPT = "LEGAL_CONCEPT"


class RelationshipType(StrEnum):
    JUDGE_PRESIDED_OVER = "JUDGE_PRESIDED_OVER"
    PETITIONER_IN = "PETITIONER_IN"
    RESPONDENT_IN = "RESPONDENT_IN"
    ADVOCATE_IN = "ADVOCATE_IN"
    DECIDED_BY = "DECIDED_BY"
    CITES = "CITES"
    REFERS_TO = "REFERS_TO"
    APPLIES = "APPLIES"
    INTERPRETS = "INTERPRETS"
    RELATED_TO = "RELATED_TO"


NonBlank = Annotated[str, Field(min_length=1)]


# --- What the LLM returns ------------------------------------------------------


class RawEntity(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    id: NonBlank  # local reference such as "e1", only meaningful inside one response
    name: NonBlank
    entity_type: NonBlank
    normalized_name: str | None = None
    description: str | None = None
    evidence: str | None = None


class RawRelationship(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    source: NonBlank  # local entity id
    target: NonBlank
    relationship_type: NonBlank
    description: str | None = None
    evidence: str | None = None


class RawExtraction(BaseModel):
    """Top-level LLM object. Items stay dicts so each one can be validated separately."""

    model_config = ConfigDict(extra="ignore")

    entities: list[dict]
    relationships: list[dict]


# --- Validated output ----------------------------------------------------------


class Entity(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    entity_id: NonBlank  # deterministic: same chunk + type + normalized name -> same id
    name: NonBlank  # the name exactly as written in the chunk (first mention)
    normalized_name: NonBlank
    entity_type: EntityType
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)  # other spellings merged into this entity
    source_document_id: NonBlank
    source_chunk_id: NonBlank
    source_text: NonBlank  # exact supporting quote from the chunk


class Relationship(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    relationship_id: NonBlank
    source_entity_id: NonBlank
    target_entity_id: NonBlank
    relationship_type: RelationshipType
    description: str | None = None
    evidence: NonBlank  # exact supporting quote from the chunk
    source_document_id: NonBlank
    source_chunk_id: NonBlank

    @model_validator(mode="after")
    def _no_self_loop(self) -> "Relationship":
        if self.source_entity_id == self.target_entity_id:
            raise ValueError("a relationship must connect two different entities")
        return self


ChunkStatus = Literal["completed", "failed"]
ExtractionStatus = Literal["completed", "partial", "failed"]


class ChunkExtraction(BaseModel):
    """Extraction result for one chunk. ``warnings`` lists LLM items that were rejected."""

    document_id: str
    chunk_id: str
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    extraction_status: ChunkStatus
    error: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "ChunkExtraction":
        entity_ids = [entity.entity_id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("duplicate entity_id in one chunk")
        for item in [*self.entities, *self.relationships]:
            if (item.source_document_id, item.source_chunk_id) != (self.document_id, self.chunk_id):
                raise ValueError("entity or relationship provenance does not match the chunk")
        known = set(entity_ids)
        for relationship in self.relationships:
            if {relationship.source_entity_id, relationship.target_entity_id} - known:
                raise ValueError(
                    f"relationship {relationship.relationship_id} references an unknown entity"
                )
        if self.extraction_status == "failed" and (self.entities or self.relationships):
            raise ValueError("a failed chunk cannot carry entities or relationships")
        if self.extraction_status == "failed" and not self.error:
            raise ValueError("a failed chunk must explain its error")
        return self


# --- API ---------------------------------------------------------------------

MAX_CHUNKS_PER_REQUEST = 250  # above the largest judgment (203 chunks)


class ExtractionRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=200)
    chunk_id: str | None = Field(default=None, min_length=1, max_length=300)
    max_chunks: int | None = Field(default=None, ge=1, le=MAX_CHUNKS_PER_REQUEST)

    @field_validator("document_id", "chunk_id")
    @classmethod
    def _not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be empty or whitespace")
        return value.strip() if value is not None else None


class ExtractionResponse(BaseModel):
    """Per-chunk results plus counters. ``completed`` only if every chunk succeeded."""

    document_id: str
    extraction_status: ExtractionStatus
    total_chunks: int  # chunks selected by the request (1 for a single chunk)
    chunks_processed: int
    chunks_succeeded: int
    chunks_failed: int
    chunks_skipped: int  # beyond max_chunks; not extracted
    total_entities: int
    total_relationships: int
    results: list[ChunkExtraction] = Field(default_factory=list)
