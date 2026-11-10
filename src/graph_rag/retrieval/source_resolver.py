"""Resolve graph source references to original ChromaDB chunks (Story 1.8).

Neo4j stores graph knowledge and, on every relationship, the ``source_document_id`` and
``source_chunk_id`` it was extracted from. ChromaDB stores the chunk text. A *source
reference* is the pair ``(document_id, chunk_id)``:

- ``chunk_id`` is the ChromaDB record ID (``<case_id>_chunk<n>``), so it is looked up by ID;
- ``document_id`` is the record's ``case_id`` metadata, and must match the stored value.

A reference is only accepted when both IDs agree with what ChromaDB holds. Missing, unknown or
mismatched references are reported, never turned into invented chunks. Read-only: this module
only calls ``chroma_store.get_chunks``.
"""

from dataclasses import dataclass, field
from typing import Any

from chromadb.api.models.Collection import Collection

from graph_rag.retrieval.graph_retrieval_models import GraphResult
from graph_rag.retrieval.hybrid_retrieval_models import UnresolvedReference
from graph_rag.retrieval.vector_retriever import DOCUMENT_ID_FIELD
from graph_rag.vector_store.chroma_store import get_chunks

ChunkKey = tuple[str | None, str]  # (document_id, chunk_id): the stable identity of a chunk


@dataclass(frozen=True)
class SourceReference:
    document_id: str
    chunk_id: str
    relationship_id: str  # first relationship that cited this chunk
    graph_rank: int  # rank of the first graph result that cited it

    @property
    def key(self) -> ChunkKey:
        return (self.document_id, self.chunk_id)


@dataclass
class ExtractedReferences:
    references: list[SourceReference] = field(default_factory=list)  # distinct, best rank first
    incomplete: list[UnresolvedReference] = field(default_factory=list)
    duplicates_removed: int = 0


@dataclass(frozen=True)
class SourceChunk:
    document_id: str
    chunk_id: str
    content: str
    metadata: dict[str, Any]

    @property
    def key(self) -> ChunkKey:
        return (self.document_id, self.chunk_id)


def extract_source_references(results: list[GraphResult]) -> ExtractedReferences:
    """Collect every evidence ``(document_id, chunk_id)`` in graph rank order, de-duplicated.

    A path's evidence covers its own edges and parallel edges from other chunks, so one path
    can cite several chunks and one chunk can support several paths. Each pair is kept once,
    at the position where the best-ranked path first cites it.
    """
    extracted = ExtractedReferences()
    seen: set[ChunkKey] = set()
    for result in sorted(results, key=lambda r: r.rank):
        for item in result.evidence:
            if not item.document_id or not item.chunk_id:
                extracted.incomplete.append(
                    UnresolvedReference(
                        document_id=item.document_id,
                        chunk_id=item.chunk_id,
                        relationship_id=item.relationship_id,
                        graph_rank=result.rank,
                        reason="missing_chunk_id" if item.document_id else "missing_document_id",
                    )
                )
                continue
            key = (item.document_id, item.chunk_id)
            if key in seen:
                extracted.duplicates_removed += 1
                continue
            seen.add(key)
            extracted.references.append(
                SourceReference(
                    document_id=item.document_id,
                    chunk_id=item.chunk_id,
                    relationship_id=item.relationship_id,
                    graph_rank=result.rank,
                )
            )
    return extracted


def _unresolved(reference: SourceReference, reason: str) -> UnresolvedReference:
    return UnresolvedReference(
        document_id=reference.document_id,
        chunk_id=reference.chunk_id,
        relationship_id=reference.relationship_id,
        graph_rank=reference.graph_rank,
        reason=reason,
    )


def resolve_source_references(
    collection: Collection, references: list[SourceReference]
) -> tuple[dict[ChunkKey, SourceChunk], list[UnresolvedReference]]:
    """Fetch the referenced chunks by ID in one read and validate their ``case_id``.

    Returns ``(resolved chunks by key, unresolved references)``. Raises ``VectorStoreError``
    if ChromaDB cannot be read; the caller decides how to report that.
    """
    if not references:
        return {}, []
    chunk_ids = list(dict.fromkeys(reference.chunk_id for reference in references))
    records = {record["id"]: record for record in get_chunks(collection, ids=chunk_ids)}

    resolved: dict[ChunkKey, SourceChunk] = {}
    unresolved: list[UnresolvedReference] = []
    for reference in references:
        record = records.get(reference.chunk_id)
        if record is None:
            unresolved.append(_unresolved(reference, "not_found"))
            continue
        stored_document_id = record["metadata"].get(DOCUMENT_ID_FIELD)
        if stored_document_id is None or str(stored_document_id) != reference.document_id:
            unresolved.append(_unresolved(reference, "document_mismatch"))
            continue
        resolved[reference.key] = SourceChunk(
            document_id=reference.document_id,
            chunk_id=reference.chunk_id,
            content=record["document"],
            metadata=record["metadata"],
        )
    return resolved, unresolved


def lookup_failed(references: list[SourceReference]) -> list[UnresolvedReference]:
    """Every reference, marked unresolved because ChromaDB could not be read."""
    return [_unresolved(reference, "lookup_failed") for reference in references]
