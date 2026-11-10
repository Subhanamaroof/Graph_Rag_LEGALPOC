"""Normalize, merge and de-duplicate vector and graph results (Story 1.8).

Both branches are turned into the same internal shape, ``FusedChunk``, keyed by the stable
chunk identity ``(document_id, chunk_id)``:

- vector chunks keep their 1-based position, cosine similarity and distance (Story 1.4);
- graph chunks are the resolved source chunks of the ranked graph paths. Their graph rank is
  the order in which the best-ranked paths first cite them; each keeps the paths, nodes,
  relationships and evidence that point to it.

A chunk found by both branches becomes ONE entry carrying both ranks and the graph context.
No database access happens here.
"""

from dataclasses import dataclass, field
from typing import Any

from graph_rag.retrieval.graph_retrieval_models import GraphResult
from graph_rag.retrieval.hybrid_retrieval_models import (
    GraphContext,
    GraphPathReference,
    RetrievalSource,
)
from graph_rag.retrieval.models import VectorSearchResponse
from graph_rag.retrieval.source_resolver import ChunkKey, SourceChunk


@dataclass
class FusedChunk:
    document_id: str | None
    chunk_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector_rank: int | None = None
    vector_score: float | None = None
    vector_distance: float | None = None
    graph_rank: int | None = None
    graph_context: GraphContext | None = None

    @property
    def key(self) -> ChunkKey:
        return (self.document_id, self.chunk_id)

    @property
    def sources(self) -> list[RetrievalSource]:
        sources: list[RetrievalSource] = []
        if self.vector_rank is not None:
            sources.append("vector")
        if self.graph_rank is not None:
            sources.append("graph")
        return sources


def normalize_vector_results(response: VectorSearchResponse) -> list[FusedChunk]:
    """Vector hits in their returned order (best first), ranked from 1. A malformed hit
    (no chunk ID) or a repeated chunk is skipped so ranks stay consistent."""
    chunks: list[FusedChunk] = []
    seen: set[ChunkKey] = set()
    for hit in response.results:
        key = (hit.document_id, hit.chunk_id)
        if not hit.chunk_id or key in seen:
            continue
        seen.add(key)
        chunks.append(
            FusedChunk(
                document_id=hit.document_id,
                chunk_id=hit.chunk_id,
                content=hit.text,
                metadata=hit.metadata,
                vector_rank=len(chunks) + 1,
                vector_score=hit.score,
                vector_distance=hit.distance,
            )
        )
    return chunks


def _add_unique(items: list, item, identity) -> None:
    if all(identity(existing) != identity(item) for existing in items):
        items.append(item)


def normalize_graph_results(
    results: list[GraphResult], resolved: dict[ChunkKey, SourceChunk]
) -> list[FusedChunk]:
    """Graph-associated chunks ordered by the best graph path that cites them, ranked from 1.

    Only resolved chunks are returned (an unresolved reference has no original text to show).
    The graph context of a chunk holds every path citing it, all nodes of those paths, and only
    the relationships and evidence whose provenance is that chunk.
    """
    by_key: dict[ChunkKey, FusedChunk] = {}
    for result in sorted(results, key=lambda r: r.rank):
        edges = {edge.relationship_id: edge for edge in result.relationships}
        cited = [item for item in result.evidence if (item.document_id, item.chunk_id) in resolved]
        for item in cited:
            key = (item.document_id, item.chunk_id)
            chunk = by_key.get(key)
            if chunk is None:
                source = resolved[key]
                chunk = FusedChunk(
                    document_id=source.document_id,
                    chunk_id=source.chunk_id,
                    content=source.content,
                    metadata=source.metadata,
                    graph_rank=len(by_key) + 1,
                    graph_context=GraphContext(),
                )
                by_key[key] = chunk
            context = chunk.graph_context
            _add_unique(
                context.paths,
                GraphPathReference(
                    graph_rank=result.rank,
                    graph_score=result.score,
                    query_entity=result.query_entity,
                    matched_entity=result.matched_entity,
                    related_node=result.related_node,
                    path=result.path,
                    hop_distance=result.hop_distance,
                ),
                lambda p: p.graph_rank,
            )
            for node in result.nodes:
                _add_unique(context.entities, node, lambda n: n.entity_id)
            _add_unique(context.evidence, item, lambda e: e.relationship_id)
            # Parallel edges (other chunks) are not in ``relationships``; their evidence is.
            if item.relationship_id in edges:
                _add_unique(
                    context.relationships, edges[item.relationship_id], lambda e: e.relationship_id
                )
    return list(by_key.values())


def merge_results(vector: list[FusedChunk], graph: list[FusedChunk]) -> list[FusedChunk]:
    """One entry per chunk key. A chunk in both lists keeps its vector fields and gains the
    graph rank and graph context; the text is the same ChromaDB record either way."""
    merged: dict[ChunkKey, FusedChunk] = {chunk.key: chunk for chunk in vector}
    for chunk in graph:
        existing = merged.get(chunk.key)
        if existing is None:
            merged[chunk.key] = chunk
        else:
            existing.graph_rank = chunk.graph_rank
            existing.graph_context = chunk.graph_context
    return list(merged.values())
