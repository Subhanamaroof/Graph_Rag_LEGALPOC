"""Graph construction: write Story 1.5 extraction results into Neo4j (Story 1.6).

Per chunk, ONE write transaction does:

1. ``MERGE`` every node on its graph ``entity_id`` and append the chunk/document to its
   provenance lists (never duplicating an ID already there).
2. ``MATCH`` both endpoints of every relationship and ``MERGE`` the edge on its
   ``relationship_id``. If any endpoint is missing, the function raises and the driver rolls
   the whole chunk back, so no half-written chunk remains.

Re-running the same chunk changes no structure: every node and edge already exists and is
matched instead of created. Created/merged counts come from Neo4j's result counters.
"""

from chromadb.api.models.Collection import Collection

from graph_rag.core.llm import JSONCompletionClient
from graph_rag.graph.extraction_service import extract_document
from graph_rag.graph.graph_builder import build_chunk_rows
from graph_rag.graph.graph_models import (
    BuildStatus,
    ChunkWriteResult,
    FailedChunk,
    GraphBuildResponse,
)
from graph_rag.graph.graph_schema import ENTITY_LABEL, NODE_LABELS, RELATIONSHIP_TYPES
from graph_rag.graph.models import ChunkExtraction, ExtractionResponse
from graph_rag.graph.neo4j_client import GraphStoreError, Neo4jClient


class MissingEndpointError(GraphStoreError):
    """A relationship's source or target node does not exist; the chunk is rolled back."""


_ALLOWED_LABELS = frozenset(NODE_LABELS.values())
_ALLOWED_TYPES = frozenset(RELATIONSHIP_TYPES)


def node_merge_query(label: str) -> str:
    # Labels cannot be query parameters, so only whitelisted labels are formatted in.
    if label not in _ALLOWED_LABELS:
        raise ValueError(f"unsupported node label {label!r}")
    return f"""
UNWIND $rows AS row
MERGE (n:{ENTITY_LABEL} {{entity_id: row.entity_id}})
ON CREATE SET n.created_at = datetime(),
              n.name = row.name,
              n.normalized_name = row.normalized_name,
              n.entity_type = row.entity_type,
              n.identity_scope = row.identity_scope,
              n.source_text = row.source_text,
              n.aliases = [],
              n.source_document_ids = [],
              n.source_chunk_ids = []
SET n:{label},
    n.updated_at = datetime(),
    n.description = coalesce(n.description, row.description),
    n.aliases = n.aliases + [a IN row.aliases WHERE NOT a IN n.aliases],
    n.source_document_ids = CASE WHEN row.document_id IN n.source_document_ids
        THEN n.source_document_ids ELSE n.source_document_ids + row.document_id END,
    n.source_chunk_ids = CASE WHEN row.chunk_id IN n.source_chunk_ids
        THEN n.source_chunk_ids ELSE n.source_chunk_ids + row.chunk_id END
RETURN count(n) AS written
"""


def relationship_merge_query(rel_type: str) -> str:
    if rel_type not in _ALLOWED_TYPES:
        raise ValueError(f"unsupported relationship type {rel_type!r}")
    return f"""
UNWIND $rows AS row
MATCH (s:{ENTITY_LABEL} {{entity_id: row.source_id}})
MATCH (t:{ENTITY_LABEL} {{entity_id: row.target_id}})
MERGE (s)-[r:{rel_type} {{relationship_id: row.relationship_id}}]->(t)
ON CREATE SET r.created_at = datetime(),
              r.relationship_type = '{rel_type}',
              r.description = row.description,
              r.evidence = row.evidence,
              r.source_document_id = row.document_id,
              r.source_chunk_id = row.chunk_id
RETURN count(r) AS written
"""


def _run(tx, query: str, rows: list[dict]) -> tuple[int, object]:
    result = tx.run(query, rows=rows)
    record = result.single()
    counters = result.consume().counters
    return (record["written"] if record else 0), counters


def write_chunk_rows(tx, nodes: dict[str, list[dict]], relationships: dict[str, list[dict]]):
    """Transaction function: write one chunk's nodes, then its edges. Raising rolls back."""
    result = ChunkWriteResult()
    for label, rows in nodes.items():
        _, counters = _run(tx, node_merge_query(label), rows)
        result.nodes_created += counters.nodes_created
        result.nodes_merged += len(rows) - counters.nodes_created
    for rel_type, rows in relationships.items():
        written, counters = _run(tx, relationship_merge_query(rel_type), rows)
        if written != len(rows):
            raise MissingEndpointError(
                f"{len(rows) - written} {rel_type} relationship(s) reference a missing node"
            )
        result.relationships_created += counters.relationships_created
        result.relationships_merged += len(rows) - counters.relationships_created
    return result


def persist_chunk(client: Neo4jClient, chunk: ChunkExtraction) -> ChunkWriteResult:
    """Write one completed chunk atomically. Raises ``GraphStoreError`` on failure."""
    rows = build_chunk_rows(chunk)
    if not rows.nodes:
        return ChunkWriteResult()
    result = client.execute_write(
        write_chunk_rows, nodes=rows.nodes, relationships=rows.relationships
    )
    result.relationships_skipped = len(chunk.relationships) - rows.relationship_count
    return result


def _status(succeeded: int, failed: int, skipped: int) -> BuildStatus:
    if succeeded == 0:
        return "failed"
    return "partial" if failed or skipped else "completed"


def persist_extraction(client: Neo4jClient, extraction: ExtractionResponse) -> GraphBuildResponse:
    """Write every completed chunk of an extraction result; one failing chunk never stops others.

    This is the service boundary for already-extracted results (no Groq call).
    """
    totals = ChunkWriteResult()
    failed: list[FailedChunk] = []
    for chunk in extraction.results:
        if chunk.extraction_status == "failed":
            failed.append(
                FailedChunk(chunk_id=chunk.chunk_id, stage="extraction", error=chunk.error or "")
            )
            continue
        try:
            written = persist_chunk(client, chunk)
        except GraphStoreError as exc:
            failed.append(FailedChunk(chunk_id=chunk.chunk_id, stage="graph", error=str(exc)))
            continue
        for name in ChunkWriteResult.model_fields:
            setattr(totals, name, getattr(totals, name) + getattr(written, name))

    processed = len(extraction.results)
    succeeded = processed - len(failed)
    return GraphBuildResponse(
        document_id=extraction.document_id,
        status=_status(succeeded, len(failed), extraction.chunks_skipped),
        total_chunks=extraction.total_chunks,
        chunks_processed=processed,
        chunks_succeeded=succeeded,
        chunks_failed=len(failed),
        chunks_skipped=extraction.chunks_skipped,
        entities_extracted=extraction.total_entities,
        relationships_extracted=extraction.total_relationships,
        failed_chunks=failed,
        **totals.model_dump(),
    )


def build_document_graph(
    client: Neo4jClient,
    llm: JSONCompletionClient,
    collection: Collection,
    document_id: str,
    *,
    max_chunks: int,
) -> GraphBuildResponse:
    """Read the judgment's chunks, extract with Story 1.5, and persist the results.

    Raises ``ChunkNotFoundError`` / ``VectorStoreError`` from the extraction step.
    """
    extraction = extract_document(llm, collection, document_id, max_chunks=max_chunks)
    return persist_extraction(client, extraction)
