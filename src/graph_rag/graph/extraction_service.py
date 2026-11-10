"""Coordinate extraction over chunks of the read-only ``judgments`` collection (Story 1.5).

A judgment (``document_id``) is the stored ``case_id``; a chunk is a ChromaDB record whose ID
looks like ``<case_id>_chunk<n>``. Chunks are only read, never modified.
"""

import re

from chromadb.api.models.Collection import Collection

from graph_rag.core.llm import JSONCompletionClient
from graph_rag.graph.entity_extractor import extract_from_chunk
from graph_rag.graph.models import ExtractionResponse, ExtractionStatus
from graph_rag.retrieval.vector_retriever import DOCUMENT_ID_FIELD
from graph_rag.vector_store.chroma_store import get_chunks

_CHUNK_NUMBER = re.compile(r"_chunk_?(\d+)$")


class ChunkNotFoundError(LookupError):
    """The requested document or chunk is not in the collection."""


def chunk_order(chunk_id: str) -> tuple[int, str]:
    """Sort key: numeric chunk position (so chunk10 follows chunk9), then the ID."""
    match = _CHUNK_NUMBER.search(chunk_id)
    return (int(match.group(1)) if match else 0, chunk_id)


def load_chunks(
    collection: Collection, document_id: str, chunk_id: str | None = None
) -> list[dict]:
    """Return the requested chunk, or all chunks of the document in reading order."""
    if chunk_id is not None:
        records = get_chunks(collection, ids=[chunk_id])
        if not records:
            raise ChunkNotFoundError(f"Chunk '{chunk_id}' was not found.")
        stored_document_id = records[0]["metadata"].get(DOCUMENT_ID_FIELD)
        if str(stored_document_id) != document_id:
            raise ChunkNotFoundError(
                f"Chunk '{chunk_id}' does not belong to document '{document_id}'."
            )
        return records
    records = get_chunks(collection, where={DOCUMENT_ID_FIELD: document_id})
    if not records:
        raise ChunkNotFoundError(f"Document '{document_id}' has no chunks in the collection.")
    return sorted(records, key=lambda record: chunk_order(record["id"]))


def _overall_status(succeeded: int, failed: int, skipped: int) -> ExtractionStatus:
    if succeeded == 0:
        return "failed"
    if failed or skipped:
        return "partial"
    return "completed"


def extract_document(
    llm: JSONCompletionClient,
    collection: Collection,
    document_id: str,
    *,
    chunk_id: str | None = None,
    max_chunks: int,
) -> ExtractionResponse:
    """Extract entities and relationships chunk by chunk.

    A failing chunk does not stop the others; it is reported with ``extraction_status:
    failed`` and its error. Chunks beyond ``max_chunks`` are counted as skipped. The overall
    status is ``completed`` only if every selected chunk was extracted successfully.
    Raises ``ChunkNotFoundError`` or ``VectorStoreError``.
    """
    records = load_chunks(collection, document_id, chunk_id)
    selected = records[:max_chunks]
    results = [
        extract_from_chunk(
            llm, document_id=document_id, chunk_id=record["id"], chunk_text=record["document"]
        )
        for record in selected
    ]
    failed = sum(result.extraction_status == "failed" for result in results)
    succeeded = len(results) - failed
    skipped = len(records) - len(selected)
    return ExtractionResponse(
        document_id=document_id,
        extraction_status=_overall_status(succeeded, failed, skipped),
        total_chunks=len(records),
        chunks_processed=len(results),
        chunks_succeeded=succeeded,
        chunks_failed=failed,
        chunks_skipped=skipped,
        total_entities=sum(len(result.entities) for result in results),
        total_relationships=sum(len(result.relationships) for result in results),
        results=results,
    )
