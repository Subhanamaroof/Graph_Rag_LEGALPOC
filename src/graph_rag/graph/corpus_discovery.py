"""Find every judgment (document) in the read-only ``judgments`` collection (Story 1.6.2).

The collection stores CHUNKS, not judgments: 7,325 records belong to far fewer judgments. The
judgment a chunk belongs to is its metadata ``case_id`` (``DOCUMENT_ID_FIELD``); there is no
``document_id`` field in this collection. Discovery pages through the collection asking only
for IDs and metadata (no chunk text, no embeddings) and groups chunk IDs by ``case_id``.
Records without a usable ``case_id`` are reported separately and never processed.
"""

from dataclasses import dataclass, field

from chromadb.api.models.Collection import Collection

from graph_rag.graph.extraction_service import chunk_order
from graph_rag.retrieval.vector_retriever import DOCUMENT_ID_FIELD
from graph_rag.vector_store.chroma_store import _CHROMA_ERRORS, VectorStoreError

# Records read per ChromaDB request. A page size, not a limit: paging continues to the end.
DISCOVERY_PAGE_SIZE = 1000
INVALID_ID_SAMPLE = 20  # how many unidentifiable record IDs to list in summaries


@dataclass
class CorpusDiscovery:
    """Chunk IDs per document (in reading order), plus records that cannot be attributed."""

    documents: dict[str, list[str]] = field(default_factory=dict)
    total_chunk_records: int = 0
    invalid_record_ids: list[str] = field(default_factory=list)

    @property
    def total_documents(self) -> int:
        return len(self.documents)

    @property
    def eligible_chunks(self) -> int:
        return sum(len(chunk_ids) for chunk_ids in self.documents.values())

    @property
    def records_missing_document_id(self) -> int:
        return len(self.invalid_record_ids)


def document_id_of(metadata: dict | None) -> str | None:
    """The judgment ID of a chunk, or ``None`` if ``case_id`` is missing or blank."""
    value = (metadata or {}).get(DOCUMENT_ID_FIELD)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def discover_corpus(
    collection: Collection, page_size: int = DISCOVERY_PAGE_SIZE
) -> CorpusDiscovery:
    """Read the whole collection page by page and group chunk IDs by document.

    Deterministic: documents are sorted by ID and chunks by their numeric position, whatever
    order ChromaDB returns them in. Raises ``VectorStoreError`` if ChromaDB cannot be read.
    """
    grouped: dict[str, list[str]] = {}
    discovery = CorpusDiscovery()
    offset = 0
    while True:
        try:
            page = collection.get(limit=page_size, offset=offset, include=["metadatas"])
        except _CHROMA_ERRORS as exc:
            raise VectorStoreError(f"Corpus discovery failed ({type(exc).__name__}).") from exc
        ids = page.get("ids") or []
        metadatas = page.get("metadatas") or [None] * len(ids)
        for record_id, metadata in zip(ids, metadatas, strict=True):
            discovery.total_chunk_records += 1
            document_id = document_id_of(metadata)
            if document_id is None:
                discovery.invalid_record_ids.append(record_id)
            else:
                grouped.setdefault(document_id, []).append(record_id)
        if len(ids) < page_size:
            break
        offset += page_size

    discovery.documents = {
        document_id: sorted(set(grouped[document_id]), key=chunk_order)
        for document_id in sorted(grouped)
    }
    discovery.invalid_record_ids.sort()
    return discovery
