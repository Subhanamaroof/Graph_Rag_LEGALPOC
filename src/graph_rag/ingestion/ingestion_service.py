"""Coordinate the ingestion pipeline for all documents.

discovery -> loading/extraction -> cleaning -> chunking (Story 1.2)
          -> embedding -> ChromaDB ``new_documents`` upsert (Story 1.3.2.2)

The embedding and storage steps run only when an ``Embedder`` and a writable collection
are passed in; without them the pipeline stops after chunking, exactly as in Story 1.2.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chromadb.api.models.Collection import Collection

from graph_rag.embeddings.embedder import Embedder, EmbeddingError
from graph_rag.ingestion.chunker import chunk_document
from graph_rag.ingestion.document_loader import (
    discover_documents,
    load_document,
    make_document_id,
)
from graph_rag.ingestion.models import (
    Chunk,
    DocumentStatus,
    IngestionError,
    IngestionResult,
    PageText,
)
from graph_rag.ingestion.text_cleaner import clean_text
from graph_rag.ingestion.text_extractor import EmptyDocumentError
from graph_rag.vector_store.chroma_store import VectorStoreError, replace_document_chunks


def process_document(
    path: Path, base_directory: Path, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Run one document through the full pipeline and return its chunks."""
    document = load_document(path, base_directory)

    cleaned_pages = [
        PageText(page_number=page.page_number, text=clean_text(page.text))
        for page in document.pages
    ]
    document = document.model_copy(
        update={
            "pages": cleaned_pages,
            "content": "\n\n".join(page.text for page in cleaned_pages if page.text),
        }
    )
    if not document.content:
        raise EmptyDocumentError("Document is empty after text extraction and cleaning")

    return chunk_document(document, chunk_size, chunk_overlap)


def chunk_metadata(chunk: Chunk, ingested_at: str) -> dict[str, Any]:
    """Flatten a chunk's identity and position into ChromaDB metadata.

    ChromaDB metadata values must be scalars or non-empty lists, so ``page_numbers`` is
    only stored for chunks that have pages (PDFs).
    """
    metadata: dict[str, Any] = {
        "document_id": chunk.document_id,
        "source_file": chunk.source_file,
        "file_type": chunk.file_type,
        "chunk_index": chunk.chunk_index,
        "start_char": chunk.metadata.get("start_char"),
        "end_char": chunk.metadata.get("end_char"),
        "ingested_at": ingested_at,
    }
    if chunk.metadata.get("page_numbers"):
        metadata["page_numbers"] = list(chunk.metadata["page_numbers"])
    return metadata


def embed_and_store(
    chunks: list[Chunk], document: DocumentStatus, embedder: Embedder, collection: Collection
) -> None:
    """Embed one document's chunks and upsert them; fills the counters on ``document``.

    Raises ``EmbeddingError`` or ``VectorStoreError``. Counters are only set after the step
    they describe succeeded, so a failed write is never reported as stored.
    """
    if not chunks:
        raise EmptyDocumentError("Document produced no chunks")
    vectors = embedder.embed_texts([chunk.text for chunk in chunks])  # same order as chunks
    document.embeddings = len(vectors)

    ingested_at = datetime.now(UTC).isoformat(timespec="seconds")
    upserted, deleted = replace_document_chunks(
        collection,
        document_id=chunks[0].document_id,
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        embeddings=vectors,
        metadatas=[chunk_metadata(chunk, ingested_at) for chunk in chunks],
    )
    document.records_upserted = upserted
    document.obsolete_chunks_deleted = deleted


def ingest_documents(
    directory: Path,
    chunk_size: int,
    chunk_overlap: int,
    *,
    embedder: Embedder | None = None,
    collection: Collection | None = None,
) -> IngestionResult:
    """Process every supported document in ``directory`` sequentially.

    A failure in one document is recorded in ``errors`` and ``documents`` and
    processing continues with the remaining documents. All generated chunks are
    returned; the API decides whether to expose them. When ``embedder`` and
    ``collection`` are given, each document's chunks are also embedded and stored.
    """
    if (embedder is None) != (collection is None):
        raise ValueError("Pass both embedder and collection, or neither")
    store = collection is not None
    collection_name = collection.name if store else None

    if not directory.is_dir():
        return IngestionResult(
            status="completed",
            documents_discovered=0,
            documents_processed=0,
            documents_failed=0,
            total_chunks=0,
            collection=collection_name,
            errors=[IngestionError(source_file=str(directory), error="Directory not found")],
            chunks=[],
        )

    paths = discover_documents(directory)
    chunks: list[Chunk] = []
    errors: list[IngestionError] = []
    documents: list[DocumentStatus] = []

    for path in paths:
        document = DocumentStatus(
            source_file=path.name, document_id=make_document_id(path, directory), status="failed"
        )
        documents.append(document)
        try:
            document_chunks = process_document(path, directory, chunk_size, chunk_overlap)
            document.chunks = len(document_chunks)
            chunks.extend(document_chunks)
            if store:
                embed_and_store(document_chunks, document, embedder, collection)
            document.status = "succeeded"
        except EmbeddingError as exc:
            document.error = f"Embedding failed: {exc}"
        except VectorStoreError as exc:
            document.error = f"Storage failed: {exc}"
        except Exception as exc:  # one bad file must not stop the others
            document.error = str(exc) or type(exc).__name__
        if document.error:
            errors.append(IngestionError(source_file=path.name, error=document.error))

    failed = len(errors)
    if failed == 0:
        status = "completed"
    elif failed < len(paths):
        status = "completed_with_errors"
    else:
        status = "failed"

    return IngestionResult(
        status=status,
        documents_discovered=len(paths),
        documents_processed=len(paths) - failed,
        documents_failed=failed,
        total_chunks=len(chunks),
        total_embeddings=sum(document.embeddings for document in documents),
        records_upserted=sum(document.records_upserted for document in documents),
        obsolete_chunks_deleted=sum(document.obsolete_chunks_deleted for document in documents),
        collection=collection_name,
        errors=errors,
        documents=documents,
        chunks=chunks,
    )
