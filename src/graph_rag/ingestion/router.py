"""HTTP endpoint that triggers the ingestion pipeline."""

from typing import Annotated

from chromadb.api.models.Collection import Collection
from fastapi import APIRouter, Depends, HTTPException, Query, status

from graph_rag.core.config import Settings, get_settings
from graph_rag.embeddings.embedder import Embedder
from graph_rag.ingestion.ingestion_service import ingest_documents
from graph_rag.ingestion.models import IngestionResult
from graph_rag.retrieval.router import get_loaded_embedder
from graph_rag.vector_store.chroma_store import VectorStoreError, get_new_documents_collection

router = APIRouter(prefix="/v1/ingestion", tags=["ingestion"])


def get_ingestion_collection(settings: Annotated[Settings, Depends(get_settings)]) -> Collection:
    """Open (or create) the ``new_documents`` collection; never the read-only one."""
    try:
        return get_new_documents_collection(settings)
    except VectorStoreError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.post("/process", response_model=IngestionResult, response_model_exclude_none=True)
def process_documents(
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[Embedder, Depends(get_loaded_embedder)],
    collection: Annotated[Collection, Depends(get_ingestion_collection)],
    include_chunks: Annotated[bool, Query(description="Return generated chunks")] = False,
) -> IngestionResult:
    """Load, clean, chunk, embed and store every document in the documents directory.

    Chunks are upserted into the ``new_documents`` ChromaDB collection (Story 1.3.2.2).
    503 if ChromaDB or the embedding model is unavailable. A failure in one document is
    reported per document and does not stop the others. Embeddings are never returned.
    """
    result = ingest_documents(
        settings.documents_directory,
        settings.chunk_size,
        settings.chunk_overlap,
        embedder=embedder,
        collection=collection,
    )
    if not include_chunks:
        result.chunks = None
    return result
