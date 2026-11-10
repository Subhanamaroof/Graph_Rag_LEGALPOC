"""HTTP endpoint for vector retrieval (Story 1.4, ChromaDB since Story 1.3.2)."""

from typing import Annotated

from chromadb.api.models.Collection import Collection
from fastapi import APIRouter, Depends, HTTPException, status

from graph_rag.core.config import Settings, get_settings
from graph_rag.embeddings.embedder import (
    Embedder,
    EmbeddingError,
    EmbeddingModelError,
    get_embedder,
)
from graph_rag.retrieval.models import VectorSearchRequest, VectorSearchResponse
from graph_rag.retrieval.vector_retriever import retrieve_similar_chunks
from graph_rag.vector_store.chroma_store import VectorSearchError, VectorStoreError, get_collection

router = APIRouter(prefix="/v1/retrieval", tags=["retrieval"])


def get_chroma_collection(settings: Annotated[Settings, Depends(get_settings)]) -> Collection:
    """Open the ChromaDB collection on first use (cached), so the app starts without it."""
    try:
        return get_collection(settings)
    except VectorStoreError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


def get_loaded_embedder(settings: Annotated[Settings, Depends(get_settings)]) -> Embedder:
    """Return the shared embedder, loading the model on the first request only."""
    embedder = get_embedder(
        settings.embedding_model_name, settings.embedding_dimension, settings.embedding_batch_size
    )
    try:
        embedder.load()
    except EmbeddingModelError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return embedder


@router.post("/vector", response_model=VectorSearchResponse)
def vector_search(
    request: VectorSearchRequest,
    collection: Annotated[Collection, Depends(get_chroma_collection)],
    embedder: Annotated[Embedder, Depends(get_loaded_embedder)],
) -> VectorSearchResponse:
    """Embed the query and return the most similar stored chunks, best match first.

    503 if ChromaDB or the embedding model is unavailable. A search with no matches
    returns 200 with an empty ``results`` list.
    """
    try:
        return retrieve_similar_chunks(
            request.query, request.top_k, embedder=embedder, collection=collection
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Query embedding failed: {exc}"
        ) from exc
    except VectorSearchError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except VectorStoreError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
