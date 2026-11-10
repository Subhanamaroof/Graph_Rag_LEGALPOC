"""HTTP endpoint for entity and relationship extraction (Story 1.5)."""

from typing import Annotated

from chromadb.api.models.Collection import Collection
from fastapi import APIRouter, Depends, HTTPException, status

from graph_rag.core.config import Settings, get_settings
from graph_rag.core.llm import JSONCompletionClient, LLMNotConfiguredError, create_groq_client
from graph_rag.graph.extraction_service import ChunkNotFoundError, extract_document
from graph_rag.graph.models import ExtractionRequest, ExtractionResponse
from graph_rag.retrieval.router import get_chroma_collection
from graph_rag.vector_store.chroma_store import VectorStoreError

router = APIRouter(prefix="/v1/extraction", tags=["extraction"])


def get_extraction_llm(
    settings: Annotated[Settings, Depends(get_settings)],
) -> JSONCompletionClient:
    try:
        return create_groq_client(settings)
    except LLMNotConfiguredError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.post("/process", response_model=ExtractionResponse)
def process_extraction(
    request: ExtractionRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    collection: Annotated[Collection, Depends(get_chroma_collection)],
    llm: Annotated[JSONCompletionClient, Depends(get_extraction_llm)],
) -> ExtractionResponse:
    """Extract entities and relationships from one chunk or from a judgment's chunks.

    Reads the ``judgments`` collection only. 404 if the document or chunk does not exist,
    503 if ChromaDB or Groq is not available. Per-chunk LLM failures are returned inside the
    200 response (``extraction_status`` ``partial`` or ``failed``), never hidden.
    """
    max_chunks = request.max_chunks or settings.extraction_max_chunks_per_document
    try:
        return extract_document(
            llm, collection, request.document_id, chunk_id=request.chunk_id, max_chunks=max_chunks
        )
    except ChunkNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except VectorStoreError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
