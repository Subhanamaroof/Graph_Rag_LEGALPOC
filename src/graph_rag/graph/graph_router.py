"""HTTP endpoints for Neo4j knowledge graph construction (Story 1.6)."""

from typing import Annotated

from chromadb.api.models.Collection import Collection
from fastapi import APIRouter, Depends, HTTPException, status

from graph_rag.core.config import Settings, get_settings
from graph_rag.core.llm import JSONCompletionClient
from graph_rag.graph.extraction_service import ChunkNotFoundError
from graph_rag.graph.graph_models import (
    GraphBuildRequest,
    GraphBuildResponse,
    GraphInitializeResponse,
)
from graph_rag.graph.graph_schema import ensure_schema, initialize_schema
from graph_rag.graph.graph_service import build_document_graph
from graph_rag.graph.neo4j_client import GraphStoreError, Neo4jClient, get_neo4j_client
from graph_rag.graph.router import get_extraction_llm
from graph_rag.retrieval.router import get_chroma_collection
from graph_rag.vector_store.chroma_store import VectorStoreError

router = APIRouter(prefix="/v1/graph", tags=["graph"])


def get_graph_client(settings: Annotated[Settings, Depends(get_settings)]) -> Neo4jClient:
    """Shared Neo4j client, connected on first use. 503 if Neo4j is unset or unreachable."""
    try:
        return get_neo4j_client(settings)
    except GraphStoreError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.post("/initialize", response_model=GraphInitializeResponse)
def initialize_graph(
    client: Annotated[Neo4jClient, Depends(get_graph_client)],
) -> GraphInitializeResponse:
    """Verify connectivity and create missing constraints/indexes. Safe to call repeatedly."""
    try:
        return GraphInitializeResponse(status="initialized", **initialize_schema(client))
    except GraphStoreError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.post("/build", response_model=GraphBuildResponse)
def build_graph(
    request: GraphBuildRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    client: Annotated[Neo4jClient, Depends(get_graph_client)],
    collection: Annotated[Collection, Depends(get_chroma_collection)],
    llm: Annotated[JSONCompletionClient, Depends(get_extraction_llm)],
) -> GraphBuildResponse:
    """Extract a judgment's chunks (Story 1.5) and write the results to Neo4j.

    404 unknown document, 503 Neo4j/ChromaDB/Groq unavailable. Per-chunk extraction or write
    failures are reported inside the 200 response (``status`` ``partial`` or ``failed``).
    """
    max_chunks = request.max_chunks or settings.extraction_max_chunks_per_document
    try:
        ensure_schema(client)
        return build_document_graph(
            client, llm, collection, request.document_id, max_chunks=max_chunks
        )
    except ChunkNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (GraphStoreError, VectorStoreError) as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
