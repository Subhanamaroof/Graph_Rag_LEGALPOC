"""HTTP endpoint for graph retrieval (Story 1.7).

Kept apart from the vector retrieval router because it depends on the Neo4j client from
``graph.graph_router``, which itself imports the vector router (an import cycle otherwise).
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from graph_rag.core.config import Settings, get_settings
from graph_rag.core.llm import JSONCompletionClient, LLMNotConfiguredError, create_groq_client
from graph_rag.graph.graph_router import get_graph_client
from graph_rag.graph.neo4j_client import Neo4jClient
from graph_rag.retrieval.graph_retrieval_models import (
    GraphRetrievalRequest,
    GraphRetrievalResponse,
)
from graph_rag.retrieval.graph_retrieval_service import retrieve_graph

router = APIRouter(prefix="/v1/retrieval", tags=["retrieval"])


def get_query_llm(
    settings: Annotated[Settings, Depends(get_settings)],
) -> JSONCompletionClient | None:
    """Groq is optional here: without it, only deterministic query analysis is used."""
    try:
        return create_groq_client(settings)
    except LLMNotConfiguredError:
        return None


@router.post("/graph", response_model=GraphRetrievalResponse)
def graph_retrieval(
    request: GraphRetrievalRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    client: Annotated[Neo4jClient, Depends(get_graph_client)],
    llm: Annotated[JSONCompletionClient | None, Depends(get_query_llm)],
) -> GraphRetrievalResponse:
    """Identify entities in the question, match them in Neo4j and return bounded paths.

    422 invalid request, 503 Neo4j not configured or unreachable. Query-time Neo4j failures,
    timeouts and "nothing found" are reported in the 200 body via ``retrieval_status``
    (``failed``, ``partial``, ``no_match``, ``no_results``) and ``errors``.
    """
    return retrieve_graph(
        request, client, llm, timeout=settings.graph_retrieval_query_timeout_seconds
    )
