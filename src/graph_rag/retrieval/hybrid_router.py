"""HTTP endpoint for hybrid retrieval (Story 1.8).

The vector and graph endpoints answer 503 when their store is down. Hybrid retrieval must not:
one store being down should only degrade the response to ``partial``. So this router does not
reuse those 503-raising dependencies; it hands the service the same underlying cached
accessors (``get_collection``, ``get_embedder``, ``get_neo4j_client``) as loaders, which each
branch calls itself. Groq stays optional, exactly as for graph retrieval.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from graph_rag.core.config import Settings, get_settings
from graph_rag.core.llm import JSONCompletionClient
from graph_rag.embeddings.embedder import EmbeddingModelError, get_embedder
from graph_rag.graph.neo4j_client import get_neo4j_client
from graph_rag.retrieval.graph_router import get_query_llm
from graph_rag.retrieval.hybrid_retrieval_models import (
    HybridRetrievalRequest,
    HybridRetrievalResponse,
)
from graph_rag.retrieval.hybrid_retrieval_service import (
    HybridResources,
    resolve_parameters,
    retrieve_hybrid,
)
from graph_rag.vector_store.chroma_store import get_collection

router = APIRouter(prefix="/v1/retrieval", tags=["retrieval"])


def get_hybrid_resources(
    settings: Annotated[Settings, Depends(get_settings)],
    llm: Annotated[JSONCompletionClient | None, Depends(get_query_llm)],
) -> HybridResources:
    embedder = get_embedder(
        settings.embedding_model_name, settings.embedding_dimension, settings.embedding_batch_size
    )
    try:
        # Load the model (once per process) before the branch timeout starts, so a cold start
        # is not reported as a timeout. A load failure is reported by the vector branch itself.
        embedder.load()
    except EmbeddingModelError:
        pass
    return HybridResources(
        load_collection=lambda: get_collection(settings),
        load_embedder=lambda: embedder,
        load_graph_client=lambda: get_neo4j_client(settings),
        llm=llm,
    )


@router.post("/hybrid", response_model=HybridRetrievalResponse)
def hybrid_retrieval(
    request: HybridRetrievalRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    resources: Annotated[HybridResources, Depends(get_hybrid_resources)],
) -> HybridRetrievalResponse:
    """Run vector and graph retrieval, resolve graph evidence to chunks, fuse with RRF.

    422 for an invalid request. Store outages, timeouts and empty results are reported in the
    200 body via ``retrieval_status`` (``success``, ``partial``, ``no_results``, ``failed``).
    Returns retrieved evidence only, never a generated answer.
    """
    return retrieve_hybrid(request, resolve_parameters(request, settings), resources)
