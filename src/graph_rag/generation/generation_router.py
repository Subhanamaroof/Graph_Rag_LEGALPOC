"""HTTP endpoint for grounded answer generation (Story 1.9).

Story 1.8 is called in-process (``retrieve_hybrid`` with the resources from its own
``get_hybrid_resources`` dependency), not over HTTP. Groq comes from the same optional client
used by retrieval; when it is missing the response says so instead of failing with 503.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from graph_rag.core.config import Settings, get_settings
from graph_rag.generation.generation_models import AnswerRequest, AnswerResponse
from graph_rag.generation.generation_service import GenerationResources, generate_answer
from graph_rag.retrieval.hybrid_retrieval_models import HybridRetrievalRequest
from graph_rag.retrieval.hybrid_retrieval_service import (
    HybridResources,
    resolve_parameters,
    retrieve_hybrid,
)
from graph_rag.retrieval.hybrid_router import get_hybrid_resources

router = APIRouter(prefix="/v1/generation", tags=["generation"])


def get_generation_resources(
    settings: Annotated[Settings, Depends(get_settings)],
    hybrid: Annotated[HybridResources, Depends(get_hybrid_resources)],
) -> GenerationResources:
    def retrieve(request: HybridRetrievalRequest):
        return retrieve_hybrid(request, resolve_parameters(request, settings), hybrid)

    return GenerationResources(
        retrieve=retrieve,
        llm=hybrid.llm,
        model_name=settings.groq_model_name,
        max_context_chars=settings.generation_max_context_chars,
        details_max_items=settings.retrieval_details_max_items,
        details_preview_chars=settings.retrieval_details_preview_chars,
    )


@router.post("/answer", response_model=AnswerResponse)
def answer(
    request: AnswerRequest,
    resources: Annotated[GenerationResources, Depends(get_generation_resources)],
) -> AnswerResponse:
    """Retrieve evidence with Story 1.8, generate a cited answer with Groq, validate citations.

    422 for an invalid request. Retrieval and generation problems are reported in the 200 body
    via ``status`` (``success``, ``partial``, ``insufficient_evidence``, ``retrieval_failed``,
    ``generation_failed``); ``answer`` is ``null`` whenever no grounded answer exists.
    """
    return generate_answer(request, resources)
