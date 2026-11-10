"""Answer generation orchestration (Story 1.9).

question → Story 1.8 retrieve_hybrid (the only evidence source)
         → sufficiency check (retrieval status, usable evidence)
         → context_builder (dedupe, budget, [SOURCE n] blocks)
         → Groq (structured JSON) → citation_validator → AnswerResponse
         (+ retrieval_details built from the same Story 1.8 response, Story 1.9.1)

This module never touches ChromaDB or Neo4j: it receives a ``retrieve`` callable that runs
Story 1.8 hybrid retrieval and only sees its response. The LLM is never called without
evidence, and nothing is returned as an answer when generation or validation fails.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import groq

from graph_rag.core.llm import JSONCompletionClient, LLMError, LLMNotConfiguredError
from graph_rag.generation.answer_generator import AnswerParseError, generate_raw_answer
from graph_rag.generation.citation_validator import validate_claims
from graph_rag.generation.context_builder import PreparedContext, build_context
from graph_rag.generation.generation_models import (
    AnswerRequest,
    AnswerResponse,
    GenerationMetadata,
    RetrievalDetails,
    RetrievalSummary,
)
from graph_rag.generation.retrieval_details import build_retrieval_details, map_claim_citations
from graph_rag.retrieval.hybrid_retrieval_models import (
    HybridRetrievalRequest,
    HybridRetrievalResponse,
)

logger = logging.getLogger(__name__)

NO_EVIDENCE_ANSWER = (
    "The available legal judgment corpus did not provide sufficient information to answer "
    "this question."
)


@dataclass
class GenerationResources:
    retrieve: Callable[[HybridRetrievalRequest], HybridRetrievalResponse]  # Story 1.8
    llm: JSONCompletionClient | None  # None when Groq is not configured
    model_name: str | None
    max_context_chars: int
    details_max_items: int = 50  # RETRIEVAL_DETAILS_MAX_ITEMS (Story 1.9.1)
    details_preview_chars: int = 400  # RETRIEVAL_DETAILS_PREVIEW_CHARS (Story 1.9.1)


def _retrieval_summary(
    response: HybridRetrievalResponse | None, context: PreparedContext | None = None
) -> RetrievalSummary:
    if response is None:
        return RetrievalSummary(errors=["Hybrid retrieval failed unexpectedly."])
    summary = RetrievalSummary(
        retrieval_status=response.retrieval_status,
        vector_results=response.vector_results_count,
        graph_results=response.graph_results_count,
        found_by_both=sum(len(r.retrieval_sources) == 2 for r in response.results),
        merged_results=response.merged_results_count,
        final_evidence_count=response.total_results,
        unresolved_references=len(response.unresolved_references),
        warnings=response.warnings,
        errors=response.errors,
    )
    if context is not None:
        summary.evidence_used = len(context.sources)
        summary.duplicates_removed = context.duplicates_removed
        summary.excluded_missing_identifier = context.excluded_missing_identifier
        summary.excluded_empty_content = context.excluded_empty_content
        summary.excluded_over_budget = context.excluded_over_budget
    return summary


def _llm_error_message(exc: LLMError) -> str:
    """Fixed, safe messages; the raw exception is only logged."""
    cause = exc.__cause__
    if isinstance(exc, LLMNotConfiguredError):
        return "Answer generation failed: Groq is not configured."
    if isinstance(cause, groq.APITimeoutError):
        return "Answer generation failed: the Groq request timed out."
    if isinstance(cause, groq.APIStatusError):
        if cause.status_code == 429:
            return "Answer generation failed: Groq rate limit reached; retry later."
        if cause.status_code == 413:
            return "Answer generation failed: the context is too large for the model."
    if isinstance(cause, groq.APIConnectionError):
        return "Answer generation failed: Groq could not be reached."
    return "Answer generation failed: the Groq request failed."


def generate_answer(request: AnswerRequest, resources: GenerationResources) -> AnswerResponse:
    metadata = GenerationMetadata(model=resources.model_name)
    details: RetrievalDetails | None = None  # set once Story 1.8 has returned a response

    def respond(status, summary, *, answer=None, warnings=(), errors=(), **extra):
        return AnswerResponse(
            question=request.question,
            status=status,
            answer=answer,
            retrieval_summary=summary,
            generation_metadata=metadata,
            warnings=list(warnings),
            errors=list(errors),
            retrieval_details=details,
            **extra,
        )

    # 1. Evidence from Story 1.8 only.
    hybrid_request = HybridRetrievalRequest(
        question=request.question, final_top_k=request.top_k, max_hops=request.max_hops
    )
    try:
        retrieval = resources.retrieve(hybrid_request)
    except Exception:  # retrieve_hybrid reports failures in its body; this is a safety net
        logger.exception("Hybrid retrieval raised during answer generation")
        return respond(
            "retrieval_failed",
            _retrieval_summary(None),
            errors=["Evidence retrieval failed; no answer was generated."],
        )

    # Explainability from this same response; nothing is retrieved again.
    details = build_retrieval_details(
        retrieval,
        max_items=resources.details_max_items,
        preview_chars=resources.details_preview_chars,
    )

    # 2. Sufficiency: never ask the LLM without evidence.
    if retrieval.retrieval_status == "failed":
        return respond(
            "retrieval_failed",
            _retrieval_summary(retrieval),
            errors=["Evidence retrieval failed; no answer was generated."],
        )
    context = build_context(retrieval.results, resources.max_context_chars)
    summary = _retrieval_summary(retrieval, context)
    if not context.sources:
        return respond("insufficient_evidence", summary, answer=NO_EVIDENCE_ANSWER)

    partial = retrieval.retrieval_status == "partial"
    warnings = []
    if partial:
        warnings.append(
            "Retrieval was partial (one retrieval branch failed or was incomplete); the answer "
            "may miss relevant evidence."
        )
    if context.excluded_over_budget:
        warnings.append(
            f"{context.excluded_over_budget} lower-ranked evidence chunk(s) were left out to "
            "stay within the context budget."
        )

    # 3. Groq generation.
    sources = context.sources if request.include_sources else []
    if resources.llm is None:
        return respond(
            "generation_failed",
            summary,
            warnings=warnings,
            errors=["Answer generation failed: Groq is not configured."],
            sources=sources,
        )
    metadata.llm_called = True
    metadata.context_characters = len(context.text)
    try:
        raw = generate_raw_answer(
            resources.llm, request.question, context.text, partial_retrieval=partial
        )
    except LLMError as exc:
        logger.warning("Answer generation LLM call failed: %s", exc)
        return respond(
            "generation_failed",
            summary,
            warnings=warnings,
            errors=[_llm_error_message(exc)],
            sources=sources,
        )
    except AnswerParseError as exc:
        logger.warning("Answer generation output rejected: %s", exc)
        return respond(
            "generation_failed",
            summary,
            warnings=warnings,
            errors=["Answer generation failed: the model returned a malformed response."],
            sources=sources,
        )

    # 4. Citation validation against the evidence actually given to the LLM.
    validated = validate_claims(raw, context.keys)
    metadata.citations_validated = True
    metadata.claims_returned = len(raw.claims)
    metadata.claims_accepted = len(validated.claims)
    metadata.invalid_citations = validated.invalid_citations
    metadata.rejected_claims = validated.rejected_claims
    details.claim_citations = map_claim_citations(
        validated.claims, validated.rejected_claims, context.sources
    )
    for source in context.sources:
        source.cited = (source.document_id, source.chunk_id) in validated.cited_keys
    if validated.invalid_citations:
        warnings.append(
            f"{len(validated.invalid_citations)} citation(s) did not match retrieved evidence "
            "and were removed."
        )
    if validated.rejected_claims:
        warnings.append(
            f"{len(validated.rejected_claims)} claim(s) without a valid citation were rejected; "
            "the answer text may contain statements not listed in claims."
        )

    if raw.insufficient_evidence:
        return respond(
            "insufficient_evidence",
            summary,
            answer=validated.answer,
            warnings=warnings,
            claims=validated.claims,
            sources=sources,
        )
    if not validated.claims:
        # A non-empty answer with no supported claim is ungrounded: do not return it.
        return respond(
            "generation_failed",
            summary,
            warnings=warnings,
            errors=["Answer generation failed: no claim in the answer had a valid citation."],
            sources=sources,
        )
    return respond(
        "partial" if partial else "success",
        summary,
        answer=validated.answer,
        warnings=warnings,
        claims=validated.claims,
        sources=sources,
    )
