"""Retrieval explainability for the answer API (Story 1.9.1).

Builds ``RetrievalDetails`` from the Story 1.8 response that answer generation already
received, plus the validated claims. It is a pure transformation: no retrieval, no database,
no LLM. Branch membership is read from the ranks Story 1.8 assigned, so a chunk's origin is
exactly how it was found:

- ``vector``: vector rank only; ``graph``: graph rank only; ``both``: both ranks.

Lists are capped at ``max_items`` and chunk text at ``preview_chars``; counts in the fusion
summary are always computed on the full, uncapped sets.
"""

from collections.abc import Callable, Iterable
from typing import Any

from graph_rag.generation.generation_models import (
    Claim,
    ClaimCitationDetail,
    ClaimEvidence,
    FusionSummary,
    RejectedClaim,
    RetrievalChunkDetail,
    RetrievalDetails,
    RetrievalOrigin,
    Source,
)
from graph_rag.retrieval.hybrid_retrieval_models import HybridResult, HybridRetrievalResponse


def retrieval_origin(sources: Iterable[str]) -> RetrievalOrigin:
    sources = set(sources)
    if {"vector", "graph"} <= sources:
        return "both"
    return "graph" if "graph" in sources else "vector"


def _metadata_value(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    return None if value is None or value == "" else str(value)


def chunk_detail(
    result: HybridResult, *, preview_chars: int, in_final: bool, with_graph: bool
) -> RetrievalChunkDetail:
    text = result.content or ""
    return RetrievalChunkDetail(
        document_id=result.document_id,
        chunk_id=result.chunk_id,
        case_id=_metadata_value(result.metadata, "case_id"),
        title=_metadata_value(result.metadata, "title"),
        court=_metadata_value(result.metadata, "court"),
        date=_metadata_value(result.metadata, "date"),
        text_preview=text[:preview_chars],
        text_length=len(text),
        text_truncated=len(text) > preview_chars,
        retrieval_origin=retrieval_origin(result.retrieval_sources),
        hybrid_rank=result.rank,
        rrf_score=result.rrf_score,
        in_final_results=in_final,
        vector_rank=result.vector_rank,
        vector_score=result.vector_score,
        vector_distance=result.vector_distance,
        graph_rank=result.graph_rank,
        graph_context=result.graph_context if with_graph else None,
    )


def _unique(items: Iterable, identity: Callable) -> list:
    seen, unique = set(), []
    for item in items:
        if identity(item) not in seen:
            seen.add(identity(item))
            unique.append(item)
    return unique


def build_retrieval_details(
    retrieval: HybridRetrievalResponse, *, max_items: int, preview_chars: int
) -> RetrievalDetails:
    # The full RRF ranking; a response built without it (older producer) falls back to the
    # final results, which are a prefix of the same ranking.
    ranked = retrieval.ranked_results or retrieval.results
    final_keys = {(r.document_id, r.chunk_id) for r in retrieval.results}

    vector = sorted((r for r in ranked if r.vector_rank is not None), key=lambda r: r.vector_rank)
    graph = sorted((r for r in ranked if r.graph_rank is not None), key=lambda r: r.graph_rank)
    both = [r for r in vector if r.graph_rank is not None]
    graph_only = [r for r in graph if r.vector_rank is None]

    truncated: list[str] = []

    def cap(name: str, items: list) -> list:
        if len(items) > max_items:
            truncated.append(name)
        return items[:max_items]

    def details(name: str, results: list[HybridResult], with_graph: bool):
        return [
            chunk_detail(
                r,
                preview_chars=preview_chars,
                in_final=(r.document_id, r.chunk_id) in final_keys,
                with_graph=with_graph,
            )
            for r in cap(name, results)
        ]

    paths = retrieval.graph_paths
    return RetrievalDetails(
        vector_status=retrieval.vector.status,
        graph_status=retrieval.graph.status,
        graph_retrieval_status=retrieval.graph.graph_retrieval_status,
        vector_results=details("vector_results", vector, with_graph=False),
        graph_results=details("graph_results", graph, with_graph=True),
        hybrid_results=details("hybrid_results", ranked, with_graph=False),
        graph_only_chunks=details("graph_only_chunks", graph_only, with_graph=True),
        graph_paths=cap("graph_paths", paths),
        graph_entities=cap(
            "graph_entities",
            _unique((n for p in paths for n in p.nodes), lambda n: n.entity_id),
        ),
        graph_relationships=cap(
            "graph_relationships",
            _unique((e for p in paths for e in p.relationships), lambda e: e.relationship_id),
        ),
        matched_entities=cap("matched_entities", retrieval.graph.matched_entities),
        unresolved_references=cap("unresolved_references", retrieval.unresolved_references),
        fusion_summary=FusionSummary(
            vector_count=len(vector),
            graph_count=len(graph),
            hybrid_count=len(ranked),
            overlap_count=len(both),
            vector_only_count=len(vector) - len(both),
            graph_only_count=len(graph_only),
            both_count=len(both),
            final_results_count=len(retrieval.results),
            rrf_k=retrieval.rrf_k,
        ),
        max_items_per_list=max_items,
        text_preview_chars=preview_chars,
        truncated_lists=truncated,
    )


def map_claim_citations(
    claims: list[Claim], rejected: list[RejectedClaim], sources: list[Source]
) -> list[ClaimEvidence]:
    """Supported claims (validated citations only) followed by rejected claims.

    Every validated citation is one of ``sources`` (validation guarantees it); a citation that
    somehow is not is left out rather than linked to an invented source.
    """
    by_key = {(s.document_id, s.chunk_id): s for s in sources}
    mapped = []
    for claim in claims:
        citations = [
            ClaimCitationDetail(
                document_id=source.document_id,
                chunk_id=source.chunk_id,
                source_number=source.source_number,
                hybrid_rank=source.rank,
                retrieval_origin=retrieval_origin(source.retrieval_sources),
            )
            for c in claim.citations
            if (source := by_key.get((c.document_id, c.chunk_id))) is not None
        ]
        mapped.append(
            ClaimEvidence(claim_text=claim.text, supported=bool(citations), citations=citations)
        )
    mapped += [
        ClaimEvidence(claim_text=r.text, supported=False, rejection_reason=r.reason)
        for r in rejected
    ]
    return mapped
