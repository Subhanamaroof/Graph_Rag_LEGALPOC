"""Hybrid retrieval orchestration (Story 1.8).

question ─┬─ vector branch: retrieve_similar_chunks (Story 1.4) ── ChromaDB chunks
          └─ graph branch:  retrieve_graph (Story 1.7) ── ranked paths ── source references
                                                         └─ resolve in ChromaDB (by chunk ID)
          → normalize → merge/de-duplicate by (document_id, chunk_id) → RRF → final_top_k

The two branches are independent: they run concurrently in two worker threads (the existing
retrieval code is synchronous) and neither needs the other's output. Each branch opens its
own resources through a loader, so ChromaDB, the embedding model or Neo4j being unavailable
fails only that branch. Everything is read-only.
"""

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass

from chromadb.api.models.Collection import Collection

from graph_rag.core.config import Settings
from graph_rag.core.llm import JSONCompletionClient
from graph_rag.embeddings.embedder import Embedder, EmbeddingError, EmbeddingModelError
from graph_rag.graph.neo4j_client import (
    GraphNotConfiguredError,
    GraphQueryTimeoutError,
    GraphStoreError,
    Neo4jClient,
)
from graph_rag.retrieval.graph_retrieval_models import GraphRetrievalRequest, GraphRetrievalResponse
from graph_rag.retrieval.graph_retrieval_service import retrieve_graph
from graph_rag.retrieval.hybrid_ranker import reciprocal_rank_fusion
from graph_rag.retrieval.hybrid_retrieval_models import (
    GraphBranchSummary,
    HybridResult,
    HybridRetrievalRequest,
    HybridRetrievalResponse,
    HybridRetrievalStatus,
    VectorBranchSummary,
)
from graph_rag.retrieval.models import VectorSearchResponse
from graph_rag.retrieval.result_fusion import (
    FusedChunk,
    merge_results,
    normalize_graph_results,
    normalize_vector_results,
)
from graph_rag.retrieval.source_resolver import (
    extract_source_references,
    lookup_failed,
    resolve_source_references,
)
from graph_rag.retrieval.vector_retriever import retrieve_similar_chunks
from graph_rag.vector_store.chroma_store import VectorSearchError, VectorStoreError

logger = logging.getLogger(__name__)


@dataclass
class HybridResources:
    """How each branch reaches its stores. Loaders are called inside the branch, so an
    unavailable store becomes a branch failure instead of failing the whole request."""

    load_collection: Callable[[], Collection]
    load_embedder: Callable[[], Embedder]
    load_graph_client: Callable[[], Neo4jClient]
    llm: JSONCompletionClient | None = None


@dataclass(frozen=True)
class HybridParameters:
    vector_top_k: int
    graph_top_k: int
    final_top_k: int
    max_hops: int
    rrf_k: int
    graph_query_timeout: float  # per Neo4j read transaction (Story 1.7 setting)
    branch_timeout: float  # how long to wait for each branch as a whole


def resolve_parameters(request: HybridRetrievalRequest, settings: Settings) -> HybridParameters:
    """Request values win; missing ones come from configuration."""
    return HybridParameters(
        vector_top_k=request.vector_top_k or settings.vector_retrieval_top_k,
        graph_top_k=request.graph_top_k or settings.graph_retrieval_top_k,
        final_top_k=request.final_top_k or settings.hybrid_final_top_k,
        max_hops=request.max_hops or settings.graph_retrieval_max_hops,
        rrf_k=settings.hybrid_rrf_k,
        graph_query_timeout=settings.graph_retrieval_query_timeout_seconds,
        branch_timeout=settings.hybrid_retrieval_timeout_seconds,
    )


# --- Safe error messages -----------------------------------------------------------------
# Only fixed messages are returned to the client; details go to the server log.


def _vector_error(exc: BaseException) -> str:
    if isinstance(exc, VectorSearchError):
        return "Vector retrieval failed: the ChromaDB search failed."
    if isinstance(exc, VectorStoreError):
        return "Vector retrieval failed: ChromaDB is unavailable."
    if isinstance(exc, EmbeddingModelError):
        return "Vector retrieval failed: the embedding model is unavailable."
    if isinstance(exc, EmbeddingError):
        return "Vector retrieval failed: the query could not be embedded."
    if isinstance(exc, ValueError):
        return "Vector retrieval failed: the question is not valid for vector search."
    logger.exception("Unexpected vector retrieval error", exc_info=exc)
    return "Vector retrieval failed unexpectedly."


def _graph_error(exc: BaseException) -> str:
    if isinstance(exc, GraphNotConfiguredError):
        return "Graph retrieval failed: Neo4j is not configured."
    if isinstance(exc, GraphQueryTimeoutError):
        return "Graph retrieval failed: a Neo4j query timed out."
    if isinstance(exc, GraphStoreError):
        return "Graph retrieval failed: Neo4j is unavailable or the query failed."
    logger.exception("Unexpected graph retrieval error", exc_info=exc)
    return "Graph retrieval failed unexpectedly."


# --- Branches ----------------------------------------------------------------------------


def run_vector_branch(
    question: str, params: HybridParameters, resources: HybridResources
) -> VectorSearchResponse:
    """Story 1.4 vector retrieval through its existing interface."""
    embedder = resources.load_embedder()
    embedder.load()
    return retrieve_similar_chunks(
        question,
        params.vector_top_k,
        embedder=embedder,
        collection=resources.load_collection(),
    )


def run_graph_branch(
    request: HybridRetrievalRequest, params: HybridParameters, resources: HybridResources
) -> GraphRetrievalResponse:
    """Story 1.7 graph retrieval through its existing interface."""
    graph_request = GraphRetrievalRequest(
        question=request.question,
        max_hops=params.max_hops,
        limit=params.graph_top_k,
        entity_types=request.entity_types,
        relationship_types=request.relationship_types,
    )
    return retrieve_graph(
        graph_request,
        resources.load_graph_client(),
        resources.llm,
        timeout=params.graph_query_timeout,
    )


def _run_branches(request, params, resources):
    """Start both branches at once; each outcome is ``(result, None)`` or ``(None, error)``."""
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid-retrieval")
    try:
        futures = {
            "vector": executor.submit(run_vector_branch, request.question, params, resources),
            "graph": executor.submit(run_graph_branch, request, params, resources),
        }
        deadline = time.monotonic() + params.branch_timeout  # one shared deadline, not one each
        outcomes = {}
        for name, future in futures.items():
            try:
                remaining = max(0.0, deadline - time.monotonic())
                outcomes[name] = (future.result(timeout=remaining), None)
            except FutureTimeoutError:
                outcomes[name] = (None, f"{name.capitalize()} retrieval timed out.")
            except Exception as exc:  # any branch error becomes a safe message
                error = _vector_error(exc) if name == "vector" else _graph_error(exc)
                outcomes[name] = (None, error)
        return outcomes["vector"], outcomes["graph"]
    finally:
        # A timed-out branch is not waited for; its thread finishes in the background.
        executor.shutdown(wait=False, cancel_futures=True)


# --- Status --------------------------------------------------------------------------------


def hybrid_status(
    *, vector_ok: bool, graph_ok: bool, incomplete: bool, results: int
) -> HybridRetrievalStatus:
    """``success`` only when both branches completed fully; ``failed`` whenever there is
    nothing usable and something went wrong; ``no_results`` when both completed empty."""
    if not vector_ok and not graph_ok:
        return "failed"
    degraded = not vector_ok or not graph_ok or incomplete
    if results == 0:
        return "failed" if degraded else "no_results"
    return "partial" if degraded else "success"


def _to_result(rank: int, chunk: FusedChunk, rrf_score: float) -> HybridResult:
    return HybridResult(
        rank=rank,
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        content=chunk.content,
        metadata=chunk.metadata,
        retrieval_sources=chunk.sources,
        rrf_score=round(rrf_score, 6),
        vector_rank=chunk.vector_rank,
        vector_score=chunk.vector_score,
        vector_distance=chunk.vector_distance,
        graph_rank=chunk.graph_rank,
        graph_context=chunk.graph_context,
    )


# --- Entry point ----------------------------------------------------------------------------


def retrieve_hybrid(
    request: HybridRetrievalRequest, params: HybridParameters, resources: HybridResources
) -> HybridRetrievalResponse:
    (vector_response, vector_error), (graph_response, graph_error) = _run_branches(
        request, params, resources
    )
    warnings: list[str] = []
    errors = [error for error in (vector_error, graph_error) if error]

    # Vector branch.
    vector_chunks = normalize_vector_results(vector_response) if vector_response else []
    vector_summary = VectorBranchSummary(
        status="failed" if vector_error else "completed",
        results_count=len(vector_chunks),
        error=vector_error,
    )

    # Graph branch: a Story 1.7 ``failed`` status is a branch failure; ``no_match`` is not.
    graph_summary = GraphBranchSummary(status="failed", errors=[graph_error] if graph_error else [])
    graph_chunks: list[FusedChunk] = []
    unresolved = []
    graph_incomplete = False
    if graph_response is not None:
        graph_summary = GraphBranchSummary(
            status="failed" if graph_response.retrieval_status == "failed" else "completed",
            graph_retrieval_status=graph_response.retrieval_status,
            analysis_method=graph_response.analysis_method,
            identified_entities=graph_response.identified_entities,
            matched_entities=graph_response.matched_entities,
            unmatched_entities=graph_response.unmatched_entities,
            paths_count=graph_response.total_results,
            warnings=graph_response.warnings,
            errors=graph_response.errors,
        )
        if graph_response.retrieval_status == "failed":
            errors.append("Graph retrieval failed: no graph query could be completed.")
        graph_incomplete = bool(graph_response.errors)  # e.g. one traversal timed out

        extracted = extract_source_references(graph_response.results)
        unresolved += extracted.incomplete
        graph_summary.source_references_count = len(extracted.references)
        graph_summary.duplicate_references_removed = extracted.duplicates_removed
        if graph_response.results and not extracted.references:
            warnings.append("Graph results carry no usable document/chunk source references.")
        try:
            resolved, missing = resolve_source_references(
                resources.load_collection(), extracted.references
            )
        except VectorStoreError:
            resolved, missing = {}, lookup_failed(extracted.references)
            graph_incomplete = True
            errors.append("Graph source resolution failed: ChromaDB is unavailable.")
        unresolved += missing
        graph_chunks = normalize_graph_results(graph_response.results, resolved)
        graph_summary.resolved_chunks_count = len(graph_chunks)

    if unresolved:
        warnings.append(
            f"{len(unresolved)} graph source reference(s) could not be resolved to a ChromaDB "
            "chunk; see unresolved_references. No content was generated for them."
        )

    # Fusion, de-duplication, RRF ranking, final limit.
    merged = {chunk.key: chunk for chunk in merge_results(vector_chunks, graph_chunks)}
    fused = reciprocal_rank_fusion(
        [[chunk.key for chunk in vector_chunks], [chunk.key for chunk in graph_chunks]],
        k=params.rrf_k,
    )
    ranked = [_to_result(rank, merged[key], score) for rank, (key, score) in enumerate(fused, 1)]
    results = ranked[: params.final_top_k]

    return HybridRetrievalResponse(
        question=request.question,
        retrieval_status=hybrid_status(
            vector_ok=vector_summary.status == "completed",
            graph_ok=graph_summary.status == "completed",
            incomplete=graph_incomplete,
            results=len(results),
        ),
        total_results=len(results),
        vector_results_count=len(vector_chunks),
        graph_results_count=len(graph_chunks),
        merged_results_count=len(merged),
        rrf_k=params.rrf_k,
        results=results,
        vector=vector_summary,
        graph=graph_summary,
        unresolved_references=unresolved,
        warnings=warnings,
        errors=errors,
        ranked_results=ranked,
        graph_paths=graph_response.results if graph_response is not None else [],
    )
