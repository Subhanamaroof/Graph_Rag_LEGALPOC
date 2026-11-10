"""Graph retrieval orchestration (Story 1.7).

question → query_analyzer (entities, intent) → entity_matcher (Neo4j nodes)
         → graph_traversal (bounded paths, per matched node) → de-duplication
         → graph_ranker (scores, provenance) → limit → GraphRetrievalResponse

Every Neo4j call is a read transaction. A Neo4j failure for one entity or one start node is
recorded in ``errors`` and the rest continues (status ``partial``); if nothing could be read,
the status is ``failed``. A question with no identifiable or matching entity is ``no_match``,
which is a normal outcome, not an error.
"""

from graph_rag.core.llm import JSONCompletionClient
from graph_rag.graph.neo4j_client import GraphQueryTimeoutError, GraphStoreError, Neo4jClient
from graph_rag.retrieval.entity_matcher import EntityMatchResult, match_entity
from graph_rag.retrieval.graph_ranker import build_result, rank_results
from graph_rag.retrieval.graph_retrieval_models import (
    GraphRetrievalRequest,
    GraphRetrievalResponse,
    MatchedEntity,
    RetrievalStatus,
)
from graph_rag.retrieval.graph_traversal import GraphPath, deduplicate_paths, traverse
from graph_rag.retrieval.query_analyzer import analyze_query

PATH_OVERFETCH = 3  # paths fetched per start node = limit * 3, so ranking has a choice
MAX_PATHS_PER_START = 300


def _error_message(stage: str, name: str, exc: GraphStoreError) -> str:
    kind = "timed out" if isinstance(exc, GraphQueryTimeoutError) else "failed"
    return f"{stage} {kind} for '{name}': {exc}"


def _status(
    *, attempted: int, failed: int, results: int, unmatched: int, errors: int
) -> RetrievalStatus:
    if attempted and failed == attempted:
        return "failed"
    if errors:
        return "partial"
    if results == 0:
        return "no_results"
    return "partial" if unmatched else "success"


def retrieve_graph(
    request: GraphRetrievalRequest,
    client: Neo4jClient,
    llm: JSONCompletionClient | None = None,
    *,
    timeout: float | None = None,
) -> GraphRetrievalResponse:
    analysis = analyze_query(request.question, llm)
    response = GraphRetrievalResponse(
        question=request.question,
        analysis_method=analysis.method,
        identified_entities=analysis.entities,
        keywords=analysis.keywords,
        relationship_intent=analysis.relationship_intent,
        retrieval_status="no_match",
        warnings=list(analysis.warnings),
    )
    if not analysis.entities:
        response.warnings.append("No legal entity was identified in the question.")
        return response

    # 1. Entity matching (one read transaction per query entity).
    matches: list[EntityMatchResult] = []
    for entity in analysis.entities:
        try:
            matches.append(match_entity(client, entity, timeout=timeout))
        except GraphStoreError as exc:
            response.errors.append(_error_message("Entity matching", entity.name, exc))
    if len(response.errors) == len(analysis.entities):
        response.retrieval_status = "failed"
        return response

    starts: dict[str, MatchedEntity] = {}  # one traversal per node, even if matched twice
    for match in matches:
        if not match.candidates:
            response.unmatched_entities.append(match.query_entity)
            continue
        if match.ambiguous:
            response.ambiguous_entities.append(match.query_entity.name)
            response.warnings.append(
                f"'{match.query_entity.name}' matched {len(match.candidates)} nodes"
                f"{' (more not shown)' if match.truncated else ''}; results from each are "
                "returned separately."
            )
        response.matched_entities.extend(match.candidates)
        for candidate in match.candidates:
            starts.setdefault(candidate.entity_id, candidate)
    if not starts:
        response.retrieval_status = "partial" if response.errors else "no_match"
        return response

    # 2. Bounded traversal from every matched node.
    type_known = {m.query_entity.name: m.query_entity.entity_type is not None for m in matches}
    path_limit = min(request.limit * PATH_OVERFETCH, MAX_PATHS_PER_START)
    rel_types = [t.value for t in request.relationship_types or []] or None
    end_types = [t.value for t in request.entity_types or []] or None
    results = []
    failed_starts = malformed = 0
    for entity_id, matched in starts.items():
        try:
            traversal = traverse(
                client,
                entity_id,
                max_hops=request.max_hops,
                relationship_types=rel_types,
                end_types=end_types,
                path_limit=path_limit,
                timeout=timeout,
            )
        except GraphStoreError as exc:
            failed_starts += 1
            response.errors.append(_error_message("Graph traversal", matched.name, exc))
            continue
        malformed += traversal.malformed
        paths: list[GraphPath] = deduplicate_paths(traversal.paths)
        if not paths:
            response.warnings.append(f"'{matched.name}' has no matching connected relationships.")
        results += [
            build_result(
                path,
                matched,
                type_known=type_known[matched.query_name],
                intent=analysis.relationship_intent,
            )
            for path in paths
        ]

    # 3. Rank, enforce the limit, report.
    ranked = rank_results(results)
    response.results = ranked[: request.limit]
    response.total_results = len(response.results)
    response.truncated = len(ranked) > request.limit
    if malformed:
        response.warnings.append(f"{malformed} malformed graph path(s) were skipped.")
    if any(not result.provenance_complete for result in response.results):
        response.warnings.append(
            "Some relationships have incomplete provenance (missing evidence, document or "
            "chunk ID); see provenance_complete on each result."
        )
    response.retrieval_status = _status(
        attempted=len(starts),
        failed=failed_starts,
        results=response.total_results,
        unmatched=len(response.unmatched_entities),
        errors=len(response.errors),
    )
    return response
