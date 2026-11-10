"""Deterministic ranking of retrieved graph paths (Story 1.7).

score = 0.35 * match + 0.30 * proximity + 0.20 * relevance + 0.10 * type_match + 0.05 * support

- match: how the start node was matched (exact 1.0, alias 0.9, partial 0.5).
- proximity: 1 / hop_distance (a direct relationship beats an indirect connection).
- relevance: share of the path's edges whose type is the question's relationship intent;
  0.5 (neutral) when no intent was identified.
- type_match: 1.0 if the query named the entity type (and so the label was enforced), else 0.5.
- support: number of edges behind the path (parallel edges from other chunks), capped at 5,
  divided by 5.

Ties are broken by hop distance, then by the readable path, so the order is reproducible.
The score only orders graph evidence; it says nothing about legal relevance or correctness.
"""

from graph_rag.graph.models import RelationshipType
from graph_rag.retrieval.graph_retrieval_models import (
    Evidence,
    GraphEdge,
    GraphNode,
    GraphResult,
    MatchedEntity,
)
from graph_rag.retrieval.graph_traversal import GraphPath

WEIGHTS = {"match": 0.35, "proximity": 0.30, "relevance": 0.20, "type_match": 0.10, "support": 0.05}
MATCH_SCORES = {"exact_normalized": 1.0, "exact_name": 1.0, "alias": 0.9, "partial": 0.5}
SUPPORT_CAP = 5


def _node_text(node: GraphNode) -> str:
    return f"({node.name}:{node.entity_type or '?'})"


def describe_path(path: GraphPath) -> str:
    """e.g. ``(Indian Penal Code:STATUTE) <-[REFERS_TO]- (Delhi High Court:COURT)``."""
    parts = [_node_text(path.nodes[0])]
    for index, edge in enumerate(path.edges):
        forward = edge.source_entity_id == path.nodes[index].entity_id
        arrow = f"-[{edge.relationship_type}]->" if forward else f"<-[{edge.relationship_type}]-"
        parts += [arrow, _node_text(path.nodes[index + 1])]
    return " ".join(parts)


def ranking_signals(
    path: GraphPath,
    matched: MatchedEntity,
    *,
    type_known: bool,
    intent: RelationshipType | None,
) -> dict[str, float]:
    if intent is None:
        relevance = 0.5
    else:
        relevance = sum(e.relationship_type == intent.value for e in path.edges) / len(path.edges)
    support = min(len(path.edges) + len(path.extra_edges), SUPPORT_CAP) / SUPPORT_CAP
    return {
        "match": MATCH_SCORES[matched.match_type],
        "proximity": 1.0 / path.hop_distance,
        "relevance": round(relevance, 4),
        "type_match": 1.0 if type_known else 0.5,
        "support": support,
    }


def _evidence(edges: list[GraphEdge]) -> list[Evidence]:
    return [
        Evidence(
            relationship_id=edge.relationship_id,
            relationship_type=edge.relationship_type,
            evidence=edge.evidence,
            document_id=edge.source_document_id,
            chunk_id=edge.source_chunk_id,
        )
        for edge in edges
    ]


def build_result(
    path: GraphPath,
    matched: MatchedEntity,
    *,
    type_known: bool,
    intent: RelationshipType | None,
) -> GraphResult:
    """Turn one de-duplicated path into an unranked result with all its provenance."""
    signals = ranking_signals(path, matched, type_known=type_known, intent=intent)
    all_edges = path.edges + path.extra_edges
    return GraphResult(
        score=round(sum(WEIGHTS[name] * value for name, value in signals.items()), 4),
        ranking_signals=signals,
        query_entity=matched.query_name,
        matched_entity=path.nodes[0],
        related_node=path.nodes[-1],
        nodes=path.nodes,
        relationships=path.edges,
        path=describe_path(path),
        hop_distance=path.hop_distance,
        supporting_edges=len(all_edges),
        evidence=_evidence(all_edges),
        source_document_ids=sorted(
            {e.source_document_id for e in all_edges if e.source_document_id}
        ),
        source_chunk_ids=sorted({e.source_chunk_id for e in all_edges if e.source_chunk_id}),
        provenance_complete=all(edge.provenance_complete for edge in all_edges),
    )


def rank_results(results: list[GraphResult]) -> list[GraphResult]:
    """Sort best first and number the ranks from 1."""
    ordered = sorted(results, key=lambda r: (-r.score, r.hop_distance, r.path))
    for rank, result in enumerate(ordered, start=1):
        result.rank = rank
    return ordered
