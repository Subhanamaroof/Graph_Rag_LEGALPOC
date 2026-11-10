"""Bounded Neo4j traversal from a matched node (Story 1.7).

Traversal is breadth-first by hop level: first every path of exactly 1 hop, then exactly 2
hops, up to ``max_hops`` (at most 3). Each level is one Cypher query with its own ``LIMIT``,
and the whole traversal stops once ``path_limit`` paths were collected, so a hub node (for
example a court mentioned in every judgment) can never return the whole graph.

Edges are walked in both directions — "which cases cited the IPC" needs the *incoming*
REFERS_TO edges of the IPC node — but every returned edge keeps its stored direction
(``source_entity_id -> target_entity_id``).

Only the hop count is formatted into the Cypher text (an ``int`` validated to 1..3);
relationship types, entity types and IDs are query parameters.
"""

from dataclasses import dataclass, field

from pydantic import ValidationError

from graph_rag.graph.graph_schema import ENTITY_LABEL, RELATIONSHIP_TYPES
from graph_rag.graph.neo4j_client import Neo4jClient
from graph_rag.retrieval.graph_retrieval_models import MAX_HOPS, GraphEdge, GraphNode


def traversal_query(hops: int) -> str:
    """Paths of exactly ``hops`` edges from the start node, filtered by edge and end type."""
    if not isinstance(hops, int) or isinstance(hops, bool) or not 1 <= hops <= MAX_HOPS:
        raise ValueError(f"hops must be an integer between 1 and {MAX_HOPS}")
    return f"""
MATCH p = (start:{ENTITY_LABEL} {{entity_id: $start_id}})-[*{hops}..{hops}]-(end:{ENTITY_LABEL})
WHERE end.entity_id <> $start_id
  AND all(r IN relationships(p) WHERE type(r) IN $relationship_types)
  AND ($end_types IS NULL OR end.entity_type IN $end_types)
RETURN [n IN nodes(p) | n {{.entity_id, .name, .entity_type, .description}}] AS nodes,
       [r IN relationships(p) | r {{.relationship_id, .description, .evidence,
            .source_document_id, .source_chunk_id, relationship_type: type(r),
            source_entity_id: startNode(r).entity_id,
            target_entity_id: endNode(r).entity_id}}] AS relationships
LIMIT $path_limit
"""


@dataclass
class GraphPath:
    """One path from the start node. ``edges[i]`` connects ``nodes[i]`` and ``nodes[i+1]``."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    # Parallel edges merged into this path by ``deduplicate_paths`` (same nodes and types).
    extra_edges: list[GraphEdge] = field(default_factory=list)

    @property
    def hop_distance(self) -> int:
        return len(self.edges)

    @property
    def key(self) -> tuple:
        """Identity of a path for de-duplication: node sequence + edge types and directions."""
        return (
            tuple(node.entity_id for node in self.nodes),
            tuple((e.relationship_type, e.source_entity_id) for e in self.edges),
        )


@dataclass
class TraversalResult:
    paths: list[GraphPath] = field(default_factory=list)
    malformed: int = 0  # records skipped because required fields were missing or inconsistent
    cycles: int = 0  # paths skipped because they revisit a node


def parse_path(record: dict) -> GraphPath | None:
    """Validate one record; ``None`` if the stored data is malformed."""
    try:
        nodes = [GraphNode.model_validate(node) for node in record["nodes"]]
        edges = [GraphEdge.model_validate(edge) for edge in record["relationships"]]
    except (KeyError, TypeError, ValidationError):
        return None
    if len(nodes) != len(edges) + 1 or not edges:
        return None
    for index, edge in enumerate(edges):
        if {edge.source_entity_id, edge.target_entity_id} != {
            nodes[index].entity_id,
            nodes[index + 1].entity_id,
        }:
            return None
    return GraphPath(nodes=nodes, edges=edges)


def fetch_paths(
    tx,
    *,
    start_id: str,
    max_hops: int,
    relationship_types: list[str],
    end_types: list[str] | None,
    path_limit: int,
) -> TraversalResult:
    """Transaction function: run hop levels 1..max_hops until ``path_limit`` paths exist."""
    result = TraversalResult()
    for hops in range(1, max_hops + 1):
        remaining = path_limit - len(result.paths)
        if remaining <= 0:
            break
        records = tx.run(
            traversal_query(hops),
            start_id=start_id,
            relationship_types=relationship_types,
            end_types=end_types,
            path_limit=remaining,
        )
        for record in records:
            path = parse_path(record.data())
            if path is None:
                result.malformed += 1
            elif len({node.entity_id for node in path.nodes}) != len(path.nodes):
                result.cycles += 1
            else:
                result.paths.append(path)
    return result


def traverse(
    client: Neo4jClient,
    start_id: str,
    *,
    max_hops: int,
    relationship_types: list[str] | None = None,
    end_types: list[str] | None = None,
    path_limit: int,
    timeout: float | None = None,
) -> TraversalResult:
    """Bounded traversal in one read transaction. Raises ``GraphStoreError`` on failure."""
    traversal_query(max_hops)  # validates the bound before touching Neo4j
    allowed = list(RELATIONSHIP_TYPES)
    types = [t for t in relationship_types if t in allowed] if relationship_types else allowed
    return client.execute_read(
        fetch_paths,
        timeout=timeout,
        start_id=start_id,
        max_hops=max_hops,
        relationship_types=types,
        end_types=end_types,
        path_limit=path_limit,
    )


def deduplicate_paths(paths: list[GraphPath]) -> list[GraphPath]:
    """Merge paths with the same nodes, edge types and directions.

    Story 1.6 stores one edge per chunk, so "A REFERS_TO B" said in three chunks is three
    parallel edges and three otherwise identical paths. They become one path whose
    ``extra_edges`` keep the other chunks' evidence. Exact repeats are dropped.
    """
    merged: dict[tuple, GraphPath] = {}
    for path in paths:
        existing = merged.get(path.key)
        if existing is None:
            merged[path.key] = GraphPath(nodes=path.nodes, edges=path.edges)
            continue
        known = {e.relationship_id for e in existing.edges + existing.extra_edges}
        existing.extra_edges += [e for e in path.edges if e.relationship_id not in known]
    return list(merged.values())
