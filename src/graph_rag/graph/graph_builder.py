"""Map one validated ``ChunkExtraction`` to node and edge rows for Neo4j (Story 1.6).

Pure Python, no database access: the rows are exactly the parameters the Cypher statements in
``graph_service`` receive, which keeps the mapping easy to test and to read.
"""

from dataclasses import dataclass, field

from graph_rag.graph.graph_schema import NODE_LABELS
from graph_rag.graph.identity import graph_entity_id, graph_relationship_id, identity_scope
from graph_rag.graph.models import ChunkExtraction


@dataclass
class ChunkGraphRows:
    """Rows grouped by node label / relationship type (labels and types cannot be parameters)."""

    nodes: dict[str, list[dict]] = field(default_factory=dict)
    relationships: dict[str, list[dict]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def node_count(self) -> int:
        return sum(len(rows) for rows in self.nodes.values())

    @property
    def relationship_count(self) -> int:
        return sum(len(rows) for rows in self.relationships.values())


def build_chunk_rows(chunk: ChunkExtraction) -> ChunkGraphRows:
    """Resolve graph identities and build node/edge rows for one completed chunk."""
    if chunk.extraction_status != "completed":
        raise ValueError(f"chunk {chunk.chunk_id} did not complete extraction")
    rows = ChunkGraphRows()
    node_rows: dict[str, dict] = {}  # graph entity_id -> row
    node_id_for: dict[str, str] = {}  # Story 1.5 entity_id -> graph entity_id

    for entity in chunk.entities:
        node_id = graph_entity_id(entity)
        node_id_for[entity.entity_id] = node_id
        names = [entity.name, *entity.aliases]
        if node_id in node_rows:  # two extracted entities resolve to one node: keep the first
            existing = node_rows[node_id]
            existing["aliases"] += [n for n in names if n not in existing["aliases"]]
            continue
        node_rows[node_id] = {
            "entity_id": node_id,
            "label": NODE_LABELS[entity.entity_type],
            "name": entity.name,
            "normalized_name": entity.normalized_name,
            "entity_type": entity.entity_type.value,
            "identity_scope": identity_scope(entity),
            "description": entity.description,
            "aliases": list(dict.fromkeys(names)),
            "source_text": entity.source_text,
            "document_id": chunk.document_id,
            "chunk_id": chunk.chunk_id,
        }
    for row in node_rows.values():
        rows.nodes.setdefault(row.pop("label"), []).append(row)

    seen: set[str] = set()
    for relationship in chunk.relationships:
        source = node_id_for.get(relationship.source_entity_id)
        target = node_id_for.get(relationship.target_entity_id)
        if source is None or target is None:  # guarded by ChunkExtraction, checked again here
            rows.warnings.append(f"{relationship.relationship_id}: endpoint not in this chunk")
            continue
        if source == target:
            rows.warnings.append(
                f"{relationship.relationship_id}: both endpoints resolve to one node; skipped"
            )
            continue
        rel_type = relationship.relationship_type.value
        edge_id = graph_relationship_id(source, rel_type, target, chunk.chunk_id)
        if edge_id in seen:
            continue
        seen.add(edge_id)
        rows.relationships.setdefault(rel_type, []).append(
            {
                "relationship_id": edge_id,
                "source_id": source,
                "target_id": target,
                "description": relationship.description,
                "evidence": relationship.evidence,
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
            }
        )
    return rows
