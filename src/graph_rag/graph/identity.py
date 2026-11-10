"""Graph entity identity: which extracted mentions become the same Neo4j node (Story 1.6).

Story 1.5 ``entity_id`` values are chunk-scoped (document + chunk + type + normalized name), so
the same court mentioned in two chunks has two different extraction IDs. A graph node needs an
identity that is stable across chunks and repeated runs, but must not merge unrelated things.

Rule: ``graph entity_id = hash(scope, entity_type, match_key(normalized_name))``

- ``match_key`` is the Story 1.5 normalized name made case-, whitespace- and quote-insensitive.
  No fuzzy matching: "Supreme Court" and "Supreme Court of India" stay different unless Story
  1.5 already normalized them to the same name.
- ``scope`` decides how far a name is trusted:
  - ``document:<id>`` for PERSON, ORGANIZATION and LEGAL_SECTION. Two people (or two "the
    Company"s, or two "Section 138"s of different Acts) with the same name in different
    judgments are not merged; inside one judgment the same name is the same party.
  - ``global`` for COURT, STATUTE, CASE, DATE, LOCATION and LEGAL_CONCEPT, whose names identify
    the thing itself, so judgments get connected through them.

Kept separate from persistence so a better resolver can replace it later.
"""

import hashlib

from graph_rag.graph.entity_extractor import normalize_for_match
from graph_rag.graph.models import Entity, EntityType

DOCUMENT_SCOPED_TYPES = frozenset(
    {EntityType.PERSON, EntityType.ORGANIZATION, EntityType.LEGAL_SECTION}
)


def identity_scope(entity: Entity) -> str:
    if entity.entity_type in DOCUMENT_SCOPED_TYPES:
        return f"document:{entity.source_document_id}"
    return "global"


def match_key(entity: Entity) -> str:
    return normalize_for_match(entity.normalized_name)


def graph_entity_id(entity: Entity) -> str:
    """Deterministic node ID: same scope + type + match key -> same node, on every run."""
    parts = (identity_scope(entity), entity.entity_type.value, match_key(entity))
    return "node_" + hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:20]


def graph_relationship_id(
    source_node_id: str, relationship_type: str, target_node_id: str, chunk_id: str
) -> str:
    """One edge per (source, type, target, chunk): each chunk's evidence is kept separately."""
    parts = (source_node_id, relationship_type, target_node_id, chunk_id)
    return "edge_" + hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:20]
