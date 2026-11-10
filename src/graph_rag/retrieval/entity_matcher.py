"""Match query entities to Neo4j nodes (Story 1.7).

Matching runs in this order and stops at the first step that finds something:

1. **Exact**: the stored ``normalized_name``, ``name`` or one of the ``aliases`` equals the
   query name, compared case- and whitespace-insensitively (the Story 1.6 identity rule).
2. **Controlled partial**: only if step 1 found nothing. The stored name must *contain* the
   whole query name on word boundaries ("Indian Penal Code" → "Indian Penal Code, 1860", but
   "Section 13" never → "Section 138"), the query name must be at least
   ``MIN_PARTIAL_LENGTH`` characters, and few candidates are returned. No edit-distance or
   other fuzzy matching.

When the type is known, only nodes with that label are considered. Several candidates are
all returned and flagged ``ambiguous``; nothing is merged and nothing is chosen silently.
All Cypher is parameterized; labels are passed as a parameter list, never formatted in.
"""

import re
from dataclasses import dataclass, field

from graph_rag.graph.entity_extractor import normalize_for_match
from graph_rag.graph.graph_schema import ENTITY_LABEL, NODE_LABELS
from graph_rag.graph.neo4j_client import Neo4jClient
from graph_rag.retrieval.graph_retrieval_models import MatchedEntity, QueryEntity

MAX_CANDIDATES = 5  # per query entity
MIN_PARTIAL_LENGTH = 4
PARTIAL_SCAN_LIMIT = 50  # rows fetched before the word-boundary filter
PROVENANCE_PREVIEW = 20  # document / chunk IDs returned per matched node

_RETURN = f"""
RETURN n.entity_id AS entity_id, n.name AS name, n.normalized_name AS normalized_name,
       n.entity_type AS entity_type,
       coalesce(n.source_document_ids, [])[0..{PROVENANCE_PREVIEW}] AS source_document_ids,
       coalesce(n.source_chunk_ids, [])[0..{PROVENANCE_PREVIEW}] AS source_chunk_ids,
       size(coalesce(n.source_document_ids, [])) AS document_count,
       size(coalesce(n.source_chunk_ids, [])) AS chunk_count"""

EXACT_MATCH_QUERY = f"""
MATCH (n:{ENTITY_LABEL})
WHERE $labels IS NULL OR any(label IN labels(n) WHERE label IN $labels)
WITH n, CASE
    WHEN toLower(trim(n.normalized_name)) IN $keys THEN 'exact_normalized'
    WHEN toLower(trim(n.name)) IN $keys THEN 'exact_name'
    WHEN any(alias IN coalesce(n.aliases, []) WHERE toLower(trim(alias)) IN $keys) THEN 'alias'
  END AS match_type
WHERE match_type IS NOT NULL
{_RETURN}, match_type
ORDER BY CASE match_type WHEN 'exact_normalized' THEN 0 WHEN 'exact_name' THEN 1 ELSE 2 END,
         n.entity_id
LIMIT $max_rows
"""

PARTIAL_MATCH_QUERY = f"""
MATCH (n:{ENTITY_LABEL})
WHERE ($labels IS NULL OR any(label IN labels(n) WHERE label IN $labels))
  AND any(key IN $keys WHERE toLower(n.normalized_name) CONTAINS key
                          OR toLower(n.name) CONTAINS key)
{_RETURN}, 'partial' AS match_type
ORDER BY size(n.normalized_name), n.entity_id
LIMIT $max_rows
"""


@dataclass
class EntityMatchResult:
    query_entity: QueryEntity
    candidates: list[MatchedEntity] = field(default_factory=list)
    truncated: bool = False  # more than MAX_CANDIDATES nodes matched

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1


def match_keys(entity: QueryEntity) -> list[str]:
    """Lower-cased, whitespace-collapsed names to compare: searched name and mention."""
    return list(dict.fromkeys(normalize_for_match(n) for n in (entity.name, entity.mention)))


def labels_for(entity: QueryEntity) -> list[str] | None:
    return [NODE_LABELS[entity.entity_type]] if entity.entity_type else None


def _fetch(tx, query: str, **params) -> list[dict]:
    return [record.data() for record in tx.run(query, **params)]


def _within_word_boundaries(stored: str | None, keys: list[str]) -> bool:
    text = normalize_for_match(stored or "")
    return any(re.search(rf"(?<!\w){re.escape(key)}(?!\w)", text) for key in keys)


def _to_candidates(rows: list[dict], entity: QueryEntity) -> list[MatchedEntity]:
    candidates = []
    for row in rows:
        if not row.get("entity_id") or not row.get("name"):
            continue  # malformed node: cannot be identified or shown
        candidates.append(MatchedEntity(query_name=entity.name, **row))
    return candidates


def find_matches(tx, entity: QueryEntity) -> EntityMatchResult:
    """Transaction function: exact match, then controlled partial match if nothing exact."""
    keys = match_keys(entity)
    labels = labels_for(entity)
    rows = _fetch(tx, EXACT_MATCH_QUERY, keys=keys, labels=labels, max_rows=MAX_CANDIDATES + 1)
    if not rows:
        partial_keys = [key for key in keys if len(key) >= MIN_PARTIAL_LENGTH]
        if partial_keys:
            scanned = _fetch(
                tx,
                PARTIAL_MATCH_QUERY,
                keys=partial_keys,
                labels=labels,
                max_rows=PARTIAL_SCAN_LIMIT,
            )
            rows = [
                row
                for row in scanned
                if _within_word_boundaries(row.get("normalized_name"), partial_keys)
                or _within_word_boundaries(row.get("name"), partial_keys)
            ][: MAX_CANDIDATES + 1]

    candidates = _to_candidates(rows, entity)
    result = EntityMatchResult(
        query_entity=entity,
        candidates=candidates[:MAX_CANDIDATES],
        truncated=len(candidates) > MAX_CANDIDATES,
    )
    for candidate in result.candidates:
        candidate.ambiguous = result.ambiguous
    return result


def match_entity(
    client: Neo4jClient, entity: QueryEntity, *, timeout: float | None = None
) -> EntityMatchResult:
    """Match one query entity in a read transaction. Raises ``GraphStoreError`` on failure."""
    return client.execute_read(find_matches, timeout=timeout, entity=entity)
