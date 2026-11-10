"""Neo4j graph data model: labels, relationship types, constraints and indexes (Story 1.6).

Every entity node carries two labels: the shared ``Entity`` label (one uniqueness constraint on
``entity_id`` for all entities) and a type label such as ``Person``. Relationship types are
exactly the Story 1.5 ``RelationshipType`` values. All statements use ``IF NOT EXISTS``, so
initialization can be run any number of times and never drops anything.
"""

from graph_rag.graph.models import EntityType, RelationshipType
from graph_rag.graph.neo4j_client import GraphStoreError, Neo4jClient

ENTITY_LABEL = "Entity"

NODE_LABELS: dict[EntityType, str] = {
    EntityType.PERSON: "Person",
    EntityType.COURT: "Court",
    EntityType.ORGANIZATION: "Organization",
    EntityType.CASE: "Case",
    EntityType.STATUTE: "Statute",
    EntityType.LEGAL_SECTION: "LegalSection",
    EntityType.DATE: "Date",
    EntityType.LOCATION: "Location",
    EntityType.LEGAL_CONCEPT: "LegalConcept",
}

# Neo4j relationship types are the validated Story 1.5 enum values; nothing else is written.
RELATIONSHIP_TYPES: tuple[str, ...] = tuple(item.value for item in RelationshipType)

# Relationship property uniqueness constraints exist from Neo4j 5.7.
_RELATIONSHIP_CONSTRAINT_VERSION = (5, 7)
_MINIMUM_VERSION = (5, 0)


def _snake(name: str) -> str:
    return "".join(f"_{c.lower()}" if c.isupper() else c for c in name).lstrip("_")


def schema_statements(server_version: tuple[int, int]) -> tuple[list[str], list[str]]:
    """Return ``(constraints, indexes)`` Cypher statements for the given server version."""
    constraints = [
        f"CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
        f"FOR (n:{ENTITY_LABEL}) REQUIRE n.entity_id IS UNIQUE"
    ]
    indexes = [
        f"CREATE INDEX entity_normalized_name IF NOT EXISTS "
        f"FOR (n:{ENTITY_LABEL}) ON (n.normalized_name)",
        f"CREATE INDEX entity_type IF NOT EXISTS FOR (n:{ENTITY_LABEL}) ON (n.entity_type)",
    ]
    for label in NODE_LABELS.values():
        indexes.append(
            f"CREATE INDEX {_snake(label)}_normalized_name IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.normalized_name)"
        )
    for rel_type in RELATIONSHIP_TYPES:
        name = f"{rel_type.lower()}_relationship_id"
        if server_version >= _RELATIONSHIP_CONSTRAINT_VERSION:
            constraints.append(
                f"CREATE CONSTRAINT {name}_unique IF NOT EXISTS "
                f"FOR ()-[r:{rel_type}]-() REQUIRE r.relationship_id IS UNIQUE"
            )
        else:
            indexes.append(
                f"CREATE INDEX {name} IF NOT EXISTS FOR ()-[r:{rel_type}]-() ON (r.relationship_id)"
            )
    return constraints, indexes


def _name_of(statement: str) -> str:
    return statement.split()[2]  # "CREATE CONSTRAINT <name> IF NOT EXISTS ..."


def initialize_schema(client: Neo4jClient) -> dict:
    """Verify connectivity, then create missing constraints and indexes. Idempotent."""
    client.verify_connectivity()
    agent = client.server_agent()
    version = client.server_version()
    if version < _MINIMUM_VERSION:
        raise GraphStoreError(f"Neo4j {agent} is not supported; Neo4j 5.x or later is required.")
    constraints, indexes = schema_statements(version)
    client.run_schema_statements(constraints + indexes)
    client.schema_ready = True
    return {
        "server_agent": agent,
        "database": client.database,
        "constraints": [_name_of(statement) for statement in constraints],
        "indexes": [_name_of(statement) for statement in indexes],
    }


def ensure_schema(client: Neo4jClient) -> None:
    """Initialize the schema once per process before the first graph write."""
    if not client.schema_ready:
        initialize_schema(client)
