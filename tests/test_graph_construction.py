"""Story 1.6 tests: Neo4j knowledge graph construction.

Neo4j is replaced by ``FakeDriver``: an in-memory graph that applies our MERGE statements'
semantics (match-or-create by ID, provenance appended, edges only between existing nodes) and
rolls back a transaction whose function raises. Groq and ChromaDB are faked as well, so no
real database or API is touched. The Cypher text itself is checked against a live server in
the manual tests.
"""

import copy
import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from neo4j.exceptions import AuthError, ServiceUnavailable
from pydantic import SecretStr

from graph_rag.core.config import Settings
from graph_rag.graph import neo4j_client as neo4j_module
from graph_rag.graph.graph_builder import build_chunk_rows
from graph_rag.graph.graph_router import get_graph_client
from graph_rag.graph.graph_schema import NODE_LABELS, initialize_schema, schema_statements
from graph_rag.graph.graph_service import (
    MissingEndpointError,
    build_document_graph,
    persist_chunk,
    persist_extraction,
    relationship_merge_query,
    write_chunk_rows,
)
from graph_rag.graph.identity import graph_entity_id
from graph_rag.graph.models import ChunkExtraction, Entity, ExtractionResponse, Relationship
from graph_rag.graph.neo4j_client import (
    GraphConnectionError,
    GraphNotConfiguredError,
    GraphStoreError,
    Neo4jClient,
)
from graph_rag.graph.router import get_extraction_llm
from graph_rag.main import app
from graph_rag.retrieval.router import get_chroma_collection

DOC = "NI138-2019-0042"
PASSWORD = "super-secret-pw"


# --- In-memory fake Neo4j -------------------------------------------------------------


class FakeResult:
    def __init__(self, written, nodes_created=0, relationships_created=0):
        self._written = written
        self._counters = SimpleNamespace(
            nodes_created=nodes_created, relationships_created=relationships_created
        )

    def single(self):
        return {"written": self._written}

    def consume(self):
        return SimpleNamespace(counters=self._counters)


class FakeTx:
    def __init__(self, graph, fail_on=None):
        self.graph = graph
        self.fail_on = fail_on

    def run(self, query, rows=()):
        if self.fail_on and self.fail_on in query:
            raise RuntimeError("simulated failure")
        if "MERGE (n:Entity" in query:
            label = re.search(r"SET n:(\w+)", query).group(1)
            created = 0
            for row in rows:
                node = self.graph.nodes.get(row["entity_id"])
                if node is None:
                    created += 1
                    node = {
                        **row,
                        "labels": {"Entity", label},
                        "aliases": [],
                        "source_document_ids": [],
                        "source_chunk_ids": [],
                    }
                    self.graph.nodes[row["entity_id"]] = node
                node["aliases"] += [a for a in row["aliases"] if a not in node["aliases"]]
                for key, value in (
                    ("source_document_ids", row["document_id"]),
                    ("source_chunk_ids", row["chunk_id"]),
                ):
                    if value not in node[key]:
                        node[key].append(value)
            return FakeResult(len(rows), nodes_created=created)
        rel_type = re.search(r"\[r:(\w+)", query).group(1)
        written = created = 0
        for row in rows:
            if row["source_id"] not in self.graph.nodes or row["target_id"] not in self.graph.nodes:
                continue
            written += 1
            if row["relationship_id"] not in self.graph.edges:
                created += 1
                self.graph.edges[row["relationship_id"]] = {**row, "type": rel_type}
        return FakeResult(written, relationships_created=created)


class FakeSession:
    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.driver.sessions_closed += 1

    def execute_write(self, work, **kwargs):
        snapshot = (copy.deepcopy(self.driver.nodes), copy.deepcopy(self.driver.edges))
        try:
            return work(FakeTx(self.driver, self.driver.fail_on), **kwargs)
        except Exception:
            self.driver.nodes, self.driver.edges = snapshot  # rollback
            raise

    def run(self, statement):
        self.driver.schema.add(statement)
        return SimpleNamespace(consume=lambda: None)


class FakeDriver:
    def __init__(self, verify_error=None, agent="Neo4j/5.27-aura"):
        self.nodes: dict[str, dict] = {}
        self.edges: dict[str, dict] = {}
        self.schema: set[str] = set()
        self.verify_error = verify_error
        self.agent = agent
        self.fail_on = None
        self.closed = False
        self.sessions_closed = 0

    def verify_connectivity(self):
        if self.verify_error:
            raise self.verify_error

    def get_server_info(self):
        return SimpleNamespace(agent=self.agent)

    def session(self, database):
        return FakeSession(self)

    def close(self):
        self.closed = True


@pytest.fixture
def driver():
    return FakeDriver()


@pytest.fixture
def client(driver):
    return Neo4jClient(driver, "neo4j")


# --- Extraction fixtures (Story 1.5 output shapes; no Groq) ---------------------------


def make_entity(doc, chunk, key, name, etype, normalized=None):
    return Entity(
        entity_id=f"ent_{chunk}_{key}",
        name=name,
        normalized_name=normalized or name,
        entity_type=etype,
        source_document_id=doc,
        source_chunk_id=chunk,
        source_text=name,
    )


def make_chunk(doc=DOC, chunk="0", relationships=True, extra_entities=()):
    chunk_id = f"{doc}_chunk{chunk}"
    entities = [
        make_entity(doc, chunk_id, "court", "Delhi High Court", "COURT"),
        make_entity(doc, chunk_id, "person", "Rajesh Kumar", "PERSON"),
        make_entity(doc, chunk_id, "statute", "Negotiable Instruments Act, 1881", "STATUTE"),
        *extra_entities,
    ]
    rels = [
        Relationship(
            relationship_id=f"rel_{chunk_id}_1",
            source_entity_id=entities[1].entity_id,
            target_entity_id=entities[0].entity_id,
            relationship_type="RELATED_TO",
            evidence="Rajesh Kumar ... before the Delhi High Court",
            source_document_id=doc,
            source_chunk_id=chunk_id,
        )
    ]
    return ChunkExtraction(
        document_id=doc,
        chunk_id=chunk_id,
        entities=entities,
        relationships=rels if relationships else [],
        extraction_status="completed",
    )


def failed(chunk="9"):
    return ChunkExtraction(
        document_id=DOC,
        chunk_id=f"{DOC}_chunk{chunk}",
        extraction_status="failed",
        error="LLM timeout",
    )


def extraction(results, skipped=0):
    return ExtractionResponse(
        document_id=DOC,
        extraction_status="completed",
        total_chunks=len(results) + skipped,
        chunks_processed=len(results),
        chunks_succeeded=sum(r.extraction_status == "completed" for r in results),
        chunks_failed=sum(r.extraction_status == "failed" for r in results),
        chunks_skipped=skipped,
        total_entities=sum(len(r.entities) for r in results),
        total_relationships=sum(len(r.relationships) for r in results),
        results=results,
    )


# --- 1-3. Configuration and connection manager ----------------------------------------


def test_neo4j_settings_load_and_hide_password():
    settings = Settings(
        _env_file=None,
        neo4j_uri="neo4j://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password=PASSWORD,
        neo4j_database="graphdb",
    )
    assert settings.neo4j_database == "graphdb"
    assert isinstance(settings.neo4j_password, SecretStr)
    assert PASSWORD not in repr(settings)


def test_client_from_settings_verifies_and_closes(driver, monkeypatch):
    calls = {}

    def factory(uri, auth):
        calls["uri"], calls["auth"] = uri, auth
        return driver

    settings = Settings(
        _env_file=None, neo4j_uri="neo4j://x", neo4j_username="u", neo4j_password=PASSWORD
    )
    client = Neo4jClient.from_settings(settings, driver_factory=factory)
    assert calls["auth"] == ("u", PASSWORD) and client.database == "neo4j"

    monkeypatch.setattr(neo4j_module, "_client", client)
    neo4j_module.close_neo4j_client()
    assert driver.closed and neo4j_module._client is None


def test_missing_configuration_is_reported():
    with pytest.raises(GraphNotConfiguredError, match="NEO4J_URI"):
        Neo4jClient.from_settings(Settings(_env_file=None))


@pytest.mark.parametrize(
    "error, message",
    [(ServiceUnavailable("unreachable"), "Cannot connect"), (AuthError("bad"), "credentials")],
)
def test_connection_failure_is_clear_and_closes_driver(error, message):
    bad = FakeDriver(verify_error=error)
    settings = Settings(
        _env_file=None, neo4j_uri="neo4j://x", neo4j_username="u", neo4j_password=PASSWORD
    )
    with pytest.raises(GraphConnectionError, match=message) as info:
        Neo4jClient.from_settings(settings, driver_factory=lambda *a, **k: bad)
    assert PASSWORD not in str(info.value)
    assert bad.closed


# --- 4-5. Schema --------------------------------------------------------------------


def test_schema_initialization_is_idempotent_and_unique(client, driver):
    first = initialize_schema(client)
    count = len(driver.schema)
    second = initialize_schema(client)
    assert first == second and len(driver.schema) == count
    assert all("IF NOT EXISTS" in s for s in driver.schema)
    assert not any(word in s for s in driver.schema for word in ("DROP", "DELETE"))
    assert "entity_id_unique" in first["constraints"]
    assert any("REQUIRE n.entity_id IS UNIQUE" in s for s in driver.schema)
    assert any("REQUIRE r.relationship_id IS UNIQUE" in s for s in driver.schema)
    assert len([i for i in first["indexes"] if i.endswith("_normalized_name")]) == 10


def test_older_server_uses_relationship_indexes():
    constraints, indexes = schema_statements((5, 4))
    assert constraints == [constraints[0]]
    assert any("ON (r.relationship_id)" in s for s in indexes)


def test_unsupported_server_version_is_rejected():
    with pytest.raises(GraphStoreError, match="not supported"):
        initialize_schema(Neo4jClient(FakeDriver(agent="Neo4j/4.4.0"), "neo4j"))


# --- 6-7. Mapping and identity ----------------------------------------------------------


def test_entities_map_to_labelled_node_rows_with_provenance():
    rows = build_chunk_rows(make_chunk())
    assert set(rows.nodes) == {"Court", "Person", "Statute"}
    person = rows.nodes["Person"][0]
    assert person["entity_type"] == "PERSON"
    assert person["document_id"] == DOC and person["chunk_id"] == f"{DOC}_chunk0"
    assert person["identity_scope"] == f"document:{DOC}"
    assert rows.nodes["Court"][0]["identity_scope"] == "global"
    assert set(NODE_LABELS.values()) >= set(rows.nodes)


def test_relationships_map_to_typed_edges_between_graph_ids():
    chunk = make_chunk()
    rows = build_chunk_rows(chunk)
    edge = rows.relationships["RELATED_TO"][0]
    assert edge["source_id"] == graph_entity_id(chunk.entities[1])
    assert edge["target_id"] == graph_entity_id(chunk.entities[0])
    assert edge["evidence"] == chunk.relationships[0].evidence
    with pytest.raises(ValueError):
        relationship_merge_query("FRIEND_OF")


def test_identity_merges_across_chunks_but_not_people_across_documents():
    a = make_chunk(chunk="0").entities
    b = make_chunk(chunk="1").entities
    other_doc = make_chunk(doc="OTHER-DOC").entities
    assert graph_entity_id(a[0]) == graph_entity_id(b[0])  # same court, two chunks
    assert graph_entity_id(a[1]) == graph_entity_id(b[1])  # same person, same judgment
    assert graph_entity_id(a[0]) == graph_entity_id(other_doc[0])  # court is global
    assert graph_entity_id(a[1]) != graph_entity_id(other_doc[1])  # namesakes stay apart


# --- 8-12. Writing nodes and edges ----------------------------------------------------


def test_nodes_and_edges_are_created(client, driver):
    result = persist_chunk(client, make_chunk())
    assert (result.nodes_created, result.nodes_merged) == (3, 0)
    assert (result.relationships_created, result.relationships_merged) == (1, 0)
    assert len(driver.nodes) == 3 and len(driver.edges) == 1
    assert driver.sessions_closed == 1


def test_existing_nodes_are_merged_and_provenance_appended(client, driver):
    persist_chunk(client, make_chunk(chunk="0"))
    result = persist_chunk(client, make_chunk(chunk="1"))
    assert (result.nodes_created, result.nodes_merged) == (0, 3)
    assert result.relationships_created == 1  # same fact, new chunk evidence: new edge
    assert len(driver.nodes) == 3
    court = next(n for n in driver.nodes.values() if n["entity_type"] == "COURT")
    assert court["source_chunk_ids"] == [f"{DOC}_chunk0", f"{DOC}_chunk1"]
    assert court["source_document_ids"] == [DOC]


def test_reingestion_is_idempotent(client, driver):
    first = persist_chunk(client, make_chunk())
    nodes, edges = copy.deepcopy(driver.nodes), copy.deepcopy(driver.edges)
    second = persist_chunk(client, make_chunk())
    assert driver.nodes == nodes and driver.edges == edges
    assert (second.nodes_created, second.nodes_merged) == (0, first.nodes_created)
    assert (second.relationships_created, second.relationships_merged) == (0, 1)


def test_relationship_evidence_and_provenance_are_kept(client, driver):
    persist_chunk(client, make_chunk())
    edge = next(iter(driver.edges.values()))
    assert edge["type"] == "RELATED_TO"
    assert edge["evidence"].startswith("Rajesh Kumar")
    assert (edge["document_id"], edge["chunk_id"]) == (DOC, f"{DOC}_chunk0")


def test_mentions_resolving_to_one_node_are_deduplicated(client, driver):
    alias = make_entity(DOC, f"{DOC}_chunk0", "court2", "delhi high court", "COURT")
    chunk = make_chunk(extra_entities=[alias])
    rows = build_chunk_rows(chunk)
    assert len(rows.nodes["Court"]) == 1
    assert rows.nodes["Court"][0]["aliases"] == ["Delhi High Court", "delhi high court"]
    persist_chunk(client, chunk)
    assert len(driver.nodes) == 3


# --- 13-14. Missing endpoints and rollback ---------------------------------------------


def test_missing_endpoint_rolls_back_the_chunk(client, driver):
    rows = build_chunk_rows(make_chunk())
    rows.relationships["RELATED_TO"][0]["target_id"] = "node_missing"
    with pytest.raises(MissingEndpointError):
        client.execute_write(write_chunk_rows, nodes=rows.nodes, relationships=rows.relationships)
    assert driver.nodes == {} and driver.edges == {}


def test_failure_mid_transaction_rolls_back(client, driver):
    driver.fail_on = "[r:RELATED_TO"
    with pytest.raises(RuntimeError):
        persist_chunk(client, make_chunk())
    assert driver.nodes == {} and driver.edges == {}


# --- 15-18. Document-level construction and summary -----------------------------------


def test_persist_extraction_summary_and_partial_failures(client, driver):
    driver.fail_on = None
    response = persist_extraction(
        client, extraction([make_chunk(chunk="0"), failed("1"), make_chunk(chunk="2")])
    )
    assert response.status == "partial"
    assert (response.chunks_processed, response.chunks_succeeded, response.chunks_failed) == (
        3,
        2,
        1,
    )
    assert response.failed_chunks[0].stage == "extraction"
    assert response.entities_extracted == 6 and response.relationships_extracted == 2
    assert (response.nodes_created, response.nodes_merged) == (3, 3)
    assert (response.relationships_created, response.relationships_merged) == (2, 0)
    assert len(driver.nodes) == 3


def test_graph_write_failure_is_reported_per_chunk(client, driver):
    class FlakyClient(Neo4jClient):
        calls = 0

        def execute_write(self, work, **kwargs):
            FlakyClient.calls += 1
            if FlakyClient.calls == 2:
                raise GraphStoreError("Neo4j write failed (ServiceUnavailable)")
            return super().execute_write(work, **kwargs)

    flaky = FlakyClient(driver, "neo4j")
    response = persist_extraction(
        flaky, extraction([make_chunk(chunk="0"), make_chunk(chunk="1")], skipped=0)
    )
    assert response.status == "partial"
    assert response.failed_chunks[0].stage == "graph"
    assert response.chunks_succeeded == 1


def test_all_chunks_failed_and_skipped_status(client):
    assert persist_extraction(client, extraction([failed()])).status == "failed"
    assert persist_extraction(client, extraction([make_chunk()], skipped=3)).status == "partial"
    assert persist_extraction(client, extraction([make_chunk()])).status == "completed"


class FakeCollection:
    name = "judgments"

    def __init__(self, records):
        self.records = records

    def get(self, ids=None, where=None, include=()):
        found = [r for r in self.records if r[2]["case_id"] == where["case_id"]]
        return {
            "ids": [r[0] for r in found],
            "documents": [r[1] for r in found],
            "metadatas": [r[2] for r in found],
        }


class CannedLLM:
    def __init__(self, answers):
        self.answers = list(answers)

    def complete_json(self, system_prompt, user_prompt):
        return self.answers.pop(0)


CHUNK_TEXT = "Rajesh Kumar appeared before the Delhi High Court."
LLM_JSON = (
    '{"entities": ['
    '{"id": "e1", "name": "Rajesh Kumar", "entity_type": "PERSON", "evidence": "Rajesh Kumar"},'
    '{"id": "e2", "name": "Delhi High Court", "entity_type": "COURT",'
    ' "evidence": "before the Delhi High Court"}],'
    '"relationships": [{"source": "e1", "target": "e2", "relationship_type": "RELATED_TO",'
    ' "evidence": "Rajesh Kumar appeared before the Delhi High Court"}]}'
)


def test_document_level_build_uses_story_1_5_extraction(client, driver):
    collection = FakeCollection(
        [
            (f"{DOC}_chunk0", CHUNK_TEXT, {"case_id": DOC}),
            (f"{DOC}_chunk1", CHUNK_TEXT, {"case_id": DOC}),
        ]
    )
    llm = CannedLLM([LLM_JSON, "not json"])
    response = build_document_graph(client, llm, collection, DOC, max_chunks=5)
    assert response.status == "partial"
    assert response.failed_chunks[0].chunk_id == f"{DOC}_chunk1"
    assert (response.nodes_created, response.relationships_created) == (2, 1)
    assert len(driver.nodes) == 2


def test_build_endpoint(client, driver):
    collection = FakeCollection([(f"{DOC}_chunk0", CHUNK_TEXT, {"case_id": DOC})])
    app.dependency_overrides[get_graph_client] = lambda: client
    app.dependency_overrides[get_chroma_collection] = lambda: collection
    app.dependency_overrides[get_extraction_llm] = lambda: CannedLLM([LLM_JSON])
    try:
        http = TestClient(app)
        init = http.post("/v1/graph/initialize")
        assert init.status_code == 200 and init.json()["status"] == "initialized"
        body = http.post("/v1/graph/build", json={"document_id": DOC}).json()
        assert body["status"] == "completed" and body["nodes_created"] == 2
        assert http.post("/v1/graph/build", json={"document_id": "nope"}).status_code == 404
        assert http.post("/v1/graph/build", json={"document_id": " "}).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_endpoints_return_503_without_neo4j(monkeypatch):
    monkeypatch.setattr(neo4j_module, "_client", None)
    app.dependency_overrides[get_graph_client] = lambda: get_graph_client(Settings(_env_file=None))
    try:
        response = TestClient(app).post("/v1/graph/initialize")
        assert response.status_code == 503 and "NEO4J_URI" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()
