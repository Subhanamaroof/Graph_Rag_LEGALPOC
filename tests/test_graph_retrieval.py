"""Story 1.7 tests: graph retrieval.

Neo4j is replaced by ``FakeGraph``: an in-memory graph whose ``FakeTx.run`` answers our three
read queries (exact match, partial match, traversal) with the same semantics — label filter,
case-insensitive equality / CONTAINS, undirected paths of exactly N distinct edges, edge-type
and end-type filters, LIMIT. Groq is replaced by a fake JSON client. Nothing real is called
and nothing is written. The Cypher text itself is checked against a live server manually.
"""

import json
import re

import pytest
from fastapi.testclient import TestClient
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from graph_rag.core.llm import LLMError
from graph_rag.graph.graph_router import get_graph_client
from graph_rag.graph.graph_schema import NODE_LABELS
from graph_rag.graph.models import EntityType, RelationshipType
from graph_rag.graph.neo4j_client import GraphQueryTimeoutError, GraphStoreError, Neo4jClient
from graph_rag.main import app
from graph_rag.retrieval.entity_matcher import MAX_CANDIDATES, match_entity
from graph_rag.retrieval.graph_ranker import rank_results
from graph_rag.retrieval.graph_retrieval_models import GraphRetrievalRequest, QueryEntity
from graph_rag.retrieval.graph_retrieval_service import retrieve_graph
from graph_rag.retrieval.graph_router import get_query_llm
from graph_rag.retrieval.graph_traversal import traversal_query, traverse
from graph_rag.retrieval.query_analyzer import analyze_query, identify_entities

DOC_A, DOC_B = "NI138-2019-0042", "IPC-2020-0007"


# --- In-memory fake Neo4j -------------------------------------------------------------


class FakeRecord:
    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data


class FakeGraph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self.queries: list[str] = []
        self.fail_for: set[str] = set()  # start IDs whose traversal raises

    def node(self, entity_id, name, entity_type, *, normalized_name=None, aliases=(), docs=()):
        self.nodes[entity_id] = {
            "entity_id": entity_id,
            "name": name,
            "normalized_name": normalized_name or name,
            "entity_type": entity_type,
            "label": NODE_LABELS[EntityType(entity_type)],
            "aliases": list(aliases),
            "source_document_ids": list(docs),
            "source_chunk_ids": [f"{d}_chunk0" for d in docs],
        }

    def edge(self, rel_id, source, rel_type, target, *, doc=DOC_A, chunk=None, evidence="quote"):
        self.edges.append(
            {
                "relationship_id": rel_id,
                "relationship_type": rel_type,
                "source_entity_id": source,
                "target_entity_id": target,
                "description": None,
                "evidence": evidence,
                "source_document_id": doc,
                "source_chunk_id": chunk if chunk is not None else f"{doc}_chunk1",
            }
        )

    # -- query emulation

    def _row(self, node, match_type):
        return {
            "entity_id": node["entity_id"],
            "name": node["name"],
            "normalized_name": node["normalized_name"],
            "entity_type": node["entity_type"],
            "source_document_ids": node["source_document_ids"],
            "source_chunk_ids": node["source_chunk_ids"],
            "document_count": len(node["source_document_ids"]),
            "chunk_count": len(node["source_chunk_ids"]),
            "match_type": match_type,
        }

    def _labelled(self, labels):
        return [n for n in self.nodes.values() if labels is None or n["label"] in labels]

    def exact(self, keys, labels, max_rows):
        rows = []
        for node in sorted(self._labelled(labels), key=lambda n: n["entity_id"]):
            if node["normalized_name"].lower().strip() in keys:
                rows.append(self._row(node, "exact_normalized"))
            elif node["name"].lower().strip() in keys:
                rows.append(self._row(node, "exact_name"))
            elif any(a.lower().strip() in keys for a in node["aliases"]):
                rows.append(self._row(node, "alias"))
        return rows[:max_rows]

    def partial(self, keys, labels, max_rows):
        rows = [
            self._row(node, "partial")
            for node in self._labelled(labels)
            if any(k in node["normalized_name"].lower() or k in node["name"].lower() for k in keys)
        ]
        return rows[:max_rows]

    def paths(self, start_id, hops, relationship_types, end_types, path_limit):
        found = []

        def walk(current, nodes, edges):
            if len(edges) == hops:
                end = self.nodes[current]
                if current != start_id and (end_types is None or end["entity_type"] in end_types):
                    found.append((nodes, edges))
                return
            for edge in self.edges:
                if edge in edges or edge["relationship_type"] not in relationship_types:
                    continue
                if edge["source_entity_id"] == current:
                    nxt = edge["target_entity_id"]
                elif edge["target_entity_id"] == current:
                    nxt = edge["source_entity_id"]
                else:
                    continue
                walk(nxt, nodes + [nxt], edges + [edge])

        walk(start_id, [start_id], [])
        keep = ("entity_id", "name", "entity_type")
        return [
            {
                "nodes": [{k: self.nodes[n][k] for k in keep} | {"description": None} for n in ns],
                "relationships": [dict(e) for e in es],
            }
            for ns, es in found[:path_limit]
        ]


class FakeTx:
    def __init__(self, graph: FakeGraph):
        self.graph = graph

    def run(self, query, **params):
        self.graph.queries.append(query)
        if "MATCH p =" in query:
            if params["start_id"] in self.graph.fail_for:
                raise GraphStoreError("Neo4j read failed (simulated).")
            hops = int(re.search(r"\*(\d)\.\.\d", query).group(1))
            rows = self.graph.paths(
                params["start_id"],
                hops,
                params["relationship_types"],
                params["end_types"],
                params["path_limit"],
            )
        elif "CONTAINS" in query:
            rows = self.graph.partial(params["keys"], params["labels"], params["max_rows"])
        else:
            rows = self.graph.exact(params["keys"], params["labels"], params["max_rows"])
        return [FakeRecord(row) for row in rows]


class FakeClient:
    """Stands in for ``Neo4jClient``: only ``execute_read`` exists, so a write is impossible."""

    def __init__(self, graph: FakeGraph):
        self.graph = graph
        self.timeouts: list[float | None] = []

    def execute_read(self, work, *, timeout=None, **kwargs):
        self.timeouts.append(timeout)
        return work(FakeTx(self.graph), **kwargs)


class FakeLLM:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, 0

    def complete_json(self, system_prompt, user_prompt):
        self.calls += 1
        if self.error:
            raise self.error
        return json.dumps(self.response) if isinstance(self.response, dict) else self.response


@pytest.fixture
def graph():
    g = FakeGraph()
    g.node("ipc", "Indian Penal Code", "STATUTE", normalized_name="Indian Penal Code, 1860",
           aliases=["IPC"], docs=[DOC_A, DOC_B])  # fmt: skip
    g.node("dhc", "Delhi High Court", "COURT", docs=[DOC_A])
    g.node("sc", "Supreme Court of India", "COURT", docs=[DOC_B])
    g.node("case_x", "State v. Ram Kumar", "CASE", docs=[DOC_A])
    g.node("case_y", "Kumar v. Union of India", "CASE", docs=[DOC_B])
    g.node("judge", "Justice A. Sharma", "PERSON", docs=[DOC_A])
    g.node("bail", "bail", "LEGAL_CONCEPT", docs=[DOC_B])
    g.node("lonely", "Limitation Act", "STATUTE", docs=[DOC_B])
    g.edge("e1", "dhc", "REFERS_TO", "ipc", evidence="under the Indian Penal Code")
    g.edge("e1b", "dhc", "REFERS_TO", "ipc", chunk=f"{DOC_A}_chunk7", evidence="IPC applies")
    g.edge("e2", "sc", "REFERS_TO", "ipc", doc=DOC_B, evidence="Section 302 IPC")
    g.edge("e3", "case_x", "CITES", "case_y", evidence="relied on Kumar v. Union of India")
    g.edge("e4", "judge", "JUDGE_PRESIDED_OVER", "case_x", evidence="Coram: A. Sharma J.")
    g.edge("e5", "case_x", "DECIDED_BY", "dhc", evidence="High Court of Delhi")
    g.edge("e6", "sc", "APPLIES", "bail", doc=DOC_B, evidence="bail was granted")
    return g


def _retrieve(graph, question, llm=None, **request):
    return retrieve_graph(
        GraphRetrievalRequest(question=question, **request), FakeClient(graph), llm
    )


# --- 1-5: query entity identification ---------------------------------------------------


@pytest.mark.parametrize("separator", ["v.", "vs", "Vs.", "versus"])
def test_identifies_case_titles(separator):
    entities = identify_entities(f"What other judgments refer to Pradeep Kumar {separator} Arun?")
    assert [(e.name, e.entity_type) for e in entities] == [
        (f"Pradeep Kumar {separator} Arun", EntityType.CASE)
    ]


def test_identifies_statute_and_intent_without_llm():
    llm = FakeLLM(error=AssertionError("must not be called"))
    analysis = analyze_query("Which cases cited the Indian Penal Code?", llm)
    assert analysis.method == "deterministic" and llm.calls == 0
    assert [(e.name, e.entity_type) for e in analysis.entities] == [
        ("Indian Penal Code", EntityType.STATUTE)
    ]
    assert analysis.relationship_intent == RelationshipType.REFERS_TO
    assert analysis.question == "Which cases cited the Indian Penal Code?"


def test_identifies_multiple_entities_and_abbreviations():
    entities = identify_entities(
        "Did State of Punjab v. Ram Kumar apply Section 138 of the NI Act and the IPC?"
    )
    found = {(e.name, e.entity_type) for e in entities}
    assert ("State of Punjab v. Ram Kumar", EntityType.CASE) in found
    assert ("Section 138", EntityType.LEGAL_SECTION) in found
    assert ("Negotiable Instruments Act", EntityType.STATUTE) in found
    assert ("Indian Penal Code", EntityType.STATUTE) in found
    assert all(e.name != "NI Act and the IPC" for e in entities)


def test_llm_output_is_validated_and_grounded():
    llm = FakeLLM(
        {
            "entities": [
                {"name": "natural justice", "entity_type": "legal_concept"},
                {"name": "Article 21", "entity_type": "LEGAL_SECTION"},  # not in the question
                {"name": "cases", "entity_type": "CASE"},  # generic word
                {"name": "audi alteram partem", "entity_type": "DOCTRINE"},  # bad type
                {"entity_type": "CASE"},  # no name
            ],
            "keywords": ["hearing"],
            "relationship_intent": "applies",
        }
    )
    analysis = analyze_query("Which cases discuss natural justice and audi alteram partem?", llm)
    assert analysis.method == "llm" and llm.calls == 1
    assert [(e.name, e.entity_type) for e in analysis.entities] == [
        ("natural justice", EntityType.LEGAL_CONCEPT),
        ("audi alteram partem", None),
    ]
    assert analysis.relationship_intent == RelationshipType.APPLIES
    assert "hearing" in analysis.keywords
    assert len(analysis.warnings) == 4


def test_no_entity_identified_returns_no_match(graph):
    response = _retrieve(graph, "what is the meaning of life?", FakeLLM({"entities": []}))
    assert response.retrieval_status == "no_match"
    assert response.identified_entities == [] and response.results == []
    assert graph.queries == []  # Neo4j never queried


@pytest.mark.parametrize(
    "llm",
    [FakeLLM(error=LLMError("Groq returned HTTP 503.")), FakeLLM("not json"), None],
)
def test_llm_failure_or_absence_falls_back_to_deterministic(llm):
    analysis = analyze_query("tell me about natural justice", llm)
    assert analysis.method == "deterministic" and analysis.entities == []
    assert len(analysis.warnings) == 1


# --- 6-9: entity matching --------------------------------------------------------------


def test_exact_and_alias_matching(graph):
    client = FakeClient(graph)
    exact = match_entity(client, QueryEntity(name="indian penal code, 1860", mention="x"))
    assert [(c.entity_id, c.match_type) for c in exact.candidates] == [("ipc", "exact_normalized")]
    alias = match_entity(client, QueryEntity(name="ipc", mention="ipc"))
    assert [(c.entity_id, c.match_type) for c in alias.candidates] == [("ipc", "alias")]


def test_controlled_partial_match_respects_word_boundaries(graph):
    graph.node("s138", "Section 138", "LEGAL_SECTION", docs=[DOC_A])
    client = FakeClient(graph)
    result = match_entity(client, QueryEntity(name="Penal Code", mention="Penal Code"))
    assert [(c.entity_id, c.match_type) for c in result.candidates] == [("ipc", "partial")]
    assert (
        match_entity(client, QueryEntity(name="Section 13", mention="Section 13")).candidates == []
    )
    assert match_entity(client, QueryEntity(name="IP", mention="IP")).candidates == []  # too short


def test_entity_type_restricts_label(graph):
    client = FakeClient(graph)
    as_court = QueryEntity(name="Indian Penal Code", mention="x", entity_type=EntityType.COURT)
    assert match_entity(client, as_court).candidates == []
    as_statute = QueryEntity(name="Indian Penal Code", mention="x", entity_type=EntityType.STATUTE)
    assert [c.entity_id for c in match_entity(client, as_statute).candidates] == ["ipc"]


def test_ambiguous_names_return_all_candidates(graph):
    graph.node("p1", "Ramesh Kumar", "PERSON", docs=[DOC_A])
    graph.node("p2", "Ramesh Kumar", "PERSON", docs=[DOC_B])
    graph.edge("e7", "p1", "PETITIONER_IN", "case_x")
    graph.edge("e8", "p2", "RESPONDENT_IN", "case_y", doc=DOC_B)
    response = _retrieve(graph, 'Which cases involve "Ramesh Kumar"?', max_hops=1)
    assert {m.entity_id for m in response.matched_entities} == {"p1", "p2"}
    assert all(m.ambiguous for m in response.matched_entities)
    assert response.ambiguous_entities == ["Ramesh Kumar"]
    assert {r.matched_entity.entity_id for r in response.results} == {"p1", "p2"}


def test_candidates_are_capped(graph):
    for i in range(MAX_CANDIDATES + 2):
        graph.node(f"p{i}", "Ram", "PERSON", docs=[f"D{i}"])
    result = match_entity(FakeClient(graph), QueryEntity(name="Ram", mention="Ram"))
    assert len(result.candidates) == MAX_CANDIDATES and result.truncated


def test_unknown_entity_returns_no_match(graph):
    response = _retrieve(graph, "Which cases cited the Foreign Exchange Management Act?")
    assert response.retrieval_status == "no_match"
    assert [e.name for e in response.unmatched_entities] == ["Foreign Exchange Management Act"]
    assert response.results == [] and response.errors == []


# --- 10-14, 17, 19: traversal -------------------------------------------------------------


def test_one_hop_traversal_preserves_direction(graph):
    response = _retrieve(graph, "Which cases cited the Indian Penal Code?", max_hops=1)
    assert response.retrieval_status == "success"
    assert {r.related_node.entity_id for r in response.results} == {"dhc", "sc"}
    for result in response.results:
        assert result.hop_distance == 1
        edge = result.relationships[0]
        assert (edge.source_entity_id, edge.target_entity_id) == (
            result.related_node.entity_id,
            "ipc",
        )
        assert "<-[REFERS_TO]-" in result.path


def test_two_hop_traversal_returns_justifying_path(graph):
    response = _retrieve(graph, "Which judge decided State v. Ram Kumar?", max_hops=2)
    two_hop = [r for r in response.results if r.related_node.entity_id == "ipc"]
    assert len(two_hop) == 1 and two_hop[0].hop_distance == 2
    assert [n.entity_id for n in two_hop[0].nodes] == ["case_x", "dhc", "ipc"]
    assert [e.relationship_type for e in two_hop[0].relationships] == ["DECIDED_BY", "REFERS_TO"]


def test_max_hops_enforced():
    with pytest.raises(ValueError):
        GraphRetrievalRequest(question="x", max_hops=4)
    with pytest.raises(ValueError):
        traversal_query(4)
    with pytest.raises(ValueError):
        traverse(FakeClient(FakeGraph()), "ipc", max_hops=0, path_limit=5)


def test_traversal_never_exceeds_requested_hops(graph):
    client = FakeClient(graph)
    result = traverse(client, "judge", max_hops=2, path_limit=100)
    assert max(p.hop_distance for p in result.paths) == 2
    assert not any("*3..3" in q for q in graph.queries)


def test_relationship_type_filter(graph):
    response = _retrieve(
        graph, "What is connected to State v. Ram Kumar?", max_hops=3,
        relationship_types=["CITES"],
    )  # fmt: skip
    assert [r.related_node.entity_id for r in response.results] == ["case_y"]
    assert all(e.relationship_type == "CITES" for r in response.results for e in r.relationships)


def test_entity_type_filter_applies_to_related_nodes(graph):
    response = _retrieve(
        graph, "What is connected to State v. Ram Kumar?", max_hops=2, entity_types=["STATUTE"]
    )
    assert {r.related_node.entity_type for r in response.results} == {"STATUTE"}


def test_parallel_edges_are_merged_not_duplicated(graph):
    response = _retrieve(graph, "Which cases cited the Indian Penal Code?", max_hops=1)
    dhc = [r for r in response.results if r.related_node.entity_id == "dhc"]
    assert len(dhc) == 1
    assert dhc[0].supporting_edges == 2
    assert {e.relationship_id for e in dhc[0].evidence} == {"e1", "e1b"}
    assert dhc[0].source_chunk_ids == [f"{DOC_A}_chunk1", f"{DOC_A}_chunk7"]


def test_cycles_are_skipped(graph):
    result = traverse(FakeClient(graph), "dhc", max_hops=3, path_limit=100)
    assert all(len({n.entity_id for n in p.nodes}) == len(p.nodes) for p in result.paths)


def test_disconnected_node_returns_no_results(graph):
    response = _retrieve(graph, "Which cases cited the Limitation Act?")
    assert response.retrieval_status == "no_results"
    assert [m.entity_id for m in response.matched_entities] == ["lonely"]
    assert response.results == []


def test_limit_enforced(graph):
    response = _retrieve(graph, "What is connected to State v. Ram Kumar?", max_hops=3, limit=2)
    assert response.total_results == 2 == len(response.results)
    assert response.truncated
    assert [r.rank for r in response.results] == [1, 2]


# --- 15-16: ranking and provenance --------------------------------------------------------


def test_ranking_prefers_direct_intent_matching_paths(graph):
    response = _retrieve(graph, "Which cases cited the Indian Penal Code?", max_hops=2)
    scores = [r.score for r in response.results]
    assert scores == sorted(scores, reverse=True)
    top = response.results[0]
    assert top.hop_distance == 1 and top.relationships[0].relationship_type == "REFERS_TO"
    assert top.related_node.entity_id == "dhc"  # two supporting edges beat one
    assert set(top.ranking_signals) == {"match", "proximity", "relevance", "type_match", "support"}
    assert all(r.hop_distance == 2 for r in response.results[2:])
    assert rank_results(list(reversed(response.results)))[0].path == top.path  # deterministic


def test_evidence_and_provenance_preserved(graph):
    response = _retrieve(graph, "Which cases cited the Indian Penal Code?", max_hops=1)
    sc = next(r for r in response.results if r.related_node.entity_id == "sc")
    assert sc.evidence[0].evidence == "Section 302 IPC"
    assert (sc.evidence[0].document_id, sc.evidence[0].chunk_id) == (DOC_B, f"{DOC_B}_chunk1")
    assert sc.source_document_ids == [DOC_B] and sc.provenance_complete
    matched = response.matched_entities[0]
    assert matched.source_document_ids == [DOC_A, DOC_B] and matched.document_count == 2


def test_incomplete_provenance_is_flagged(graph):
    graph.edge("e9", "case_y", "REFERS_TO", "lonely", doc=DOC_B, chunk="", evidence="")
    response = _retrieve(graph, "Which cases cited the Limitation Act?")
    assert response.results[0].provenance_complete is False
    assert (
        response.results[0].evidence[0].evidence is None
        or not response.results[0].evidence[0].evidence
    )
    assert any("incomplete provenance" in w for w in response.warnings)


def test_malformed_graph_records_are_skipped(graph):
    graph.edge(None, "case_y", "REFERS_TO", "lonely", doc=DOC_B)  # no relationship_id
    response = _retrieve(graph, "Which cases cited the Limitation Act?")
    assert response.results == [] and any("malformed" in w for w in response.warnings)


# --- 18: Neo4j failures -------------------------------------------------------------------


def test_traversal_failure_for_one_start_is_partial(graph):
    graph.fail_for = {"case_x"}
    response = _retrieve(graph, "Did Kumar v. Union of India cite State v. Ram Kumar?", max_hops=1)
    assert response.retrieval_status == "partial"
    assert response.results and len(response.errors) == 1


def test_total_neo4j_failure_is_failed_not_success(graph):
    class BrokenClient:
        def execute_read(self, *args, **kwargs):
            raise GraphQueryTimeoutError("Neo4j read query timed out after 10 s.")

    request = GraphRetrievalRequest(question="Which cases cited the Indian Penal Code?")
    response = retrieve_graph(request, BrokenClient())
    assert response.retrieval_status == "failed" and response.results == []
    assert "timed out" in response.errors[0]


class _Session:
    def __init__(self, error):
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute_read(self, work, **kwargs):
        raise self.error


class _Driver:
    def __init__(self, error):
        self.error = error

    def session(self, database):
        return _Session(self.error)


def test_execute_read_maps_timeout_and_driver_errors():
    timeout = Neo4jError._hydrate_neo4j(
        code="Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration", message="t/o"
    )
    with pytest.raises(GraphQueryTimeoutError):
        Neo4jClient(_Driver(timeout), "neo4j").execute_read(lambda tx: None, timeout=1)
    with pytest.raises(GraphStoreError):
        Neo4jClient(_Driver(ServiceUnavailable("down")), "neo4j").execute_read(lambda tx: None)


# --- 20: API -------------------------------------------------------------------------------


@pytest.fixture
def api(graph):
    app.dependency_overrides[get_graph_client] = lambda: FakeClient(graph)
    app.dependency_overrides[get_query_llm] = lambda: None
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_api_returns_structured_response(api):
    response = api.post(
        "/v1/retrieval/graph",
        json={"question": "Which cases cited the Indian Penal Code?", "max_hops": 1, "limit": 5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_status"] == "success" and body["total_results"] == 2
    assert {"question", "identified_entities", "matched_entities", "unmatched_entities",
            "results", "total_results", "retrieval_status"} <= set(body)  # fmt: skip
    result = body["results"][0]
    assert {"matched_entity", "nodes", "relationships", "path", "hop_distance", "evidence",
            "source_document_ids", "source_chunk_ids", "score"} <= set(result)  # fmt: skip


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "   "},
        {"question": "x", "max_hops": 0},
        {"question": "x", "max_hops": 4},
        {"question": "x", "limit": 0},
        {"question": "x", "limit": 101},
        {"question": "x", "entity_types": ["JUDGE"]},
        {"question": "x", "relationship_types": ["LIKES"]},
        {"question": "x", "relationship_types": []},
    ],
)
def test_api_rejects_invalid_requests(api, payload):
    assert api.post("/v1/retrieval/graph", json=payload).status_code == 422


def test_api_returns_503_when_neo4j_unavailable(monkeypatch):
    from graph_rag.graph import neo4j_client as neo4j_module

    def unavailable(_settings):
        raise neo4j_module.GraphNotConfiguredError("Neo4j is not configured. Set NEO4J_URI.")

    monkeypatch.setattr("graph_rag.graph.graph_router.get_neo4j_client", unavailable)
    app.dependency_overrides[get_query_llm] = lambda: None
    try:
        response = TestClient(app).post("/v1/retrieval/graph", json={"question": "IPC?"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503 and "NEO4J_URI" in response.json()["detail"]
