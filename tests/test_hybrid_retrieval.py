"""Story 1.8 tests: hybrid retrieval (vector + graph, fused with RRF).

ChromaDB is ``FakeCollection`` (answers ``query`` and ``get`` from an in-memory dict and has no
write methods), the embedding model is ``FakeEmbedder``, and Story 1.7 ``retrieve_graph`` is
replaced by a recorder returning a prepared response. The real ``retrieve_similar_chunks`` runs
against the fakes. No external service is called and no data is written.
"""

import time

import pytest
from fastapi.testclient import TestClient

from graph_rag.core.config import Settings
from graph_rag.embeddings.embedder import EmbeddingModelError
from graph_rag.graph.neo4j_client import GraphNotConfiguredError
from graph_rag.main import app
from graph_rag.retrieval import hybrid_retrieval_service as service
from graph_rag.retrieval.graph_retrieval_models import (
    Evidence,
    GraphEdge,
    GraphNode,
    GraphResult,
    GraphRetrievalResponse,
)
from graph_rag.retrieval.hybrid_ranker import reciprocal_rank_fusion, rrf_term
from graph_rag.retrieval.hybrid_retrieval_models import HybridRetrievalRequest
from graph_rag.retrieval.hybrid_retrieval_service import (
    HybridParameters,
    HybridResources,
    hybrid_status,
    resolve_parameters,
    retrieve_hybrid,
)
from graph_rag.retrieval.hybrid_router import get_hybrid_resources
from graph_rag.retrieval.models import RetrievedChunk, VectorSearchResponse
from graph_rag.retrieval.result_fusion import (
    merge_results,
    normalize_graph_results,
    normalize_vector_results,
)
from graph_rag.retrieval.source_resolver import (
    extract_source_references,
    resolve_source_references,
)
from graph_rag.vector_store.chroma_store import VectorStoreError

D1, D2 = "CASE-001", "CASE-002"
A, B, C, D = f"{D1}_chunk1", f"{D1}_chunk2", f"{D1}_chunk3", f"{D2}_chunk4"
QUESTION = "Which cases refer to the Indian Penal Code?"


# --- Fakes -------------------------------------------------------------------------------


class FakeEmbedder:
    def __init__(self, error=None):
        self.error, self.texts = error, []

    def load(self):
        if self.error:
            raise self.error

    def embed_text(self, text):
        self.texts.append(text)
        return [0.1] * 384


class FakeCollection:
    """In-memory chunks. ``order`` is what a similarity query returns. Read-only."""

    def __init__(self, order=(A, B, C), *, fail=False):
        self.records = {
            A: ("text A", {"case_id": D1, "court": "Delhi HC"}),
            B: ("text B", {"case_id": D1, "court": "Delhi HC"}),
            C: ("text C", {"case_id": D1, "court": "Delhi HC"}),
            D: ("text D", {"case_id": D2, "court": "SC"}),
        }
        self.order, self.fail = list(order), fail
        self.get_calls = []

    def query(self, query_embeddings, n_results, include):
        if self.fail:
            raise RuntimeError("simulated chroma failure with secret details")
        ids = self.order[:n_results]
        return {
            "ids": [ids],
            "documents": [[self.records[i][0] for i in ids]],
            "metadatas": [[self.records[i][1] for i in ids]],
            "distances": [[0.1 * (n + 1) for n in range(len(ids))]],
        }

    def get(self, ids=None, where=None, include=None):
        self.get_calls.append(list(ids or []))
        if self.fail:
            raise RuntimeError("simulated chroma failure")
        found = [i for i in ids if i in self.records]
        return {
            "ids": found,
            "documents": [self.records[i][0] for i in found],
            "metadatas": [self.records[i][1] for i in found],
        }


def node(entity_id, name="N", entity_type="STATUTE"):
    return GraphNode(entity_id=entity_id, name=name, entity_type=entity_type)


def graph_result(rank, refs, *, extra=()):
    """A one-hop path whose edge(s) cite ``refs`` [(relationship_id, document_id, chunk_id)].
    ``extra`` refs are parallel edges: evidence only, not in ``relationships``."""
    start, end = node("ipc", "Indian Penal Code"), node(f"case{rank}", f"Case {rank}", "CASE")
    edges = [
        GraphEdge(
            relationship_id=rel,
            relationship_type="REFERS_TO",
            source_entity_id=end.entity_id,
            target_entity_id=start.entity_id,
            evidence=f"quote {rel}",
            source_document_id=doc,
            source_chunk_id=chunk,
        )
        for rel, doc, chunk in refs
    ]
    evidence = [
        Evidence(
            relationship_id=rel,
            relationship_type="REFERS_TO",
            evidence=f"quote {rel}",
            document_id=doc,
            chunk_id=chunk,
        )
        for rel, doc, chunk in list(refs) + list(extra)
    ]
    return GraphResult(
        rank=rank,
        score=round(1.0 - rank / 10, 4),
        query_entity="Indian Penal Code",
        matched_entity=start,
        related_node=end,
        nodes=[start, end],
        relationships=edges[:1],
        path=f"(Indian Penal Code:STATUTE) <-[REFERS_TO]- (Case {rank}:CASE)",
        hop_distance=1,
        supporting_edges=len(evidence),
        evidence=evidence,
    )


def graph_response(results=(), status="success", errors=()):
    return GraphRetrievalResponse(
        question=QUESTION,
        analysis_method="deterministic",
        results=list(results),
        total_results=len(results),
        retrieval_status=status,
        errors=list(errors),
    )


DEFAULT_GRAPH = [graph_result(1, [("r1", D1, B)]), graph_result(2, [("r2", D2, D)])]


class GraphRecorder:
    def __init__(self, response=None, error=None, delay=0.0):
        self.response = response if response is not None else graph_response(DEFAULT_GRAPH)
        self.error, self.delay, self.calls = error, delay, []

    def __call__(self, request, client, llm=None, *, timeout=None):
        self.calls.append({"request": request, "client": client, "llm": llm, "timeout": timeout})
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.response


PARAMS = HybridParameters(
    vector_top_k=3,
    graph_top_k=10,
    final_top_k=10,
    max_hops=2,
    rrf_k=60,
    graph_query_timeout=7.0,
    branch_timeout=5.0,
)


def resources(collection=None, embedder=None, graph_client_error=None, llm=None):
    collection = collection or FakeCollection()
    embedder = embedder or FakeEmbedder()

    def load_graph_client():
        if graph_client_error:
            raise graph_client_error
        return "neo4j-client"

    return HybridResources(
        load_collection=lambda: collection,
        load_embedder=lambda: embedder,
        load_graph_client=load_graph_client,
        llm=llm,
    )


@pytest.fixture
def graph(monkeypatch):
    recorder = GraphRecorder()
    monkeypatch.setattr(service, "retrieve_graph", recorder)
    return recorder


def run(params=PARAMS, res=None, **request):
    request = HybridRetrievalRequest(question=QUESTION, **request)
    return retrieve_hybrid(request, params, res or resources())


def by_chunk(response):
    return {result.chunk_id: result for result in response.results}


# --- Vector integration -------------------------------------------------------------------


def test_vector_branch_reuses_existing_retriever_and_embedder(graph, monkeypatch):
    calls = []
    original = service.retrieve_similar_chunks

    def spy(query, top_k, *, embedder, collection):
        calls.append((query, top_k))
        return original(query, top_k, embedder=embedder, collection=collection)

    monkeypatch.setattr(service, "retrieve_similar_chunks", spy)
    embedder = FakeEmbedder()
    response = run(res=resources(embedder=embedder))
    assert calls == [(QUESTION, 3)]
    assert embedder.texts == [QUESTION]  # embedded once, by the existing embedder
    assert response.vector.status == "completed" and response.vector_results_count == 3


def test_vector_metadata_scores_and_ranks_preserved(graph):
    result = by_chunk(run())[A]
    assert (result.document_id, result.chunk_id, result.content) == (D1, A, "text A")
    assert result.metadata == {"case_id": D1, "court": "Delhi HC"}
    assert result.vector_rank == 1
    assert result.vector_distance == pytest.approx(0.1)
    assert result.vector_score == pytest.approx(0.9)  # 1 - distance, Story 1.4 semantics
    assert by_chunk(run())[C].vector_rank == 3


def test_empty_vector_results_still_return_graph_chunks(graph):
    response = run(res=resources(collection=FakeCollection(order=())))
    assert response.vector_results_count == 0
    assert {r.chunk_id for r in response.results} == {B, D}
    assert response.retrieval_status == "success"


# --- Graph integration --------------------------------------------------------------------


def test_graph_branch_reuses_retrieve_graph_with_request_values(graph):
    llm = object()
    run(
        params=HybridParameters(**{**PARAMS.__dict__, "max_hops": 3}),
        res=resources(llm=llm),
        entity_types=["CASE"],
        relationship_types=["REFERS_TO"],
    )
    call = graph.calls[0]
    assert call["request"].question == QUESTION
    assert call["request"].limit == PARAMS.graph_top_k
    assert call["request"].max_hops == 3
    assert [t.value for t in call["request"].entity_types] == ["CASE"]
    assert [t.value for t in call["request"].relationship_types] == ["REFERS_TO"]
    assert call["client"] == "neo4j-client" and call["llm"] is llm
    assert call["timeout"] == PARAMS.graph_query_timeout  # Story 1.7 timeout reused


def test_graph_context_preserves_entities_relationships_and_evidence(graph):
    context = by_chunk(run())[D].graph_context
    assert [p.graph_rank for p in context.paths] == [2]
    assert {e.entity_id for e in context.entities} == {"ipc", "case2"}
    assert [r.relationship_id for r in context.relationships] == ["r2"]
    assert [(e.document_id, e.chunk_id) for e in context.evidence] == [(D2, D)]


def test_source_references_extracted_in_rank_order_with_multiple_refs_and_duplicates():
    results = [
        graph_result(2, [("r3", D2, D)], extra=[("r4", D1, B)]),
        graph_result(1, [("r1", D1, B)], extra=[("r2", D1, A)]),
    ]
    extracted = extract_source_references(results)
    assert [(r.document_id, r.chunk_id) for r in extracted.references] == [
        (D1, B),
        (D1, A),
        (D2, D),
    ]
    assert extracted.duplicates_removed == 1  # (D1, B) cited by both paths
    assert extracted.references[0].graph_rank == 1


def test_incomplete_references_reported_not_resolved():
    extracted = extract_source_references(
        [graph_result(1, [("r1", None, B)], extra=[("r2", D1, None)])]
    )
    assert extracted.references == []
    assert [u.reason for u in extracted.incomplete] == ["missing_document_id", "missing_chunk_id"]


def test_graph_no_match_returns_vector_results(graph):
    graph.response = graph_response(status="no_match")
    response = run()
    assert response.retrieval_status == "success"
    assert response.graph.status == "completed"
    assert response.graph.graph_retrieval_status == "no_match"
    assert [r.retrieval_sources for r in response.results] == [["vector"]] * 3


def test_graph_results_without_source_references_warn(graph):
    graph.response = graph_response([graph_result(1, [("r1", None, None)])])
    response = run()
    assert response.graph_results_count == 0
    assert any("no usable document/chunk" in w for w in response.warnings)


# --- Source resolution -----------------------------------------------------------------------


def test_references_resolve_to_original_chunks_with_metadata():
    collection = FakeCollection()
    refs = extract_source_references(DEFAULT_GRAPH).references
    resolved, unresolved = resolve_source_references(collection, refs)
    assert unresolved == []
    assert resolved[(D2, D)].content == "text D"
    assert resolved[(D2, D)].metadata == {"case_id": D2, "court": "SC"}
    assert collection.get_calls == [[B, D]]  # one read, de-duplicated IDs


def test_unknown_or_mismatched_references_are_not_fabricated():
    refs = extract_source_references(
        [graph_result(1, [("r1", D1, "CASE-001_chunk99")], extra=[("r2", D2, A)])]
    ).references
    resolved, unresolved = resolve_source_references(FakeCollection(), refs)
    assert resolved == {}
    assert [u.reason for u in unresolved] == ["not_found", "document_mismatch"]


def test_missing_chunks_reported_in_response(graph):
    graph.response = graph_response([graph_result(1, [("r1", D1, "CASE-001_chunk99")])])
    response = run()
    assert [u.chunk_id for u in response.unresolved_references] == ["CASE-001_chunk99"]
    assert all(r.chunk_id != "CASE-001_chunk99" for r in response.results)
    assert response.retrieval_status == "success"  # a data gap, not a system failure


def test_source_resolution_failure_is_partial(graph):
    collection = FakeCollection()
    res = resources(collection=collection)
    original_get = collection.get

    def failing_get(**kwargs):
        raise RuntimeError("down")

    collection.get = failing_get
    response = run(res=res)
    collection.get = original_get
    assert response.retrieval_status == "partial"
    assert {u.reason for u in response.unresolved_references} == {"lookup_failed"}
    assert response.vector_results_count == 3 and response.graph_results_count == 0


# --- Fusion -------------------------------------------------------------------------------------


def vector_response(*ids):
    return VectorSearchResponse(
        query=QUESTION,
        total_results=len(ids),
        results=[
            RetrievedChunk(
                document_id=D1, chunk_id=i, text=f"t{i}", metadata={}, score=0.9, distance=0.1
            )
            for i in ids
        ],
    )


def test_fusion_merges_duplicates_and_keeps_both_sources():
    vector = normalize_vector_results(vector_response(A, B, C))
    resolved, _ = resolve_source_references(
        FakeCollection(), extract_source_references(DEFAULT_GRAPH).references
    )
    graph = normalize_graph_results(DEFAULT_GRAPH, resolved)
    merged = {chunk.chunk_id: chunk for chunk in merge_results(vector, graph)}
    assert len(merged) == 4
    assert merged[B].sources == ["vector", "graph"]
    assert (merged[B].vector_rank, merged[B].graph_rank) == (2, 1)
    assert merged[B].graph_context.evidence[0].relationship_id == "r1"  # graph context kept
    assert merged[A].sources == ["vector"] and merged[D].sources == ["graph"]
    assert merged[D].key == (D2, D)


def test_vector_normalization_skips_repeated_chunks():
    chunks = normalize_vector_results(vector_response(A, A, B))
    assert [(c.chunk_id, c.vector_rank) for c in chunks] == [(A, 1), (B, 2)]


def test_chunk_cited_by_several_paths_keeps_all_paths():
    results = [graph_result(1, [("r1", D1, B)]), graph_result(2, [("r2", D1, B)])]
    resolved, _ = resolve_source_references(
        FakeCollection(), extract_source_references(results).references
    )
    (chunk,) = normalize_graph_results(results, resolved)
    assert [p.graph_rank for p in chunk.graph_context.paths] == [1, 2]
    assert [e.relationship_id for e in chunk.graph_context.evidence] == ["r1", "r2"]


# --- Ranking ------------------------------------------------------------------------------------


def test_rrf_matches_story_example():
    fused = reciprocal_rank_fusion([["A", "B", "C"], ["B", "D"]], k=60)
    assert [key for key, _ in fused] == ["B", "A", "D", "C"]
    scores = dict(fused)
    assert scores["B"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["A"] == pytest.approx(1 / 61)
    assert scores["D"] == pytest.approx(1 / 62)


def test_rrf_uses_one_based_ranks():
    assert rrf_term(1, 60) == pytest.approx(1 / 61)
    with pytest.raises(ValueError):
        rrf_term(0, 60)


def test_rrf_ties_are_deterministic():
    first = reciprocal_rank_fusion([["X", "Y"], ["Z", "W"]], k=60)
    second = reciprocal_rank_fusion([["Z", "W"], ["X", "Y"]], k=60)
    assert [k for k, _ in first] == [k for k, _ in second] == ["X", "Z", "W", "Y"]


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_rrf_k_rejected(k):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["A"]], k=k)


def test_rrf_rejects_repeated_item_in_one_list():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["A", "A"]])


def test_hybrid_ranking_and_final_limit(graph):
    response = run()
    assert [r.chunk_id for r in response.results] == [B, A, D, C]
    assert [r.rank for r in response.results] == [1, 2, 3, 4]
    assert response.results[0].rrf_score == pytest.approx(1 / 62 + 1 / 61, abs=1e-6)
    limited = run(params=HybridParameters(**{**PARAMS.__dict__, "final_top_k": 2}))
    assert [r.chunk_id for r in limited.results] == [B, A]
    assert limited.total_results == 2 and limited.merged_results_count == 4


def test_configuration_defaults_and_validation():
    settings = Settings(_env_file=None)
    params = resolve_parameters(HybridRetrievalRequest(question=QUESTION, vector_top_k=4), settings)
    assert (params.vector_top_k, params.graph_top_k, params.final_top_k) == (4, 10, 10)
    assert (params.rrf_k, params.max_hops) == (60, 2)
    with pytest.raises(ValueError):
        Settings(_env_file=None, hybrid_rrf_k=0)
    with pytest.raises(ValueError):
        Settings(_env_file=None, vector_retrieval_top_k=21)


# --- Error handling ------------------------------------------------------------------------------


def test_chromadb_failure_returns_graph_branch_unresolved_as_failed(graph):
    response = run(res=resources(collection=FakeCollection(fail=True)))
    assert response.vector.status == "failed"
    assert response.vector.error == "Vector retrieval failed: the ChromaDB search failed."
    assert "secret" not in response.model_dump_json()  # raw exception text never exposed
    assert response.retrieval_status == "failed"  # graph chunks cannot be resolved either


def test_neo4j_unavailable_returns_vector_only_partial(graph):
    response = run(res=resources(graph_client_error=GraphNotConfiguredError("x")))
    assert response.retrieval_status == "partial"
    assert response.graph.status == "failed"
    assert response.errors == ["Graph retrieval failed: Neo4j is not configured."]
    assert [r.retrieval_sources for r in response.results] == [["vector"]] * 3


def test_graph_status_failed_counts_as_branch_failure(graph):
    graph.response = graph_response(status="failed", errors=["Entity matching failed"])
    response = run()
    assert response.graph.status == "failed" and response.retrieval_status == "partial"


def test_embedding_failure_returns_graph_only_partial(graph):
    response = run(res=resources(embedder=FakeEmbedder(error=EmbeddingModelError("no model"))))
    assert response.retrieval_status == "partial"
    assert {r.chunk_id for r in response.results} == {B, D}
    assert all(r.retrieval_sources == ["graph"] for r in response.results)


def test_graph_partial_errors_make_response_partial(graph):
    graph.response = graph_response(DEFAULT_GRAPH, status="partial", errors=["timed out"])
    assert run().retrieval_status == "partial"


def test_both_branches_empty_is_no_results(graph):
    graph.response = graph_response(status="no_match")
    response = run(res=resources(collection=FakeCollection(order=())))
    assert response.retrieval_status == "no_results" and response.results == []


def test_both_branches_failing_is_failed(graph):
    graph.error = VectorStoreError("unused")  # any exception type
    response = run(res=resources(collection=FakeCollection(fail=True)))
    assert response.retrieval_status == "failed"
    assert len(response.errors) == 2


def test_branch_timeout_is_reported(graph):
    graph.delay = 0.5
    params = HybridParameters(**{**PARAMS.__dict__, "branch_timeout": 0.1})
    response = run(params=params)
    assert response.graph.errors == ["Graph retrieval timed out."]
    assert response.retrieval_status == "partial" and response.vector_results_count == 3


@pytest.mark.parametrize(
    ("vector_ok", "graph_ok", "incomplete", "results", "expected"),
    [
        (True, True, False, 3, "success"),
        (True, False, False, 3, "partial"),
        (True, True, True, 3, "partial"),
        (True, True, False, 0, "no_results"),
        (True, False, False, 0, "failed"),
        (False, False, False, 0, "failed"),
    ],
)
def test_status_rules(vector_ok, graph_ok, incomplete, results, expected):
    assert (
        hybrid_status(
            vector_ok=vector_ok, graph_ok=graph_ok, incomplete=incomplete, results=results
        )
        == expected
    )


# --- API ---------------------------------------------------------------------------------


@pytest.fixture
def api(graph):
    holder = {"resources": resources()}
    app.dependency_overrides[get_hybrid_resources] = lambda: holder["resources"]
    yield TestClient(app), holder
    app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "body",
    [
        {"question": "   "},
        {"question": "q", "vector_top_k": 21},
        {"question": "q", "graph_top_k": 0},
        {"question": "q", "final_top_k": 101},
        {"question": "q", "max_hops": 4},
        {"question": "q", "entity_types": ["NOT_A_TYPE"]},
    ],
)
def test_api_request_validation(api, body):
    client, _ = api
    assert client.post("/v1/retrieval/hybrid", json=body).status_code == 422


def test_api_success_response(api):
    client, _ = api
    response = client.post("/v1/retrieval/hybrid", json={"question": QUESTION, "vector_top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_status"] == "success"
    assert (body["vector_results_count"], body["graph_results_count"]) == (3, 2)
    assert body["merged_results_count"] == 4 and body["rrf_k"] == 60
    assert body["results"][0]["retrieval_sources"] == ["vector", "graph"]
    assert "answer" not in body  # retrieval only, no generation


def test_api_partial_response(api):
    client, holder = api
    holder["resources"] = resources(graph_client_error=GraphNotConfiguredError("x"))
    body = client.post("/v1/retrieval/hybrid", json={"question": QUESTION}).json()
    assert body["retrieval_status"] == "partial" and body["graph"]["status"] == "failed"


def test_api_no_results_response(api, graph):
    client, holder = api
    graph.response = graph_response(status="no_match")
    holder["resources"] = resources(collection=FakeCollection(order=()))
    body = client.post("/v1/retrieval/hybrid", json={"question": QUESTION}).json()
    assert body["retrieval_status"] == "no_results" and body["results"] == []


def test_api_failed_response(api, graph):
    client, holder = api
    graph.error = RuntimeError("password=hunter2")
    holder["resources"] = resources(collection=FakeCollection(fail=True))
    response = client.post("/v1/retrieval/hybrid", json={"question": QUESTION})
    assert response.status_code == 200
    assert response.json()["retrieval_status"] == "failed"
    assert "hunter2" not in response.text
