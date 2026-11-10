"""Story 1.9.1 tests: retrieval details in the answer API.

The real Story 1.8 ``retrieve_hybrid`` runs over an in-memory ChromaDB fake and a recorded
Story 1.7 graph response; Groq is a fake JSON client. Fixture: vector search returns A, B, C;
graph paths cite B (path 1) and D (path 2). So B is found by both, A and C by vector only,
and D by graph only. No external service is called.
"""

import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from graph_rag.core.llm import LLMError
from graph_rag.generation.generation_models import AnswerRequest, AnswerResponse
from graph_rag.generation.generation_router import get_generation_resources
from graph_rag.generation.generation_service import GenerationResources, generate_answer
from graph_rag.generation.retrieval_details import build_retrieval_details
from graph_rag.graph.neo4j_client import GraphNotConfiguredError
from graph_rag.main import app
from graph_rag.retrieval import hybrid_retrieval_service as service
from graph_rag.retrieval.graph_retrieval_models import (
    Evidence,
    GraphEdge,
    GraphNode,
    GraphResult,
    GraphRetrievalResponse,
    MatchedEntity,
)
from graph_rag.retrieval.hybrid_retrieval_models import (
    HybridRetrievalRequest,
    HybridRetrievalResponse,
)
from graph_rag.retrieval.hybrid_retrieval_service import (
    HybridParameters,
    HybridResources,
    retrieve_hybrid,
)
from graph_rag.retrieval.hybrid_router import get_hybrid_resources

D1, D2 = "CASE-001", "CASE-002"
A, B, C, D = f"{D1}_chunk1", f"{D1}_chunk2", f"{D1}_chunk3", f"{D2}_chunk4"
QUESTION = "Which cases refer to the Indian Penal Code?"
GENERATION_DIR = Path(__file__).parents[1] / "src" / "graph_rag" / "generation"
PARAMS = HybridParameters(
    vector_top_k=3,
    graph_top_k=10,
    final_top_k=10,
    max_hops=2,
    rrf_k=60,
    graph_query_timeout=7.0,
    branch_timeout=5.0,
)


# --- Fakes ----------------------------------------------------------------------------------


class FakeEmbedder:
    def __init__(self):
        self.texts = []

    def load(self):
        pass

    def embed_text(self, text):
        self.texts.append(text)
        return [0.1] * 384


class FakeCollection:
    def __init__(self, order=(A, B, C)):
        self.records = {
            A: ("text A " * 10, {"case_id": D1, "court": "Delhi HC", "title": "X v Y"}),
            B: ("text B", {"case_id": D1, "court": "Delhi HC", "date": "2019-01-02"}),
            C: ("text C", {"case_id": D1}),
            D: ("text D", {"case_id": D2, "court": "SC", "source": "/secret/path.pdf"}),
        }
        self.order, self.queries = list(order), 0

    def query(self, query_embeddings, n_results, include):
        self.queries += 1
        ids = self.order[:n_results]
        return {
            "ids": [ids],
            "documents": [[self.records[i][0] for i in ids]],
            "metadatas": [[self.records[i][1] for i in ids]],
            "distances": [[0.1 * (n + 1) for n in range(len(ids))]],
        }

    def get(self, ids=None, where=None, include=None):
        found = [i for i in ids if i in self.records]
        return {
            "ids": found,
            "documents": [self.records[i][0] for i in found],
            "metadatas": [self.records[i][1] for i in found],
        }


IPC = GraphNode(entity_id="ipc", name="Indian Penal Code", entity_type="STATUTE")


def path(rank, rel, doc, chunk):
    case = GraphNode(entity_id=f"case{rank}", name=f"Case {rank}", entity_type="CASE")
    edge = GraphEdge(
        relationship_id=rel,
        relationship_type="REFERS_TO",
        source_entity_id=case.entity_id,
        target_entity_id=IPC.entity_id,
        evidence=f"quote {rel}",
        source_document_id=doc,
        source_chunk_id=chunk,
    )
    return GraphResult(
        rank=rank,
        score=1.0 - rank / 10,
        query_entity="Indian Penal Code",
        matched_entity=IPC,
        related_node=case,
        nodes=[IPC, case],
        relationships=[edge],
        path=f"(Indian Penal Code:STATUTE) <-[REFERS_TO]- (Case {rank}:CASE)",
        hop_distance=1,
        supporting_edges=1,
        evidence=[
            Evidence(
                relationship_id=rel,
                relationship_type="REFERS_TO",
                evidence=f"quote {rel}",
                document_id=doc,
                chunk_id=chunk,
            )
        ],
    )


MATCHED = MatchedEntity(
    query_name="Indian Penal Code", entity_id="ipc", name="Indian Penal Code", match_type="alias"
)


def graph_response(results, status="success"):
    return GraphRetrievalResponse(
        question=QUESTION,
        analysis_method="deterministic",
        matched_entities=[MATCHED] if results else [],
        results=results,
        total_results=len(results),
        retrieval_status=status,
    )


class GraphRecorder:
    def __init__(self, response):
        self.response, self.calls = response, 0

    def __call__(self, request, client, llm=None, *, timeout=None):
        self.calls += 1
        return self.response


class FakeLLM:
    def __init__(self, output=None, error=None):
        self.output, self.error, self.calls = output, error, 0

    def complete_json(self, system_prompt, user_prompt):
        self.calls += 1
        if self.error:
            raise self.error
        return self.output


def cite(doc, chunk):
    return {"document_id": doc, "chunk_id": chunk}


def llm_json(claims, answer="Answer.", insufficient=False):
    return json.dumps({"answer": answer, "claims": claims, "insufficient_evidence": insufficient})


GOOD_CLAIMS = [
    {"text": "Case 1 refers to the IPC.", "citations": [cite(D1, B), cite(D1, "ghost")]},
    {"text": "Case 2 also does.", "citations": [cite(D2, D)]},
    {"text": "Unsupported statement.", "citations": [cite(D2, "nowhere")]},
]


class Pipeline:
    """Real Story 1.8 + Story 1.9 with fakes at the store / LLM boundary, counting calls."""

    def __init__(
        self,
        monkeypatch,
        *,
        graph_paths=None,
        vector_order=(A, B, C),
        graph_error=None,
        llm=None,
        max_items=50,
        preview_chars=400,
    ):
        paths = [path(1, "r1", D1, B), path(2, "r2", D2, D)] if graph_paths is None else graph_paths
        self.graph = GraphRecorder(graph_response(paths))
        monkeypatch.setattr(service, "retrieve_graph", self.graph)
        self.collection, self.embedder = FakeCollection(vector_order), FakeEmbedder()
        self.hybrid_calls = 0
        self.llm = llm if llm is not None else FakeLLM(llm_json(GOOD_CLAIMS))

        def load_graph_client():
            if graph_error:
                raise graph_error
            return "neo4j-client"

        self.hybrid = HybridResources(
            load_collection=lambda: self.collection,
            load_embedder=lambda: self.embedder,
            load_graph_client=load_graph_client,
        )

        def retrieve(request):
            self.hybrid_calls += 1
            params = replace(PARAMS, final_top_k=request.final_top_k or PARAMS.final_top_k)
            return retrieve_hybrid(request, params, self.hybrid)

        self.resources = GenerationResources(
            retrieve=retrieve,
            llm=self.llm,
            model_name="test-model",
            max_context_chars=12000,
            details_max_items=max_items,
            details_preview_chars=preview_chars,
        )

    def ask(self, **request) -> AnswerResponse:
        return generate_answer(AnswerRequest(question=QUESTION, **request), self.resources)


def keys(items):
    return [(item.document_id, item.chunk_id) for item in items]


# --- A. Retrieval details ---------------------------------------------------------------------


def test_branch_and_hybrid_results_exposed_with_origins(monkeypatch):
    response = Pipeline(monkeypatch).ask()
    details = response.retrieval_details
    assert response.status == "success"
    assert (details.vector_status, details.graph_status) == ("completed", "completed")

    assert keys(details.vector_results) == [(D1, A), (D1, B), (D1, C)]
    assert [r.vector_rank for r in details.vector_results] == [1, 2, 3]
    assert [r.vector_distance for r in details.vector_results] == pytest.approx([0.1, 0.2, 0.3])
    assert [r.vector_score for r in details.vector_results] == pytest.approx([0.9, 0.8, 0.7])

    assert keys(details.graph_results) == [(D1, B), (D2, D)]
    assert [r.graph_rank for r in details.graph_results] == [1, 2]

    # B is in both lists, so it collects two RRF terms and ranks first; one entry per chunk.
    assert keys(details.hybrid_results)[0] == (D1, B)
    assert sorted(keys(details.hybrid_results)) == sorted([(D1, A), (D1, B), (D1, C), (D2, D)])
    assert [r.hybrid_rank for r in details.hybrid_results] == [1, 2, 3, 4]
    assert details.hybrid_results[0].rrf_score == pytest.approx(1 / 62 + 1 / 61, abs=1e-6)
    origin = {r.chunk_id: r.retrieval_origin for r in details.hybrid_results}
    assert origin == {A: "vector", B: "both", C: "vector", D: "graph"}
    assert keys(details.graph_only_chunks) == [(D2, D)]


def test_hybrid_ranks_match_story_1_8_sources(monkeypatch):
    response = Pipeline(monkeypatch).ask()
    hybrid = {r.chunk_id: r for r in response.retrieval_details.hybrid_results}
    for source in response.sources:
        detail = hybrid[source.chunk_id]
        assert (detail.hybrid_rank, detail.rrf_score) == (source.rank, source.rrf_score)
        assert detail.document_id == source.document_id


def test_metadata_preview_and_missing_optional_metadata(monkeypatch):
    details = Pipeline(monkeypatch, preview_chars=50).ask().retrieval_details
    by_chunk = {r.chunk_id: r for r in details.hybrid_results}
    a, c = by_chunk[A], by_chunk[C]
    assert (a.case_id, a.title, a.court) == (D1, "X v Y", "Delhi HC")
    assert len(a.text_preview) == 50 and a.text_truncated and a.text_length == 70
    assert (c.title, c.court, c.date) == (None, None, None)  # absent, not invented
    assert by_chunk[B].date == "2019-01-02" and not by_chunk[B].text_truncated
    assert "source" not in by_chunk[D].model_dump()  # raw metadata (file paths) not exposed


# --- B. Fusion summary ------------------------------------------------------------------------


def test_fusion_summary_counts(monkeypatch):
    summary = Pipeline(monkeypatch).ask().retrieval_details.fusion_summary
    assert summary.model_dump() == {
        "vector_count": 3,
        "graph_count": 2,
        "hybrid_count": 4,
        "overlap_count": 1,
        "vector_only_count": 2,
        "graph_only_count": 1,
        "both_count": 1,
        "final_results_count": 4,
        "rrf_k": 60,
    }


def test_counts_cover_full_merge_beyond_final_top_k(monkeypatch):
    details = Pipeline(monkeypatch).ask(top_k=2).retrieval_details
    assert details.fusion_summary.hybrid_count == 4
    assert details.fusion_summary.final_results_count == 2
    assert [r.in_final_results for r in details.hybrid_results] == [True, True, False, False]
    assert keys(details.graph_only_chunks) == [(D2, D)]  # still reported outside final_top_k


def test_no_evidence_skips_llm_and_reports_empty_summary(monkeypatch):
    pipeline = Pipeline(monkeypatch, graph_paths=[], vector_order=())
    response = pipeline.ask()
    assert response.status == "insufficient_evidence" and pipeline.llm.calls == 0
    details = response.retrieval_details
    assert details.fusion_summary.model_dump(exclude={"rrf_k"}) == dict.fromkeys(
        details.fusion_summary.model_dump(exclude={"rrf_k"}), 0
    )
    assert details.hybrid_results == details.graph_paths == details.claim_citations == []


def test_partial_retrieval_exposes_only_available_branch(monkeypatch):
    pipeline = Pipeline(monkeypatch, graph_error=GraphNotConfiguredError("no creds"))
    response = pipeline.ask()
    details = response.retrieval_details
    assert response.status == "partial"
    assert (details.vector_status, details.graph_status) == ("completed", "failed")
    assert details.graph_results == details.graph_paths == details.graph_entities == []
    assert details.fusion_summary.graph_count == 0 and details.fusion_summary.vector_count == 3
    assert {r.retrieval_origin for r in details.hybrid_results} == {"vector"}


def test_empty_vector_results(monkeypatch):
    details = Pipeline(monkeypatch, vector_order=()).ask().retrieval_details
    assert details.vector_results == []
    assert keys(details.graph_only_chunks) == keys(details.graph_results) == [(D1, B), (D2, D)]
    assert details.fusion_summary.graph_only_count == 2 and details.fusion_summary.both_count == 0


# --- C. Graph explainability ------------------------------------------------------------------


def test_graph_entities_relationships_and_provenance(monkeypatch):
    details = Pipeline(monkeypatch).ask().retrieval_details
    assert [n.entity_id for n in details.graph_entities] == ["ipc", "case1", "case2"]
    assert [
        (e.relationship_id, e.source_document_id, e.source_chunk_id)
        for e in details.graph_relationships
    ] == [("r1", D1, B), ("r2", D2, D)]
    assert [p.rank for p in details.graph_paths] == [1, 2]
    assert [m.entity_id for m in details.matched_entities] == ["ipc"]

    graph_b = details.graph_results[0]
    assert [p.graph_rank for p in graph_b.graph_context.paths] == [1]
    assert [(e.document_id, e.chunk_id) for e in graph_b.graph_context.evidence] == [(D1, B)]
    assert details.graph_only_chunks[0].graph_context.relationships[0].relationship_id == "r2"
    # Graph context is kept in the graph lists only; join other lists by (document_id, chunk_id).
    assert all(r.graph_context is None for r in details.vector_results + details.hybrid_results)


def test_missing_graph_provenance_is_not_fabricated(monkeypatch):
    broken = path(1, "r9", D1, None)  # relationship without a source chunk
    details = Pipeline(monkeypatch, graph_paths=[broken]).ask().retrieval_details
    assert details.graph_results == details.graph_only_chunks == []
    assert details.graph_relationships[0].source_chunk_id is None
    assert [(u.relationship_id, u.reason) for u in details.unresolved_references] == [
        ("r9", "missing_chunk_id")
    ]
    assert {r.retrieval_origin for r in details.hybrid_results} == {"vector"}


def test_no_graph_paths_gives_empty_graph_details(monkeypatch):
    details = Pipeline(monkeypatch, graph_paths=[]).ask().retrieval_details
    assert details.graph_entities == details.graph_relationships == details.graph_paths == []
    assert details.graph_retrieval_status == "success"


# --- D. Claim and citation mapping ------------------------------------------------------------


def test_claims_map_to_validated_sources_only(monkeypatch):
    response = Pipeline(monkeypatch).ask()
    mapped = response.retrieval_details.claim_citations
    sources = {s.chunk_id: s for s in response.sources}

    first, second, rejected = mapped
    assert first.supported and keys(first.citations) == [(D1, B)]  # "ghost" removed
    assert first.citations[0].source_number == sources[B].source_number
    assert first.citations[0].retrieval_origin == "both"
    assert second.citations[0].retrieval_origin == "graph"
    assert (rejected.supported, rejected.citations) == (False, [])
    assert rejected.rejection_reason == "all_citations_invalid"
    # The mapping mirrors the existing validated claims exactly.
    assert [keys(c.citations) for c in response.claims] == [keys(first.citations), [(D2, D)]]
    invalid = {(i.document_id, i.chunk_id) for i in response.generation_metadata.invalid_citations}
    assert invalid == {(D1, "ghost"), (D2, "nowhere")}
    cited = {c.chunk_id for claim in mapped for c in claim.citations}
    assert cited.isdisjoint({"ghost", "nowhere"})


def test_answer_still_withheld_when_no_claim_is_valid(monkeypatch):
    llm = FakeLLM(llm_json([{"text": "Made up.", "citations": [cite(D1, "ghost")]}]))
    response = Pipeline(monkeypatch, llm=llm).ask()
    assert response.status == "generation_failed" and response.answer is None
    assert response.claims == []
    (claim,) = response.retrieval_details.claim_citations
    assert not claim.supported and claim.rejection_reason == "all_citations_invalid"


def test_claim_without_citations_is_represented(monkeypatch):
    claims = [
        {"text": "Supported.", "citations": [cite(D1, A)]},
        {"text": "No citation.", "citations": []},
    ]
    response = Pipeline(monkeypatch, llm=FakeLLM(llm_json(claims))).ask()
    supported, unsupported = response.retrieval_details.claim_citations
    assert supported.supported and supported.citations[0].retrieval_origin == "vector"
    assert unsupported.rejection_reason == "no_citations" and unsupported.citations == []


# --- E. Backward compatibility ----------------------------------------------------------------


def test_existing_fields_and_request_schema_unchanged():
    assert set(AnswerResponse.model_fields) == {
        "question",
        "status",
        "answer",
        "claims",
        "sources",
        "retrieval_summary",
        "generation_metadata",
        "warnings",
        "errors",
        "retrieval_details",
    }
    assert AnswerResponse.model_fields["retrieval_details"].is_required() is False
    assert set(AnswerRequest.model_fields) == {"question", "top_k", "max_hops", "include_sources"}


def test_api_returns_details_and_consumers_can_ignore_them(monkeypatch):
    pipeline = Pipeline(monkeypatch)
    app.dependency_overrides[get_generation_resources] = lambda: pipeline.resources
    try:
        body = TestClient(app).post("/v1/generation/answer", json={"question": QUESTION}).json()
    finally:
        app.dependency_overrides.clear()
    assert body["status"] == "success" and body["claims"][0]["citations"] == [cite(D1, B)]
    assert body["retrieval_details"]["fusion_summary"]["graph_only_count"] == 1
    legacy = {k: v for k, v in body.items() if k != "retrieval_details"}
    assert AnswerResponse.model_validate(legacy).retrieval_details is None


def test_hybrid_endpoint_json_is_unchanged(monkeypatch):
    pipeline = Pipeline(monkeypatch)
    app.dependency_overrides[get_hybrid_resources] = lambda: pipeline.hybrid
    try:
        body = TestClient(app).post("/v1/retrieval/hybrid", json={"question": QUESTION}).json()
    finally:
        app.dependency_overrides.clear()
    assert body["retrieval_status"] == "success"
    assert "ranked_results" not in body and "graph_paths" not in body


def test_fallback_to_final_results_without_ranked_results(monkeypatch):
    request = HybridRetrievalRequest(question=QUESTION)
    full = retrieve_hybrid(request, PARAMS, Pipeline(monkeypatch).hybrid)
    legacy = HybridRetrievalResponse.model_validate(full.model_dump())  # excluded fields dropped
    assert legacy.ranked_results == [] and legacy.graph_paths == []
    details = build_retrieval_details(legacy, max_items=50, preview_chars=400)
    assert keys(details.hybrid_results) == keys(full.results)
    assert details.graph_entities == []  # not available, so not invented


# --- F. Execution behaviour -------------------------------------------------------------------


def test_each_branch_retrieval_and_llm_run_once(monkeypatch):
    pipeline = Pipeline(monkeypatch)
    pipeline.ask()
    assert pipeline.hybrid_calls == 1
    assert pipeline.collection.queries == 1 and len(pipeline.embedder.texts) == 1
    assert pipeline.graph.calls == 1
    assert pipeline.llm.calls == 1


def test_details_module_has_no_store_or_llm_access():
    tree = ast.parse((GENERATION_DIR / "retrieval_details.py").read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
    forbidden = ("chromadb", "neo4j", "groq", "graph_rag.vector_store", "graph_rag.graph")
    forbidden += ("graph_rag.core.llm", "graph_rag.retrieval.hybrid_retrieval_service")
    assert not any(name.startswith(forbidden) for name in imported), imported


# --- G. Failure states and limits -------------------------------------------------------------


def test_generation_failure_keeps_retrieval_details(monkeypatch):
    pipeline = Pipeline(monkeypatch, llm=FakeLLM(error=LLMError("boom")))
    response = pipeline.ask()
    assert response.status == "generation_failed" and response.answer is None
    assert response.retrieval_details.fusion_summary.hybrid_count == 4
    assert response.retrieval_details.claim_citations == []


def test_retrieval_failure_states(monkeypatch):
    pipeline = Pipeline(
        monkeypatch, vector_order=(), graph_error=GraphNotConfiguredError("no creds")
    )
    pipeline.collection.query = lambda **_: (_ for _ in ()).throw(RuntimeError("down"))
    response = pipeline.ask()
    assert response.status == "retrieval_failed" and pipeline.llm.calls == 0
    details = response.retrieval_details
    assert (details.vector_status, details.graph_status) == ("failed", "failed")
    assert details.fusion_summary.hybrid_count == 0

    def raising(request):
        raise RuntimeError("unexpected")

    pipeline.resources.retrieve = raising
    assert pipeline.ask().retrieval_details is None  # no Story 1.8 response to explain


def test_list_limits_cap_items_but_not_counts(monkeypatch):
    details = Pipeline(monkeypatch, max_items=1).ask().retrieval_details
    assert len(details.vector_results) == len(details.hybrid_results) == 1
    assert len(details.graph_entities) == 1
    assert details.fusion_summary.vector_count == 3 and details.fusion_summary.hybrid_count == 4
    assert {"vector_results", "graph_results", "hybrid_results", "graph_entities"} <= set(
        details.truncated_lists
    )
    assert details.max_items_per_list == 1
