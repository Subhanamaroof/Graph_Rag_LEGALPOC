"""Story 1.10 — Streamlit frontend tests.

HTTP is mocked with ``httpx.MockTransport``; the Streamlit app runs in ``AppTest`` with the API
client patched. No backend, Groq, ChromaDB or Neo4j is used.
"""

import copy
import json
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from frontend import state
from frontend.api.client import ApiError, GraphRagApiClient, HealthStatus
from frontend.components.answer import STATUS_DISPLAY, citation_references, status_display
from frontend.components.common import chunk_text, origin_label, text_lookup
from frontend.components.evidence import graph_rows, hybrid_rows, vector_rows
from frontend.components.graph_insights import (
    build_dot,
    compute_contribution,
    contribution_observations,
    entity_rows,
    group_entities,
    relationship_rows,
)
from frontend.components.sources import claim_rows, split_sources

APP_PATH = str(Path(__file__).resolve().parent.parent / "frontend" / "app.py")

# --- Fixture: an answer response shaped like Story 1.9 / 1.9.1 --------------------------------

COURT = {"entity_id": "e-court", "name": "Supreme Court of India", "entity_type": "Court"}
IPC = {"entity_id": "e-ipc", "name": "Indian Penal Code", "entity_type": "Statute"}
S302 = {"entity_id": "e-302", "name": "Section 302", "entity_type": "LegalSection"}
EDGE = {
    "relationship_id": "r1",
    "relationship_type": "APPLIES",
    "source_entity_id": "e-court",
    "target_entity_id": "e-302",
    "evidence": "the Court applied Section 302",
    "source_document_id": "DOC2",
    "source_chunk_id": "DOC2_chunk1",
}
EDGE2 = {
    "relationship_id": "r2",
    "relationship_type": "REFERS_TO",
    "source_entity_id": "e-302",
    "target_entity_id": "e-ipc",
    "evidence": None,
    "source_document_id": "DOC3",
    "source_chunk_id": "DOC3_chunk0",
}
PATH = {
    "graph_rank": 1,
    "graph_score": 0.9,
    "query_entity": "Section 302",
    "matched_entity": S302,
    "related_node": COURT,
    "path": "(Supreme Court of India)-[:APPLIES]->(Section 302)",
    "hop_distance": 1,
}


def chunk(doc, chunk_id, origin, hybrid_rank, *, vrank=None, grank=None, in_final=True, ctx=None):
    return {
        "document_id": doc,
        "chunk_id": chunk_id,
        "title": "State v. Accused",
        "text_preview": f"preview of {chunk_id}",
        "text_length": 900,
        "text_truncated": True,
        "retrieval_origin": origin,
        "hybrid_rank": hybrid_rank,
        "rrf_score": 1 / (60 + hybrid_rank),
        "in_final_results": in_final,
        "vector_rank": vrank,
        "vector_score": 0.81234 if vrank else None,
        "vector_distance": 0.18766 if vrank else None,
        "graph_rank": grank,
        "graph_context": ctx,
    }


GRAPH_CTX = {"paths": [PATH], "entities": [COURT, S302], "relationships": [EDGE], "evidence": []}
V_ONLY = chunk("DOC1", "DOC1_chunk0", "vector", 2, vrank=1)
BOTH = chunk("DOC3", "DOC3_chunk0", "both", 1, vrank=2, grank=2)
G_ONLY = chunk("DOC2", "DOC2_chunk1", "graph", 3, grank=1, ctx=GRAPH_CTX)


def source(n, doc, chunk_id, origin, cited):
    return {
        "source_number": n,
        "document_id": doc,
        "chunk_id": chunk_id,
        "rank": n,
        "retrieval_sources": ["vector"],
        "retrieval_source": origin,
        "rrf_score": 0.016,
        "content": f"FULL TEXT of {chunk_id}",
        "metadata": {"court": "Supreme Court"},
        "cited": cited,
    }


SAMPLE = {
    "question": "Which section did the court apply?",
    "status": "success",
    "answer": "The Court applied Section 302 [DOC2, DOC2_chunk1].",
    "claims": [
        {
            "text": "The Court applied Section 302.",
            "citations": [{"document_id": "DOC2", "chunk_id": "DOC2_chunk1"}],
        }
    ],
    "sources": [
        source(1, "DOC3", "DOC3_chunk0", "vector_and_graph", False),
        source(2, "DOC1", "DOC1_chunk0", "vector", False),
        source(3, "DOC2", "DOC2_chunk1", "graph", True),
    ],
    "retrieval_summary": {
        "retrieval_status": "success",
        "vector_results": 2,
        "graph_results": 2,
        "found_by_both": 1,
        "final_evidence_count": 3,
        "evidence_used": 3,
        "duplicates_removed": 0,
    },
    "generation_metadata": {
        "model": "test-model",
        "llm_called": True,
        "claims_returned": 2,
        "claims_accepted": 1,
    },
    "warnings": [],
    "errors": [],
    "retrieval_details": {
        "vector_status": "completed",
        "graph_status": "completed",
        "graph_retrieval_status": "success",
        "vector_results": [V_ONLY, BOTH],
        "graph_results": [G_ONLY, BOTH],
        "hybrid_results": [BOTH, V_ONLY, G_ONLY],
        "graph_only_chunks": [G_ONLY],
        "graph_paths": [],
        "graph_entities": [COURT, S302, IPC],
        "graph_relationships": [EDGE, EDGE2],
        "matched_entities": [
            {
                "query_name": "Section 302",
                "entity_id": "e-302",
                "name": "Section 302",
                "normalized_name": "section 302",
                "entity_type": "LegalSection",
                "match_type": "exact_name",
                "source_document_ids": ["DOC2"],
                "source_chunk_ids": ["DOC2_chunk1"],
                "document_count": 1,
                "chunk_count": 1,
            }
        ],
        "unresolved_references": [],
        "claim_citations": [
            {
                "claim_text": "The Court applied Section 302.",
                "supported": True,
                "citations": [
                    {
                        "document_id": "DOC2",
                        "chunk_id": "DOC2_chunk1",
                        "source_number": 3,
                        "hybrid_rank": 3,
                        "retrieval_origin": "graph",
                    }
                ],
            },
            {
                "claim_text": "Unsupported claim.",
                "supported": False,
                "citations": [],
                "rejection_reason": "all_citations_invalid",
            },
        ],
        "fusion_summary": {
            "vector_count": 2,
            "graph_count": 2,
            "hybrid_count": 3,
            "overlap_count": 1,
            "vector_only_count": 1,
            "graph_only_count": 1,
            "both_count": 1,
            "final_results_count": 3,
            "rrf_k": 60,
        },
        "max_items_per_list": 50,
        "text_preview_chars": 400,
        "truncated_lists": [],
    },
}


def sample(**changes):
    data = copy.deepcopy(SAMPLE)
    data.update(changes)
    return data


def empty_details():
    details = copy.deepcopy(SAMPLE["retrieval_details"])
    for key in (
        "vector_results",
        "graph_results",
        "hybrid_results",
        "graph_only_chunks",
        "graph_entities",
        "graph_relationships",
        "matched_entities",
        "claim_citations",
    ):
        details[key] = []
    details["fusion_summary"] = {"rrf_k": 60}
    details["graph_retrieval_status"] = "no_match"
    return details


# --- 1–4. API client ----------------------------------------------------------------------------


def make_client(handler) -> GraphRagApiClient:
    return GraphRagApiClient("http://backend", transport=httpx.MockTransport(handler))


def test_health_check_connected():
    client = make_client(
        lambda r: httpx.Response(200, json={"status": "ok", "application": "Graph RAG POC"})
    )
    assert client.check_health() == HealthStatus(True, "Connected", "Graph RAG POC")


def test_health_check_connection_error_and_bad_status():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    down = make_client(refuse).check_health()
    assert not down.connected and "Cannot reach the backend" in down.message
    odd = make_client(lambda r: httpx.Response(200, json={"status": "degraded"})).check_health()
    assert not odd.connected


def test_generate_answer_sends_expected_request():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=SAMPLE)

    data = make_client(handler).generate_answer("What applies?", top_k=7)
    assert seen["path"] == "/v1/generation/answer"
    assert seen["body"] == {"question": "What applies?", "include_sources": True, "top_k": 7}
    assert data["status"] == "success"


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (httpx.Response(500, json={"detail": "boom"}), "http"),
        (httpx.Response(422, json={"detail": [{"msg": "question too long"}]}), "http"),
        (httpx.Response(200, content=b"not json"), "invalid_json"),
        (httpx.Response(200, json={"status": "weird"}), "schema"),
        (httpx.Response(200, json=[1, 2]), "schema"),
        (httpx.Response(200, json={"status": "success"}), "schema"),  # missing summaries
    ],
)
def test_generate_answer_errors(response, kind):
    with pytest.raises(ApiError) as exc:
        make_client(lambda r: response).generate_answer("q")
    assert exc.value.kind == kind
    if response.status_code == 422:
        assert "question too long" in exc.value.message


def test_generate_answer_timeout_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ApiError) as exc:
        make_client(handler).generate_answer("q")
    assert exc.value.kind == "timeout" and "not retried" in exc.value.message
    assert len(calls) == 1


# --- 5–7. Status, citations, source classification ---------------------------------------------


def test_every_backend_status_has_a_display():
    for status in (
        "success",
        "partial",
        "insufficient_evidence",
        "retrieval_failed",
        "generation_failed",
    ):
        assert status in STATUS_DISPLAY
    assert status_display("success")[1] == "success"
    assert status_display("retrieval_failed")[1] == "error"


def test_citations_come_only_from_validated_response():
    assert citation_references(SAMPLE) == [("DOC2", "DOC2_chunk1")]
    cited, uncited = split_sources(SAMPLE)
    assert [s["chunk_id"] for s in cited] == ["DOC2_chunk1"]
    assert [s["source_number"] for s in uncited] == [1, 2]
    rows = claim_rows(SAMPLE)
    assert rows[0]["supported"] and "[SOURCE 3] DOC2 · DOC2_chunk1 (Graph)" in rows[0]["citations"]
    assert rows[1] == {
        "claim": "Unsupported claim.",
        "supported": False,
        "citations": [],
        "rejection_reason": "all_citations_invalid",
    }


def test_claim_rows_fall_back_to_story_1_9_claims():
    rows = claim_rows(sample(retrieval_details=None))
    assert rows == [
        {
            "claim": "The Court applied Section 302.",
            "supported": True,
            "citations": ["DOC2 · DOC2_chunk1"],
            "rejection_reason": None,
        }
    ]


def test_source_classification_labels():
    assert origin_label("vector") == "Vector"
    assert origin_label("graph") == "Graph"
    assert origin_label("both") == "Vector + Graph"
    assert origin_label("vector_and_graph") == "Vector + Graph"
    assert origin_label(None) == "Unknown"


# --- 8. Vector and graph result display ---------------------------------------------------------


def test_vector_graph_and_hybrid_rows_use_backend_values():
    details = SAMPLE["retrieval_details"]
    vrows = vector_rows(details)
    assert [r["Chunk ID"] for r in vrows] == ["DOC1_chunk0", "DOC3_chunk0"]
    assert vrows[0]["Similarity"] == 0.8123 and vrows[1]["Origin"] == "Vector + Graph"
    grows = graph_rows(details)
    assert grows[0]["Graph path"] == PATH["path"] and grows[0]["Origin"] == "Graph"
    assert grows[1]["Graph path"] is None  # no graph_context: nothing invented
    hrows = hybrid_rows(details)
    assert [r["Hybrid rank"] for r in hrows] == [1, 2, 3]
    assert hrows[0]["Vector rank"] == 2 and hrows[0]["Graph rank"] == 2


def test_chunk_text_prefers_full_source_text_then_preview():
    lookup = text_lookup(SAMPLE["sources"])
    assert chunk_text(G_ONLY, lookup) == ("FULL TEXT of DOC2_chunk1", False)
    other = chunk("DOCX", "DOCX_chunk9", "vector", 9, vrank=9)
    assert chunk_text(other, lookup) == ("preview of DOCX_chunk9", True)


# --- 9–10. Entities and relationships -----------------------------------------------------------


def test_entities_grouped_by_type_in_schema_order():
    groups = group_entities([IPC, {"entity_id": "x", "name": "X"}, COURT, S302])
    assert list(groups) == ["Court", "Statute", "LegalSection", "Untyped"]


def test_entity_rows_fill_provenance_only_from_matched_entities():
    rows = {r["Entity ID"]: r for r in entity_rows(SAMPLE["retrieval_details"])}
    assert rows["e-302"]["Normalized name"] == "section 302"
    assert rows["e-302"]["Source chunk IDs"] == "DOC2_chunk1"
    assert rows["e-ipc"]["Normalized name"] == "—"  # not in the API: shown as missing
    assert rows["e-ipc"]["Source document IDs"] == "—"


def test_relationship_rows_map_names_and_mark_missing_evidence():
    details = SAMPLE["retrieval_details"]
    names = {e["entity_id"]: e["name"] for e in details["graph_entities"]}
    rows = relationship_rows([EDGE, EDGE2, {**EDGE, "source_entity_id": "e-unknown"}], names)
    assert rows[0]["Source entity"] == "Supreme Court of India"
    assert rows[0]["Relationship"] == "APPLIES" and rows[0]["Source chunk ID"] == "DOC2_chunk1"
    assert rows[1]["Supporting evidence"] == "—"
    assert rows[2]["Source entity"] == "e-unknown"  # unknown names keep the real ID


def test_graph_view_contains_only_returned_nodes_and_edges():
    dot = build_dot(SAMPLE["retrieval_details"])
    assert dot.count("->") == 2
    assert "Supreme Court of India" in dot and "Indian Penal Code" in dot
    assert build_dot(empty_details()) is None


# --- 11–13. Contribution, missing fields, empty results -----------------------------------------


def test_contribution_counts_come_from_response_fields():
    c = compute_contribution(SAMPLE)
    assert c["vector_count"] == 2 and c["graph_count"] == 2
    assert c["graph_only_count"] == 1 and c["both_count"] == 1
    assert c["duplicates_merged"] == 1 and c["final_results_count"] == 3
    assert c["graph_only_in_final"] == 1
    assert c["supported_claims"] == 1 and c["claims_citing_graph_only"] == 1
    observations = contribution_observations(SAMPLE)
    assert any("added 1 chunk(s) that vector retrieval did not return" in o for o in observations)
    assert any("APPLIES" in o and "DOC2_chunk1" in o for o in observations)
    assert any("found by both" in o for o in observations)
    assert any("1 of 1 supported claim(s) cite at least one graph-only" in o for o in observations)


def test_contribution_when_graph_adds_no_unique_evidence():
    data = sample()
    d = data["retrieval_details"]
    d["graph_only_chunks"] = []
    d["fusion_summary"].update(graph_only_count=0, graph_count=1)
    d["claim_citations"] = [
        {
            **d["claim_citations"][0],
            "citations": [{**d["claim_citations"][0]["citations"][0], "retrieval_origin": "both"}],
        }
    ]
    observations = contribution_observations(data)
    assert any("no unique evidence" in o for o in observations)
    assert any("No supported claim cites a graph-only chunk" in o for o in observations)


def test_missing_optional_graph_fields():
    data = sample()
    d = data["retrieval_details"]
    d["graph_only_chunks"] = [{**G_ONLY, "graph_context": None}]
    for key in ("graph_entities", "matched_entities"):
        d.pop(key)
    observations = contribution_observations(data)  # no path: no per-chunk observation
    assert not any("was reached from" in o for o in observations)
    assert entity_rows(d) == []
    assert graph_rows({"graph_results": [{"chunk_id": "c"}]})[0]["Graph path"] is None
    assert compute_contribution(sample(retrieval_details=None)) is None
    assert contribution_observations(sample(retrieval_details=None)) == []


def test_empty_retrieval_results():
    data = sample(
        retrieval_details=empty_details(),
        sources=[],
        claims=[],
        answer=None,
        status="insufficient_evidence",
    )
    c = compute_contribution(data)
    assert c["vector_count"] == 0 and c["graph_count"] == 0 and c["supported_claims"] == 0
    assert contribution_observations(data) == [
        "Graph retrieval added no evidence for this question (graph retrieval status: no_match)."
    ]
    assert vector_rows(data["retrieval_details"]) == []
    assert split_sources(data) == ([], [])


# --- 14–16. App rendering and session state -----------------------------------------------------


@pytest.fixture
def app(monkeypatch):
    responses = []
    monkeypatch.setattr(
        GraphRagApiClient,
        "check_health",
        lambda self: HealthStatus(True, "Connected", "Graph RAG POC"),
    )

    def fake_generate(self, question, top_k=None):
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(GraphRagApiClient, "generate_answer", fake_generate)
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.responses = responses
    return at.run()


def ask(at, question, response):
    at.responses.append(response)
    at.chat_input[0].set_value(question).run()
    return at


def all_text(at):
    return " ".join(
        str(e.value)
        for kind in ("markdown", "caption", "success", "info", "warning", "error")
        for e in getattr(at, kind)
    )


def test_app_renders_answer_status_citations_and_evidence(app):
    ask(app, "Which section did the court apply?", sample())
    assert not app.exception
    text = all_text(app)
    assert "The Court applied Section 302" in text
    assert any("Answer generated" in s.value for s in app.success)
    labels = [e.label for e in app.expander]
    assert any("[SOURCE 3] DOC2 · DOC2_chunk1 — Graph" in label for label in labels)
    assert any("Vector #1 · DOC1 · DOC1_chunk0 — Vector" in label for label in labels)
    assert [t.label for t in app.tabs][:4] == [
        "Sources & citations",
        "How Graph Retrieval Contributed",
        "Retrieval evidence",
        "Entities & relationships",
    ]
    assert "Graph retrieval added 1 chunk(s)" in text


def test_app_null_answer_shows_status_explanation(app):
    response = sample(
        status="insufficient_evidence",
        answer=None,
        claims=[],
        errors=[],
        retrieval_details=empty_details(),
    )
    ask(app, "Unknown topic?", response)
    assert not app.exception
    assert any("Insufficient evidence" in i.value for i in app.info)
    assert "The Court applied Section 302" not in all_text(app)


def test_app_shows_api_error_without_answer(app):
    ask(app, "q", ApiError("connection", "Cannot reach the backend at http://x."))
    assert not app.exception
    assert any("Cannot reach the backend" in e.value for e in app.error)
    assert app.session_state["messages"][-1]["error"].startswith("Cannot reach")


def test_session_state_keeps_conversation_order():
    ss = {}
    state.init_state(ss, default_top_k=10)
    assert ss["messages"] == [] and ss["top_k"] == 10 and not ss["is_generating"]
    state.add_user_message(ss, "q1")
    state.add_assistant_message(ss, response={"status": "success"})
    state.add_user_message(ss, "q2")
    state.add_assistant_message(ss, error="boom")
    assert [m["role"] for m in ss["messages"]] == ["user", "assistant", "user", "assistant"]
    state.init_state(ss, default_top_k=3)  # re-running the script keeps existing state
    assert len(ss["messages"]) == 4 and ss["top_k"] == 10


def test_new_chat_and_clear_chat():
    ss = {}
    state.init_state(ss, default_top_k=10)
    ss["top_k"] = 4
    state.add_user_message(ss, "q")
    old_id = ss["conversation_id"]
    state.clear_chat(ss)
    assert ss["messages"] == [] and ss["top_k"] == 4 and ss["conversation_id"] == old_id
    state.add_user_message(ss, "q")
    state.new_chat(ss, default_top_k=10)
    assert ss["messages"] == [] and ss["top_k"] == 10 and ss["conversation_id"] != old_id


def test_app_new_chat_button_clears_conversation(app):
    ask(app, "q", sample())
    assert len(app.session_state["messages"]) == 2
    next(b for b in app.button if b.label == "New Chat").click().run()
    assert app.session_state["messages"] == []
    assert not app.chat_message  # nothing rendered after the reset
