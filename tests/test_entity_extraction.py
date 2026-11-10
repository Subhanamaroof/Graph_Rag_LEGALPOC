"""Story 1.5 tests: legal entity and relationship extraction.

The Groq LLM is replaced by ``FakeLLM`` (canned JSON answers) and the ChromaDB collection by
``FakeCollection`` (an in-memory ``get``). No API key, network or real database is used.
"""

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from graph_rag.core.config import Settings
from graph_rag.core.llm import GroqJSONClient, LLMError, LLMNotConfiguredError, create_groq_client
from graph_rag.graph.entity_extractor import extract_from_chunk, parse_llm_output
from graph_rag.graph.extraction_service import ChunkNotFoundError, extract_document
from graph_rag.graph.models import ChunkExtraction, Entity, Relationship
from graph_rag.graph.router import get_extraction_llm
from graph_rag.main import app
from graph_rag.retrieval.router import get_chroma_collection

DOC = "NI138-2019-0042"
CHUNK = f"{DOC}_chunk0"
TEXT = (
    "The complainant, Rajesh Kumar, filed a complaint against State Bank of India before the "
    "Delhi High Court. A notice under Section 138 of the Negotiable Instruments Act, 1881 was "
    "issued on 12.03.2018. Justice A. Sharma presided over the matter."
)


def entity(eid, name, etype, evidence, normalized=None):
    item = {"id": eid, "name": name, "entity_type": etype, "evidence": evidence}
    if normalized:
        item["normalized_name"] = normalized
    return item


E_PERSON = entity("e1", "Rajesh Kumar", "PERSON", "The complainant, Rajesh Kumar")
E_SECTION = entity("e2", "Section 138", "LEGAL_SECTION", "Section 138 of the Negotiable")
E_STATUTE = entity(
    "e3", "Negotiable Instruments Act, 1881", "STATUTE", "Negotiable Instruments Act, 1881"
)
E_COURT = entity("e4", "Delhi High Court", "COURT", "before the Delhi High Court")
E_DATE = entity("e5", "12.03.2018", "DATE", "issued on 12.03.2018")
E_ORG = entity("e6", "State Bank of India", "ORGANIZATION", "against State Bank of India")
REL_SECTION_OF_ACT = {
    "source": "e2",
    "target": "e3",
    "relationship_type": "RELATED_TO",
    "description": "Section 138 belongs to the Act",
    "evidence": "Section 138 of the Negotiable Instruments Act, 1881",
}


class FakeLLM:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, str) else json.dumps(answer)


def answer(entities=(), relationships=()):
    return {"entities": list(entities), "relationships": list(relationships)}


class FakeCollection:
    """Read-only stand-in: answers ``get`` by IDs or ``case_id`` and records writes (none)."""

    name = "judgments"

    def __init__(self, records):
        self.records = records  # (id, text, metadata)
        self.writes = 0

    def get(self, ids=None, where=None, include=None):
        rows = self.records
        if ids is not None:
            rows = [row for row in rows if row[0] in ids]
        if where is not None:
            rows = [row for row in rows if row[2].get("case_id") == where["case_id"]]
        return {
            "ids": [row[0] for row in rows],
            "documents": [row[1] for row in rows],
            "metadatas": [row[2] for row in rows],
        }

    def upsert(self, **_):
        self.writes += 1

    add = update = delete = upsert


def records(count, document_id=DOC):
    return [(f"{document_id}_chunk{i}", TEXT, {"case_id": document_id}) for i in range(count)]


def run(llm, text=TEXT):
    return extract_from_chunk(llm, document_id=DOC, chunk_id=CHUNK, chunk_text=text)


# --- Single-chunk extraction and validation ----------------------------------------


def test_valid_entity_extraction_keeps_original_name_and_provenance():
    result = run(FakeLLM(answer([E_PERSON])))
    assert result.extraction_status == "completed"
    (person,) = result.entities
    assert (person.name, person.normalized_name, person.entity_type) == (
        "Rajesh Kumar",
        "Rajesh Kumar",
        "PERSON",
    )
    assert (person.source_document_id, person.source_chunk_id) == (DOC, CHUNK)
    assert person.source_text == "The complainant, Rajesh Kumar"
    assert person.entity_id.startswith("ent_")


def test_valid_relationship_extraction_links_entity_ids():
    result = run(FakeLLM(answer([E_SECTION, E_STATUTE], [REL_SECTION_OF_ACT])))
    section, statute = result.entities
    (relationship,) = result.relationships
    assert relationship.source_entity_id == section.entity_id
    assert relationship.target_entity_id == statute.entity_id
    assert relationship.relationship_type == "RELATED_TO"
    assert relationship.evidence == REL_SECTION_OF_ACT["evidence"]
    assert (relationship.source_document_id, relationship.source_chunk_id) == (DOC, CHUNK)


def test_multiple_entity_types_in_one_chunk():
    result = run(FakeLLM(answer([E_PERSON, E_SECTION, E_STATUTE, E_COURT, E_DATE, E_ORG])))
    assert {e.entity_type.value for e in result.entities} == {
        "PERSON",
        "LEGAL_SECTION",
        "STATUTE",
        "COURT",
        "DATE",
        "ORGANIZATION",
    }
    assert result.warnings == []


def test_empty_extraction_response():
    result = run(FakeLLM(answer()))
    assert result.extraction_status == "completed"
    assert result.entities == [] and result.relationships == []


def test_blank_chunk_skips_llm():
    llm = FakeLLM(answer([E_PERSON]))
    result = run(llm, text="   ")
    assert llm.calls == [] and result.entities == [] and result.extraction_status == "completed"


def test_invalid_entity_type_is_rejected_with_warning():
    bad = entity("e9", "Rajesh Kumar", "PLANET", "Rajesh Kumar")
    result = run(FakeLLM(answer([bad, E_COURT])))
    assert [e.name for e in result.entities] == ["Delhi High Court"]
    assert "PLANET" not in str(result.entities) and "entity[0]" in result.warnings[0]


def test_invalid_relationship_type_is_rejected():
    rel = {**REL_SECTION_OF_ACT, "relationship_type": "OWNS"}
    result = run(FakeLLM(answer([E_SECTION, E_STATUTE], [rel])))
    assert len(result.entities) == 2 and result.relationships == []
    assert "relationship[0]" in result.warnings[0]


def test_relationship_with_unknown_entity_is_rejected():
    rel = {**REL_SECTION_OF_ACT, "target": "e42"}
    result = run(FakeLLM(answer([E_SECTION], [rel])))
    assert result.relationships == []
    assert "unknown entity" in result.warnings[0]


@pytest.mark.parametrize("evidence", [None, "", "Section 138 was struck down by Parliament"])
def test_relationship_without_grounded_evidence_is_rejected(evidence):
    rel = {**REL_SECTION_OF_ACT, "evidence": evidence}
    result = run(FakeLLM(answer([E_SECTION, E_STATUTE], [rel])))
    assert result.relationships == []
    assert "evidence" in result.warnings[0]


def test_entity_without_evidence_or_not_in_text_is_rejected():
    no_evidence = entity("e1", "Rajesh Kumar", "PERSON", None)
    hallucinated = entity("e2", "Ramesh Gupta", "PERSON", "Ramesh Gupta")
    result = run(FakeLLM(answer([no_evidence, hallucinated])))
    assert result.entities == []
    assert "evidence" in result.warnings[0] and "not found" in result.warnings[1]


def test_evidence_matching_ignores_case_whitespace_and_curly_quotes():
    item = entity("e1", "rajesh  kumar", "person", "the complainant,\n Rajesh Kumar")
    result = run(FakeLLM(answer([item])))
    assert result.entities[0].entity_type == "PERSON"
    assert result.entities[0].name == "rajesh kumar"


def test_duplicate_mentions_are_merged_and_aliases_kept():
    first = entity("e1", "Delhi High Court", "COURT", "before the Delhi High Court")
    second = entity("e2", "High Court", "COURT", "Delhi High Court", normalized="Delhi High Court")
    rel = {
        "source": "e2",
        "target": "e3",
        "relationship_type": "REFERS_TO",
        "evidence": "Section 138 of the Negotiable Instruments Act, 1881",
    }
    result = run(FakeLLM(answer([first, second, E_STATUTE], [rel])))
    court = result.entities[0]
    assert len(result.entities) == 2
    assert court.aliases == ["High Court"]
    assert result.relationships[0].source_entity_id == court.entity_id  # remapped from e2


def test_same_name_different_type_is_not_merged():
    as_court = entity("e1", "Delhi High Court", "COURT", "Delhi High Court")
    as_location = entity("e2", "Delhi", "LOCATION", "Delhi High Court", normalized="Delhi")
    result = run(FakeLLM(answer([as_court, as_location])))
    assert len(result.entities) == 2


def test_entity_ids_are_deterministic():
    first = run(FakeLLM(answer([E_PERSON])))
    second = run(FakeLLM(answer([E_PERSON])))
    assert first.entities[0].entity_id == second.entities[0].entity_id


# --- LLM response parsing and failures -----------------------------------------------


def test_parse_accepts_code_fenced_json():
    parsed = parse_llm_output('```json\n{"entities": [], "relationships": []}\n```')
    assert parsed.entities == [] and parsed.relationships == []


@pytest.mark.parametrize(
    "raw", ["not json", "[]", '{"entities": []}', '{"entities": {}, "relationships": []}']
)
def test_malformed_llm_response_fails_the_chunk(raw):
    result = run(FakeLLM(raw))
    assert result.extraction_status == "failed"
    assert result.error and result.entities == []


def test_llm_failure_fails_the_chunk():
    result = run(FakeLLM(LLMError("Groq returned HTTP 429.")))
    assert result.extraction_status == "failed"
    assert result.error == "Groq returned HTTP 429."


def test_prompt_contains_chunk_text_and_rules():
    llm = FakeLLM(answer())
    run(llm)
    system_prompt, user_prompt = llm.calls[0]
    assert TEXT in user_prompt
    assert "LEGAL_SECTION" in system_prompt and "JUDGE_PRESIDED_OVER" in system_prompt
    assert "Appearing in the same" in system_prompt


# --- Schema guarantees ---------------------------------------------------------------


def make_entity(entity_id="ent_1", chunk_id=CHUNK):
    return Entity(
        entity_id=entity_id,
        name="Section 138",
        normalized_name="Section 138",
        entity_type="LEGAL_SECTION",
        source_document_id=DOC,
        source_chunk_id=chunk_id,
        source_text="Section 138",
    )


def make_relationship(target="ent_2"):
    return Relationship(
        relationship_id="rel_1",
        source_entity_id="ent_1",
        target_entity_id=target,
        relationship_type="REFERS_TO",
        evidence="Section 138",
        source_document_id=DOC,
        source_chunk_id=CHUNK,
    )


def test_schema_rejects_invalid_types_and_missing_evidence():
    with pytest.raises(ValidationError):
        Entity(**{**make_entity().model_dump(), "entity_type": "PLANET"})
    with pytest.raises(ValidationError):
        Relationship(**{**make_relationship().model_dump(), "relationship_type": "OWNS"})
    with pytest.raises(ValidationError):
        Relationship(**{**make_relationship().model_dump(), "evidence": "  "})


def test_schema_rejects_unknown_entity_reference_and_wrong_provenance():
    with pytest.raises(ValidationError, match="unknown entity"):
        ChunkExtraction(
            document_id=DOC,
            chunk_id=CHUNK,
            entities=[make_entity()],
            relationships=[make_relationship()],
            extraction_status="completed",
        )
    with pytest.raises(ValidationError, match="provenance"):
        ChunkExtraction(
            document_id=DOC,
            chunk_id=CHUNK,
            entities=[make_entity(chunk_id="other")],
            extraction_status="completed",
        )


# --- Groq client -----------------------------------------------------------------------


def test_groq_client_requires_configuration():
    with pytest.raises(LLMNotConfiguredError):
        create_groq_client(Settings(_env_file=None, groq_api_key=None, groq_model_name=None))


def test_groq_client_uses_json_mode_and_rejects_empty_content():
    class Completions:
        def __init__(self, content):
            self.content, self.kwargs = content, None

        def create(self, **kwargs):
            self.kwargs = kwargs
            message = type("M", (), {"content": self.content})
            return type("C", (), {"choices": [type("Ch", (), {"message": message})]})

    completions = Completions('{"entities": [], "relationships": []}')
    fake = type("G", (), {"chat": type("Chat", (), {"completions": completions})})
    client = GroqJSONClient("key", "model-x", 5, client=fake)
    assert client.complete_json("sys", "user").startswith("{")
    assert completions.kwargs["response_format"] == {"type": "json_object"}
    assert completions.kwargs["temperature"] == 0 and completions.kwargs["model"] == "model-x"
    completions.content = ""
    with pytest.raises(LLMError, match="empty"):
        client.complete_json("sys", "user")


# --- Document-level service ------------------------------------------------------------


def test_single_chunk_extraction():
    collection = FakeCollection(records(3))
    response = extract_document(
        FakeLLM(answer([E_PERSON])), collection, DOC, chunk_id=f"{DOC}_chunk1", max_chunks=20
    )
    assert response.extraction_status == "completed"
    assert response.total_chunks == response.chunks_processed == 1
    assert response.results[0].chunk_id == f"{DOC}_chunk1"
    assert collection.writes == 0


def test_chunk_from_another_document_is_not_found():
    collection = FakeCollection(records(1) + records(1, document_id="OTHER-1"))
    with pytest.raises(ChunkNotFoundError, match="does not belong"):
        extract_document(
            FakeLLM(answer()), collection, DOC, chunk_id="OTHER-1_chunk0", max_chunks=20
        )
    with pytest.raises(ChunkNotFoundError):
        extract_document(FakeLLM(answer()), collection, "MISSING", max_chunks=20)


def test_document_extraction_orders_chunks_and_reports_partial_failure():
    rows = [(f"{DOC}_chunk{i}", TEXT, {"case_id": DOC}) for i in (10, 2, 0)]
    llm = FakeLLM(answer([E_PERSON]), "not json", answer([E_COURT]))
    response = extract_document(llm, FakeCollection(rows), DOC, max_chunks=20)
    assert [r.chunk_id for r in response.results] == [
        f"{DOC}_chunk0",
        f"{DOC}_chunk2",
        f"{DOC}_chunk10",
    ]
    assert response.extraction_status == "partial"
    assert (response.chunks_succeeded, response.chunks_failed) == (2, 1)
    assert response.results[1].extraction_status == "failed" and response.results[1].error
    assert response.total_entities == 2
    # Each chunk keeps its own provenance even for the same entity name.
    ids = {r.entities[0].source_chunk_id for r in response.results if r.entities}
    assert ids == {f"{DOC}_chunk0", f"{DOC}_chunk10"}


def test_document_extraction_all_failed_and_max_chunks():
    collection = FakeCollection(records(5))
    failed = extract_document(FakeLLM(LLMError("down")), collection, DOC, max_chunks=20)
    assert failed.extraction_status == "failed" and failed.chunks_failed == 5
    limited = extract_document(FakeLLM(answer()), collection, DOC, max_chunks=2)
    assert limited.extraction_status == "partial"
    assert (limited.total_chunks, limited.chunks_processed, limited.chunks_skipped) == (5, 2, 3)


# --- API -------------------------------------------------------------------------------


@pytest.fixture
def client():
    def use(llm, collection):
        app.dependency_overrides[get_extraction_llm] = lambda: llm
        app.dependency_overrides[get_chroma_collection] = lambda: collection
        return TestClient(app)

    yield use
    app.dependency_overrides.clear()


def test_api_single_chunk(client):
    api = client(
        FakeLLM(answer([E_SECTION, E_STATUTE], [REL_SECTION_OF_ACT])), FakeCollection(records(2))
    )
    body = api.post("/v1/extraction/process", json={"document_id": DOC, "chunk_id": CHUNK}).json()
    assert body["extraction_status"] == "completed"
    assert (body["total_entities"], body["total_relationships"]) == (2, 1)
    assert body["results"][0]["entities"][0]["source_chunk_id"] == CHUNK


def test_api_document_partial_failure(client):
    api = client(FakeLLM(answer(), LLMError("timeout")), FakeCollection(records(2)))
    response = api.post("/v1/extraction/process", json={"document_id": DOC})
    body = response.json()
    assert response.status_code == 200
    assert body["extraction_status"] == "partial"
    assert [r["extraction_status"] for r in body["results"]] == ["completed", "failed"]


def test_api_errors(client):
    api = client(FakeLLM(answer()), FakeCollection(records(1)))
    assert api.post("/v1/extraction/process", json={"document_id": "NOPE"}).status_code == 404
    assert api.post("/v1/extraction/process", json={"document_id": "  "}).status_code == 422
    assert api.post("/v1/extraction/process", json={}).status_code == 422
    too_many = {"document_id": DOC, "max_chunks": 1000}
    assert api.post("/v1/extraction/process", json=too_many).status_code == 422


def test_api_groq_not_configured_returns_503(client):
    api = client(FakeLLM(answer()), FakeCollection(records(1)))
    del app.dependency_overrides[get_extraction_llm]
    from graph_rag.core.config import get_settings

    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, groq_api_key=None, groq_model_name=None
    )
    response = api.post("/v1/extraction/process", json={"document_id": DOC})
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]
