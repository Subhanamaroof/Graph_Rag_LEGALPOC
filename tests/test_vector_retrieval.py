"""Story 1.4 tests (migrated to ChromaDB in Story 1.3.2): query embedding, similarity
search, result mapping and the retrieval API.

No model download and no real database: the Sentence Transformers model and the ChromaDB
collection are replaced with small fakes that record what they were asked to do.
"""

import pytest
from chromadb.errors import ChromaError
from fastapi.testclient import TestClient

from graph_rag.core.config import Settings, get_settings
from graph_rag.embeddings import embedder as embedder_module
from graph_rag.embeddings.embedder import Embedder, EmbeddingError
from graph_rag.main import app
from graph_rag.retrieval.models import MAX_TOP_K
from graph_rag.retrieval.router import get_chroma_collection, get_loaded_embedder
from graph_rag.retrieval.vector_retriever import retrieve_similar_chunks
from graph_rag.vector_store.chroma_store import VectorSearchError

DIMENSION = 384

METADATA = {
    "case_id": "NI138-2019-0042",
    "court": "High Court",
    "date": "2019-05-01",
    "section": "Judgment",
    "title": "State v. Example",
}
RECORD = ("NI138-2019-0042_chunk0", "The accused was granted bail.", METADATA, 0.25)


class FakeModel:
    def __init__(self, fail: bool = False, dimension: int = DIMENSION) -> None:
        self.fail = fail
        self.dimension = dimension
        self.encoded: list[list[str]] = []

    def encode(self, texts: list[str], **_: object) -> list[list[float]]:
        self.encoded.append(texts)
        if self.fail:
            raise RuntimeError("model crashed")
        return [[0.01 * (i % 10) for i in range(self.dimension)] for _ in texts]


class FakeCollection:
    """Records ``query`` calls and answers in ChromaDB's nested result format."""

    name = "judgments"

    def __init__(self, records: list[tuple] | None = None, error: Exception | None = None) -> None:
        self.records = records if records is not None else [RECORD]
        self.error = error
        self.queries: list[dict] = []

    def query(self, **kwargs: object) -> dict:
        self.queries.append(kwargs)
        if self.error:
            raise self.error
        hits = self.records[: kwargs["n_results"]]
        return {
            "ids": [[h[0] for h in hits]],
            "documents": [[h[1] for h in hits]],
            "metadatas": [[h[2] for h in hits]],
            "distances": [[h[3] for h in hits]],
        }


def make_embedder(monkeypatch: pytest.MonkeyPatch, model: FakeModel | None = None) -> Embedder:
    model = model or FakeModel()
    monkeypatch.setattr(embedder_module, "_load_sentence_transformer", lambda name: model)
    return Embedder("fake-model", DIMENSION)


def retrieve(embedder: Embedder, collection: FakeCollection, query: str = "bail?", top_k=5):
    return retrieve_similar_chunks(query, top_k, embedder=embedder, collection=collection)


# --- Query embedding ----------------------------------------------------------


def test_query_is_embedded_once_as_384_dim_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel()
    collection = FakeCollection()
    retrieve(make_embedder(monkeypatch, model), collection, query="  When is   bail granted?  ")

    assert model.encoded == [["When is bail granted?"]]  # same clean_text as chunks
    (query_vector,) = collection.queries[0]["query_embeddings"]
    assert len(query_vector) == DIMENSION
    assert all(isinstance(value, float) for value in query_vector)


def test_wrong_query_vector_dimension_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    collection = FakeCollection()
    with pytest.raises(EmbeddingError, match="384"):
        retrieve(make_embedder(monkeypatch, FakeModel(dimension=128)), collection)
    assert collection.queries == []  # nothing is searched with a bad vector


# --- ChromaDB query ------------------------------------------------------------


def test_query_requests_text_metadata_distances_but_never_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = FakeCollection()
    retrieve(make_embedder(monkeypatch), collection, top_k=7)

    call = collection.queries[0]
    assert call["n_results"] == 7
    assert set(call["include"]) == {"documents", "metadatas", "distances"}


@pytest.mark.parametrize("top_k", [0, MAX_TOP_K + 1])
def test_out_of_range_top_k_is_rejected(monkeypatch: pytest.MonkeyPatch, top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        retrieve(make_embedder(monkeypatch), FakeCollection(), top_k=top_k)


# --- Result mapping and edge cases ----------------------------------------------


def test_hits_map_case_id_and_convert_cosine_distance(monkeypatch: pytest.MonkeyPatch) -> None:
    second = ("NI138-2019-0042_chunk1", "Second chunk.", METADATA, 0.4)
    response = retrieve(make_embedder(monkeypatch), FakeCollection([RECORD, second]))

    assert response.total_results == 2
    first = response.results[0]
    assert [r.chunk_id for r in response.results] == [RECORD[0], second[0]]  # Chroma order
    assert first.document_id == "NI138-2019-0042"  # explicit mapping from case_id
    assert first.text == RECORD[1]
    assert first.metadata == METADATA  # all stored fields preserved
    assert first.distance == 0.25
    assert first.score == pytest.approx(0.75)  # cosine similarity = 1 - cosine distance


def test_missing_case_id_is_not_invented(monkeypatch: pytest.MonkeyPatch) -> None:
    record = ("x_chunk0", "text", {"court": "High Court"}, 0.1)
    response = retrieve(make_embedder(monkeypatch), FakeCollection([record]))
    assert response.results[0].document_id is None


def test_no_matches_returns_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    response = retrieve(make_embedder(monkeypatch), FakeCollection(records=[]))
    assert response.total_results == 0
    assert response.results == []


def test_blank_query_is_rejected_before_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel()
    with pytest.raises(ValueError, match="query"):
        retrieve(make_embedder(monkeypatch, model), FakeCollection(), query=" \n\t ")
    assert model.encoded == []


def test_embedding_failure_raises_embedding_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(EmbeddingError, match="failed"):
        retrieve(make_embedder(monkeypatch, FakeModel(fail=True)), FakeCollection())


def test_chroma_query_failure_raises_search_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(VectorSearchError):
        retrieve(make_embedder(monkeypatch), FakeCollection(error=ChromaError("boom")))


# --- API --------------------------------------------------------------------------


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch):
    """TestClient with the fake model and a configurable fake collection."""
    state = {"collection": FakeCollection(), "model": FakeModel()}
    app.dependency_overrides[get_chroma_collection] = lambda: state["collection"]
    app.dependency_overrides[get_loaded_embedder] = lambda: make_embedder(
        monkeypatch, state["model"]
    )
    try:
        yield TestClient(app), state
    finally:
        app.dependency_overrides.clear()


def test_api_success_response(api) -> None:
    client, _ = api
    response = client.post("/v1/retrieval/vector", json={"query": "bail", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "bail"
    assert body["total_results"] == 1
    assert body["results"][0] == {
        "document_id": "NI138-2019-0042",
        "chunk_id": RECORD[0],
        "text": RECORD[1],
        "metadata": METADATA,
        "score": 0.75,
        "distance": 0.25,
    }
    assert "embedding" not in response.text


def test_api_default_top_k_is_5(api) -> None:
    client, state = api
    client.post("/v1/retrieval/vector", json={"query": "bail"})
    assert state["collection"].queries[0]["n_results"] == 5


def test_api_empty_result(api) -> None:
    client, state = api
    state["collection"] = FakeCollection(records=[])
    response = client.post("/v1/retrieval/vector", json={"query": "bail"})
    assert response.status_code == 200
    assert response.json() == {"query": "bail", "total_results": 0, "results": []}


@pytest.mark.parametrize(
    "payload",
    [{}, {"query": ""}, {"query": "   "}, {"query": "x", "top_k": 0}, {"query": "x", "top_k": 21}],
)
def test_api_rejects_invalid_input(api, payload: dict) -> None:
    client, _ = api
    assert client.post("/v1/retrieval/vector", json=payload).status_code == 422


@pytest.mark.parametrize(
    ("collection", "model", "code"),
    [
        (FakeCollection(error=ChromaError("x")), FakeModel(), 502),
        (FakeCollection(), FakeModel(fail=True), 500),
    ],
    ids=["query-failed", "embedding-failed"],
)
def test_api_error_status_codes(api, collection, model, code: int) -> None:
    client, state = api
    state["collection"], state["model"] = collection, model
    response = client.post("/v1/retrieval/vector", json={"query": "bail"})

    assert response.status_code == code
    assert response.json()["detail"]


def test_api_returns_503_when_database_is_missing(tmp_path) -> None:
    missing = tmp_path / "no-chroma-here"
    app.dependency_overrides[get_settings] = lambda: Settings(chroma_persist_directory=missing)
    try:
        response = TestClient(app).post("/v1/retrieval/vector", json={"query": "bail"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "CHROMA_PERSIST_DIRECTORY" in response.json()["detail"]
    assert not missing.exists()  # no empty database was created
