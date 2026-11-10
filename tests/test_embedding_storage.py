"""Story 1.3 tests: embedding generation.

The MongoDB storage and ingestion-pipeline tests were removed in Story 1.3.2, when MongoDB
was dropped. No real model download: the Sentence Transformers model is a small fake.
"""

import pytest

from graph_rag.embeddings import embedder as embedder_module
from graph_rag.embeddings.embedder import Embedder, EmbeddingError, EmbeddingModelError

DIMENSION = 384


class FakeModel:
    """Stands in for SentenceTransformer: deterministic 384-dim vectors."""

    def __init__(self, fail_on: str | None = None, dimension: int = DIMENSION) -> None:
        self.fail_on = fail_on
        self.dimension = dimension
        self.calls = 0

    def encode(self, texts: list[str], **_: object) -> list[list[float]]:
        self.calls += 1
        if self.fail_on and any(self.fail_on in text for text in texts):
            raise RuntimeError("model crashed")
        return [[float(len(text) % 7 + i) / 1000 for i in range(self.dimension)] for text in texts]


def make_embedder(monkeypatch: pytest.MonkeyPatch, model: FakeModel | None = None) -> Embedder:
    model = model or FakeModel()
    monkeypatch.setattr(embedder_module, "_load_sentence_transformer", lambda name: model)
    return Embedder("fake-model", DIMENSION)


# --- Embedding generation ---------------------------------------------------


def test_single_chunk_embedding_has_384_floats(monkeypatch: pytest.MonkeyPatch) -> None:
    vector = make_embedder(monkeypatch).embed_text("Graph RAG combines vectors and graphs.")

    assert len(vector) == DIMENSION
    assert all(isinstance(value, float) for value in vector)


def test_multiple_chunks_load_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[str] = []
    model = FakeModel()

    def fake_loader(name: str) -> FakeModel:
        loads.append(name)
        return model

    monkeypatch.setattr(embedder_module, "_load_sentence_transformer", fake_loader)
    embedder = Embedder("fake-model", DIMENSION)

    first = embedder.embed_texts(["one", "two", "three"])
    second = embedder.embed_texts(["four"])

    assert len(first) == 3 and len(second) == 1
    assert loads == ["fake-model"]


def test_wrong_dimension_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = make_embedder(monkeypatch, FakeModel(dimension=10))

    with pytest.raises(EmbeddingError, match="384-dimensional"):
        embedder.embed_texts(["text"])


def test_empty_chunks_are_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = make_embedder(monkeypatch)

    assert embedder.embed_texts([]) == []
    with pytest.raises(EmbeddingError, match="empty text"):
        embedder.embed_texts(["valid", "   "])


def test_embedding_failures_raise_embedding_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(EmbeddingError, match="generation failed"):
        make_embedder(monkeypatch, FakeModel(fail_on="boom")).embed_texts(["boom"])

    def broken_loader(name: str) -> None:
        raise OSError("model not found")

    monkeypatch.setattr(embedder_module, "_load_sentence_transformer", broken_loader)
    with pytest.raises(EmbeddingModelError, match="Could not load embedding model"):
        Embedder("missing-model", DIMENSION).load()
