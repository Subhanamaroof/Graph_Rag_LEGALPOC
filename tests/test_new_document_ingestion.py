"""Story 1.3.2.2 tests: ingesting new documents into the ChromaDB ``new_documents`` collection.

Every test uses a throwaway ChromaDB database in ``tmp_path`` (or a temporary COPY of
``data/chroma/``), so the project database is never written. The Sentence Transformers model
is replaced by a small deterministic fake, so no model is downloaded.
"""

import hashlib
import math
import shutil
from pathlib import Path

import chromadb
import pymupdf
import pytest
from fastapi.testclient import TestClient

from graph_rag.core.config import Settings, get_settings
from graph_rag.embeddings import embedder as embedder_module
from graph_rag.embeddings.embedder import Embedder
from graph_rag.ingestion import ingestion_service
from graph_rag.ingestion.ingestion_service import ingest_documents
from graph_rag.ingestion.router import get_ingestion_collection
from graph_rag.main import app
from graph_rag.retrieval.router import get_loaded_embedder
from graph_rag.vector_store.chroma_store import (
    VectorStoreError,
    get_new_documents_collection,
    open_collection,
    open_writable_collection,
    search_similar_chunks,
)

DIMENSION = 384
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROJECT_DB = Path(__file__).resolve().parents[1] / "data" / "chroma"


class FakeModel:
    """Stands in for SentenceTransformer: a deterministic unit vector per text."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.loads = 0

    def encode(self, texts: list[str], **_: object) -> list[list[float]]:
        if self.fail_on and any(self.fail_on in text for text in texts):
            raise RuntimeError("model crashed")
        vectors = []
        for text in texts:
            seed = hashlib.sha256(text.encode("utf-8")).digest()
            raw = [seed[i % len(seed)] / 255 + i / DIMENSION for i in range(DIMENSION)]
            norm = math.sqrt(sum(value * value for value in raw))
            vectors.append([value / norm for value in raw])
        return vectors


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> FakeModel:
    model = FakeModel()

    def load(_name: str) -> FakeModel:
        model.loads += 1
        return model

    monkeypatch.setattr(embedder_module, "_load_sentence_transformer", load)
    return model


@pytest.fixture
def embedder(fake_model: FakeModel) -> Embedder:
    return Embedder(MODEL, DIMENSION)


def unit_vector(position: int) -> list[float]:
    vector = [0.0] * DIMENSION
    vector[position] = 1.0
    return vector


@pytest.fixture
def db(tmp_path: Path) -> Path:
    """A temp database that already holds a small ``judgments`` collection."""
    path = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(path))
    judgments = client.create_collection(
        "judgments", embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
    )
    judgments.add(
        ids=[f"CASE-1_chunk{i}" for i in range(3)],
        documents=[f"judgment text {i}" for i in range(3)],
        metadatas=[{"case_id": "CASE-1", "court": "HC"}] * 3,
        embeddings=[unit_vector(i) for i in range(3)],
    )
    return path


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    path = tmp_path / "documents"
    path.mkdir()
    return path


def new_documents(db: Path):
    return open_writable_collection(db, "new_documents", MODEL)


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def make_pdf(path: Path, page_texts: list[str]) -> Path:
    pdf = pymupdf.open()
    for text in page_texts:
        pdf.new_page().insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()
    return path


def snapshot(collection) -> dict:
    return collection.get(include=["documents", "metadatas", "embeddings"])


def long_text(paragraphs: int) -> str:
    return "\n\n".join(
        f"Paragraph {i}. " + "Graph RAG links chunks. " * 25 for i in range(paragraphs)
    )


# --- Pipeline -----------------------------------------------------------------


def test_story_1_2_loader_and_chunker_are_reused(
    monkeypatch: pytest.MonkeyPatch, docs: Path, db: Path, embedder: Embedder
) -> None:
    calls: list[str] = []
    real_load, real_chunk = ingestion_service.load_document, ingestion_service.chunk_document
    monkeypatch.setattr(
        ingestion_service, "load_document", lambda *a: calls.append("load") or real_load(*a)
    )
    monkeypatch.setattr(
        ingestion_service, "chunk_document", lambda *a: calls.append("chunk") or real_chunk(*a)
    )
    write(docs / "a.txt", "Alpha document.")

    ingest_documents(docs, 800, 150, embedder=embedder, collection=new_documents(db))

    assert calls == ["load", "chunk"]


def test_multiple_documents_are_embedded_and_stored_with_metadata(
    docs: Path, db: Path, embedder: Embedder, fake_model: FakeModel
) -> None:
    write(docs / "a.txt", long_text(4))
    write(docs / "b.md", "# Title\n\nMarkdown body.")
    make_pdf(docs / "c.pdf", ["First page text", "Second page text"])
    collection = new_documents(db)

    result = ingest_documents(docs, 800, 150, embedder=embedder, collection=collection)

    assert result.status == "completed"
    assert result.documents_processed == 3
    assert result.total_chunks == result.total_embeddings == result.records_upserted
    assert result.total_chunks > 3  # the long text file produced several chunks
    assert result.collection == "new_documents"
    assert fake_model.loads == 1  # model loaded once, not once per chunk

    stored = collection.get(include=["documents", "metadatas", "embeddings"])
    assert collection.count() == result.total_chunks
    assert sorted(stored["ids"]) == sorted(chunk.chunk_id for chunk in result.chunks)
    assert all(len(vector) == DIMENSION for vector in stored["embeddings"])

    by_id = dict(zip(stored["ids"], stored["metadatas"], strict=True))
    texts = dict(zip(stored["ids"], stored["documents"], strict=True))
    for chunk in result.chunks:
        metadata = by_id[chunk.chunk_id]
        assert texts[chunk.chunk_id] == chunk.text
        assert metadata["document_id"] == chunk.document_id
        assert metadata["source_file"] == chunk.source_file
        assert metadata["file_type"] == chunk.file_type
        assert metadata["chunk_index"] == chunk.chunk_index
        assert "ingested_at" in metadata
    pdf_chunk = next(chunk for chunk in result.chunks if chunk.file_type == "pdf")
    assert by_id[pdf_chunk.chunk_id]["page_numbers"] == [1, 2]

    # Each vector belongs to its own chunk text.
    first = stored["ids"].index(result.chunks[0].chunk_id)
    expected = fake_model.encode([result.chunks[0].text])[0]
    assert list(stored["embeddings"][first]) == pytest.approx(expected, abs=1e-6)


def test_reingestion_does_not_create_duplicates(docs: Path, db: Path, embedder: Embedder) -> None:
    write(docs / "a.txt", long_text(3))
    collection = new_documents(db)

    first = ingest_documents(docs, 800, 150, embedder=embedder, collection=collection)
    second = ingest_documents(docs, 800, 150, embedder=embedder, collection=collection)

    assert collection.count() == first.total_chunks == second.records_upserted
    assert second.obsolete_chunks_deleted == 0


def test_shorter_document_removes_only_its_own_obsolete_chunks(
    docs: Path, db: Path, embedder: Embedder
) -> None:
    write(docs / "a.txt", long_text(6))
    write(docs / "b.txt", long_text(3))
    collection = new_documents(db)
    first = ingest_documents(docs, 800, 150, embedder=embedder, collection=collection)
    b_ids = {chunk.chunk_id for chunk in first.chunks if chunk.source_file == "b.txt"}

    write(docs / "a.txt", "Now a short document.")
    second = ingest_documents(docs, 800, 150, embedder=embedder, collection=collection)

    a_status = next(document for document in second.documents if document.source_file == "a.txt")
    a_chunks_before = first.documents[0].chunks
    assert a_status.chunks == 1
    assert a_status.obsolete_chunks_deleted == a_chunks_before - 1
    assert collection.count() == 1 + len(b_ids)
    assert b_ids <= set(collection.get(include=[])["ids"])


# --- Failures -----------------------------------------------------------------


def test_empty_input_directory(docs: Path, db: Path, embedder: Embedder) -> None:
    collection = new_documents(db)

    result = ingest_documents(docs, 800, 150, embedder=embedder, collection=collection)

    assert (result.status, result.documents_discovered, result.records_upserted) == (
        "completed",
        0,
        0,
    )
    assert collection.count() == 0


def test_invalid_document_does_not_stop_others(docs: Path, db: Path, embedder: Embedder) -> None:
    write(docs / "good.txt", "Valid content.")
    (docs / "broken.pdf").write_bytes(b"not really a pdf")
    write(docs / "ignored.docx", "unsupported formats are not discovered")

    result = ingest_documents(docs, 800, 150, embedder=embedder, collection=new_documents(db))

    assert result.status == "completed_with_errors"
    assert result.documents_discovered == 2
    assert [d.status for d in result.documents] == ["failed", "succeeded"]
    assert result.records_upserted == 1


def test_embedding_failure_is_reported_and_not_stored(
    docs: Path, db: Path, embedder: Embedder, fake_model: FakeModel
) -> None:
    fake_model.fail_on = "explode"
    write(docs / "a.txt", "Please explode here.")
    write(docs / "b.txt", "Fine document.")
    collection = new_documents(db)

    result = ingest_documents(docs, 800, 150, embedder=embedder, collection=collection)

    failed = result.documents[0]
    assert failed.status == "failed" and failed.error.startswith("Embedding failed")
    assert failed.records_upserted == 0
    assert result.status == "completed_with_errors"
    assert collection.count() == 1


class FailingCollection:
    """A collection whose writes fail, to check that failures are never reported as stored."""

    name = "new_documents"

    def upsert(self, **_: object) -> None:
        raise RuntimeError("disk full")


def test_storage_failure_is_never_reported_as_success(docs: Path, embedder: Embedder) -> None:
    write(docs / "a.txt", "Some content.")

    result = ingest_documents(docs, 800, 150, embedder=embedder, collection=FailingCollection())

    document = result.documents[0]
    assert result.status == "failed"
    assert document.status == "failed"
    assert document.error.startswith("Storage failed")
    assert document.records_upserted == 0 and result.records_upserted == 0


def test_missing_database_is_reported_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(VectorStoreError, match="not found"):
        open_writable_collection(missing, "new_documents", MODEL)
    assert not missing.exists()


# --- Collection safety ----------------------------------------------------------


def test_judgments_collection_is_unchanged(docs: Path, db: Path, embedder: Embedder) -> None:
    judgments = open_collection(db, "judgments")
    before = snapshot(judgments)
    write(docs / "a.txt", long_text(3))

    ingest_documents(docs, 800, 150, embedder=embedder, collection=new_documents(db))
    ingest_documents(docs, 800, 150, embedder=embedder, collection=new_documents(db))

    after = snapshot(judgments)
    assert after["ids"] == before["ids"]
    assert after["documents"] == before["documents"]
    assert after["metadatas"] == before["metadatas"]
    assert [list(v) for v in after["embeddings"]] == [list(v) for v in before["embeddings"]]


def test_read_only_collection_cannot_be_used_for_ingestion(db: Path) -> None:
    with pytest.raises(VectorStoreError, match="read-only"):
        open_writable_collection(db, "judgments", MODEL)
    settings = Settings(
        chroma_persist_directory=db,
        chroma_collection_name="legal",
        chroma_new_documents_collection="legal",
    )
    with pytest.raises(VectorStoreError, match="read-only"):
        get_new_documents_collection(settings)


def test_stored_vectors_are_searchable_with_the_query_embedder(
    docs: Path, db: Path, embedder: Embedder
) -> None:
    write(docs / "a.txt", "Neo4j stores the knowledge graph.")
    write(docs / "b.txt", "ChromaDB stores the chunk vectors.")
    collection = new_documents(db)
    ingest_documents(docs, 800, 150, embedder=embedder, collection=collection)

    query_vector = embedder.embed_text("ChromaDB stores the chunk vectors.")
    hits = search_similar_chunks(collection, query_vector, top_k=2)

    assert hits[0]["metadata"]["source_file"] == "b.txt"
    assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-5)


def test_real_judgments_copy_is_unchanged(tmp_path: Path, docs: Path, embedder: Embedder) -> None:
    if not (PROJECT_DB / "chroma.sqlite3").is_file():
        pytest.skip("data/chroma/ is not present")
    copy = tmp_path / "chroma_copy"
    shutil.copytree(PROJECT_DB, copy)
    judgments = open_collection(copy, "judgments")
    count_before = judgments.count()
    sample_before = judgments.get(limit=5, include=["documents", "metadatas"])
    write(docs / "a.txt", long_text(2))

    result = ingest_documents(docs, 800, 150, embedder=embedder, collection=new_documents(copy))

    assert result.records_upserted == result.total_chunks > 0
    assert judgments.count() == count_before == 7325
    assert judgments.get(limit=5, include=["documents", "metadatas"]) == sample_before


# --- API ------------------------------------------------------------------------


@pytest.fixture
def api(docs: Path, db: Path):
    settings = Settings(documents_directory=docs, chroma_persist_directory=db)
    app.dependency_overrides[get_settings] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_api_ingests_and_reports_statistics(api, docs: Path, db: Path, embedder: Embedder) -> None:
    app.dependency_overrides[get_loaded_embedder] = lambda: embedder
    write(docs / "a.txt", long_text(2))
    write(docs / "b.md", "# Notes\n\nShort.")

    summary = api.post("/v1/ingestion/process").json()
    detailed = api.post("/v1/ingestion/process?include_chunks=true").json()

    assert summary["status"] == "completed"
    assert summary["collection"] == "new_documents"
    assert summary["documents_discovered"] == summary["documents_processed"] == 2
    assert summary["total_chunks"] == summary["total_embeddings"] == summary["records_upserted"]
    assert {"chunks", "embeddings", "records_upserted"} <= summary["documents"][0].keys()
    assert "chunks" not in summary
    assert "embedding" not in detailed["chunks"][0]
    assert "embedding" not in str(detailed["chunks"][0].keys())
    assert new_documents(db).count() == summary["total_chunks"]


def test_api_returns_503_when_database_is_missing(tmp_path: Path, embedder: Embedder) -> None:
    settings = Settings(documents_directory=tmp_path, chroma_persist_directory=tmp_path / "missing")
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_loaded_embedder] = lambda: embedder
    try:
        response = TestClient(app).post("/v1/ingestion/process")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "not found" in response.json()["detail"]


def test_api_returns_503_when_model_cannot_load(
    monkeypatch: pytest.MonkeyPatch, docs: Path, db: Path
) -> None:
    def broken(_name: str) -> None:
        raise OSError("no network")

    monkeypatch.setattr(embedder_module, "_load_sentence_transformer", broken)
    # A model name no other test uses, so the cached shared embedder is not reused.
    settings = Settings(
        documents_directory=docs,
        chroma_persist_directory=db,
        embedding_model_name="unavailable/test-model",
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_ingestion_collection] = lambda: None
    write(docs / "a.txt", "Content.")
    try:
        response = TestClient(app).post("/v1/ingestion/process")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "Could not load embedding model" in response.json()["detail"]
