"""Story 1.3.2 tests: the read-only ChromaDB connection layer.

Unit tests build tiny throwaway ChromaDB databases in ``tmp_path``. The integration tests
run against a temporary COPY of ``data/chroma/`` so the project database is never opened
by the test suite; they are skipped when that directory is absent.
"""

import shutil
from pathlib import Path

import chromadb
import pytest

from graph_rag.vector_store import chroma_store
from graph_rag.vector_store.chroma_store import (
    CollectionNotFoundError,
    VectorStoreError,
    check_compatibility,
    open_collection,
    search_similar_chunks,
    summarize_collection,
)

DIMENSION = 384
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROJECT_DB = Path(__file__).resolve().parents[1] / "data" / "chroma"


def unit_vector(position: int) -> list[float]:
    vector = [0.0] * DIMENSION
    vector[position] = 1.0
    return vector


@pytest.fixture
def small_db(tmp_path: Path) -> Path:
    """A 3-record cosine collection shaped like ``judgments`` (made with our own vectors)."""
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection(
        "judgments", embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
    )
    collection.add(
        ids=[f"CASE-1_chunk{i}" for i in range(3)],
        documents=[f"chunk text {i}" for i in range(3)],
        metadatas=[
            {"case_id": "CASE-1", "court": "HC", "date": "2020", "section": "s", "title": "t"}
        ]
        * 3,
        embeddings=[unit_vector(i) for i in range(3)],
    )
    return tmp_path


def test_opens_existing_collection_and_reads_contents(small_db: Path) -> None:
    collection = open_collection(small_db, "judgments")
    summary = summarize_collection(collection)

    assert summary.name == "judgments"
    assert summary.count == 3
    assert summary.space == "cosine"
    assert summary.has_text
    assert summary.metadata_fields == ["case_id", "court", "date", "section", "title"]
    assert summary.embedding_dimensions == [DIMENSION, DIMENSION]


def test_client_is_created_once_per_directory(small_db: Path) -> None:
    open_collection(small_db, "judgments")
    open_collection(small_db, "judgments")
    assert chroma_store._get_client.cache_info().currsize >= 1
    assert chroma_store._get_client(str(small_db.resolve())) is chroma_store._get_client(
        str(small_db.resolve())
    )


def test_missing_collection_is_reported_not_created(small_db: Path) -> None:
    with pytest.raises(CollectionNotFoundError, match="CHROMA_COLLECTION_NAME"):
        open_collection(small_db, "does_not_exist")
    client = chroma_store._get_client(str(small_db.resolve()))
    assert [c.name for c in client.list_collections()] == ["judgments"]


def test_missing_database_directory_is_reported_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "nothing"
    with pytest.raises(VectorStoreError, match="CHROMA_PERSIST_DIRECTORY"):
        open_collection(missing, "judgments")
    assert not missing.exists()


def test_search_returns_nearest_first_with_cosine_distance(small_db: Path) -> None:
    collection = open_collection(small_db, "judgments")
    hits = search_similar_chunks(collection, unit_vector(1), top_k=2)

    assert hits[0]["id"] == "CASE-1_chunk1"
    assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-6)  # identical direction
    assert hits[1]["distance"] == pytest.approx(1.0, abs=1e-6)  # orthogonal: cos = 0
    assert hits[0]["document"] == "chunk text 1"
    assert hits[0]["metadata"]["case_id"] == "CASE-1"
    assert "embedding" not in hits[0]


def test_compatibility_rejects_non_cosine_space_and_other_models(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    l2 = client.create_collection("l2_space", embedding_function=None)  # Chroma default: l2
    with pytest.raises(VectorStoreError, match="'l2' distance"):
        check_compatibility(l2, MODEL)

    class Other:
        name = "judgments"
        configuration_json = {
            "hnsw": {"space": "cosine"},
            "embedding_function": {"config": {"model_name": "all-mpnet-base-v2"}},
        }

    with pytest.raises(VectorStoreError, match="all-mpnet-base-v2"):
        check_compatibility(Other(), MODEL)


# --- Integration: temporary copy of the real copied database ----------------------


@pytest.fixture(scope="module")
def judgments_copy(tmp_path_factory: pytest.TempPathFactory):
    if not (PROJECT_DB / "chroma.sqlite3").is_file():
        pytest.skip("data/chroma/ is not present")
    copy = tmp_path_factory.mktemp("chroma_copy") / "chroma"
    shutil.copytree(PROJECT_DB, copy)
    collection = open_collection(copy, "judgments")
    check_compatibility(collection, MODEL)
    return collection


def test_real_collection_contents(judgments_copy) -> None:
    summary = summarize_collection(judgments_copy)

    assert summary.count == 7325
    assert summary.space == "cosine"
    assert summary.has_text
    assert {"case_id", "court", "date", "section", "title"} <= set(summary.metadata_fields)
    assert summary.embedding_dimensions == [DIMENSION, DIMENSION]


def test_real_collection_similarity_search(judgments_copy) -> None:
    stored = judgments_copy.get(limit=1, include=["embeddings"])
    record_id, vector = stored["ids"][0], list(map(float, stored["embeddings"][0]))

    hits = search_similar_chunks(judgments_copy, vector, top_k=3)

    assert len(hits) == 3
    assert hits[0]["id"] == record_id  # a stored vector is its own nearest neighbour
    assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-5)
    assert all(hit["document"] and hit["metadata"].get("case_id") for hit in hits)
