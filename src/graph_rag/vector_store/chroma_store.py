"""Access to the local ChromaDB database in ``data/chroma/``.

Two collections live in the same database:

- ``judgments`` (Story 1.3.2): legal judgment chunks built elsewhere and copied into this
  project. It is only OPENED and READ: never created, reset, deleted, recreated or written.
- ``new_documents`` (Story 1.3.2.2): chunks of documents ingested by this project. It is the
  only collection this module writes to, and only one document's records at a time.

Developer check (prints counts and dimensions, never chunk text or vectors):

    uv run python -m graph_rag.vector_store.chroma_store                # judgments
    uv run python -m graph_rag.vector_store.chroma_store new_documents  # ingested documents
"""

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import ChromaError, NotFoundError

from graph_rag.core.config import Settings, get_settings

# The only distance metric this project knows how to turn into a similarity score.
SUPPORTED_SPACE = "cosine"

# Collections that ingestion must never write to, whatever the configuration says.
PROTECTED_COLLECTIONS = frozenset({"judgments"})

# Errors ChromaDB can raise for a failed read or write.
_CHROMA_ERRORS = (ChromaError, ValueError, RuntimeError, OSError)


class VectorStoreError(Exception):
    """Raised when ChromaDB cannot be opened or queried."""


class CollectionNotFoundError(VectorStoreError):
    """The configured collection does not exist. It is never created as a replacement."""


class VectorSearchError(VectorStoreError):
    """A similarity query against the collection failed."""


@lru_cache
def _get_client(persist_directory: str) -> Any:
    """One shared persistent client per directory for the whole process."""
    return chromadb.PersistentClient(path=persist_directory)


def close_client() -> None:
    """Forget the cached client and collections (called on application shutdown)."""
    _cached_collection.cache_clear()
    _cached_writable_collection.cache_clear()
    _get_client.cache_clear()


def open_collection(persist_directory: Path, collection_name: str) -> Collection:
    """Open an EXISTING collection. Raises ``VectorStoreError`` instead of creating anything.

    The directory is checked first because ``PersistentClient`` would otherwise create a
    brand-new empty database at a wrong path. ``embedding_function=None`` stops Chroma
    from loading its own copy of the model: queries are embedded by our ``Embedder``.
    """
    if not (persist_directory / "chroma.sqlite3").is_file():
        raise VectorStoreError(
            f"ChromaDB database not found at '{persist_directory}'. "
            "Check CHROMA_PERSIST_DIRECTORY (see .env.example)."
        )
    try:
        client = _get_client(str(persist_directory.resolve()))
        return client.get_collection(collection_name, embedding_function=None)
    except NotFoundError as exc:
        raise CollectionNotFoundError(
            f"ChromaDB collection '{collection_name}' does not exist in '{persist_directory}'. "
            "Check CHROMA_COLLECTION_NAME. No replacement collection is created."
        ) from exc
    except (ChromaError, OSError, ValueError) as exc:
        raise VectorStoreError(f"Cannot open ChromaDB ({type(exc).__name__}: {exc}).") from exc


def collection_space(collection: Collection) -> str:
    """The distance metric the collection's HNSW index was built with (e.g. ``cosine``)."""
    hnsw = (collection.configuration_json or {}).get("hnsw") or {}
    return hnsw.get("space", "l2")  # Chroma's default when nothing was configured


def check_compatibility(collection: Collection, model_name: str) -> None:
    """Fail clearly if stored vectors and our queries would not live in the same space.

    Two checks: the collection uses cosine distance, and the model recorded with the
    collection is the same model we embed queries with.
    """
    space = collection_space(collection)
    if space != SUPPORTED_SPACE:
        raise VectorStoreError(
            f"Collection '{collection.name}' uses '{space}' distance; only "
            f"'{SUPPORTED_SPACE}' is supported."
        )
    stored_function = (collection.configuration_json or {}).get("embedding_function") or {}
    stored_model = (stored_function.get("config") or {}).get("model_name")
    # Chroma stores the short name ("all-MiniLM-L6-v2"); settings use the full Hub ID.
    if stored_model and stored_model.split("/")[-1] != model_name.split("/")[-1]:
        raise VectorStoreError(
            f"Collection '{collection.name}' was embedded with '{stored_model}', but "
            f"EMBEDDING_MODEL_NAME is '{model_name}'. Query vectors would not be comparable."
        )


@lru_cache
def _cached_collection(persist_directory: Path, collection_name: str, model_name: str):
    collection = open_collection(persist_directory, collection_name)
    check_compatibility(collection, model_name)
    return collection


def get_collection(settings: Settings) -> Collection:
    """Open (once) and return the configured, compatibility-checked collection."""
    return _cached_collection(
        settings.chroma_persist_directory,
        settings.chroma_collection_name,
        settings.embedding_model_name,
    )


def open_writable_collection(
    persist_directory: Path,
    collection_name: str,
    model_name: str,
    protected: frozenset[str] = PROTECTED_COLLECTIONS,
) -> Collection:
    """Open the ingestion collection, creating it (empty) inside the EXISTING database.

    Refuses protected collection names, so ingestion can never be pointed at ``judgments``.
    The database directory must already exist: no second ChromaDB directory is created.
    ``get_or_create_collection`` never deletes or overwrites existing records.
    """
    if collection_name in protected:
        raise VectorStoreError(
            f"Collection '{collection_name}' is read-only and cannot receive new documents. "
            "Check CHROMA_NEW_DOCUMENTS_COLLECTION."
        )
    if not (persist_directory / "chroma.sqlite3").is_file():
        raise VectorStoreError(
            f"ChromaDB database not found at '{persist_directory}'. "
            "Check CHROMA_PERSIST_DIRECTORY (see .env.example)."
        )
    try:
        client = _get_client(str(persist_directory.resolve()))
        collection = client.get_or_create_collection(
            collection_name,
            embedding_function=None,  # vectors always come from our own Embedder
            configuration={"hnsw": {"space": SUPPORTED_SPACE}},
            metadata={"embedding_model": model_name},
        )
    except _CHROMA_ERRORS as exc:
        raise VectorStoreError(f"Cannot open ChromaDB ({type(exc).__name__}: {exc}).") from exc

    space = collection_space(collection)
    stored_model = (collection.metadata or {}).get("embedding_model")
    if space != SUPPORTED_SPACE or (stored_model and stored_model != model_name):
        raise VectorStoreError(
            f"Collection '{collection_name}' uses '{space}' distance and model "
            f"'{stored_model}'; expected '{SUPPORTED_SPACE}' and '{model_name}'."
        )
    return collection


@lru_cache
def _cached_writable_collection(
    persist_directory: Path, collection_name: str, model_name: str, protected: frozenset[str]
) -> Collection:
    return open_writable_collection(persist_directory, collection_name, model_name, protected)


def get_new_documents_collection(settings: Settings) -> Collection:
    """Open (once) the collection for newly ingested documents.

    The configured read-only collection (``CHROMA_COLLECTION_NAME``) is protected too.
    """
    return _cached_writable_collection(
        settings.chroma_persist_directory,
        settings.chroma_new_documents_collection,
        settings.embedding_model_name,
        PROTECTED_COLLECTIONS | {settings.chroma_collection_name},
    )


def replace_document_chunks(
    collection: Collection,
    document_id: str,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
) -> tuple[int, int]:
    """Store one document's chunks and remove that document's chunks that no longer exist.

    1. ``upsert`` inserts new IDs and overwrites existing IDs, so re-ingesting a document
       never creates duplicates.
    2. Records whose metadata ``document_id`` matches but whose ID is not in ``ids`` are
       left over from a longer, older version of the document; they are deleted by ID.
       Records of other documents are never selected.

    Upserting first means a failed write leaves the previous version intact.
    Returns ``(records_upserted, obsolete_records_deleted)``.
    """
    if collection.name in PROTECTED_COLLECTIONS:
        raise VectorStoreError(f"Refusing to write to read-only collection '{collection.name}'.")
    if not ids:
        raise VectorStoreError(f"Document '{document_id}' has no chunks to store.")
    try:
        collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    except _CHROMA_ERRORS as exc:
        raise VectorStoreError(f"Storing chunks failed ({type(exc).__name__}: {exc}).") from exc
    try:
        stored_ids = collection.get(where={"document_id": document_id}, include=[])["ids"]
        obsolete_ids = sorted(set(stored_ids) - set(ids))
        if obsolete_ids:
            collection.delete(ids=obsolete_ids)
    except _CHROMA_ERRORS as exc:
        raise VectorStoreError(
            f"Chunks were stored, but removing obsolete chunks failed ({type(exc).__name__})."
        ) from exc
    return len(ids), len(obsolete_ids)


def search_similar_chunks(
    collection: Collection, query_vector: list[float], top_k: int
) -> list[dict[str, Any]]:
    """Return the ``top_k`` nearest records, closest first, as plain dicts.

    Chroma's HNSW index finds the nearest stored vectors and returns their *distance*
    (smaller = closer). Embedding arrays are not requested, so they are never returned.
    """
    try:
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except (ChromaError, ValueError, RuntimeError) as exc:
        raise VectorSearchError(f"ChromaDB query failed ({type(exc).__name__}).") from exc

    # Results are nested one level per query; we send exactly one query.
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0] or [None] * len(ids)
    metadatas = (result.get("metadatas") or [[]])[0] or [None] * len(ids)
    distances = (result.get("distances") or [[]])[0]
    return [
        {"id": record_id, "document": document, "metadata": metadata or {}, "distance": distance}
        for record_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        )
    ]


def get_chunks(
    collection: Collection,
    *,
    ids: list[str] | None = None,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Read records by ID or by metadata filter (Story 1.5). Read-only: never writes.

    Returns ``{"id", "document", "metadata"}`` dicts; embeddings are not requested.
    """
    try:
        result = collection.get(ids=ids, where=where, include=["documents", "metadatas"])
    except _CHROMA_ERRORS as exc:
        raise VectorStoreError(f"Reading chunks failed ({type(exc).__name__}).") from exc
    ids_found = result.get("ids") or []
    documents = result.get("documents") or [None] * len(ids_found)
    metadatas = result.get("metadatas") or [None] * len(ids_found)
    return [
        {"id": record_id, "document": document or "", "metadata": metadata or {}}
        for record_id, document, metadata in zip(ids_found, documents, metadatas, strict=True)
    ]


@dataclass(frozen=True)
class CollectionSummary:
    """What the integrity check found. Contains no chunk text and no vectors."""

    name: str
    count: int
    space: str
    sample_ids: list[str]
    has_text: bool
    metadata_fields: list[str]
    embedding_dimensions: list[int]


def summarize_collection(collection: Collection, sample_size: int = 2) -> CollectionSummary:
    """Read a small sample to confirm text, metadata and vectors are present."""
    try:
        sample = collection.get(limit=sample_size, include=["documents", "metadatas", "embeddings"])
        count = collection.count()
    except (ChromaError, ValueError, RuntimeError) as exc:
        raise VectorStoreError(f"Cannot read ChromaDB collection ({type(exc).__name__}).") from exc
    documents = sample.get("documents") or []
    metadatas = sample.get("metadatas") or []
    embeddings = sample.get("embeddings")
    return CollectionSummary(
        name=collection.name,
        count=count,
        space=collection_space(collection),
        sample_ids=list(sample.get("ids") or []),
        has_text=bool(documents) and all(bool(text) for text in documents),
        metadata_fields=sorted({key for metadata in metadatas if metadata for key in metadata}),
        embedding_dimensions=[len(vector) for vector in embeddings]
        if embeddings is not None
        else [],
    )


def main() -> None:
    settings = get_settings()
    if len(sys.argv) > 1:  # inspect another collection by name, without creating it
        collection = open_collection(settings.chroma_persist_directory, sys.argv[1])
    else:
        collection = get_collection(settings)
    summary = summarize_collection(collection)
    print(f"collection={summary.name} count={summary.count} space={summary.space}")
    print(f"sample_ids={summary.sample_ids} has_text={summary.has_text}")
    print(f"metadata_fields={summary.metadata_fields}")
    print(
        f"embedding_dimensions={summary.embedding_dimensions} "
        f"(expected {settings.embedding_dimension})"
    )


if __name__ == "__main__":
    main()
