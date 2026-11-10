"""Vector retrieval: query -> query embedding -> ChromaDB similarity search -> top-k chunks."""

from typing import Any

from chromadb.api.models.Collection import Collection

from graph_rag.embeddings.embedder import Embedder
from graph_rag.ingestion.text_cleaner import clean_text
from graph_rag.retrieval.models import MAX_TOP_K, RetrievedChunk, VectorSearchResponse
from graph_rag.vector_store.chroma_store import search_similar_chunks

# Metadata key that identifies the legal judgment a chunk came from.
DOCUMENT_ID_FIELD = "case_id"


def to_retrieved_chunk(hit: dict[str, Any]) -> RetrievedChunk:
    """Map one ChromaDB hit to the API shape.

    The collection uses cosine distance (``1 - cosine similarity``), so
    ``score = 1 - distance`` recovers the cosine similarity exactly.
    """
    metadata = hit["metadata"]
    case_id = metadata.get(DOCUMENT_ID_FIELD)
    return RetrievedChunk(
        document_id=str(case_id) if case_id is not None else None,
        chunk_id=hit["id"],
        text=hit["document"] or "",
        metadata=metadata,
        score=1.0 - hit["distance"],
        distance=hit["distance"],
    )


def retrieve_similar_chunks(
    query: str,
    top_k: int,
    *,
    embedder: Embedder,
    collection: Collection,
) -> VectorSearchResponse:
    """Return the ``top_k`` stored chunks whose embeddings are closest to the query's.

    The query is cleaned with the same ``clean_text`` used for chunks and embedded with
    the same (already loaded) model as the stored vectors, so both live in one space.
    Raises ``ValueError`` for invalid input, ``EmbeddingError`` if embedding fails and
    ``VectorStoreError`` (or a subclass) if ChromaDB cannot serve the search.
    """
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
    cleaned_query = clean_text(query)
    if not cleaned_query:
        raise ValueError("query must not be empty or whitespace")

    query_vector = embedder.embed_text(cleaned_query)  # validated: 384 finite floats

    hits = search_similar_chunks(collection, query_vector, top_k)
    results = [to_retrieved_chunk(hit) for hit in hits]
    return VectorSearchResponse(query=query, total_results=len(results), results=results)
