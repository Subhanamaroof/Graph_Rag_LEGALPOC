"""Data structures shared by the ingestion modules.

``Chunk`` is the common unit consumed later by both the vector pipeline (ChromaDB)
and the knowledge graph pipeline (Neo4j), so its fields must stay stable.
"""

from typing import Any

from pydantic import BaseModel, Field


class PageText(BaseModel):
    """A piece of extracted text and the page it came from.

    PDFs produce one ``PageText`` per page. TXT and Markdown have no pages, so they
    produce a single ``PageText`` with ``page_number=None``.
    """

    page_number: int | None = None
    text: str


class LoadedDocument(BaseModel):
    """A source document after loading and text extraction."""

    document_id: str
    source_file: str
    file_type: str
    content: str
    pages: list[PageText]
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A piece of a document small enough to embed or send to an LLM."""

    chunk_id: str
    document_id: str
    source_file: str
    file_type: str
    chunk_index: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionError(BaseModel):
    """A problem with one document (or the input directory)."""

    source_file: str
    error: str


class DocumentStatus(BaseModel):
    """Per-document outcome of one ingestion run."""

    source_file: str
    document_id: str | None = None
    status: str  # "succeeded" or "failed"
    chunks: int = 0
    embeddings: int = 0
    records_upserted: int = 0
    obsolete_chunks_deleted: int = 0
    error: str | None = None


class IngestionResult(BaseModel):
    """Summary of one ingestion run. ``chunks`` is only filled when requested.

    ``total_chunks`` counts chunks generated. Since Story 1.3.2.2 the chunks are also
    embedded and upserted into ``collection`` (``new_documents``); records are only counted
    after ChromaDB accepted the write.
    """

    status: str
    documents_discovered: int
    documents_processed: int
    documents_failed: int
    total_chunks: int
    total_embeddings: int = 0
    records_upserted: int = 0
    obsolete_chunks_deleted: int = 0
    collection: str | None = None
    errors: list[IngestionError] = Field(default_factory=list)
    documents: list[DocumentStatus] = Field(default_factory=list)
    chunks: list[Chunk] | None = None
