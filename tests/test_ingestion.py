"""Story 1.2 tests: loading, extraction, cleaning, chunking, ingestion service and API."""

from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from graph_rag.core.config import Settings, get_settings
from graph_rag.ingestion.chunker import chunk_document
from graph_rag.ingestion.document_loader import discover_documents, load_document
from graph_rag.ingestion.ingestion_service import ingest_documents, process_document
from graph_rag.ingestion.models import LoadedDocument, PageText
from graph_rag.ingestion.router import get_ingestion_collection
from graph_rag.ingestion.text_cleaner import clean_text
from graph_rag.ingestion.text_extractor import EmptyDocumentError
from graph_rag.main import app
from graph_rag.retrieval.router import get_loaded_embedder


def make_pdf(path: Path, page_texts: list[str]) -> Path:
    pdf = pymupdf.open()
    for text in page_texts:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()
    return path


def make_document(text: str, pages: list[PageText] | None = None) -> LoadedDocument:
    return LoadedDocument(
        document_id="doc_test",
        source_file="test.txt",
        file_type="txt",
        content=text,
        pages=pages or [PageText(text=text)],
    )


# --- Loading and extraction -------------------------------------------------


def test_txt_file_loading(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Hello world.\nSecond line.", encoding="utf-8", newline="\n")

    document = load_document(path, tmp_path)

    assert document.file_type == "txt"
    assert document.source_file == "notes.txt"
    assert document.content == "Hello world.\nSecond line."
    assert document.pages[0].page_number is None


def test_txt_invalid_utf8_is_handled(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_bytes(b"caf\xe9 menu")

    document = load_document(path, tmp_path)

    assert document.metadata["decoding_errors"] is True
    assert "menu" in document.content


def test_markdown_structure_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text(
        "# Title\n\nIntro paragraph.\n\n- item one\n- item two\n", encoding="utf-8", newline="\n"
    )

    document = load_document(path, tmp_path)

    assert document.file_type == "md"
    assert "# Title" in document.content
    assert "- item one\n- item two" in document.content


def test_pdf_text_extraction_keeps_page_numbers(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "report.pdf", ["First page text", "Second page text"])

    document = load_document(path, tmp_path)

    assert [page.page_number for page in document.pages] == [1, 2]
    assert "First page text" in document.pages[0].text
    assert "Second page text" in document.pages[1].text
    assert document.metadata["page_count"] == 2


def test_empty_pdf_is_reported(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "scanned.pdf", [""])

    with pytest.raises(EmptyDocumentError):
        load_document(path, tmp_path)


def test_empty_text_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n  ", encoding="utf-8", newline="\n")

    with pytest.raises(EmptyDocumentError):
        process_document(path, tmp_path, chunk_size=800, chunk_overlap=150)


def test_unsupported_extensions_and_subfolders_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8", newline="\n")
    (tmp_path / "b.md").write_text("b", encoding="utf-8", newline="\n")
    (tmp_path / "c.docx").write_text("c", encoding="utf-8", newline="\n")
    (tmp_path / "d.json").write_text("{}", encoding="utf-8", newline="\n")
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "e.txt").write_text("e", encoding="utf-8", newline="\n")

    names = [path.name for path in discover_documents(tmp_path)]

    assert names == ["a.txt", "b.md"]


def test_missing_directory_returns_no_documents(tmp_path: Path) -> None:
    assert discover_documents(tmp_path / "missing") == []


# --- Cleaning ---------------------------------------------------------------


def test_text_cleaning() -> None:
    raw = "# Heading\r\n\r\nSome    text,\twith  spaces.   \r\n\n\n\n\nNext paragraph!​\n  - nested"

    cleaned = clean_text(raw)

    assert cleaned == "# Heading\n\nSome text, with spaces.\n\nNext paragraph!\n  - nested"


# --- Chunking ---------------------------------------------------------------


def test_chunk_size_is_respected() -> None:
    text = " ".join(f"Sentence number {i} talks about robots." for i in range(200))

    chunks = chunk_document(make_document(text), chunk_size=200, chunk_overlap=40)

    assert len(chunks) > 1
    assert all(0 < len(chunk.text) <= 200 for chunk in chunks)


def test_consecutive_chunks_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(300))

    chunks = chunk_document(make_document(text), chunk_size=100, chunk_overlap=30)

    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.metadata["start_char"] < previous.metadata["end_char"]
        shared = text[current.metadata["start_char"] : previous.metadata["end_char"]]
        assert shared and shared in previous.text and current.text.startswith(shared)


def test_chunks_keep_metadata_and_pdf_pages(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "report.pdf", ["Alpha page one content", "Beta page two content"])

    chunks = process_document(path, tmp_path, chunk_size=30, chunk_overlap=5)
    document_id = load_document(path, tmp_path).document_id

    for index, chunk in enumerate(chunks):
        assert chunk.document_id == document_id
        assert chunk.source_file == "report.pdf"
        assert chunk.file_type == "pdf"
        assert chunk.chunk_index == index
        assert chunk.text.strip()
    assert chunks[0].metadata["page_numbers"] == [1]
    assert chunks[-1].metadata["page_numbers"] == [2]


# --- Deterministic IDs ------------------------------------------------------


def test_document_id_is_consistent(tmp_path: Path) -> None:
    path = tmp_path / "doc.txt"
    path.write_text("Stable content.", encoding="utf-8", newline="\n")
    other = tmp_path / "other.txt"
    other.write_text("Stable content.", encoding="utf-8", newline="\n")

    first = load_document(path, tmp_path).document_id

    assert first == load_document(path, tmp_path).document_id
    assert first != load_document(other, tmp_path).document_id


def test_chunk_ids_are_consistent(tmp_path: Path) -> None:
    path = tmp_path / "doc.txt"
    path.write_text(" ".join(f"token{i}" for i in range(500)), encoding="utf-8", newline="\n")

    first = [c.chunk_id for c in process_document(path, tmp_path, 200, 40)]
    second = [c.chunk_id for c in process_document(path, tmp_path, 200, 40)]

    assert first == second
    assert len(set(first)) == len(first)


# --- Service and API --------------------------------------------------------


def test_invalid_document_does_not_stop_others(tmp_path: Path) -> None:
    (tmp_path / "good.txt").write_text("Valid content here.", encoding="utf-8", newline="\n")
    (tmp_path / "broken.pdf").write_bytes(b"not really a pdf")

    result = ingest_documents(tmp_path, chunk_size=800, chunk_overlap=150)

    assert result.status == "completed_with_errors"
    assert result.documents_discovered == 2
    assert result.documents_processed == 1
    assert result.documents_failed == 1
    assert result.errors[0].source_file == "broken.pdf"
    assert result.total_chunks == 1


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        Settings(chunk_size=100, chunk_overlap=100)


def test_ingestion_api_response(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("First document.", encoding="utf-8", newline="\n")
    (tmp_path / "b.md").write_text("# Second\n\nDocument.", encoding="utf-8", newline="\n")
    app.dependency_overrides[get_settings] = lambda: Settings(documents_directory=tmp_path)
    # Since Story 1.3.2.2 the endpoint also embeds and stores; keep this test chunk-only.
    app.dependency_overrides[get_loaded_embedder] = lambda: None
    app.dependency_overrides[get_ingestion_collection] = lambda: None
    client = TestClient(app)

    try:
        summary = client.post("/v1/ingestion/process").json()
        detailed = client.post("/v1/ingestion/process?include_chunks=true").json()
    finally:
        app.dependency_overrides.clear()

    assert {
        "status": "completed",
        "documents_discovered": 2,
        "documents_processed": 2,
        "documents_failed": 0,
        "total_chunks": 2,
        "errors": [],
    }.items() <= summary.items()
    assert "chunks" not in summary
    assert len(detailed["chunks"]) == 2
    assert {
        "chunk_id",
        "document_id",
        "source_file",
        "file_type",
        "chunk_index",
        "text",
        "metadata",
    } <= detailed["chunks"][0].keys()
