"""Recursive, paragraph-aware chunking with page tracking and deterministic chunk IDs."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from graph_rag.ingestion.models import Chunk, LoadedDocument

# Tried in order: paragraphs, lines, sentences, words, and finally single characters.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
PAGE_SEPARATOR = "\n\n"


def make_chunk_id(document_id: str, chunk_index: int) -> str:
    """Deterministic chunk ID: same document + same position -> same ID."""
    return f"{document_id}_chunk_{chunk_index:04d}"


def _join_pages(document: LoadedDocument) -> tuple[str, list[tuple[int, int, int]]]:
    """Join page texts into one string and record each page's character span.

    Returns the full text and a list of ``(page_number, start, end)`` spans.
    Pages without a page number (TXT/Markdown) produce no spans.
    """
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    position = 0
    for page in document.pages:
        if not page.text:
            continue
        if parts:
            parts.append(PAGE_SEPARATOR)
            position += len(PAGE_SEPARATOR)
        if page.page_number is not None:
            spans.append((page.page_number, position, position + len(page.text)))
        parts.append(page.text)
        position += len(page.text)
    return "".join(parts), spans


def chunk_document(document: LoadedDocument, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split an (already cleaned) document into overlapping chunks.

    Each chunk records its character span in the joined text and the page numbers
    that span overlaps, so PDF chunks can later be cited by page.
    """
    full_text, page_spans = _join_pages(document)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        keep_separator="end",  # keep "." with its sentence instead of the next one
    )

    chunks: list[Chunk] = []
    search_from = 0
    for piece in splitter.split_text(full_text):
        if not piece.strip():
            continue
        # Locate the piece in the original text to recover its character span.
        start = full_text.find(piece, search_from)
        if start == -1:
            start = full_text.find(piece)
        end = start + len(piece)
        search_from = start + 1

        page_numbers = [
            number
            for number, page_start, page_end in page_spans
            if start < page_end and end > page_start
        ]

        index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(document.document_id, index),
                document_id=document.document_id,
                source_file=document.source_file,
                file_type=document.file_type,
                chunk_index=index,
                text=piece,
                metadata={"page_numbers": page_numbers, "start_char": start, "end_char": end},
            )
        )
    return chunks
