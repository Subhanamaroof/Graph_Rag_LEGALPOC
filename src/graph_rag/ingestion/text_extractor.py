"""Format-specific text extraction for TXT, Markdown and text-based PDF files."""

from pathlib import Path

import pymupdf

from graph_rag.ingestion.models import PageText


class DocumentProcessingError(Exception):
    """Raised when a single document cannot be turned into usable text."""


class EmptyDocumentError(DocumentProcessingError):
    """Raised when a document contains no extractable text."""


def extract_plain_text(path: Path) -> tuple[list[PageText], dict[str, bool]]:
    """Read a TXT or Markdown file as UTF-8.

    Markdown is kept as-is: headings (``#``), lists and paragraphs are meaningful
    structure for both chunking and later entity extraction.

    Invalid UTF-8 bytes are replaced with U+FFFD instead of failing the whole file;
    this is reported through the returned metadata.
    """
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")  # "-sig" drops a leading byte-order mark
        had_decoding_errors = False
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
        had_decoding_errors = True
    return [PageText(page_number=None, text=text)], {"decoding_errors": had_decoding_errors}


def extract_pdf_text(path: Path) -> tuple[list[PageText], dict[str, int]]:
    """Extract text page by page with PyMuPDF, keeping 1-based page numbers.

    Only the PDF's embedded text layer is read; scanned (image-only) PDFs would need
    OCR, which is out of scope, so they surface as ``EmptyDocumentError``.
    """
    with pymupdf.open(path) as pdf:
        pages = [PageText(page_number=page.number + 1, text=page.get_text("text")) for page in pdf]
        page_count = pdf.page_count

    if not any(page.text.strip() for page in pages):
        raise EmptyDocumentError(
            "PDF has no extractable text (it may be scanned/image-only; OCR is not supported)"
        )
    return pages, {"page_count": page_count}


def extract_text(path: Path, file_type: str) -> tuple[list[PageText], dict]:
    """Dispatch to the extractor for ``file_type`` ("txt", "md" or "pdf")."""
    if file_type in ("txt", "md"):
        return extract_plain_text(path)
    if file_type == "pdf":
        return extract_pdf_text(path)
    raise DocumentProcessingError(f"Unsupported file type: {file_type}")
