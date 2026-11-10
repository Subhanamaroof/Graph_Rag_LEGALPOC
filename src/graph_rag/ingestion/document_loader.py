"""Discover supported documents and load them into ``LoadedDocument`` objects."""

import hashlib
from pathlib import Path

from graph_rag.ingestion.models import LoadedDocument
from graph_rag.ingestion.text_extractor import extract_text

SUPPORTED_EXTENSIONS = {".txt": "txt", ".md": "md", ".pdf": "pdf"}


def discover_documents(directory: Path) -> list[Path]:
    """Return supported files directly inside ``directory``, sorted by name.

    Discovery is intentionally non-recursive so that sub-folders (for example
    generated output) are never picked up. Hidden files and unsupported
    extensions are ignored. A missing directory yields an empty list.
    """
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def make_document_id(path: Path, base_directory: Path) -> str:
    """Derive a stable ID from the file's path relative to the documents directory.

    The same file always hashes to the same ID across runs; no randomness is used.
    """
    relative_path = path.resolve().relative_to(base_directory.resolve()).as_posix()
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
    return f"doc_{digest[:16]}"


def load_document(path: Path, base_directory: Path) -> LoadedDocument:
    """Extract the text of one file and wrap it with its identifying metadata.

    The original file is only read, never modified.
    """
    file_type = SUPPORTED_EXTENSIONS[path.suffix.lower()]
    pages, extraction_metadata = extract_text(path, file_type)
    return LoadedDocument(
        document_id=make_document_id(path, base_directory),
        source_file=path.name,
        file_type=file_type,
        content="\n\n".join(page.text for page in pages),
        pages=pages,
        metadata={"file_size_bytes": path.stat().st_size, **extraction_metadata},
    )
