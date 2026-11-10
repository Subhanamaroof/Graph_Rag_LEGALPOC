"""Shared display helpers: retrieval-origin labels and safe text rendering."""

import html
from typing import Any

import streamlit as st

# Backend origin values (Story 1.9 ``retrieval_source`` and Story 1.9.1 ``retrieval_origin``)
# mapped to display labels. Unknown values are shown as-is, never re-labelled.
ORIGIN_LABELS = {
    "vector": "Vector",
    "graph": "Graph",
    "both": "Vector + Graph",
    "vector_and_graph": "Vector + Graph",
}


def origin_label(origin: str | None) -> str:
    if origin is None:
        return "Unknown"
    return ORIGIN_LABELS.get(origin, origin)


def escape_markdown(text: str) -> str:
    """Stop ``$`` in legal text (e.g. amounts) from being rendered as LaTeX.

    HTML is already escaped because ``st.markdown`` is never called with
    ``unsafe_allow_html=True`` on backend text.
    """
    return text.replace("$", "\\$")


def render_chunk_text(text: str) -> None:
    """Original chunk text, HTML-escaped, whitespace preserved."""
    st.markdown(f'<div class="chunk-text">{html.escape(text)}</div>', unsafe_allow_html=True)


def value_or_dash(value: Any) -> Any:
    return "—" if value is None or value == "" else value


def text_lookup(sources: list[dict[str, Any]]) -> dict[tuple, str]:
    """Full chunk text by ``(document_id, chunk_id)`` from the evidence given to the LLM."""
    return {
        (s.get("document_id"), s.get("chunk_id")): s.get("content", "")
        for s in sources
        if s.get("content")
    }


def chunk_text(detail: dict[str, Any], lookup: dict[tuple, str]) -> tuple[str, bool]:
    """``(text, is_preview)``: the full text when the chunk was also an LLM source, otherwise
    the backend's ``text_preview`` (flagged when the backend truncated it)."""
    full = lookup.get((detail.get("document_id"), detail.get("chunk_id")))
    if full:
        return full, False
    return detail.get("text_preview", ""), bool(detail.get("text_truncated"))
