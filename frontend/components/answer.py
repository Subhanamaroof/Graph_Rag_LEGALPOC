"""The generated answer, its Story 1.9 status and its generation/retrieval metadata."""

from typing import Any

import streamlit as st

from frontend.components.common import escape_markdown

# (label, Streamlit alert kind, explanation). The explanation describes what the backend
# status means; the backend's own errors and warnings are shown with it.
STATUS_DISPLAY: dict[str, tuple[str, str, str]] = {
    "success": (
        "Answer generated",
        "success",
        "Every claim kept in the answer cites retrieved evidence that passed validation.",
    ),
    "partial": (
        "Partial answer",
        "warning",
        "An answer was produced, but retrieval was incomplete, the evidence only partly "
        "answers the question, or some claims were removed because their citations failed "
        "validation.",
    ),
    "insufficient_evidence": (
        "Insufficient evidence",
        "info",
        "The retrieved judgment passages do not contain enough information to answer this "
        "question, so no answer was generated.",
    ),
    "retrieval_failed": (
        "Retrieval failed",
        "error",
        "Evidence retrieval failed, so no answer was generated.",
    ),
    "generation_failed": (
        "Generation failed",
        "error",
        "Evidence was retrieved, but the language model did not produce a validly cited answer.",
    ),
}


def status_display(status: str) -> tuple[str, str, str]:
    return STATUS_DISPLAY.get(status, (status, "warning", "Unrecognised answer status."))


def citation_references(response: dict[str, Any]) -> list[tuple[str, str]]:
    """Distinct validated ``(document_id, chunk_id)`` citations, in first-use order."""
    seen: list[tuple[str, str]] = []
    for claim in response.get("claims", []):
        for citation in claim.get("citations", []):
            key = (citation.get("document_id"), citation.get("chunk_id"))
            if key not in seen:
                seen.append(key)
    return seen


def metadata_line(response: dict[str, Any]) -> str:
    summary = response.get("retrieval_summary", {})
    meta = response.get("generation_metadata", {})
    parts = [f"Retrieval: {summary.get('retrieval_status') or 'not completed'}"]
    parts.append(
        f"Evidence used: {summary.get('evidence_used', 0)} of "
        f"{summary.get('final_evidence_count', 0)} retrieved chunks"
    )
    if meta.get("llm_called"):
        parts.append(
            f"Claims accepted: {meta.get('claims_accepted', 0)} of {meta.get('claims_returned', 0)}"
        )
    if meta.get("model"):
        parts.append(f"Model: {meta['model']}")
    return " · ".join(parts)


def _alert(kind: str, text: str) -> None:
    {"success": st.success, "warning": st.warning, "info": st.info, "error": st.error}[kind](text)


def render_answer(response: dict[str, Any]) -> None:
    label, kind, explanation = status_display(response["status"])
    answer = response.get("answer")

    if answer:
        _alert(kind, f"**{label}**")
        st.markdown(escape_markdown(answer))
    else:
        _alert(kind, f"**{label}** — {explanation}")
        for error in response.get("errors", []):
            st.markdown(f"- {escape_markdown(error)}")

    citations = citation_references(response)
    if citations:
        refs = ", ".join(f"`{doc}` · `{chunk}`" for doc, chunk in citations)
        st.caption(f"Validated citations ({len(citations)}): {refs}")
    st.caption(metadata_line(response))

    warnings = response.get("warnings", []) + (response.get("errors", []) if answer else [])
    if warnings:
        with st.expander(f"Backend warnings and errors ({len(warnings)})"):
            for warning in warnings:
                st.markdown(f"- {escape_markdown(warning)}")
