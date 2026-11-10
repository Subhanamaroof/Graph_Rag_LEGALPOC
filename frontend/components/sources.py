"""Source citations: the validated claims and the original judgment chunks they cite.

Everything comes from the backend's validated response (``claims``, ``sources`` and
``retrieval_details.claim_citations``). The frontend never creates a citation.
"""

from typing import Any

import streamlit as st

from frontend.components.common import escape_markdown, origin_label, render_chunk_text

REJECTION_REASONS = {
    "no_citations": "the model gave no citation",
    "all_citations_invalid": "none of its citations matched the retrieved evidence",
}


def split_sources(response: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """``(cited, uncited)`` evidence sources, each in ``[SOURCE n]`` order."""
    sources = sorted(response.get("sources", []), key=lambda s: s.get("source_number", 0))
    return [s for s in sources if s.get("cited")], [s for s in sources if not s.get("cited")]


def source_title(source: dict[str, Any]) -> str:
    return (
        f"[SOURCE {source.get('source_number')}] {source.get('document_id')} · "
        f"{source.get('chunk_id')} — {origin_label(source.get('retrieval_source'))}"
    )


def claim_rows(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Claims with their citations. Prefers the Story 1.9.1 mapping (which includes rejected
    claims and ``[SOURCE n]`` labels); falls back to the Story 1.9 ``claims`` list."""
    details = response.get("retrieval_details") or {}
    if details.get("claim_citations"):
        return [
            {
                "claim": c.get("claim_text", ""),
                "supported": c.get("supported", False),
                "citations": [
                    f"[SOURCE {x.get('source_number')}] {x.get('document_id')} · "
                    f"{x.get('chunk_id')} ({origin_label(x.get('retrieval_origin'))})"
                    for x in c.get("citations", [])
                ],
                "rejection_reason": c.get("rejection_reason"),
            }
            for c in details["claim_citations"]
        ]
    return [
        {
            "claim": c.get("text", ""),
            "supported": True,
            "citations": [
                f"{x.get('document_id')} · {x.get('chunk_id')}" for x in c.get("citations", [])
            ],
            "rejection_reason": None,
        }
        for c in response.get("claims", [])
    ]


def _render_source(source: dict[str, Any]) -> None:
    with st.expander(source_title(source)):
        meta = source.get("metadata") or {}
        facts = [f"{k.title()}: {meta[k]}" for k in ("title", "court", "date") if meta.get(k)]
        facts.append(f"Hybrid rank {source.get('rank')} · RRF {source.get('rrf_score', 0):.4f}")
        st.caption(" · ".join(facts))
        render_chunk_text(source.get("content", ""))


def render_sources(response: dict[str, Any]) -> None:
    rows = claim_rows(response)
    if rows:
        st.markdown("**Claims and their citations**")
        for row in rows:
            if row["supported"]:
                st.markdown(f"- {escape_markdown(row['claim'])}")
                st.caption("  " + "; ".join(row["citations"]))
            else:
                reason = REJECTION_REASONS.get(row["rejection_reason"], row["rejection_reason"])
                st.markdown(f"- ~~{escape_markdown(row['claim'])}~~")
                st.caption(f"  Removed from the answer: {reason}.")

    cited, uncited = split_sources(response)
    if not cited and not uncited:
        st.info("No evidence sources were returned for this question.")
        return
    st.markdown(f"**Cited judgment passages ({len(cited)})**")
    if not cited:
        st.caption("No passage was cited by a validated claim.")
    for source in cited:
        _render_source(source)
    if uncited:
        st.markdown(f"**Other evidence given to the model, not cited ({len(uncited)})**")
        for source in uncited:
            _render_source(source)
