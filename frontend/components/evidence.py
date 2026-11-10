"""Retrieval evidence: vector results, graph results and the hybrid (RRF) ranking.

Reads ``retrieval_details`` (Story 1.9.1), which is copied from the single Story 1.8 response
of the same request. Ranks, scores and origins are displayed exactly as returned; nothing is
re-computed here.
"""

from typing import Any

import streamlit as st

from frontend.components.common import (
    chunk_text,
    origin_label,
    render_chunk_text,
    text_lookup,
)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def vector_rows(details: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "Vector rank": d.get("vector_rank"),
            "Document ID": d.get("document_id"),
            "Chunk ID": d.get("chunk_id"),
            "Similarity": _round(d.get("vector_score")),
            "Distance": _round(d.get("vector_distance")),
            "Origin": origin_label(d.get("retrieval_origin")),
            "Hybrid rank": d.get("hybrid_rank"),
            "In final top-k": d.get("in_final_results"),
        }
        for d in details.get("vector_results", [])
    ]


def best_graph_path(detail: dict[str, Any]) -> dict[str, Any] | None:
    paths = (detail.get("graph_context") or {}).get("paths") or []
    return paths[0] if paths else None


def graph_rows(details: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for d in details.get("graph_results", []):
        path = best_graph_path(d)
        rows.append(
            {
                "Graph rank": d.get("graph_rank"),
                "Document ID": d.get("document_id"),
                "Chunk ID": d.get("chunk_id"),
                "Origin": origin_label(d.get("retrieval_origin")),
                "Graph path": path.get("path") if path else None,
                "Path score": _round(path.get("graph_score")) if path else None,
                "Hybrid rank": d.get("hybrid_rank"),
                "In final top-k": d.get("in_final_results"),
            }
        )
    return rows


def hybrid_rows(details: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "Hybrid rank": d.get("hybrid_rank"),
            "RRF score": round(d.get("rrf_score", 0.0), 5),
            "Origin": origin_label(d.get("retrieval_origin")),
            "Vector rank": d.get("vector_rank"),
            "Graph rank": d.get("graph_rank"),
            "In final top-k": d.get("in_final_results"),
            "Document ID": d.get("document_id"),
            "Chunk ID": d.get("chunk_id"),
        }
        for d in details.get("hybrid_results", [])
    ]


def _truncation_note(details: dict[str, Any], name: str) -> None:
    if name in details.get("truncated_lists", []):
        st.caption(
            f"List capped at {details.get('max_items_per_list')} items by the backend; "
            "counts above include every result."
        )


def _render_chunk(detail: dict[str, Any], rank_label: str, lookup: dict) -> None:
    title = (
        f"{rank_label} · {detail.get('document_id')} · {detail.get('chunk_id')} — "
        f"{origin_label(detail.get('retrieval_origin'))}"
    )
    with st.expander(title):
        facts = [f"{k.title()}: {detail[k]}" for k in ("title", "court", "date") if detail.get(k)]
        if facts:
            st.caption(" · ".join(facts))
        path = best_graph_path(detail)
        if path:
            st.caption(f"Reached through graph path: {path.get('path')}")
        text, is_preview = chunk_text(detail, lookup)
        render_chunk_text(text)
        if is_preview:
            st.caption(
                f"Preview: first {len(text)} of {detail.get('text_length')} characters "
                "(this chunk was not part of the model context)."
            )


def render_evidence(response: dict[str, Any]) -> None:
    details = response.get("retrieval_details")
    if not details:
        st.warning(
            "This response has no retrieval details (retrieval raised an error before "
            "returning results), so per-branch evidence cannot be shown."
        )
        return
    lookup = text_lookup(response.get("sources", []))
    fusion = details.get("fusion_summary", {})
    st.caption(
        f"Vector branch: {details.get('vector_status')} · Graph branch: "
        f"{details.get('graph_status')} (graph retrieval status: "
        f"{details.get('graph_retrieval_status') or 'not run'})"
    )

    st.markdown(f"#### A. Vector retrieval — {fusion.get('vector_count', 0)} results")
    if details.get("vector_results"):
        st.dataframe(vector_rows(details), hide_index=True, width="stretch")
        _truncation_note(details, "vector_results")
        for d in details["vector_results"]:
            _render_chunk(d, f"Vector #{d.get('vector_rank')}", lookup)
    else:
        st.info("Vector retrieval returned no chunks.")

    st.markdown(f"#### B. Graph retrieval — {fusion.get('graph_count', 0)} results")
    if details.get("graph_results"):
        st.dataframe(graph_rows(details), hide_index=True, width="stretch")
        _truncation_note(details, "graph_results")
        for d in details["graph_results"]:
            _render_chunk(d, f"Graph #{d.get('graph_rank')}", lookup)
    else:
        st.info("Graph retrieval returned no chunks for this question.")

    st.markdown(f"#### C. Hybrid results — {fusion.get('final_results_count', 0)} final")
    cols = st.columns(4)
    cols[0].metric("Merged chunks", fusion.get("hybrid_count", 0))
    cols[1].metric("Vector only", fusion.get("vector_only_count", 0))
    cols[2].metric("Graph only", fusion.get("graph_only_count", 0))
    cols[3].metric("Found by both", fusion.get("both_count", 0))
    summary = response.get("retrieval_summary", {})
    st.caption(
        f"{fusion.get('overlap_count', 0)} duplicate chunk(s) found by both branches were merged "
        f"into one result each. The context builder removed "
        f"{summary.get('duplicates_removed', 0)} further duplicate(s) and placed "
        f"{summary.get('evidence_used', 0)} chunk(s) in the model context."
    )
    rrf_k = fusion.get("rrf_k")
    if rrf_k is not None:
        st.caption(
            f"Reciprocal Rank Fusion (computed by the backend): RRF score = Σ 1 / ({rrf_k} + "
            "rank) over the branches that found the chunk. Chunks within the final top-k were "
            "offered to the answer model."
        )
    if details.get("hybrid_results"):
        st.dataframe(hybrid_rows(details), hide_index=True, width="stretch")
        _truncation_note(details, "hybrid_results")
    else:
        st.info("No hybrid results.")
