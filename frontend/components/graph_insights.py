"""Graph entities, relationships, a graph view and "How Graph Retrieval Contributed".

Every node, edge, count and observation is derived from ``retrieval_details`` (Story 1.9.1).
Nothing is invented: when a field is missing it is shown as missing, and observations are
only emitted when the response supports them. The contribution section explains which
evidence graph retrieval supplied; it does not claim a measured answer-quality improvement,
because the backend generates a single answer from the fused evidence.
"""

from typing import Any

import streamlit as st

from frontend.components.common import escape_markdown, value_or_dash

ENTITY_TYPE_ORDER = [
    "Person",
    "Court",
    "Organization",
    "Case",
    "Statute",
    "LegalSection",
    "Date",
    "Location",
    "LegalConcept",
]
UNTYPED = "Untyped"
MAX_CHUNK_OBSERVATIONS = 5

# --- Entities and relationships ---------------------------------------------------------------


def group_entities(entities: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Entities grouped by type: known types in schema order, then others, then untyped."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for entity in entities:
        groups.setdefault(entity.get("entity_type") or UNTYPED, []).append(entity)

    def order(entity_type: str) -> tuple[int, str]:
        if entity_type in ENTITY_TYPE_ORDER:
            return ENTITY_TYPE_ORDER.index(entity_type), entity_type
        return (len(ENTITY_TYPE_ORDER) + (entity_type == UNTYPED), entity_type)

    return {t: groups[t] for t in sorted(groups, key=order)}


def entity_names(details: dict[str, Any]) -> dict[str, str]:
    """``entity_id -> name`` from path nodes and matched entities."""
    names = {e["entity_id"]: e.get("name") for e in details.get("matched_entities", [])}
    names.update({e["entity_id"]: e.get("name") for e in details.get("graph_entities", [])})
    return {k: v for k, v in names.items() if v}


def entity_rows(details: dict[str, Any]) -> list[dict[str, Any]]:
    """Path entities. Path nodes carry no normalized name or provenance in the API; those are
    taken from the matched-entity record of the same ``entity_id`` when there is one."""
    matched = {m["entity_id"]: m for m in details.get("matched_entities", [])}
    rows = []
    for entity in details.get("graph_entities", []):
        m = matched.get(entity["entity_id"], {})
        rows.append(
            {
                "Name": entity.get("name"),
                "Type": value_or_dash(entity.get("entity_type")),
                "Normalized name": value_or_dash(m.get("normalized_name")),
                "Matched query entity": value_or_dash(m.get("query_name")),
                "Source document IDs": value_or_dash(", ".join(m.get("source_document_ids", []))),
                "Source chunk IDs": value_or_dash(", ".join(m.get("source_chunk_ids", []))),
                "Description": value_or_dash(entity.get("description")),
                "Entity ID": entity["entity_id"],
            }
        )
    return rows


def matched_entity_rows(details: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "Query mention": m.get("query_name"),
            "Matched entity": m.get("name"),
            "Type": value_or_dash(m.get("entity_type")),
            "Normalized name": value_or_dash(m.get("normalized_name")),
            "Match": m.get("match_type"),
            "Ambiguous": m.get("ambiguous", False),
            "Documents": m.get("document_count", 0),
            "Chunks": m.get("chunk_count", 0),
            "Source document IDs": value_or_dash(", ".join(m.get("source_document_ids", []))),
        }
        for m in details.get("matched_entities", [])
    ]


def relationship_rows(
    relationships: list[dict[str, Any]], names: dict[str, str]
) -> list[dict[str, Any]]:
    """Edges in their stored direction; unknown endpoint names fall back to the entity ID."""
    return [
        {
            "Source entity": names.get(r["source_entity_id"], r["source_entity_id"]),
            "Relationship": r.get("relationship_type"),
            "Target entity": names.get(r["target_entity_id"], r["target_entity_id"]),
            "Supporting evidence": value_or_dash(r.get("evidence")),
            "Source document ID": value_or_dash(r.get("source_document_id")),
            "Source chunk ID": value_or_dash(r.get("source_chunk_id")),
        }
        for r in relationships
    ]


def _dot_label(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def build_dot(details: dict[str, Any]) -> str | None:
    """Graphviz DOT for the returned path entities and relationships only, or ``None``."""
    entities = details.get("graph_entities", [])
    relationships = details.get("graph_relationships", [])
    if not relationships:
        return None
    names = entity_names(details)
    matched = {m["entity_id"] for m in details.get("matched_entities", [])}
    node_ids = [e["entity_id"] for e in entities]
    for r in relationships:  # an endpoint missing from a capped entity list is still real
        for endpoint in (r["source_entity_id"], r["target_entity_id"]):
            if endpoint not in node_ids:
                node_ids.append(endpoint)
    types = {e["entity_id"]: e.get("entity_type") for e in entities}

    lines = [
        "digraph G {",
        "  rankdir=LR;",
        '  node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="#94A3B8",'
        ' fontname="Helvetica", fontsize=10];',
        '  edge [color="#64748B", fontname="Helvetica", fontsize=9];',
    ]
    for index, node_id in enumerate(node_ids):
        label = _dot_label(names.get(node_id, node_id))
        if types.get(node_id):
            label += f"\\n({_dot_label(types[node_id])})"
        style = ', fillcolor="#DBEAFE", color="#2563EB"' if node_id in matched else ""
        lines.append(f'  n{index} [label="{label}"{style}];')
    index_of = {node_id: i for i, node_id in enumerate(node_ids)}
    for r in relationships:
        lines.append(
            f"  n{index_of[r['source_entity_id']]} -> n{index_of[r['target_entity_id']]}"
            f' [label="{_dot_label(r.get("relationship_type", ""))}"];'
        )
    lines.append("}")
    return "\n".join(lines)


# --- Graph RAG contribution -------------------------------------------------------------------


def compute_contribution(response: dict[str, Any]) -> dict[str, Any] | None:
    """Counts describing what each branch contributed, from actual response fields only."""
    details = response.get("retrieval_details")
    if not details:
        return None
    fusion = details.get("fusion_summary", {})
    graph_only = details.get("graph_only_chunks", [])

    supported = [c for c in details.get("claim_citations", []) if c.get("supported")]

    def cited_with(origin: str) -> set[tuple]:
        return {
            (x.get("document_id"), x.get("chunk_id"))
            for c in supported
            for x in c.get("citations", [])
            if x.get("retrieval_origin") == origin
        }

    def claims_citing(origin: str) -> int:
        return sum(
            any(x.get("retrieval_origin") == origin for x in c.get("citations", []))
            for c in supported
        )

    return {
        "vector_count": fusion.get("vector_count", 0),
        "graph_count": fusion.get("graph_count", 0),
        "vector_only_count": fusion.get("vector_only_count", 0),
        "graph_only_count": fusion.get("graph_only_count", 0),
        "both_count": fusion.get("both_count", 0),
        "duplicates_merged": fusion.get("overlap_count", 0),
        "hybrid_count": fusion.get("hybrid_count", 0),
        "final_results_count": fusion.get("final_results_count", 0),
        "graph_only_in_final": sum(1 for c in graph_only if c.get("in_final_results")),
        "graph_only_listed": len(graph_only),
        "supported_claims": len(supported),
        "claims_citing_graph_only": claims_citing("graph"),
        "claims_citing_both": claims_citing("both"),
        "graph_only_chunks_cited": len(cited_with("graph")),
        "both_chunks_cited": len(cited_with("both")),
    }


def _chunk_observation(chunk: dict[str, Any]) -> str | None:
    context = chunk.get("graph_context") or {}
    paths = context.get("paths") or []
    if not paths:
        return None
    path = paths[0]
    rel_types = sorted(
        {r.get("relationship_type") for r in context.get("relationships", []) if r} - {None}
    )
    text = (
        f"Graph-only chunk `{chunk.get('chunk_id')}` (document `{chunk.get('document_id')}`) "
        f"was reached from the matched entity “{path.get('matched_entity', {}).get('name')}” "
        f"via the path {path.get('path')}"
    )
    if rel_types:
        text += f"; relationships extracted from this chunk: {', '.join(rel_types)}"
    typed = [
        f"{e.get('name')} ({e['entity_type']})"
        for e in context.get("entities", [])
        if e.get("entity_type")
    ]
    if typed:
        text += f"; connected entities: {', '.join(typed[:6])}"
    return text + "."


def contribution_observations(response: dict[str, Any]) -> list[str]:
    """Factual statements, each backed by a field in the response."""
    details = response.get("retrieval_details")
    c = compute_contribution(response)
    if not details or c is None:
        return []
    observations: list[str] = []

    if details.get("graph_status") == "failed":
        observations.append(
            "Graph retrieval failed for this question; all evidence came from vector retrieval."
        )
    elif c["graph_count"] == 0:
        observations.append(
            "Graph retrieval added no evidence for this question (graph retrieval status: "
            f"{details.get('graph_retrieval_status') or 'unknown'})."
        )
    elif c["graph_only_count"] == 0:
        observations.append(
            "Graph retrieval added no unique evidence: every chunk it found was also found by "
            "vector retrieval."
        )
    else:
        observations.append(
            f"Graph retrieval added {c['graph_only_count']} chunk(s) that vector retrieval did "
            f"not return; {c['graph_only_in_final']} of the listed graph-only chunks were within "
            "the final top-k offered to the answer model."
        )
        in_final = [ch for ch in details.get("graph_only_chunks", []) if ch.get("in_final_results")]
        for chunk in in_final[:MAX_CHUNK_OBSERVATIONS]:
            if text := _chunk_observation(chunk):
                observations.append(text)

    if c["both_count"]:
        observations.append(
            f"{c['both_count']} chunk(s) were found by both vector and graph retrieval; "
            "Reciprocal Rank Fusion added both rank contributions, so these chunks rank higher "
            "than they would with a single branch."
        )

    if c["supported_claims"]:
        if c["claims_citing_graph_only"]:
            observations.append(
                f"{c['claims_citing_graph_only']} of {c['supported_claims']} supported claim(s) "
                f"cite at least one graph-only chunk ({c['graph_only_chunks_cited']} distinct "
                "chunk(s)). Those passages were not in the vector results, so they reached the "
                "answer through graph retrieval."
            )
        else:
            observations.append(
                "No supported claim cites a graph-only chunk; the answer's citations point to "
                "chunks that vector retrieval also returned."
            )
        if c["claims_citing_both"]:
            observations.append(
                f"{c['claims_citing_both']} supported claim(s) cite chunks found by both branches."
            )
    return observations


def _graph_relationships_for_unique_evidence(details: dict[str, Any]) -> list[dict[str, Any]]:
    seen, edges = set(), []
    for chunk in details.get("graph_only_chunks", []):
        for edge in (chunk.get("graph_context") or {}).get("relationships", []):
            if edge.get("relationship_id") not in seen:
                seen.add(edge.get("relationship_id"))
                edges.append(edge)
    return edges


def render_contribution(response: dict[str, Any]) -> None:
    c = compute_contribution(response)
    details = response.get("retrieval_details")
    if c is None or not details:
        summary = response.get("retrieval_summary", {})
        st.warning(
            "The response has no retrieval details, so graph contribution cannot be shown. "
            f"Reported counts — vector: {summary.get('vector_results', 0)}, graph: "
            f"{summary.get('graph_results', 0)}, both: {summary.get('found_by_both', 0)}."
        )
        return

    st.markdown("##### A. Vector contribution")
    st.caption(
        "Vector retrieval embeds the question and finds chunks whose stored embeddings are "
        "semantically similar (cosine similarity in ChromaDB)."
    )
    cols = st.columns(3)
    cols[0].metric("Vector results", c["vector_count"])
    cols[1].metric("Vector only", c["vector_only_count"])
    cols[2].metric("Also found by graph", c["both_count"])

    st.markdown("##### B. Graph contribution")
    st.caption(
        "Graph retrieval matches entities named in the question to the Neo4j knowledge graph, "
        "follows their relationships, and returns the judgment chunks those relationships were "
        "extracted from."
    )
    cols = st.columns(3)
    cols[0].metric("Graph results", c["graph_count"])
    cols[1].metric("Graph only", c["graph_only_count"])
    cols[2].metric("Matched entities", len(details.get("matched_entities", [])))
    names = entity_names(details)
    unique_edges = _graph_relationships_for_unique_evidence(details)
    if unique_edges:
        st.markdown("Relationships that supplied graph-only evidence:")
        st.dataframe(relationship_rows(unique_edges, names), hide_index=True, width="stretch")

    st.markdown("##### C. Combined evidence")
    cols = st.columns(4)
    cols[0].metric("Duplicates merged", c["duplicates_merged"])
    cols[1].metric("Merged chunks", c["hybrid_count"])
    cols[2].metric("Final hybrid results", c["final_results_count"])
    cols[3].metric("Unique graph evidence", c["graph_only_count"])

    st.markdown("##### D. Effect on the answer")
    for observation in contribution_observations(response):
        st.markdown(f"- {escape_markdown(observation)}")
    st.caption(
        "The backend generates one answer from the fused vector + graph evidence. No separate "
        "vector-only answer was generated, so these observations describe which evidence graph "
        "retrieval supplied — not a measured improvement in answer quality."
    )


def render_graph_insights(response: dict[str, Any]) -> None:
    details = response.get("retrieval_details")
    if not details:
        st.warning("The response has no retrieval details, so no graph data can be shown.")
        return

    matched = matched_entity_rows(details)
    st.markdown(f"**Entities matched from the question ({len(matched)})**")
    if matched:
        st.dataframe(matched, hide_index=True, width="stretch")
    else:
        st.caption("No question entity matched a node in the knowledge graph.")

    entities = details.get("graph_entities", [])
    st.markdown(f"**Entities on the retrieved graph paths ({len(entities)})**")
    if entities:
        rows_by_id = {row["Entity ID"]: row for row in entity_rows(details)}
        for entity_type, group in group_entities(entities).items():
            st.markdown(f"*{entity_type}* ({len(group)})")
            st.dataframe(
                [rows_by_id[e["entity_id"]] for e in group], hide_index=True, width="stretch"
            )
        st.caption(
            "Path entities carry no normalized name or source IDs in the API; those columns are "
            "filled only for entities that were also matched from the question."
        )
    else:
        st.caption("No graph paths were returned.")

    relationships = details.get("graph_relationships", [])
    st.markdown(f"**Relationships on the retrieved graph paths ({len(relationships)})**")
    if relationships:
        st.dataframe(
            relationship_rows(relationships, entity_names(details)),
            hide_index=True,
            width="stretch",
        )
        dot = build_dot(details)
        if dot:
            st.markdown("**Graph view** (matched entities highlighted in blue)")
            st.graphviz_chart(dot, width="stretch")
    else:
        st.caption("No relationships were returned.")

    unresolved = details.get("unresolved_references", [])
    if unresolved:
        st.markdown(f"**Graph references without an original chunk ({len(unresolved)})**")
        st.dataframe(unresolved, hide_index=True, width="stretch")

    truncated = [
        n for n in details.get("truncated_lists", []) if n.startswith(("graph", "matched"))
    ]
    if truncated:
        st.caption(
            f"Capped by the backend at {details.get('max_items_per_list')} items: "
            f"{', '.join(truncated)}."
        )
