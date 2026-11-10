"""Chat conversation: message history and the per-answer evidence tabs."""

from typing import Any

import streamlit as st

from frontend.components.answer import render_answer
from frontend.components.common import escape_markdown
from frontend.components.evidence import render_evidence
from frontend.components.graph_insights import render_contribution, render_graph_insights
from frontend.components.sources import render_sources

EVIDENCE_TABS = [
    "Sources & citations",
    "How Graph Retrieval Contributed",
    "Retrieval evidence",
    "Entities & relationships",
]


def render_assistant_message(message: dict[str, Any]) -> None:
    if message.get("error"):
        st.error(message["error"])
        return
    response = message["response"]
    render_answer(response)
    sources, contribution, evidence, graph = st.tabs(EVIDENCE_TABS)
    with sources:
        render_sources(response)
    with contribution:
        render_contribution(response)
    with evidence:
        render_evidence(response)
    with graph:
        render_graph_insights(response)


def render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(escape_markdown(message["content"]))
        else:
            render_assistant_message(message)


def render_history(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        render_message(message)
