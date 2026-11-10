"""Legal Graph RAG — Streamlit frontend (Story 1.10).

Run from the project root (FastAPI must be running separately):

    uv run streamlit run frontend/app.py

Each question is sent once to ``POST /v1/generation/answer``; the answer, citations and the
retrieval details are rendered from that single response.
"""

import html
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # `streamlit run` only puts frontend/ on sys.path
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from frontend import state  # noqa: E402
from frontend.api.client import ApiError, GraphRagApiClient, HealthStatus  # noqa: E402
from frontend.components.chat import render_assistant_message, render_history  # noqa: E402
from frontend.components.common import escape_markdown  # noqa: E402
from frontend.config import MAX_UI_TOP_K, FrontendSettings, get_settings  # noqa: E402

CSS_PATH = Path(__file__).parent / "styles" / "custom.css"


def render_header(health: HealthStatus) -> None:
    pill = (
        '<span class="status-pill connected">● Backend connected</span>'
        if health.connected
        else '<span class="status-pill disconnected">● Backend unavailable</span>'
    )
    st.markdown(
        '<div class="app-header"><div><h1>Legal Graph RAG</h1>'
        "<p>AI-powered legal judgment research assistant</p></div>"
        f"{pill}</div>",
        unsafe_allow_html=True,
    )
    if not health.connected:
        st.error(health.message)


def render_sidebar(health: HealthStatus, settings: FrontendSettings) -> None:
    ss = st.session_state
    with st.sidebar:
        st.markdown("### Legal Graph RAG")
        st.caption(
            "Answers are generated only from retrieved judgment passages. Every answer shows "
            "which chunks were found by vector search, by the knowledge graph, or by both."
        )
        cols = st.columns(2)
        cols[0].button(
            "New Chat",
            on_click=state.new_chat,
            args=(ss, settings.default_top_k),
            width="stretch",
        )
        cols[1].button("Clear Chat", on_click=state.clear_chat, args=(ss,), width="stretch")

        st.markdown("#### Retrieval settings")
        st.slider(
            "Evidence chunks (top_k)",
            min_value=1,
            max_value=MAX_UI_TOP_K,
            key="top_k",
            help="Number of fused hybrid results offered to the answer model.",
        )

        st.markdown("#### Backend")
        if health.connected:
            st.success(f"Connected to {health.application or 'backend'}")
        else:
            st.error("Not connected")
        st.caption(f"API: {html.escape(settings.api_base_url)}")
        st.caption(
            f"Session {ss.conversation_id} · {len(ss.messages)} message(s). "
            "History is kept in this browser session only; each question is answered "
            "independently."
        )


def handle_question(question: str, client: GraphRagApiClient) -> None:
    ss = st.session_state
    state.add_user_message(ss, question)
    with st.chat_message("user"):
        st.markdown(escape_markdown(question))
    with st.chat_message("assistant"):
        ss.is_generating = True
        try:
            with st.spinner("Retrieving evidence and generating a grounded answer…"):
                response = client.generate_answer(question, top_k=ss.top_k)
            message = state.add_assistant_message(ss, response=response)
        except ApiError as exc:
            message = state.add_assistant_message(ss, error=exc.message)
        finally:
            ss.is_generating = False
        render_assistant_message(message)


def main() -> None:
    st.set_page_config(page_title="Legal Graph RAG", page_icon="⚖️", layout="wide")
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

    settings = get_settings()
    client = GraphRagApiClient.from_settings(settings)
    state.init_state(st.session_state, settings.default_top_k)
    health = client.check_health()

    render_sidebar(health, settings)
    render_header(health)
    if not st.session_state.messages:
        st.caption(
            "Ask a question about the judgments in the corpus, e.g. which statutes a case "
            "applied or which earlier judgments it cited."
        )
    render_history(st.session_state.messages)

    question = st.chat_input(
        "Ask a legal research question…", disabled=st.session_state.is_generating
    )
    if question and question.strip():
        handle_question(question.strip(), client)


main()
