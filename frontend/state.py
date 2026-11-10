"""Session state for the current browser session (Story 1.10).

Works on any mutable mapping, so it is tested with a plain dict and used with
``st.session_state``. Nothing is persisted: closing the tab or starting a new chat discards the
conversation. Each question is answered independently; no history is sent to the backend.
"""

import uuid
from collections.abc import MutableMapping
from typing import Any


def init_state(state: MutableMapping[str, Any], default_top_k: int) -> None:
    state.setdefault("messages", [])
    state.setdefault("conversation_id", uuid.uuid4().hex[:8])
    state.setdefault("is_generating", False)
    state.setdefault("top_k", default_top_k)


def add_user_message(state: MutableMapping[str, Any], question: str) -> dict[str, Any]:
    message = {"role": "user", "content": question}
    state["messages"].append(message)
    return message


def add_assistant_message(
    state: MutableMapping[str, Any],
    *,
    response: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Store either the backend response or the error message shown instead of it."""
    message = {"role": "assistant", "response": response, "error": error}
    state["messages"].append(message)
    return message


def clear_chat(state: MutableMapping[str, Any]) -> None:
    """Remove the messages; keep the conversation and the settings."""
    state["messages"] = []
    state["is_generating"] = False


def new_chat(state: MutableMapping[str, Any], default_top_k: int) -> None:
    """Start over: no messages, a new conversation id and default settings."""
    clear_chat(state)
    state["conversation_id"] = uuid.uuid4().hex[:8]
    state["top_k"] = default_top_k
