"""Minimal Groq chat client that returns a JSON string (Story 1.5).

Only one operation is needed so far: send a system + user prompt and get back the model's
JSON text. Parsing and validation are the caller's job, so the LLM is never trusted blindly.
"""

from typing import Any, Protocol

import groq

from graph_rag.core.config import Settings


class LLMError(Exception):
    """The LLM call failed or returned nothing usable.

    ``retryable`` marks transient failures (rate limit, server error, network) that may succeed
    if repeated later; ``retry_after`` is the server's suggested wait in seconds, if it sent one.
    """

    def __init__(self, message: str, *, retryable: bool = False, retry_after: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


class LLMNotConfiguredError(LLMError):
    """GROQ_API_KEY or GROQ_MODEL_NAME is missing."""


class JSONCompletionClient(Protocol):
    """Anything that can turn a prompt into JSON text (the real Groq client or a test fake)."""

    def complete_json(self, system_prompt: str, user_prompt: str) -> str: ...


class GroqJSONClient:
    """Calls Groq's chat completions API in JSON mode with temperature 0."""

    def __init__(self, api_key: str, model: str, timeout: float, client: Any | None = None):
        self.model = model
        self._client = client or groq.Groq(api_key=api_key, timeout=timeout)

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
        except groq.APIStatusError as exc:
            raise LLMError(
                f"Groq returned HTTP {exc.status_code}.",
                retryable=exc.status_code == 429 or exc.status_code >= 500,
                retry_after=_retry_after_seconds(exc.response),
            ) from exc
        except groq.APIConnectionError as exc:  # connection errors, timeouts
            raise LLMError(f"Groq request failed ({type(exc).__name__}).", retryable=True) from exc
        except groq.APIError as exc:
            raise LLMError(f"Groq request failed ({type(exc).__name__}).") from exc
        content = completion.choices[0].message.content if completion.choices else None
        if not content or not content.strip():
            raise LLMError("Groq returned an empty response.")
        return content


def _retry_after_seconds(response: Any) -> float | None:
    """The ``retry-after`` header in seconds, if the server sent a numeric one."""
    try:
        return max(0.0, float(response.headers["retry-after"]))
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def create_groq_client(settings: Settings) -> GroqJSONClient:
    """Build the client from settings; ``LLMNotConfiguredError`` if credentials are missing."""
    if settings.groq_api_key is None or not settings.groq_model_name:
        raise LLMNotConfiguredError(
            "Groq is not configured. Set GROQ_API_KEY and GROQ_MODEL_NAME (see .env.example)."
        )
    return GroqJSONClient(
        settings.groq_api_key.get_secret_value(),
        settings.groq_model_name,
        settings.groq_timeout_seconds,
    )
