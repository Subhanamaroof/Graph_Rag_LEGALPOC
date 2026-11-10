"""HTTP client for the Graph RAG backend.

The answer workflow calls only ``POST /v1/generation/answer``: since Story 1.9.1 its response
already carries the retrieval details (vector, graph and hybrid results, graph paths, entities,
relationships and fusion counts), so no separate retrieval call is made.

Every failure is raised as ``ApiError`` with a user-facing message. Answer generation is never
retried: after a timeout the backend may still be generating, and a retry would pay for a
second Groq call.
"""

from dataclasses import dataclass
from typing import Any

import httpx

from frontend.config import FrontendSettings

ANSWER_STATUSES = frozenset(
    {"success", "partial", "insufficient_evidence", "retrieval_failed", "generation_failed"}
)
_REQUIRED_DICT_FIELDS = ("retrieval_summary", "generation_metadata")
_OPTIONAL_LIST_FIELDS = ("claims", "sources", "warnings", "errors")


class ApiError(Exception):
    """``kind`` is one of ``connection``, ``timeout``, ``http``, ``invalid_json``, ``schema``."""

    def __init__(self, kind: str, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class HealthStatus:
    connected: bool
    message: str
    application: str | None = None


def validate_answer_response(data: Any) -> dict[str, Any]:
    """Check the fields the UI depends on; optional fields may be missing."""
    if not isinstance(data, dict):
        raise ApiError("schema", "The backend returned an unexpected response (not an object).")
    if data.get("status") not in ANSWER_STATUSES:
        raise ApiError("schema", f"The backend returned an unknown status: {data.get('status')!r}.")
    for field in _REQUIRED_DICT_FIELDS:
        if not isinstance(data.get(field), dict):
            raise ApiError("schema", f"The backend response is missing '{field}'.")
    for field in _OPTIONAL_LIST_FIELDS:
        if field in data and not isinstance(data[field], list):
            raise ApiError("schema", f"The backend response field '{field}' is not a list.")
    answer = data.get("answer")
    if answer is not None and not isinstance(answer, str):
        raise ApiError("schema", "The backend response field 'answer' is not text.")
    details = data.get("retrieval_details")
    if details is not None and not isinstance(details, dict):
        raise ApiError("schema", "The backend response field 'retrieval_details' is invalid.")
    return data


def _http_error_message(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, list):  # FastAPI validation errors
        detail = "; ".join(str(item.get("msg", item)) for item in detail if item)
    if response.status_code == 422:
        return f"The question was rejected by the backend: {detail or 'invalid request'}."
    suffix = f": {detail}" if detail else "."
    return f"The backend returned HTTP {response.status_code}{suffix}"


class GraphRagApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        health_timeout: float = 5.0,
        answer_timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.health_timeout = health_timeout
        self.answer_timeout = answer_timeout
        self._transport = transport  # injected in tests (httpx.MockTransport)

    @classmethod
    def from_settings(cls, settings: FrontendSettings) -> "GraphRagApiClient":
        return cls(
            settings.api_base_url,
            health_timeout=settings.health_timeout_seconds,
            answer_timeout=settings.answer_timeout_seconds,
        )

    def _request(self, method: str, path: str, *, timeout: float, json: dict | None = None) -> Any:
        try:
            with httpx.Client(
                base_url=self.base_url, timeout=timeout, transport=self._transport
            ) as client:
                response = client.request(method, path, json=json)
        except httpx.TimeoutException as exc:
            raise ApiError(
                "timeout",
                f"The backend did not respond within {timeout:g} seconds. The request was not "
                "retried; the server may still be processing it.",
            ) from exc
        except httpx.RequestError as exc:
            raise ApiError(
                "connection",
                f"Cannot reach the backend at {self.base_url}. Is the FastAPI server running?",
            ) from exc
        if response.status_code >= 400:
            raise ApiError("http", _http_error_message(response), response.status_code)
        try:
            return response.json()
        except ValueError as exc:
            raise ApiError(
                "invalid_json", "The backend returned a response that is not JSON."
            ) from exc

    def check_health(self) -> HealthStatus:
        """``GET /health`` — a liveness check that touches no database or LLM."""
        try:
            data = self._request("GET", "/health", timeout=self.health_timeout)
        except ApiError as exc:
            return HealthStatus(connected=False, message=exc.message)
        if not isinstance(data, dict) or data.get("status") != "ok":
            return HealthStatus(False, "The backend responded but did not report status 'ok'.")
        return HealthStatus(True, "Connected", data.get("application"))

    def generate_answer(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        """``POST /v1/generation/answer`` — one retrieval and at most one LLM call."""
        payload: dict[str, Any] = {"question": question, "include_sources": True}
        if top_k is not None:
            payload["top_k"] = top_k
        data = self._request(
            "POST", "/v1/generation/answer", timeout=self.answer_timeout, json=payload
        )
        return validate_answer_response(data)
