"""Call Groq for a grounded answer and parse its structured output (Story 1.9).

The LLM output is parsed into ``RawAnswer``, a deliberately loose shape: citation fields may
be missing so that a bad citation is *reported* by the citation validator instead of failing
the whole answer. Anything that is not this shape is a malformed response.
"""

import json

from pydantic import BaseModel, Field, ValidationError, field_validator

from graph_rag.core.llm import JSONCompletionClient
from graph_rag.generation.prompts import SYSTEM_PROMPT, build_user_prompt


class AnswerParseError(Exception):
    """The LLM response is not valid JSON in the expected answer shape."""


class RawCitation(BaseModel):
    document_id: str | None = None
    chunk_id: str | None = None

    @field_validator("document_id", "chunk_id", mode="before")
    @classmethod
    def _blank_is_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        if isinstance(value, int | float):  # tolerate numeric IDs; validated against evidence
            return str(value)
        return value.strip() if isinstance(value, str) else value


class RawClaim(BaseModel):
    text: str = Field(min_length=1)
    citations: list[RawCitation] = Field(default_factory=list)


class RawAnswer(BaseModel):
    answer: str = Field(min_length=1)
    claims: list[RawClaim] = Field(default_factory=list)
    insufficient_evidence: bool = False


def parse_answer(raw: str) -> RawAnswer:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AnswerParseError(f"LLM response is not valid JSON ({exc.msg}).") from exc
    if not isinstance(data, dict):
        raise AnswerParseError("LLM response must be a JSON object.")
    try:
        answer = RawAnswer.model_validate(data)
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in err["loc"]) for err in exc.errors()})
        raise AnswerParseError(f"LLM response has invalid fields: {', '.join(fields)}.") from exc
    if not answer.answer.strip():
        raise AnswerParseError("LLM response has an empty answer.")
    return answer


def generate_raw_answer(
    llm: JSONCompletionClient, question: str, context: str, *, partial_retrieval: bool
) -> RawAnswer:
    """One Groq call (JSON mode, temperature 0). Raises ``LLMError`` or ``AnswerParseError``."""
    raw = llm.complete_json(
        SYSTEM_PROMPT, build_user_prompt(question, context, partial_retrieval=partial_retrieval)
    )
    return parse_answer(raw)
