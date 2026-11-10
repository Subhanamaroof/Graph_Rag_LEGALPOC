"""Validate LLM citations against the evidence actually retrieved (Story 1.9).

The prompt asks the LLM to cite correctly; this module makes sure it did. A citation is valid
only if its exact ``(document_id, chunk_id)`` pair is one of the evidence sources given to
the LLM. Invalid citations are removed and reported, never repaired or replaced. A claim left
with no valid citation is rejected. Inline ``[document_id, chunk_id]`` markers in the answer
text get the same check, so the final text cannot point to a chunk that was not retrieved.
"""

import re
from dataclasses import dataclass, field

from graph_rag.generation.answer_generator import RawAnswer, RawCitation
from graph_rag.generation.generation_models import (
    Citation,
    Claim,
    InvalidCitation,
    RejectedClaim,
)

# "[DOC, CHUNK]": two comma-separated identifiers without spaces, so ordinary bracketed prose
# such as "[Section 138, NI Act]" is not mistaken for a citation.
INLINE_CITATION = re.compile(r"\[\s*([^\[\],\s]+)\s*,\s*([^\[\],\s]+)\s*\]")


@dataclass
class ValidatedAnswer:
    answer: str
    claims: list[Claim] = field(default_factory=list)
    invalid_citations: list[InvalidCitation] = field(default_factory=list)
    rejected_claims: list[RejectedClaim] = field(default_factory=list)

    @property
    def cited_keys(self) -> set[tuple[str, str]]:
        return {(c.document_id, c.chunk_id) for claim in self.claims for c in claim.citations}


def _check(citation: RawCitation, evidence: set[tuple[str, str]]) -> InvalidCitation | None:
    reason = None
    if not citation.document_id:
        reason = "missing_document_id"
    elif not citation.chunk_id:
        reason = "missing_chunk_id"
    elif (citation.document_id, citation.chunk_id) not in evidence:
        reason = "not_in_evidence"
    if reason is None:
        return None
    return InvalidCitation(
        document_id=citation.document_id,
        chunk_id=citation.chunk_id,
        reason=reason,
        location="claim",
    )


def validate_claims(raw: RawAnswer, evidence: set[tuple[str, str]]) -> ValidatedAnswer:
    result = ValidatedAnswer(answer=raw.answer)
    for raw_claim in raw.claims:
        valid: list[Citation] = []
        for raw_citation in raw_claim.citations:
            problem = _check(raw_citation, evidence)
            if problem:
                result.invalid_citations.append(problem)
                continue
            citation = Citation(
                document_id=raw_citation.document_id, chunk_id=raw_citation.chunk_id
            )
            if citation not in valid:  # the same chunk cited twice counts once
                valid.append(citation)
        if valid:
            result.claims.append(Claim(text=raw_claim.text, citations=valid))
        else:
            reason = "all_citations_invalid" if raw_claim.citations else "no_citations"
            result.rejected_claims.append(RejectedClaim(text=raw_claim.text, reason=reason))
    result.answer = strip_invalid_inline_citations(raw.answer, evidence, result.invalid_citations)
    return result


def strip_invalid_inline_citations(
    text: str, evidence: set[tuple[str, str]], report: list[InvalidCitation]
) -> str:
    """Remove ``[doc, chunk]`` markers that do not match retrieved evidence."""

    def replace(match: re.Match[str]) -> str:
        key = (match.group(1), match.group(2))
        if key in evidence:
            return match.group(0)
        report.append(
            InvalidCitation(
                document_id=key[0],
                chunk_id=key[1],
                reason="not_in_evidence",
                location="answer_text",
            )
        )
        return ""

    cleaned = INLINE_CITATION.sub(replace, text)
    return re.sub(r"[ \t]+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", cleaned)).strip()
