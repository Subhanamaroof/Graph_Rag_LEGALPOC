"""Extract entities and relationships from one chunk with Groq, then validate them (Story 1.5).

Flow for one chunk:

1. One LLM call returns entities and relationships together as JSON.
2. ``parse_llm_output`` checks the top-level shape (malformed JSON -> the chunk fails).
3. ``build_chunk_extraction`` validates every item on its own. An item is rejected, with a
   warning, when its type is unknown, its name or evidence is not in the chunk text, or it
   references an unknown entity. Valid items survive.
4. Duplicate mentions (same type + normalized name) are merged into one entity.
5. Document and chunk provenance are attached and deterministic IDs are assigned.
"""

import hashlib
import json
import re

from pydantic import ValidationError

from graph_rag.core.llm import JSONCompletionClient, LLMError
from graph_rag.graph.models import (
    ChunkExtraction,
    Entity,
    RawEntity,
    RawExtraction,
    RawRelationship,
    Relationship,
)
from graph_rag.graph.prompts import SYSTEM_PROMPT, build_user_prompt


class ExtractionParseError(ValueError):
    """The LLM response is not a JSON object with ``entities`` and ``relationships`` lists."""


_QUOTE_TRANSLATION = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def normalize_for_match(text: str) -> str:
    """Case-, whitespace- and quote-insensitive form used to check that a quote is in the text."""
    return " ".join(text.translate(_QUOTE_TRANSLATION).casefold().split())


def clean_name(name: str) -> str:
    """Collapse internal whitespace and drop trailing punctuation; the meaning is unchanged."""
    return " ".join(name.split()).strip(" ,;:")


def is_grounded(quote: str | None, normalized_chunk: str) -> bool:
    if not quote:
        return False
    normalized_quote = normalize_for_match(quote).strip("\"' ")
    return bool(normalized_quote) and normalized_quote in normalized_chunk


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def parse_llm_output(raw: str) -> RawExtraction:
    text = _FENCE.sub("", raw.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionParseError(f"LLM response is not valid JSON ({exc.msg}).") from exc
    try:
        return RawExtraction.model_validate(data)
    except ValidationError as exc:
        raise ExtractionParseError(
            "LLM response must be an object with 'entities' and 'relationships' lists."
        ) from exc


def _first_error(exc: ValidationError) -> str:
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error["loc"]) or "item"
    return f"{location}: {error['msg']}"


def build_chunk_extraction(
    raw: RawExtraction, *, document_id: str, chunk_id: str, chunk_text: str
) -> ChunkExtraction:
    """Turn the parsed LLM output into a validated ``ChunkExtraction``."""
    normalized_chunk = normalize_for_match(chunk_text)
    warnings: list[str] = []
    entities: dict[str, Entity] = {}  # entity_id -> entity, in first-mention order
    local_to_entity_id: dict[str, str] = {}  # LLM's "e1" -> our entity_id

    for index, item in enumerate(raw.entities):
        try:
            candidate = RawEntity.model_validate(item)
        except ValidationError as exc:
            warnings.append(f"entity[{index}] rejected: {_first_error(exc)}")
            continue
        name = clean_name(candidate.name)
        if not is_grounded(name, normalized_chunk):
            warnings.append(f"entity[{index}] '{name}' rejected: name not found in chunk text")
            continue
        if not is_grounded(candidate.evidence, normalized_chunk):
            warnings.append(f"entity[{index}] '{name}' rejected: evidence missing or not in chunk")
            continue
        normalized_name = clean_name(candidate.normalized_name or "") or name
        entity_id = _stable_id(
            "ent",
            document_id,
            chunk_id,
            candidate.entity_type.upper(),
            normalized_name.casefold(),
        )
        if entity_id in entities:  # duplicate mention of an entity already extracted
            existing = entities[entity_id]
            if name != existing.name and name not in existing.aliases:
                existing.aliases.append(name)
            local_to_entity_id[candidate.id] = entity_id
            continue
        try:
            entity = Entity(
                entity_id=entity_id,
                name=name,
                normalized_name=normalized_name,
                entity_type=candidate.entity_type.upper(),
                description=candidate.description or None,
                source_document_id=document_id,
                source_chunk_id=chunk_id,
                source_text=candidate.evidence,
            )
        except ValidationError as exc:
            warnings.append(f"entity[{index}] '{name}' rejected: {_first_error(exc)}")
            continue
        entities[entity_id] = entity
        local_to_entity_id[candidate.id] = entity_id

    relationships: dict[str, Relationship] = {}
    for index, item in enumerate(raw.relationships):
        try:
            candidate = RawRelationship.model_validate(item)
        except ValidationError as exc:
            warnings.append(f"relationship[{index}] rejected: {_first_error(exc)}")
            continue
        source_id = local_to_entity_id.get(candidate.source)
        target_id = local_to_entity_id.get(candidate.target)
        if source_id is None or target_id is None:
            warnings.append(f"relationship[{index}] rejected: references an unknown entity")
            continue
        if not is_grounded(candidate.evidence, normalized_chunk):
            warnings.append(f"relationship[{index}] rejected: evidence missing or not in chunk")
            continue
        relationship_type = candidate.relationship_type.upper()
        relationship_id = _stable_id("rel", chunk_id, source_id, relationship_type, target_id)
        if relationship_id in relationships:
            continue  # the same fact stated twice
        try:
            relationships[relationship_id] = Relationship(
                relationship_id=relationship_id,
                source_entity_id=source_id,
                target_entity_id=target_id,
                relationship_type=relationship_type,
                description=candidate.description or None,
                evidence=candidate.evidence,
                source_document_id=document_id,
                source_chunk_id=chunk_id,
            )
        except ValidationError as exc:
            warnings.append(f"relationship[{index}] rejected: {_first_error(exc)}")

    return ChunkExtraction(
        document_id=document_id,
        chunk_id=chunk_id,
        entities=list(entities.values()),
        relationships=list(relationships.values()),
        warnings=warnings,
        extraction_status="completed",
    )


def failed_chunk(document_id: str, chunk_id: str, error: str) -> ChunkExtraction:
    return ChunkExtraction(
        document_id=document_id, chunk_id=chunk_id, extraction_status="failed", error=error
    )


def extract_from_chunk(
    llm: JSONCompletionClient, *, document_id: str, chunk_id: str, chunk_text: str
) -> ChunkExtraction:
    """Run the whole flow for one chunk. Never raises: failures become a ``failed`` result."""
    if not chunk_text.strip():
        return ChunkExtraction(
            document_id=document_id,
            chunk_id=chunk_id,
            warnings=["chunk has no text"],
            extraction_status="completed",
        )
    try:
        raw_output = llm.complete_json(SYSTEM_PROMPT, build_user_prompt(chunk_text))
        parsed = parse_llm_output(raw_output)
    except (LLMError, ExtractionParseError) as exc:
        return failed_chunk(document_id, chunk_id, str(exc))
    return build_chunk_extraction(
        parsed, document_id=document_id, chunk_id=chunk_id, chunk_text=chunk_text
    )
