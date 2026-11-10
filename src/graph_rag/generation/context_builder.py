"""Turn Story 1.8 hybrid results into numbered, source-labelled LLM context (Story 1.9).

Rules:
- Story 1.8's ranking is the ordering; nothing is re-ranked.
- A result needs both ``document_id`` and ``chunk_id`` (a citation needs both) and non-blank
  text; others are excluded and counted.
- Duplicates by ``(document_id, chunk_id)`` keep the first (best-ranked) occurrence.
- Chunk text is copied verbatim. A chunk is never cut: chunks are added whole, in rank order,
  until the next one would exceed the character budget.
- Only a few metadata fields help an answer (title, court, date, section); the rest is
  dropped. Graph paths are shown as retrieval hints, not as citable text.
"""

from dataclasses import dataclass, field

from graph_rag.generation.generation_models import Source
from graph_rag.retrieval.hybrid_retrieval_models import HybridResult

CONTEXT_METADATA_FIELDS = ("title", "court", "date", "section")
MAX_GRAPH_PATHS_PER_SOURCE = 3


@dataclass
class PreparedContext:
    sources: list[Source] = field(default_factory=list)
    text: str = ""
    duplicates_removed: int = 0
    excluded_missing_identifier: int = 0
    excluded_empty_content: int = 0
    excluded_over_budget: int = 0

    @property
    def keys(self) -> set[tuple[str, str]]:
        return {(source.document_id, source.chunk_id) for source in self.sources}


def _retrieval_label(sources: list[str]) -> str:
    if "vector" in sources and "graph" in sources:
        return "vector_and_graph"
    return sources[0] if sources else "vector"


def _to_source(number: int, result: HybridResult) -> Source:
    return Source(
        source_number=number,
        document_id=result.document_id or "",
        chunk_id=result.chunk_id,
        rank=result.rank,
        retrieval_sources=result.retrieval_sources,
        retrieval_source=_retrieval_label(result.retrieval_sources),
        rrf_score=result.rrf_score,
        vector_rank=result.vector_rank,
        graph_rank=result.graph_rank,
        content=result.content,
        metadata={k: result.metadata[k] for k in CONTEXT_METADATA_FIELDS if k in result.metadata},
    )


def format_source(source: Source, result: HybridResult) -> str:
    """One evidence block. Identifiers come from retrieval, never from the LLM."""
    lines = [
        f"[SOURCE {source.source_number}]",
        f"Document ID: {source.document_id}",
        f"Chunk ID: {source.chunk_id}",
        f"Retrieved by: {source.retrieval_source.replace('_', ' ')}",
    ]
    lines += [f"{key.capitalize()}: {value}" for key, value in source.metadata.items()]
    if result.graph_context and result.graph_context.paths:
        paths = [p.path for p in result.graph_context.paths[:MAX_GRAPH_PATHS_PER_SOURCE]]
        lines.append("Graph connections (retrieval hints, not evidence): " + " | ".join(paths))
    lines += ["Content:", source.content, f"[END SOURCE {source.source_number}]"]
    return "\n".join(lines)


def build_context(results: list[HybridResult], max_chars: int) -> PreparedContext:
    context = PreparedContext()
    seen: set[tuple[str, str]] = set()
    blocks: list[str] = []
    used = 0
    for result in results:  # already in Story 1.8 rank order
        if not result.document_id or not result.chunk_id:
            context.excluded_missing_identifier += 1
            continue
        key = (result.document_id, result.chunk_id)
        if key in seen:
            context.duplicates_removed += 1
            continue
        if not result.content or not result.content.strip():
            context.excluded_empty_content += 1
            continue
        if context.excluded_over_budget:  # budget already reached: keep the ranking a prefix
            context.excluded_over_budget += 1
            continue
        source = _to_source(len(context.sources) + 1, result)
        block = format_source(source, result)
        separator = 2 if blocks else 0
        if used + separator + len(block) > max_chars:
            context.excluded_over_budget += 1
            continue
        seen.add(key)
        blocks.append(block)
        used += separator + len(block)
        context.sources.append(source)
    context.text = "\n\n".join(blocks)
    return context
