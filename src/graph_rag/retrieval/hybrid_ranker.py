"""Reciprocal Rank Fusion (Story 1.8).

    RRF(d) = sum over ranked lists r containing d of  1 / (k + rank_r(d))

``rank_r(d)`` is the 1-based position of ``d`` in list ``r``. Only positions are used, so a
cosine similarity and a graph path score — which live on different scales — are never added
or compared. A chunk found by both lists collects two terms and so tends to rise. ``k``
(default 60) damps the gap between the first few positions.

RRF orders retrieval evidence; it says nothing about legal authority or correctness.
This module has no database or retrieval code, so it can be tested on plain lists.
"""

from collections.abc import Hashable, Sequence


def _check_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("RRF k must be a positive integer")


def rrf_term(rank: int, k: int) -> float:
    """The contribution of one list in which the item sits at 1-based ``rank``."""
    _check_k(k)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("ranks are 1-based positive integers")
    return 1.0 / (k + rank)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Hashable]], k: int = 60
) -> list[tuple[Hashable, float]]:
    """Fuse ranked lists of item keys into one list of ``(key, rrf_score)``, best first.

    Ties are broken by the best single-list position, then by the key itself, so the order is
    reproducible. Each list must list an item at most once.
    """
    _check_k(k)
    scores: dict[Hashable, float] = {}
    best_rank: dict[Hashable, int] = {}
    for ranked in ranked_lists:
        if len(set(ranked)) != len(ranked):
            raise ValueError("a ranked list must not contain the same item twice")
        for position, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + rrf_term(position, k)
            best_rank[key] = min(best_rank.get(key, position), position)
    return sorted(
        scores.items(), key=lambda item: (-item[1], best_rank[item[0]], _sort_key(item[0]))
    )


def _sort_key(key: Hashable) -> tuple:
    """Comparable form of a key; ``None`` parts (e.g. a missing document_id) sort first."""
    parts = key if isinstance(key, tuple) else (key,)
    return tuple((part is not None, str(part) if part is not None else "") for part in parts)
