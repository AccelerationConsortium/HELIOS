"""Candidate pool memory: recall of similar historical candidates.

The orchestrator already *persists* every executed candidate (params, outcome,
status, failure reason) in the ``campaign_candidates`` table. What was missing
is the *read* side: given a proposed point, what similar points has this
campaign already tried, and how did they turn out?

This module supplies that recall. It is read-only and fail-open: with no
history it returns an empty list, so callers can attach "evidence: similar
runs" to a decision trace without changing any decision behaviour.

Distance reuses :func:`app.optimization.candidate_pool._distance` so similarity
is computed exactly the same way the live pool scores diversity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.optimization.candidate_pool import _distance
from app.services.campaign_state import load_all_candidates


@dataclass(frozen=True)
class SimilarCandidate:
    """A historical candidate near a query point, with its recorded outcome."""

    params: dict[str, Any]
    distance: float
    kpi: float | None
    status: str
    error: str | None
    round_number: int
    candidate_index: int


def recall_similar_candidates(
    campaign_id: str,
    params: dict[str, Any],
    space: Any,
    *,
    k: int = 3,
) -> list[SimilarCandidate]:
    """Return the ``k`` historical candidates nearest to ``params``.

    Ordered by ascending distance (ties broken by round then index). Returns an
    empty list when the campaign has no recorded candidates (fail-open).
    """
    rows = load_all_candidates(campaign_id)
    if not rows:
        return []

    scored = [
        SimilarCandidate(
            params=row["params"],
            distance=_distance(params, row["params"], space),
            kpi=row.get("kpi_value"),
            status=row["status"],
            error=row.get("error"),
            round_number=row["round_number"],
            candidate_index=row["candidate_index"],
        )
        for row in rows
    ]
    scored.sort(key=lambda c: (c.distance, c.round_number, c.candidate_index))
    return scored[:k]
