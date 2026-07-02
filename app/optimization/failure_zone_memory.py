"""Cross-campaign failure-zone memory: a read-only prior over failed regions.

Each campaign already records its failed candidates (coordinate + error) in the
``campaign_candidates`` table. What was missing is the ability to learn from
*other* campaigns: given a proposed point, which parameter regions have failed
before — in this lab, on other campaigns — and why?

This module supplies that recall. It is read-only and fail-open: with no
failure history it returns an empty list, so callers can attach failure-region
evidence (or, in a later phase, a soft prior) without changing any decision
behaviour. The current campaign is excluded by default so a campaign does not
treat its own in-progress failures as external priors unless asked to.

Distance reuses :func:`app.optimization.candidate_pool._distance`, the same
similarity used by the live candidate pool, so failure proximity is measured
consistently with diversity scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.optimization.candidate_pool import _distance
from app.services.campaign_state import load_failed_candidates


@dataclass(frozen=True)
class FailureZone:
    """A historical failed point near a query point, with its recorded reason."""

    params: dict[str, Any]
    distance: float
    error: str | None
    campaign_id: str
    round_number: int
    candidate_index: int


def recall_failure_zones(
    campaign_id: str,
    params: dict[str, Any],
    space: Any,
    *,
    k: int = 3,
    include_current: bool = False,
) -> list[FailureZone]:
    """Return the ``k`` historical failed points nearest to ``params``.

    Drawn from failed candidates across *other* campaigns by default; pass
    ``include_current=True`` to also consider the current campaign's failures.
    Ordered by ascending distance (ties broken by campaign, round, index).
    Returns an empty list when no failure history exists (fail-open).
    """
    exclude = None if include_current else campaign_id
    rows = load_failed_candidates(exclude_campaign_id=exclude)
    if not rows:
        return []

    zones = [
        FailureZone(
            params=row["params"],
            distance=_distance(params, row["params"], space),
            error=row.get("error"),
            campaign_id=row["campaign_id"],
            round_number=row["round_number"],
            candidate_index=row["candidate_index"],
        )
        for row in rows
    ]
    zones.sort(
        key=lambda z: (z.distance, z.campaign_id, z.round_number, z.candidate_index)
    )
    return zones[:k]
