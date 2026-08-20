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
    applicability_context: dict[str, Any]
    applicability_status: str
    applicability_mismatches: tuple[str, ...]
    safe_to_reuse: bool


def recall_similar_candidates(
    campaign_id: str,
    params: dict[str, Any],
    space: Any,
    *,
    k: int = 3,
    current_context: dict[str, Any] | None = None,
) -> list[SimilarCandidate]:
    """Return the ``k`` historical candidates nearest to ``params``.

    Ordered by ascending distance (ties broken by round then index). Returns an
    empty list when the campaign has no recorded candidates (fail-open).
    """
    rows = load_all_candidates(campaign_id)
    if not rows:
        return []

    scored = []
    for row in rows:
        applicability_status, mismatches, safe_to_reuse = _compare_applicability(
            row.get("applicability_context"), current_context
        )
        scored.append(
            SimilarCandidate(
                params=row["params"],
                distance=_distance(params, row["params"], space),
                kpi=row.get("kpi_value"),
                status=row["status"],
                error=row.get("error"),
                round_number=row["round_number"],
                candidate_index=row["candidate_index"],
                applicability_context=dict(row.get("applicability_context") or {}),
                applicability_status=applicability_status,
                applicability_mismatches=mismatches,
                safe_to_reuse=safe_to_reuse,
            )
        )
    scored.sort(key=lambda c: (c.distance, c.round_number, c.candidate_index))
    return scored[:k]


_APPLICABILITY_KEYS = (
    "objective_kpi",
    "direction",
    "current_objective_level",
    "material_family",
    "active_experimental_node_id",
    "protocol_pattern_id",
    "instrument_id",
    "calibration_id",
)


def _compare_applicability(
    stored: Any, current: dict[str, Any] | None
) -> tuple[str, tuple[str, ...], bool]:
    stored_context = dict(stored) if isinstance(stored, dict) else {}
    current_context = dict(current or {})
    if not stored_context:
        return "unknown_missing_stored_context", ("stored_context",), False
    if not current_context:
        return "unknown_missing_current_context", ("current_context",), False

    missing = tuple(
        key
        for key in _APPLICABILITY_KEYS
        if (key in stored_context) != (key in current_context)
    )
    if missing:
        return "unknown_incomplete_context", missing, False

    mismatches = tuple(
        key
        for key in _APPLICABILITY_KEYS
        if key in stored_context
        and key in current_context
        and stored_context[key] != current_context[key]
    )
    if mismatches:
        return "mismatch", mismatches, False

    required = {"objective_kpi", "direction"}
    if not required.issubset(stored_context) or not required.issubset(current_context):
        missing = tuple(
            sorted(
                (required - stored_context.keys())
                | (required - current_context.keys())
            )
        )
        return "unknown_incomplete_context", missing, False
    return "compatible", (), True
