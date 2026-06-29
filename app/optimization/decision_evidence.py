"""Phase ⑤: attach memory recall to a decision as structured evidence.

This layer is *evidence only*. It consumes the read-only memory layers --
:func:`recall_similar_candidates` (within-campaign history) and
:func:`recall_failure_zones` (cross-campaign failed regions) -- and packages
what they return so it can be attached to the existing provenance record.

It never participates in ranking, gating, or selection. Recall is fail-open at
two levels: each recall call is guarded, and an empty result yields no evidence
so callers attach nothing when there is no history.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.optimization.candidate_memory import (
    SimilarCandidate,
    recall_similar_candidates,
)
from app.optimization.failure_zone_memory import (
    FailureZone,
    recall_failure_zones,
)
from app.services.candidate_gen import space_from_dimensions


@dataclass(frozen=True)
class DecisionEvidence:
    """Recall evidence for a single proposed point."""

    params: dict[str, Any]
    similar_candidates: list[SimilarCandidate]
    failure_zones: list[FailureZone]

    def is_empty(self) -> bool:
        return not self.similar_candidates and not self.failure_zones

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "similar_candidates": [
                {
                    "params": s.params,
                    "distance": s.distance,
                    "kpi": s.kpi,
                    "status": s.status,
                    "error": s.error,
                    "round_number": s.round_number,
                    "candidate_index": s.candidate_index,
                }
                for s in self.similar_candidates
            ],
            "failure_zones": [
                {
                    "params": z.params,
                    "distance": z.distance,
                    "error": z.error,
                    "campaign_id": z.campaign_id,
                    "round_number": z.round_number,
                    "candidate_index": z.candidate_index,
                }
                for z in self.failure_zones
            ],
        }


def build_decision_evidence(
    campaign_id: str,
    params: dict[str, Any],
    space: Any,
    *,
    k_similar: int = 3,
    k_failures: int = 3,
    include_current_failures: bool = False,
) -> DecisionEvidence:
    """Recall similar candidates + nearby failure zones for one point (fail-open)."""
    try:
        similar = recall_similar_candidates(campaign_id, params, space, k=k_similar)
    except Exception:
        similar = []
    try:
        zones = recall_failure_zones(
            campaign_id,
            params,
            space,
            k=k_failures,
            include_current=include_current_failures,
        )
    except Exception:
        zones = []
    return DecisionEvidence(
        params=params, similar_candidates=similar, failure_zones=zones
    )


def evidence_for_decision(
    request: Any,
    decision: Any,
    *,
    k_similar: int = 3,
    k_failures: int = 3,
) -> dict[str, Any] | None:
    """Build structured evidence for a decision's accepted candidates.

    Returns ``{"candidates": [...]}`` with one entry per accepted candidate that
    has any recalled evidence, or ``None`` when there is nothing to attach.
    """
    items: list[dict[str, Any]] = []
    for params in decision.final_candidates:
        ev = build_decision_evidence(
            request.campaign_id,
            params,
            request.space,
            k_similar=k_similar,
            k_failures=k_failures,
        )
        if not ev.is_empty():
            items.append(ev.to_dict())
    return {"candidates": items} if items else None


def evidence_for_candidates(
    campaign_id: str,
    candidates: list[dict[str, Any]],
    dimensions: list[dict[str, Any]],
    protocol_template: dict[str, Any] | None = None,
    *,
    k_similar: int = 3,
    k_failures: int = 3,
) -> dict[str, Any] | None:
    """Build structured evidence for a round's candidates (production wiring).

    Takes raw candidate param dicts and raw dimension dicts (as the orchestrator
    holds them), builds the search space, and returns ``{"candidates": [...]}``
    with one entry per candidate that has recalled evidence, or ``None`` when
    there is nothing to attach. Fail-open: any error yields ``None``.
    """
    try:
        space = space_from_dimensions(dimensions, protocol_template)
    except Exception:
        return None

    items: list[dict[str, Any]] = []
    for params in candidates:
        ev = build_decision_evidence(
            campaign_id,
            params,
            space,
            k_similar=k_similar,
            k_failures=k_failures,
        )
        if not ev.is_empty():
            items.append(ev.to_dict())
    return {"candidates": items} if items else None
