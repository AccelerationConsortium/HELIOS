"""Campaign mode transition table — shadow-only scientific-activity classifier.

This module formalizes the per-round "what kind of scientific activity does the
campaign need next?" question into a deterministic ``CampaignMode`` decision.
It consumes the evolving ``ObjectiveState`` (Phase 1) and the distributional
``FailureAttributionDistribution`` (Phase 2) and applies an explicit, ordered
transition table.

Boundaries (intentional):

* ``CampaignMode`` is a *mode* vocabulary, distinct from the existing
  ``CampaignDecisionAction`` and from ``ObjectiveState.stop_recommended``. The
  ``STOP_RECOMMENDED`` mode merely *reflects* an objective-state stop signal; it
  does not stop anything and does not map onto ``CampaignDecisionAction``.
* Decisions are ``shadow_only=True`` by default. This module does not touch the
  orchestrator, strategy selector, candidate selection, or the decision layer.
* The table is deterministic; conflicting rules are resolved by explicit
  priority order (lowest ``priority_rank`` wins).
* Every decision carries evidence including objective-state and
  failure-distribution fields, and is replayable via pydantic serialization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.services.decision_models import CampaignDecisionEvidence
from app.services.failure_attribution import (
    FailureAttributionCategory,
    FailureAttributionDistribution,
)
from app.services.objective_models import ProxyGapLevel
from app.services.objective_state import ObjectiveState

__all__ = [
    "CampaignMode",
    "CampaignModeContext",
    "CampaignModeDecision",
    "CampaignModeTransitionTable",
    "decide_campaign_mode",
]

#: Attribution confidence below this is treated as "cannot attribute" and routes
#: to human observation. Matches the existing decision-layer threshold.
_LOW_CONFIDENCE_THRESHOLD = 0.5


class CampaignMode(StrEnum):
    """Scientific-activity mode proposed for the next round."""

    BO_OPTIMIZATION = "bo_optimization"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    FAILURE_DIAGNOSIS = "failure_diagnosis"
    LITERATURE_CONTEXT_SEEKING = "literature_context_seeking"
    HUMAN_OBSERVATION_REQUEST = "human_observation_request"
    STOP_RECOMMENDED = "stop_recommended"


@dataclass(frozen=True)
class CampaignModeContext:
    """Inputs to the campaign mode transition table for one round."""

    campaign_id: str
    round_index: int
    objective_state: ObjectiveState | None = None
    failure_attribution: FailureAttributionDistribution | None = None
    literature_missing: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class CampaignModeDecision(BaseModel):
    """A shadow-only campaign mode proposal with provenance."""

    campaign_id: str
    round_index: int = Field(ge=0)
    mode: CampaignMode
    priority_rank: int = Field(ge=1)
    reason: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: list[CampaignDecisionEvidence] = Field(default_factory=list)
    shadow_only: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class CampaignModeTransitionTable:
    """Deterministic, priority-ordered campaign mode classifier."""

    def decide(
        self,
        context: CampaignModeContext,
        *,
        now: datetime | None = None,
    ) -> CampaignModeDecision:
        timestamp = now or datetime.now(UTC)
        objective = context.objective_state
        attribution = context.failure_attribution
        evidence = _build_evidence(objective, attribution)

        mode, rank, reason, confidence = _select_mode(
            objective=objective,
            attribution=attribution,
            literature_missing=context.literature_missing,
        )

        return CampaignModeDecision(
            campaign_id=context.campaign_id,
            round_index=context.round_index,
            mode=mode,
            priority_rank=rank,
            reason=reason,
            confidence=confidence,
            evidence=evidence,
            shadow_only=True,
            created_at=timestamp,
        )


def decide_campaign_mode(
    context: CampaignModeContext,
    *,
    now: datetime | None = None,
) -> CampaignModeDecision:
    """Decide a campaign mode with the default transition table."""
    return CampaignModeTransitionTable().decide(context, now=now)


def _select_mode(
    *,
    objective: ObjectiveState | None,
    attribution: FailureAttributionDistribution | None,
    literature_missing: bool,
) -> tuple[CampaignMode, int, str, float]:
    """Apply the explicit priority order. Lowest rank that matches wins."""
    # Rank 1: an objective-state stop signal overrides every other consideration.
    if objective is not None and objective.stop_recommended:
        reason = objective.stop_reason or "objective state recommends stopping"
        return (
            CampaignMode.STOP_RECOMMENDED,
            1,
            f"Objective state recommends stopping: {reason}.",
            objective.objective_confidence,
        )

    # Rank 2: a failure we cannot attribute confidently needs human observation.
    if attribution is not None and attribution.confidence < _LOW_CONFIDENCE_THRESHOLD:
        return (
            CampaignMode.HUMAN_OBSERVATION_REQUEST,
            2,
            (
                f"Failure attribution confidence {attribution.confidence:.3g} is "
                f"below {_LOW_CONFIDENCE_THRESHOLD}; human observation is needed "
                "before routing."
            ),
            _clamp_unit(1.0 - attribution.confidence),
        )

    # Rank 3: a confident instrument failure routes to calibration.
    if (
        attribution is not None
        and attribution.confidence >= _LOW_CONFIDENCE_THRESHOLD
        and attribution.dominant_category == FailureAttributionCategory.INSTRUMENT
    ):
        return (
            CampaignMode.CALIBRATION,
            3,
            (
                "Dominant failure attribution is instrument "
                f"(p={attribution.dominant_probability:.3g}); calibration is needed."
            ),
            attribution.dominant_probability,
        )

    # Rank 4: any other confident failure routes to diagnosis (fail closed: a
    # failure is never treated as an optimization signal).
    if attribution is not None and attribution.confidence >= _LOW_CONFIDENCE_THRESHOLD:
        return (
            CampaignMode.FAILURE_DIAGNOSIS,
            4,
            (
                "Dominant failure attribution is "
                f"{attribution.dominant_category.value} "
                f"(p={attribution.dominant_probability:.3g}); diagnosis is needed."
            ),
            attribution.dominant_probability,
        )

    # Rank 5: a high proxy gap means the active proxy diverges from the true
    # objective, so validating the proxy takes priority over seeking context.
    if (
        objective is not None
        and objective.proxy_gap is not None
        and objective.proxy_gap.level == ProxyGapLevel.HIGH
    ):
        return (
            CampaignMode.VALIDATION,
            5,
            (
                "Active objective has a high proxy gap "
                f"(score={objective.proxy_gap.score:.3g}); validate the proxy "
                "before more optimization."
            ),
            objective.proxy_gap.score,
        )

    # Rank 6: missing external context routes to literature seeking.
    if literature_missing:
        return (
            CampaignMode.LITERATURE_CONTEXT_SEEKING,
            6,
            "External literature context is missing; seek context before optimizing.",
            0.6,
        )

    # Rank 7: default to parameter optimization.
    confidence = objective.objective_confidence if objective is not None else 0.5
    return (
        CampaignMode.BO_OPTIMIZATION,
        7,
        "No diagnostic, validation, or context signal; continue parameter optimization.",
        confidence,
    )


def _build_evidence(
    objective: ObjectiveState | None,
    attribution: FailureAttributionDistribution | None,
) -> list[CampaignDecisionEvidence]:
    evidence: list[CampaignDecisionEvidence] = []
    if objective is not None:
        evidence.append(
            CampaignDecisionEvidence(
                source="objective_state",
                kind="objective_state",
                summary=(
                    f"Objective confidence {objective.objective_confidence:.3g}, "
                    f"revision {objective.revision}."
                ),
                payload={
                    "objective_confidence": objective.objective_confidence,
                    "proxy_gap_level": (
                        objective.proxy_gap.level.value
                        if objective.proxy_gap is not None
                        else None
                    ),
                    "stop_recommended": objective.stop_recommended,
                    "stop_reason": objective.stop_reason,
                    "revision": objective.revision,
                    "rounds_observed": objective.rounds_observed,
                    "consecutive_failure_count": objective.consecutive_failure_count,
                },
            )
        )
    if attribution is not None:
        evidence.append(
            CampaignDecisionEvidence(
                source="failure_attribution",
                kind="failure_attribution",
                summary=(
                    f"Dominant attribution {attribution.dominant_category.value} "
                    f"(p={attribution.dominant_probability:.3g}), "
                    f"confidence {attribution.confidence:.3g}."
                ),
                payload={
                    "dominant_category": attribution.dominant_category.value,
                    "dominant_probability": attribution.dominant_probability,
                    "confidence": attribution.confidence,
                    "probabilities": {
                        category.value: probability
                        for category, probability in attribution.probabilities.items()
                    },
                    "source_likely_cause": attribution.source_likely_cause,
                    "source_failure_type": attribution.source_failure_type,
                },
            )
        )
    return evidence


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))
