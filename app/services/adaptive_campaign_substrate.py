"""Adaptive campaign substrate — single shadow artifact bundling Phase 1-5.

This is the observation bridge for the campaign-level adaptive decision
substrate. It assembles the free-standing Phase 1-5 shadow modules into one
replayable artifact for a single round::

    ObjectiveState (+ FailureAttribution)
        -> CampaignModeDecision        (Phase 3)
        -> DynamicActionSpaceSnapshot  (Phase 4)
        -> ValueOfInformationSnapshot  (Phase 5)

It is strictly an aggregator / observation bridge:

* It only *reads* the existing Phase 1-5 modules and composes their pure
  builders. It does not import or touch the orchestrator, strategy selector,
  decision layer, candidate selection, or action execution.
* The whole artifact and every nested component stay ``shadow_only=True``.
* The Value-of-Information ranking is advisory only: the artifact records it for
  observation and explicitly flags ``voi_ranking_advisory_only`` — it is never
  used to select or reorder actions here.
* Deterministic and JSON-safe: a single injected ``now`` timestamps the whole
  chain, and the artifact is replayable via pydantic serialization.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.services.campaign_mode import (
    CampaignModeContext,
    CampaignModeDecision,
    decide_campaign_mode,
)
from app.services.decision_models import CampaignDecisionEvidence
from app.services.dynamic_action_space import (
    ActionSpec,
    DynamicActionSpaceSnapshot,
    build_action_space_snapshot,
)
from app.services.failure_attribution import FailureAttributionDistribution
from app.services.objective_state import ObjectiveState
from app.services.value_of_information import (
    ActionValueSignals,
    ValueOfInformationSnapshot,
    VoIWeights,
    score_value_of_information,
)

__all__ = [
    "AdaptiveCampaignSubstrateSnapshot",
    "build_adaptive_campaign_substrate_snapshot",
]


class AdaptiveCampaignSubstrateSnapshot(BaseModel):
    """A single shadow artifact bundling the Phase 1-5 substrate for one round."""

    campaign_id: str
    round_index: int = Field(ge=0)
    objective_state: ObjectiveState | None = None
    failure_attribution: FailureAttributionDistribution | None = None
    campaign_mode_decision: CampaignModeDecision
    dynamic_action_space_snapshot: DynamicActionSpaceSnapshot
    value_of_information_snapshot: ValueOfInformationSnapshot
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


def build_adaptive_campaign_substrate_snapshot(
    *,
    campaign_id: str,
    round_index: int,
    objective_state: ObjectiveState | None = None,
    failure_attribution: FailureAttributionDistribution | None = None,
    actions: list[ActionSpec] | None = None,
    available_capabilities: list[str] | None = None,
    value_signals: list[ActionValueSignals] | None = None,
    literature_missing: bool = False,
    safety_summary: dict[str, Any] | None = None,
    weights: VoIWeights | None = None,
    now: datetime | None = None,
) -> AdaptiveCampaignSubstrateSnapshot:
    """Assemble a single shadow substrate artifact from real-round inputs.

    Read-only composition of the Phase 3-5 builders. Does not affect routing,
    selection, or execution; the returned artifact is observation-only.
    """
    timestamp = now or datetime.now(UTC)

    mode_decision = decide_campaign_mode(
        CampaignModeContext(
            campaign_id=campaign_id,
            round_index=round_index,
            objective_state=objective_state,
            failure_attribution=failure_attribution,
            literature_missing=literature_missing,
            safety_summary=dict(safety_summary or {}),
        ),
        now=timestamp,
    )

    action_space_snapshot = build_action_space_snapshot(
        mode_decision=mode_decision,
        actions=actions or [],
        available_capabilities=available_capabilities or [],
        failure_attribution=failure_attribution,
        objective_state=objective_state,
        now=timestamp,
    )

    voi_snapshot = score_value_of_information(
        action_space_snapshot=action_space_snapshot,
        value_signals=value_signals or [],
        objective_state=objective_state,
        weights=weights,
        now=timestamp,
    )

    evidence = _build_evidence(
        objective_state=objective_state,
        failure_attribution=failure_attribution,
        mode_decision=mode_decision,
        action_space_snapshot=action_space_snapshot,
        voi_snapshot=voi_snapshot,
    )

    return AdaptiveCampaignSubstrateSnapshot(
        campaign_id=campaign_id,
        round_index=round_index,
        objective_state=objective_state,
        failure_attribution=failure_attribution,
        campaign_mode_decision=mode_decision,
        dynamic_action_space_snapshot=action_space_snapshot,
        value_of_information_snapshot=voi_snapshot,
        evidence=evidence,
        shadow_only=True,
        created_at=timestamp,
        metadata={"voi_ranking_advisory_only": True},
    )


def _build_evidence(
    *,
    objective_state: ObjectiveState | None,
    failure_attribution: FailureAttributionDistribution | None,
    mode_decision: CampaignModeDecision,
    action_space_snapshot: DynamicActionSpaceSnapshot,
    voi_snapshot: ValueOfInformationSnapshot,
) -> list[CampaignDecisionEvidence]:
    top_ranked = voi_snapshot.ranking[0] if voi_snapshot.ranking else None
    return [
        CampaignDecisionEvidence(
            source="adaptive_campaign_substrate",
            kind="substrate_chain",
            summary=(
                f"mode={mode_decision.mode.value} (rank {mode_decision.priority_rank}); "
                f"{len(action_space_snapshot.assessments)} action(s); "
                f"advisory top-ranked={top_ranked}."
            ),
            payload={
                "mode": mode_decision.mode.value,
                "mode_priority_rank": mode_decision.priority_rank,
                "objective_confidence": (
                    objective_state.objective_confidence
                    if objective_state is not None
                    else None
                ),
                "dominant_failure_category": (
                    failure_attribution.dominant_category.value
                    if failure_attribution is not None
                    else None
                ),
                "preferred_actions": list(action_space_snapshot.preferred_actions),
                "proposed_disabled_actions": list(
                    action_space_snapshot.proposed_disabled_actions
                ),
                "voi_ranking": list(voi_snapshot.ranking),
                "voi_ranking_advisory_only": True,
            },
        )
    ]
