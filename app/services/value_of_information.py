"""Value-of-Information / future-value shadow scoring.

This is the final ring of the campaign-level adaptive decision *substrate*:

    ObjectiveState (+ FailureAttribution) -> CampaignModeDecision
        -> DynamicActionSpaceSnapshot -> Value-of-Information scoring

It turns the per-action shadow assessments from ``dynamic_action_space`` into a
non-myopic ``decision_value`` per candidate, following the planning-runtime
design::

    decision_value =
        immediate_objective_gain
      + proxy_gap_reduction
      + information_gain
      + hypothesis_value
      + future_option_value
      - cost
      - risk
      - latency_penalty

Boundaries (intentional):

* Strictly shadow. It produces an *advisory* ranking and never changes
  candidate selection, strategy selection, routing, or any execution.
* It is the aggregation layer, not a surrogate: the model-derived estimate
  terms (immediate reward, information gain, hypothesis resolution, proxy-gap
  reduction, future option value, reversibility) are supplied as read-only
  ``ActionValueSignals`` (defaulting to 0), while ``risk`` / ``cost`` /
  ``latency`` are read from the upstream action-space snapshot.
* Deterministic; every score carries a reason and evidence; the whole snapshot
  is JSON-safe and replayable via pydantic.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.services.campaign_mode import CampaignMode
from app.services.decision_models import CampaignDecisionEvidence
from app.services.dynamic_action_space import (
    ActionAssessment,
    DynamicActionSpaceSnapshot,
)
from app.services.objective_state import ObjectiveState

__all__ = [
    "ActionValueSignals",
    "CandidateActionScore",
    "ValueOfInformationScorer",
    "ValueOfInformationSnapshot",
    "VoIWeights",
    "score_value_of_information",
]


class VoIWeights(BaseModel):
    """Weights for the non-myopic decision-value aggregation."""

    immediate: float = 1.0
    proxy_gap: float = 0.8
    information: float = 0.6
    hypothesis: float = 0.6
    future_option: float = 0.5
    cost: float = 0.4
    risk: float = 0.7
    latency: float = 0.3


class ActionValueSignals(BaseModel):
    """Read-only model-derived value estimates for one action.

    Defaults are zero so an action with no surrogate signal is scored purely on
    its cost / risk / latency. ``reversibility`` is recorded for provenance and
    reserved for future option-value derivation.
    """

    name: str
    immediate_reward: float = 0.0
    expected_information_gain: float = 0.0
    expected_hypothesis_resolution: float = 0.0
    expected_proxy_gap_reduction: float = 0.0
    expected_future_option_value: float = 0.0
    reversibility: float = Field(default=1.0, ge=0.0, le=1.0)


class CandidateActionScore(BaseModel):
    """Non-myopic value decomposition for a single candidate action."""

    name: str
    immediate_reward: float
    expected_information_gain: float
    expected_hypothesis_resolution: float
    expected_proxy_gap_reduction: float
    expected_future_option_value: float
    expected_failure_risk: float
    cost: float
    latency: float
    reversibility: float
    cost_normalized: float
    latency_normalized: float
    decision_value: float
    reason: str
    evidence: list[CampaignDecisionEvidence] = Field(default_factory=list)


class ValueOfInformationSnapshot(BaseModel):
    """A shadow-only value-of-information snapshot for one round."""

    campaign_id: str
    round_index: int = Field(ge=0)
    mode: CampaignMode
    scores: list[CandidateActionScore] = Field(default_factory=list)
    ranking: list[str] = Field(default_factory=list)
    weights: VoIWeights = Field(default_factory=VoIWeights)
    shadow_only: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class ValueOfInformationScorer:
    """Deterministically score candidate actions by non-myopic decision value."""

    def score(
        self,
        *,
        action_space_snapshot: DynamicActionSpaceSnapshot,
        value_signals: list[ActionValueSignals],
        objective_state: ObjectiveState | None = None,
        weights: VoIWeights | None = None,
        now: datetime | None = None,
    ) -> ValueOfInformationSnapshot:
        timestamp = now or datetime.now(UTC)
        active_weights = weights or VoIWeights()
        signals_by_name = {signal.name: signal for signal in value_signals}

        assessments = action_space_snapshot.assessments
        max_cost = max((_value(a.cost) for a in assessments), default=0.0)
        max_latency = max((_value(a.latency) for a in assessments), default=0.0)

        scores = [
            _score_action(
                assessment=assessment,
                signals=signals_by_name.get(assessment.name, ActionValueSignals(name=assessment.name)),
                weights=active_weights,
                max_cost=max_cost,
                max_latency=max_latency,
            )
            for assessment in assessments
        ]

        ranking = [
            score.name
            for score in sorted(scores, key=lambda s: (-s.decision_value, s.name))
        ]

        metadata: dict[str, Any] = {}
        if objective_state is not None and objective_state.proxy_gap is not None:
            metadata["proxy_gap_baseline"] = objective_state.proxy_gap.score

        return ValueOfInformationSnapshot(
            campaign_id=action_space_snapshot.campaign_id,
            round_index=action_space_snapshot.round_index,
            mode=action_space_snapshot.mode,
            scores=scores,
            ranking=ranking,
            weights=active_weights,
            shadow_only=True,
            created_at=timestamp,
            metadata=metadata,
        )


def score_value_of_information(
    *,
    action_space_snapshot: DynamicActionSpaceSnapshot,
    value_signals: list[ActionValueSignals],
    objective_state: ObjectiveState | None = None,
    weights: VoIWeights | None = None,
    now: datetime | None = None,
) -> ValueOfInformationSnapshot:
    """Score candidate actions with the default scorer."""
    return ValueOfInformationScorer().score(
        action_space_snapshot=action_space_snapshot,
        value_signals=value_signals,
        objective_state=objective_state,
        weights=weights,
        now=now,
    )


def _score_action(
    *,
    assessment: ActionAssessment,
    signals: ActionValueSignals,
    weights: VoIWeights,
    max_cost: float,
    max_latency: float,
) -> CandidateActionScore:
    cost = _value(assessment.cost)
    latency = _value(assessment.latency)
    risk = assessment.risk
    cost_norm = _round(cost / max_cost) if max_cost > 0 else 0.0
    latency_norm = _round(latency / max_latency) if max_latency > 0 else 0.0

    decision_value = _round(
        weights.immediate * signals.immediate_reward
        + weights.proxy_gap * signals.expected_proxy_gap_reduction
        + weights.information * signals.expected_information_gain
        + weights.hypothesis * signals.expected_hypothesis_resolution
        + weights.future_option * signals.expected_future_option_value
        - weights.cost * cost_norm
        - weights.risk * risk
        - weights.latency * latency_norm
    )

    reason = (
        f"decision_value={decision_value:.3g} "
        f"(immediate={signals.immediate_reward:.3g}, "
        f"info={signals.expected_information_gain:.3g}, "
        f"hypothesis={signals.expected_hypothesis_resolution:.3g}, "
        f"proxy_gap_reduction={signals.expected_proxy_gap_reduction:.3g}, "
        f"future_option={signals.expected_future_option_value:.3g}, "
        f"risk={risk:.3g}, cost_norm={cost_norm:.3g}, latency_norm={latency_norm:.3g})"
    )

    evidence = [
        CampaignDecisionEvidence(
            source="value_of_information",
            kind="candidate_action_score",
            summary=reason,
            payload={
                "name": assessment.name,
                "action_label": assessment.label.value,
                "immediate_reward": signals.immediate_reward,
                "expected_information_gain": signals.expected_information_gain,
                "expected_hypothesis_resolution": signals.expected_hypothesis_resolution,
                "expected_proxy_gap_reduction": signals.expected_proxy_gap_reduction,
                "expected_future_option_value": signals.expected_future_option_value,
                "expected_failure_risk": risk,
                "cost_normalized": cost_norm,
                "latency_normalized": latency_norm,
                "reversibility": signals.reversibility,
                "decision_value": decision_value,
            },
        )
    ]

    return CandidateActionScore(
        name=assessment.name,
        immediate_reward=signals.immediate_reward,
        expected_information_gain=signals.expected_information_gain,
        expected_hypothesis_resolution=signals.expected_hypothesis_resolution,
        expected_proxy_gap_reduction=signals.expected_proxy_gap_reduction,
        expected_future_option_value=signals.expected_future_option_value,
        expected_failure_risk=risk,
        cost=cost,
        latency=latency,
        reversibility=signals.reversibility,
        cost_normalized=cost_norm,
        latency_normalized=latency_norm,
        decision_value=decision_value,
        reason=reason,
        evidence=evidence,
    )


def _value(value: float | None) -> float:
    return float(value) if value is not None else 0.0


def _round(value: float) -> float:
    return round(value, 10)
