"""Outcome accounting for contextual shadow campaign decisions.

This module binds a shadow ``CampaignDecisionTrace`` to observed post-decision
outcomes and computes deterministic reward summaries. It is pure DTO and
scoring logic: no database writes, external calls, promotion gates, or policy
updates happen here.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.services.decision_trace import CampaignDecisionTrace

__all__ = [
    "CampaignDecisionAccounting",
    "CampaignDecisionAccountingBuilder",
    "CampaignDecisionOutcome",
    "CampaignDecisionOutcomeBuilder",
    "CampaignDecisionReward",
    "CampaignDecisionRewardCalculator",
    "build_campaign_decision_accounting",
    "build_campaign_decision_outcome",
    "calculate_campaign_decision_reward",
]


class CampaignDecisionOutcome(BaseModel):
    """Observed post-decision outcome linked to a shadow decision trace."""

    trace_id: str
    campaign_id: str
    round_index: int = Field(ge=0)
    observed_action: str | None = None
    observed_backend: str | None = None
    candidate_count: int | None = None
    execution_success: bool | None = None
    failure_count: int = 0
    safety_incident_count: int = 0
    objective_delta: float | None = None
    proxy_gap_delta: float | None = None
    validation_success: bool | None = None
    context_request_fulfilled: bool | None = None
    human_override: bool | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class CampaignDecisionReward(BaseModel):
    """Deterministic reward/regret summary for a decision outcome."""

    trace_id: str
    reward: float = Field(ge=-1.0, le=1.0)
    regret: float | None = None
    safety_penalty: float = 0.0
    failure_penalty: float = 0.0
    objective_reward: float = 0.0
    proxy_gap_reward: float = 0.0
    validation_reward: float = 0.0
    context_reward: float = 0.0
    rationale: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CampaignDecisionAccounting(BaseModel):
    """Trace, observed outcome, and reward summary bundled for replay."""

    trace: CampaignDecisionTrace
    outcome: CampaignDecisionOutcome
    reward: CampaignDecisionReward
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class CampaignDecisionOutcomeBuilder:
    """Build outcomes from decision traces and observed execution summaries."""

    def build(
        self,
        *,
        trace: CampaignDecisionTrace,
        observed_action: str | None = None,
        observed_backend: str | None = None,
        candidate_count: int | None = None,
        execution_success: bool | None = None,
        failure_count: int = 0,
        safety_incident_count: int = 0,
        objective_delta: float | None = None,
        proxy_gap_delta: float | None = None,
        validation_success: bool | None = None,
        context_request_fulfilled: bool | None = None,
        human_override: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CampaignDecisionOutcome:
        return CampaignDecisionOutcome(
            trace_id=trace.trace_id,
            campaign_id=trace.campaign_id,
            round_index=trace.round_index,
            observed_action=observed_action,
            observed_backend=observed_backend,
            candidate_count=candidate_count,
            execution_success=execution_success,
            failure_count=failure_count,
            safety_incident_count=safety_incident_count,
            objective_delta=objective_delta,
            proxy_gap_delta=proxy_gap_delta,
            validation_success=validation_success,
            context_request_fulfilled=context_request_fulfilled,
            human_override=human_override,
            metadata=deepcopy(dict(metadata or {})),
        )


class CampaignDecisionRewardCalculator:
    """Calculate deterministic reward components from observed outcomes."""

    def calculate(self, outcome: CampaignDecisionOutcome) -> CampaignDecisionReward:
        execution_reward = _execution_reward(outcome.execution_success)
        failure_penalty = _round_component(-0.1 * outcome.failure_count)
        safety_penalty = _round_component(-0.5 * outcome.safety_incident_count)
        objective_reward = _objective_reward(outcome.objective_delta)
        proxy_gap_reward = _proxy_gap_reward(outcome.proxy_gap_delta)
        validation_reward = _validation_reward(outcome.validation_success)
        context_reward = (
            0.1 if outcome.context_request_fulfilled is True else 0.0
        )
        raw_reward = (
            execution_reward
            + failure_penalty
            + safety_penalty
            + objective_reward
            + proxy_gap_reward
            + validation_reward
            + context_reward
        )
        reward = _clamp(raw_reward)
        regret = max(0.0, -reward)
        return CampaignDecisionReward(
            trace_id=outcome.trace_id,
            reward=reward,
            regret=regret,
            safety_penalty=safety_penalty,
            failure_penalty=failure_penalty,
            objective_reward=objective_reward,
            proxy_gap_reward=proxy_gap_reward,
            validation_reward=validation_reward,
            context_reward=context_reward,
            rationale=_reward_rationale(
                execution_reward=execution_reward,
                failure_penalty=failure_penalty,
                safety_penalty=safety_penalty,
                objective_reward=objective_reward,
                proxy_gap_reward=proxy_gap_reward,
                validation_reward=validation_reward,
                context_reward=context_reward,
                raw_reward=raw_reward,
                reward=reward,
            ),
            metadata={
                "execution_reward": execution_reward,
                "raw_reward": raw_reward,
                "clamped": raw_reward != reward,
            },
        )


class CampaignDecisionAccountingBuilder:
    """Bind a trace and outcome with calculated reward accounting."""

    def build(
        self,
        *,
        trace: CampaignDecisionTrace,
        outcome: CampaignDecisionOutcome,
        metadata: dict[str, Any] | None = None,
    ) -> CampaignDecisionAccounting:
        reward = CampaignDecisionRewardCalculator().calculate(outcome)
        return CampaignDecisionAccounting(
            trace=trace.model_copy(deep=True),
            outcome=outcome.model_copy(deep=True),
            reward=reward,
            metadata=deepcopy(dict(metadata or {})),
        )


def build_campaign_decision_outcome(**kwargs: Any) -> CampaignDecisionOutcome:
    """Build a CampaignDecisionOutcome with the default builder."""
    return CampaignDecisionOutcomeBuilder().build(**kwargs)


def calculate_campaign_decision_reward(
    outcome: CampaignDecisionOutcome,
) -> CampaignDecisionReward:
    """Calculate a CampaignDecisionReward with the default calculator."""
    return CampaignDecisionRewardCalculator().calculate(outcome)


def build_campaign_decision_accounting(**kwargs: Any) -> CampaignDecisionAccounting:
    """Build CampaignDecisionAccounting with the default builder."""
    return CampaignDecisionAccountingBuilder().build(**kwargs)


def _execution_reward(execution_success: bool | None) -> float:
    if execution_success is True:
        return 0.2
    if execution_success is False:
        return -0.3
    return 0.0


def _objective_reward(objective_delta: float | None) -> float:
    if objective_delta is None:
        return 0.0
    if objective_delta > 0:
        return _round_component(min(objective_delta, 1.0) * 0.3)
    return _round_component(objective_delta * 0.3)


def _proxy_gap_reward(proxy_gap_delta: float | None) -> float:
    if proxy_gap_delta is None:
        return 0.0
    if proxy_gap_delta < 0:
        return _round_component(abs(proxy_gap_delta) * 0.3)
    return _round_component(-proxy_gap_delta * 0.3)


def _validation_reward(validation_success: bool | None) -> float:
    if validation_success is True:
        return 0.2
    if validation_success is False:
        return -0.2
    return 0.0


def _reward_rationale(
    *,
    execution_reward: float,
    failure_penalty: float,
    safety_penalty: float,
    objective_reward: float,
    proxy_gap_reward: float,
    validation_reward: float,
    context_reward: float,
    raw_reward: float,
    reward: float,
) -> str:
    components = {
        "execution": execution_reward,
        "failure": failure_penalty,
        "safety": safety_penalty,
        "objective": objective_reward,
        "proxy_gap": proxy_gap_reward,
        "validation": validation_reward,
        "context": context_reward,
    }
    nonzero = [
        f"{name}={value:.3g}"
        for name, value in components.items()
        if value != 0.0
    ]
    if not nonzero:
        nonzero = ["no nonzero outcome components"]
    suffix = " Reward was clamped." if raw_reward != reward else ""
    return f"Reward components: {', '.join(nonzero)}.{suffix}"


def _clamp(value: float) -> float:
    return _round_component(max(-1.0, min(1.0, value)))


def _round_component(value: float) -> float:
    return round(value, 10)
