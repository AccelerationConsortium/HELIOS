"""Replay summarization for contextual shadow decisions.

This module aggregates ``CampaignDecisionAccounting`` records into deterministic
metrics for later replay and promotion-gate work. It is pure aggregation logic:
no database reads, external calls, training, or promotion decisions happen here.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from app.services.decision_outcome import CampaignDecisionAccounting

__all__ = [
    "CampaignDecisionReplayAnalyzer",
    "CampaignDecisionReplaySummary",
    "summarize_campaign_decision_replay",
]


class CampaignDecisionReplaySummary(BaseModel):
    """Aggregate replay metrics for contextual shadow decision accounting."""

    replay_id: str
    accounting_count: int = Field(ge=0)
    campaign_ids: list[str] = Field(default_factory=list)
    round_range: dict[str, int] = Field(default_factory=dict)
    average_reward: float = 0.0
    positive_reward_count: int = 0
    negative_reward_count: int = 0
    neutral_reward_count: int = 0
    action_distribution: dict[str, int] = Field(default_factory=dict)
    route_change_count: int = 0
    route_change_rate: float = 0.0
    average_safety_penalty: float = 0.0
    average_failure_penalty: float = 0.0
    average_objective_reward: float = 0.0
    average_proxy_gap_reward: float = 0.0
    average_validation_reward: float = 0.0
    average_context_reward: float = 0.0
    context_request_fulfillment_rate: float | None = None
    validation_success_rate: float | None = None
    human_override_rate: float | None = None
    rationale: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class CampaignDecisionReplayAnalyzer:
    """Aggregate decision accounting records into replay-level metrics."""

    def analyze(
        self,
        accountings: list[CampaignDecisionAccounting],
        *,
        replay_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CampaignDecisionReplaySummary:
        accounting_count = len(accountings)
        replay_id = replay_id or f"cdr-{uuid4().hex}"
        if accounting_count == 0:
            return CampaignDecisionReplaySummary(
                replay_id=replay_id,
                accounting_count=0,
                rationale="No decision accounting records were provided for replay.",
                metadata=deepcopy(dict(metadata or {})),
            )

        rewards = [accounting.reward.reward for accounting in accountings]
        route_change_count = sum(
            1 for accounting in accountings if accounting.trace.would_change_route
        )
        action_distribution = _action_distribution(accountings)
        round_numbers = [accounting.trace.round_index for accounting in accountings]
        average_reward = _mean(rewards)
        route_change_rate = _round_metric(route_change_count / accounting_count)

        return CampaignDecisionReplaySummary(
            replay_id=replay_id,
            accounting_count=accounting_count,
            campaign_ids=sorted(
                {accounting.trace.campaign_id for accounting in accountings}
            ),
            round_range={
                "min_round": min(round_numbers),
                "max_round": max(round_numbers),
            },
            average_reward=average_reward,
            positive_reward_count=sum(1 for reward in rewards if reward > 0),
            negative_reward_count=sum(1 for reward in rewards if reward < 0),
            neutral_reward_count=sum(1 for reward in rewards if reward == 0),
            action_distribution=action_distribution,
            route_change_count=route_change_count,
            route_change_rate=route_change_rate,
            average_safety_penalty=_mean(
                [accounting.reward.safety_penalty for accounting in accountings]
            ),
            average_failure_penalty=_mean(
                [accounting.reward.failure_penalty for accounting in accountings]
            ),
            average_objective_reward=_mean(
                [accounting.reward.objective_reward for accounting in accountings]
            ),
            average_proxy_gap_reward=_mean(
                [accounting.reward.proxy_gap_reward for accounting in accountings]
            ),
            average_validation_reward=_mean(
                [accounting.reward.validation_reward for accounting in accountings]
            ),
            average_context_reward=_mean(
                [accounting.reward.context_reward for accounting in accountings]
            ),
            context_request_fulfillment_rate=_optional_bool_rate(
                [
                    accounting.outcome.context_request_fulfilled
                    for accounting in accountings
                ]
            ),
            validation_success_rate=_optional_bool_rate(
                [accounting.outcome.validation_success for accounting in accountings]
            ),
            human_override_rate=_optional_bool_rate(
                [accounting.outcome.human_override for accounting in accountings]
            ),
            rationale=(
                f"Summarized {accounting_count} decision accounting records; "
                f"average_reward={average_reward:.3g}; "
                f"route_change_rate={route_change_rate:.3g}."
            ),
            metadata=deepcopy(dict(metadata or {})),
        )


def summarize_campaign_decision_replay(
    accountings: list[CampaignDecisionAccounting],
    **kwargs: Any,
) -> CampaignDecisionReplaySummary:
    """Summarize decision accounting records with the default analyzer."""
    return CampaignDecisionReplayAnalyzer().analyze(accountings, **kwargs)


def _action_distribution(
    accountings: list[CampaignDecisionAccounting],
) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for accounting in accountings:
        action = accounting.trace.decision_plan.action_type.value
        distribution[action] = distribution.get(action, 0) + 1
    return distribution


def _optional_bool_rate(values: list[bool | None]) -> float | None:
    observed = [value for value in values if value is not None]
    if not observed:
        return None
    return _round_metric(sum(1 for value in observed if value) / len(observed))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return _round_metric(sum(values) / len(values))


def _round_metric(value: float) -> float:
    return round(value, 10)
