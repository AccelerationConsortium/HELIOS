"""Loop engineering DTOs, reward accounting, and replay summaries.

This module makes HELIOS's closed-loop behavior explicit without changing the
live runtime path.  It is a pure service layer: no database writes, no hardware
calls, no policy promotion, and no orchestration side effects happen here.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "LoopDecision",
    "LoopEpisode",
    "LoopIteration",
    "LoopIterationBuilder",
    "LoopOutcome",
    "LoopReplayAnalyzer",
    "LoopReplaySummary",
    "LoopReward",
    "LoopRewardCalculator",
    "LoopSignal",
    "LoopSpec",
    "build_loop_iteration",
    "calculate_loop_reward",
    "summarize_loop_replay",
]

SignalSeverity = Literal["info", "warning", "critical"]
LoopDecisionAuthority = Literal["shadow", "advisory", "canary", "live"]


class LoopSpec(BaseModel):
    """Loop-level contract for one scientific campaign."""

    objective_name: str
    direction: Literal["minimize", "maximize"] = "minimize"
    budget: int | None = Field(default=None, ge=1)
    target_value: float | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    allowed_actions: tuple[str, ...] = ()
    stop_conditions: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoopSignal(BaseModel):
    """One observation available to the loop before a decision."""

    name: str
    source: str
    value: Any | None = None
    severity: SignalSeverity = "info"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoopDecision(BaseModel):
    """A decision made by the loop for one iteration."""

    action: str
    rationale: str
    authority: LoopDecisionAuthority = "shadow"
    selected_backend: str | None = None
    selected_candidate_id: str | None = None
    candidate_count: int | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoopOutcome(BaseModel):
    """Observed result after a loop decision is executed or shadow-scored."""

    execution_success: bool | None = None
    objective_delta: float | None = None
    validation_success: bool | None = None
    recovery_attempted: bool = False
    recovery_success: bool | None = None
    failure_count: int = Field(default=0, ge=0)
    safety_incident_count: int = Field(default=0, ge=0)
    observations: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    failure_attribution: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoopReward(BaseModel):
    """Deterministic score for one loop iteration."""

    iteration_id: str
    reward: float = Field(ge=-1.0, le=1.0)
    regret: float | None = None
    execution_reward: float = 0.0
    objective_reward: float = 0.0
    validation_reward: float = 0.0
    recovery_reward: float = 0.0
    failure_penalty: float = 0.0
    safety_penalty: float = 0.0
    rationale: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoopIteration(BaseModel):
    """One complete observe-decide-act-evaluate unit."""

    iteration_id: str
    episode_id: str
    campaign_id: str
    round_index: int = Field(ge=0)
    spec: LoopSpec
    signals: list[LoopSignal] = Field(default_factory=list)
    decision: LoopDecision
    outcome: LoopOutcome
    reward: LoopReward
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class LoopEpisode(BaseModel):
    """A replayable closed-loop campaign episode."""

    episode_id: str
    campaign_id: str
    spec: LoopSpec
    iterations: list[LoopIteration] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class LoopReplaySummary(BaseModel):
    """Aggregate replay metrics over loop iterations."""

    replay_id: str
    iteration_count: int = Field(ge=0)
    campaign_ids: list[str] = Field(default_factory=list)
    round_range: dict[str, int] = Field(default_factory=dict)
    average_reward: float = 0.0
    positive_reward_count: int = 0
    negative_reward_count: int = 0
    neutral_reward_count: int = 0
    execution_success_rate: float | None = None
    validation_success_rate: float | None = None
    recovery_success_rate: float | None = None
    safety_incident_count: int = 0
    failure_count: int = 0
    average_objective_reward: float = 0.0
    average_failure_penalty: float = 0.0
    average_safety_penalty: float = 0.0
    action_distribution: dict[str, int] = Field(default_factory=dict)
    backend_distribution: dict[str, int] = Field(default_factory=dict)
    critical_signal_count: int = 0
    rationale: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class LoopRewardCalculator:
    """Calculate deterministic reward components from a loop outcome."""

    def calculate(
        self,
        *,
        iteration_id: str,
        outcome: LoopOutcome,
    ) -> LoopReward:
        execution_reward = _execution_reward(outcome.execution_success)
        objective_reward = _objective_reward(outcome.objective_delta)
        validation_reward = _validation_reward(outcome.validation_success)
        recovery_reward = _recovery_reward(
            attempted=outcome.recovery_attempted,
            success=outcome.recovery_success,
        )
        failure_penalty = _round_component(-0.1 * outcome.failure_count)
        safety_penalty = _round_component(-0.5 * outcome.safety_incident_count)
        raw_reward = (
            execution_reward
            + objective_reward
            + validation_reward
            + recovery_reward
            + failure_penalty
            + safety_penalty
        )
        reward = _clamp(raw_reward)
        regret = max(0.0, -reward)
        return LoopReward(
            iteration_id=iteration_id,
            reward=reward,
            regret=regret,
            execution_reward=execution_reward,
            objective_reward=objective_reward,
            validation_reward=validation_reward,
            recovery_reward=recovery_reward,
            failure_penalty=failure_penalty,
            safety_penalty=safety_penalty,
            rationale=_reward_rationale(
                execution_reward=execution_reward,
                objective_reward=objective_reward,
                validation_reward=validation_reward,
                recovery_reward=recovery_reward,
                failure_penalty=failure_penalty,
                safety_penalty=safety_penalty,
                raw_reward=raw_reward,
                reward=reward,
            ),
            metadata={
                "raw_reward": raw_reward,
                "clamped": raw_reward != reward,
            },
        )


class LoopIterationBuilder:
    """Build immutable loop iterations from spec, signals, decision, and outcome."""

    def build(
        self,
        *,
        campaign_id: str,
        round_index: int,
        spec: LoopSpec,
        decision: LoopDecision,
        outcome: LoopOutcome,
        signals: list[LoopSignal] | None = None,
        episode_id: str | None = None,
        iteration_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LoopIteration:
        iteration_id = iteration_id or f"lei-{uuid4().hex}"
        reward = LoopRewardCalculator().calculate(
            iteration_id=iteration_id,
            outcome=outcome,
        )
        return LoopIteration(
            iteration_id=iteration_id,
            episode_id=episode_id or f"lep-{campaign_id}",
            campaign_id=campaign_id,
            round_index=round_index,
            spec=spec.model_copy(deep=True),
            signals=deepcopy(list(signals or [])),
            decision=decision.model_copy(deep=True),
            outcome=outcome.model_copy(deep=True),
            reward=reward,
            metadata=deepcopy(dict(metadata or {})),
        )


class LoopReplayAnalyzer:
    """Aggregate loop iterations into replay-level evidence."""

    def analyze(
        self,
        iterations: list[LoopIteration],
        *,
        replay_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LoopReplaySummary:
        replay_id = replay_id or f"ler-{uuid4().hex}"
        count = len(iterations)
        if count == 0:
            return LoopReplaySummary(
                replay_id=replay_id,
                iteration_count=0,
                rationale="No loop iterations were provided for replay.",
                metadata=deepcopy(dict(metadata or {})),
            )

        rewards = [iteration.reward.reward for iteration in iterations]
        round_numbers = [iteration.round_index for iteration in iterations]
        average_reward = _mean(rewards)
        action_distribution = _count_by(
            iteration.decision.action for iteration in iterations
        )
        backend_distribution = _count_by(
            iteration.decision.selected_backend
            for iteration in iterations
            if iteration.decision.selected_backend
        )
        critical_signal_count = sum(
            1
            for iteration in iterations
            for signal in iteration.signals
            if signal.severity == "critical"
        )
        failure_count = sum(i.outcome.failure_count for i in iterations)
        safety_incident_count = sum(i.outcome.safety_incident_count for i in iterations)

        return LoopReplaySummary(
            replay_id=replay_id,
            iteration_count=count,
            campaign_ids=sorted({iteration.campaign_id for iteration in iterations}),
            round_range={
                "min_round": min(round_numbers),
                "max_round": max(round_numbers),
            },
            average_reward=average_reward,
            positive_reward_count=sum(1 for reward in rewards if reward > 0),
            negative_reward_count=sum(1 for reward in rewards if reward < 0),
            neutral_reward_count=sum(1 for reward in rewards if reward == 0),
            execution_success_rate=_optional_bool_rate(
                [iteration.outcome.execution_success for iteration in iterations]
            ),
            validation_success_rate=_optional_bool_rate(
                [iteration.outcome.validation_success for iteration in iterations]
            ),
            recovery_success_rate=_optional_bool_rate(
                [
                    iteration.outcome.recovery_success
                    for iteration in iterations
                    if iteration.outcome.recovery_attempted
                ]
            ),
            safety_incident_count=safety_incident_count,
            failure_count=failure_count,
            average_objective_reward=_mean(
                [iteration.reward.objective_reward for iteration in iterations]
            ),
            average_failure_penalty=_mean(
                [iteration.reward.failure_penalty for iteration in iterations]
            ),
            average_safety_penalty=_mean(
                [iteration.reward.safety_penalty for iteration in iterations]
            ),
            action_distribution=action_distribution,
            backend_distribution=backend_distribution,
            critical_signal_count=critical_signal_count,
            rationale=(
                f"Summarized {count} loop iterations; "
                f"average_reward={average_reward:.3g}; "
                f"failures={failure_count}; safety_incidents={safety_incident_count}."
            ),
            metadata=deepcopy(dict(metadata or {})),
        )


def build_loop_iteration(**kwargs: Any) -> LoopIteration:
    """Build a loop iteration with the default builder."""
    return LoopIterationBuilder().build(**kwargs)


def calculate_loop_reward(
    *,
    iteration_id: str,
    outcome: LoopOutcome,
) -> LoopReward:
    """Calculate a loop reward with the default calculator."""
    return LoopRewardCalculator().calculate(
        iteration_id=iteration_id,
        outcome=outcome,
    )


def summarize_loop_replay(
    iterations: list[LoopIteration],
    **kwargs: Any,
) -> LoopReplaySummary:
    """Summarize loop iterations with the default analyzer."""
    return LoopReplayAnalyzer().analyze(iterations, **kwargs)


def _execution_reward(value: bool | None) -> float:
    if value is True:
        return 0.2
    if value is False:
        return -0.3
    return 0.0


def _objective_reward(delta: float | None) -> float:
    if delta is None:
        return 0.0
    return _round_component(float(delta) * 0.3)


def _validation_reward(value: bool | None) -> float:
    if value is True:
        return 0.2
    if value is False:
        return -0.2
    return 0.0


def _recovery_reward(*, attempted: bool, success: bool | None) -> float:
    if not attempted:
        return 0.0
    if success is True:
        return 0.1
    if success is False:
        return -0.1
    return 0.0


def _reward_rationale(
    *,
    execution_reward: float,
    objective_reward: float,
    validation_reward: float,
    recovery_reward: float,
    failure_penalty: float,
    safety_penalty: float,
    raw_reward: float,
    reward: float,
) -> str:
    parts = [
        f"execution={execution_reward:.3g}",
        f"objective={objective_reward:.3g}",
        f"validation={validation_reward:.3g}",
        f"recovery={recovery_reward:.3g}",
        f"failure={failure_penalty:.3g}",
        f"safety={safety_penalty:.3g}",
    ]
    if raw_reward != reward:
        parts.append(f"clamped={reward:.3g}")
    return "; ".join(parts)


def _optional_bool_rate(values: list[bool | None]) -> float | None:
    observed = [value for value in values if value is not None]
    if not observed:
        return None
    return _round_metric(sum(1 for value in observed if value) / len(observed))


def _count_by(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return _round_metric(sum(values) / len(values))


def _round_component(value: float) -> float:
    return round(value, 10)


def _round_metric(value: float) -> float:
    return round(value, 10)


def _clamp(value: float) -> float:
    return _round_component(max(-1.0, min(1.0, value)))
