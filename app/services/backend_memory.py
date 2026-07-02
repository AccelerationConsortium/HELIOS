"""Backend performance memory and contextual bandit utilities."""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from app.services.strategy_models import (
    BackendPerformance,
    CampaignSnapshot,
    ContextualBanditDecision,
    FailureType,
    ShadowBanditEvaluationRecord,
    StrategyReward,
)


def problem_context_key(snapshot: CampaignSnapshot) -> str:
    """Bucket a campaign snapshot for strategy credit assignment."""
    dim_bucket = "high_dim" if snapshot.n_dimensions >= 10 else "low_dim"
    categorical = "categorical" if snapshot.has_categorical else "continuous"
    context = snapshot.campaign_context
    level = (
        getattr(context.current_objective_level, "value", context.current_objective_level)
        if context is not None
        else "performance"
    )
    noise = "unknown_noise"
    if snapshot.last_batch_kpis:
        mean = sum(snapshot.last_batch_kpis) / len(snapshot.last_batch_kpis)
        spread = max(snapshot.last_batch_kpis) - min(snapshot.last_batch_kpis)
        noise = "high_noise" if abs(mean) > 1e-9 and spread / abs(mean) > 0.25 else "low_noise"
    return f"{dim_bucket}:{categorical}:{level}:{noise}"


def strategy_reward(
    *,
    objective_improvement: float = 0.0,
    uncertainty_reduction: float = 0.0,
    constraint_satisfaction: float = 1.0,
    data_quality_gain: float = 0.0,
    novelty: float = 0.0,
    failure_penalty: float = 0.0,
    cost_penalty: float = 0.0,
    time_penalty: float = 0.0,
) -> float:
    """Composite reward for backend/action credit assignment."""
    return make_strategy_reward(
        objective_improvement=objective_improvement,
        information_gain=uncertainty_reduction,
        constraint_satisfaction=constraint_satisfaction,
        data_quality_gain=data_quality_gain,
        novelty=novelty,
        failure_penalty=failure_penalty,
        cost_penalty=cost_penalty,
        time_penalty=time_penalty,
    ).composite_reward


def make_strategy_reward(
    *,
    objective_improvement: float = 0.0,
    information_gain: float = 0.0,
    constraint_satisfaction: float = 1.0,
    data_quality_gain: float = 0.0,
    novelty: float = 0.0,
    failure_penalty: float = 0.0,
    cost_penalty: float = 0.0,
    time_penalty: float = 0.0,
    reward_version: str = "strategy_reward_v1",
) -> StrategyReward:
    """Build a logging-only StrategyReward with a stable composite formula."""
    reward = (
        0.35 * objective_improvement
        + 0.20 * information_gain
        + 0.15 * constraint_satisfaction
        + 0.15 * data_quality_gain
        + 0.10 * novelty
        - 0.20 * failure_penalty
        - 0.10 * cost_penalty
        - 0.05 * time_penalty
    )
    return StrategyReward(
        objective_improvement=objective_improvement,
        information_gain=information_gain,
        constraint_satisfaction=constraint_satisfaction,
        data_quality_gain=data_quality_gain,
        novelty=novelty,
        failure_penalty=failure_penalty,
        cost_penalty=cost_penalty,
        time_penalty=time_penalty,
        composite_reward=max(-1.0, min(1.0, float(reward))),
        reward_version=reward_version,
    )


def shadow_record_from_bandit(
    decision: ContextualBanditDecision,
    *,
    reward: StrategyReward | None = None,
    outcome: str | None = None,
) -> ShadowBanditEvaluationRecord:
    """Create a logging-only bandit-vs-actual evaluation record."""
    return ShadowBanditEvaluationRecord(
        actual_action=decision.actual_action,
        actual_backend=decision.actual_backend,
        suggested_action=decision.suggested_action,
        suggested_backend=decision.suggested_backend,
        agrees_with_actual=decision.agrees_with_actual,
        bandit_confidence=decision.confidence,
        actual_reward=reward.composite_reward if reward is not None else None,
        outcome=outcome,
    )


class BackendPerformanceMemory:
    """In-memory credit store keyed by problem context and backend."""

    def __init__(self, records: dict[str, dict[str, Any]] | None = None) -> None:
        self._records: dict[str, BackendPerformance] = {}
        for key, raw in (records or {}).items():
            self._records[key] = BackendPerformance(**raw)

    @staticmethod
    def _key(context_key: str, backend_name: str, action_type: str) -> str:
        return f"{context_key}|{action_type}|{backend_name}"

    def record(
        self,
        *,
        context_key: str,
        backend_name: str,
        action_type: str,
        reward: float,
        success: bool = True,
        constraint_violation: bool = False,
        latency: float = 0.0,
        cost: float = 0.0,
        used_at: str | None = None,
    ) -> BackendPerformance:
        key = self._key(context_key, backend_name, action_type)
        old = self._records.get(key)
        n_old = old.num_calls if old else 0
        n = n_old + 1
        old_success = old.success_rate if old else 0.0
        old_improvement = old.mean_improvement if old else 0.0
        old_failure = old.failure_rate if old else 0.0
        old_constraint = old.constraint_violation_rate if old else 0.0
        perf = BackendPerformance(
            backend_name=backend_name,
            action_type=action_type,
            problem_fingerprint=context_key,
            num_calls=n,
            success_rate=(old_success * n_old + float(success)) / n,
            mean_improvement=(old_improvement * n_old + reward) / n,
            failure_rate=(old_failure * n_old + float(not success)) / n,
            constraint_violation_rate=(
                old_constraint * n_old + float(constraint_violation)
            ) / n,
            latency=(old.latency * n_old + latency) / n if old else latency,
            cost=(old.cost * n_old + cost) / n if old else cost,
            last_used_at=used_at,
        )
        self._records[key] = perf
        return perf

    def record_failure_event(
        self,
        *,
        context_key: str,
        backend_name: str,
        action_type: str,
        failure_type: FailureType | str,
        used_at: str | None = None,
    ) -> BackendPerformance:
        """Record typed failure attribution without over-penalizing optimizers."""
        ftype = getattr(failure_type, "value", failure_type)
        penalize_backend = ftype in (
            FailureType.CONSTRAINT.value,
            FailureType.BACKEND.value,
        )
        scientific_negative = ftype == FailureType.SCIENTIFIC_NEGATIVE.value
        return self.record(
            context_key=context_key,
            backend_name=backend_name,
            action_type=action_type,
            reward=0.0 if scientific_negative else -0.5 if penalize_backend else 0.0,
            success=not penalize_backend,
            constraint_violation=ftype == FailureType.CONSTRAINT.value,
            used_at=used_at,
        )

    def bias_for(self, context_key: str, backend_name: str, action_type: str) -> float:
        perf = self._records.get(self._key(context_key, backend_name, action_type))
        if perf is None or perf.num_calls == 0:
            return 0.0
        return (
            0.4 * perf.mean_improvement
            + 0.3 * perf.success_rate
            - 0.2 * perf.failure_rate
            - 0.1 * perf.constraint_violation_rate
        )

    def to_json(self) -> dict[str, dict[str, Any]]:
        return {key: asdict(value) for key, value in self._records.items()}


def audit_failure_attribution(
    failure_type: FailureType | str,
) -> dict[str, Any]:
    """Explain whether a failure type should penalize optimizer memory."""
    ftype = getattr(failure_type, "value", failure_type)
    penalizes = ftype in (FailureType.CONSTRAINT.value, FailureType.BACKEND.value)
    evidence_only = ftype == FailureType.SCIENTIFIC_NEGATIVE.value
    return {
        "failure_type": ftype,
        "penalizes_backend": penalizes,
        "evidence_only": evidence_only,
        "reason": (
            "optimizer-owned failure"
            if penalizes
            else "scientific evidence, not execution failure"
            if evidence_only
            else "execution/measurement context, not optimizer-owned"
        ),
    }


class ContextualStrategyBandit:
    """Conservative UCB bandit over action/backend arms."""

    def __init__(
        self,
        stats: dict[str, dict[str, float]] | None = None,
        *,
        exploration_c: float = 1.0,
    ) -> None:
        self._stats = stats or {}
        self.exploration_c = exploration_c

    def select(
        self,
        *,
        context_key: str,
        arms: tuple[str, ...],
        priors: dict[str, float] | None = None,
        actual_action: str = "",
        actual_backend: str = "",
    ) -> ContextualBanditDecision:
        prior = priors or {}
        total_n = sum(int(v.get("n", 0)) for v in self._stats.values())
        scores: list[dict[str, Any]] = []
        for arm in arms:
            key = f"{context_key}|{arm}"
            stat = self._stats.get(key, {"reward": 0.0, "n": 0.0})
            n = int(stat.get("n", 0))
            if n == 0:
                score = math.inf
                mean = 0.0
            else:
                mean = float(stat.get("reward", 0.0)) / n
                score = mean + self.exploration_c * math.sqrt(math.log(max(total_n, 1)) / n)
            if math.isfinite(score):
                score += prior.get(arm, 0.0)
            scores.append({
                "arm": arm,
                "mean_reward": mean,
                "n": n,
                "score": score if math.isfinite(score) else None,
                "_rank_score": score,
            })

        ranked = sorted(
            scores,
            key=lambda row: (
                -float("inf") if row["_rank_score"] == math.inf else -row["_rank_score"],
                arms.index(row["arm"]),
            ),
        )
        selected = ranked[0]["arm"]
        public_scores = tuple(
            {k: v for k, v in row.items() if k != "_rank_score"}
            for row in scores
        )
        suggested_action, suggested_backend = _split_arm(selected)
        actual_arm = (
            f"{actual_action}:{actual_backend}"
            if actual_action and actual_backend
            else ""
        )
        top_score = ranked[0]["_rank_score"]
        second_score = ranked[1]["_rank_score"] if len(ranked) > 1 else top_score
        confidence = _confidence(top_score, second_score)
        return ContextualBanditDecision(
            selected_arm=selected,
            context_key=context_key,
            arm_scores=public_scores,
            reason="ucb selection with rule-based priors",
            actual_action=actual_action,
            actual_backend=actual_backend,
            suggested_action=suggested_action,
            suggested_backend=suggested_backend,
            agrees_with_actual=selected == actual_arm if actual_arm else False,
            confidence=confidence,
        )

    def update(self, *, context_key: str, arm: str, reward: float) -> None:
        key = f"{context_key}|{arm}"
        stat = self._stats.setdefault(key, {"reward": 0.0, "n": 0.0})
        stat["reward"] = float(stat.get("reward", 0.0)) + reward
        stat["n"] = float(stat.get("n", 0.0)) + 1.0

    def to_json(self) -> dict[str, dict[str, float]]:
        return self._stats


def _split_arm(arm: str) -> tuple[str, str]:
    if ":" not in arm:
        return "", arm
    action, backend = arm.split(":", 1)
    return action, backend


def _confidence(top_score: float, second_score: float) -> float:
    if top_score == math.inf:
        return 0.0
    if second_score == math.inf:
        return 0.0
    gap = max(0.0, top_score - second_score)
    return round(min(1.0, gap / (abs(top_score) + 1e-9)), 4)
