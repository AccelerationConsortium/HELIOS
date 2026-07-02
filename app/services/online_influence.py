"""Guarded online soft influence utilities."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
from typing import Any

from app.services.policy_evaluation import PolicyEvaluationRunner
from app.services.strategy_models import (
    CampaignSnapshot,
    OnlineInfluenceMode,
    OnlineInfluenceOutcome,
    OnlineInfluenceRolloutConfig,
    PolicyInfluenceConfig,
    RankingInfluenceRecord,
)


def _value(value: Any) -> str:
    return str(getattr(value, "value", value))


def effective_policy_influence_config(
    base: PolicyInfluenceConfig,
    rollout: OnlineInfluenceRolloutConfig,
    snapshot: CampaignSnapshot,
) -> tuple[PolicyInfluenceConfig, str]:
    """Resolve the execution-time influence config for one round."""
    mode = _value(rollout.mode)
    if not rollout.enabled or mode == OnlineInfluenceMode.OFF.value:
        return base, "online influence disabled"
    if rollout.allowed_campaign_ids and snapshot.campaign_id not in rollout.allowed_campaign_ids:
        return PolicyInfluenceConfig(), "campaign not allow-listed for online influence"
    level = _value(
        snapshot.campaign_context.current_objective_level
        if snapshot.campaign_context is not None
        else "performance"
    )
    allowed_levels = tuple(_value(item) for item in rollout.allowed_objective_levels)
    if allowed_levels and level not in allowed_levels:
        return PolicyInfluenceConfig(), "objective level not allow-listed for online influence"
    if rollout.max_rounds is not None and snapshot.round_number > rollout.max_rounds:
        return PolicyInfluenceConfig(), "online influence max round exceeded"
    if mode == OnlineInfluenceMode.SAFE_SOFT.value:
        return PolicyInfluenceConfig(
            enable_action_policy_rerank=True,
            enable_backend_memory_rerank=True,
            enable_bandit_rerank=bool(
                rollout.enable_bandit_soft_influence and base.enable_bandit_rerank
            ),
            bandit_offline_eval_passed=base.bandit_offline_eval_passed,
            bandit_calibration_score=base.bandit_calibration_score,
            enable_transition_guard_penalty=False,
            max_action_policy_weight=min(base.max_action_policy_weight, rollout.max_action_policy_weight),
            max_backend_memory_weight=min(base.max_backend_memory_weight, rollout.max_backend_memory_weight),
            max_bandit_weight=min(base.max_bandit_weight, rollout.max_bandit_weight),
            max_transition_guard_weight=0.0,
            max_total_score_delta=min(base.max_total_score_delta, rollout.max_total_score_delta),
            allow_action_policy_hard_veto=False,
        ), "safe-soft online influence enabled"
    if mode == OnlineInfluenceMode.EVALUATION.value:
        return PolicyInfluenceConfig(
            enable_bandit_rerank=bool(
                rollout.enable_bandit_soft_influence and base.enable_bandit_rerank
            ),
            bandit_offline_eval_passed=base.bandit_offline_eval_passed,
            bandit_calibration_score=base.bandit_calibration_score,
            max_bandit_weight=min(base.max_bandit_weight, rollout.max_bandit_weight),
            max_total_score_delta=min(base.max_total_score_delta, rollout.max_total_score_delta),
            allow_action_policy_hard_veto=False,
        ), "evaluation mode allows explicit bandit tiny soft influence"
    if mode == OnlineInfluenceMode.SHADOW.value:
        return PolicyInfluenceConfig(), "shadow mode records evaluation without execution influence"
    return PolicyInfluenceConfig(), f"unknown online influence mode: {mode}"


def build_online_influence_outcome(
    *,
    rollout: OnlineInfluenceRolloutConfig,
    baseline_top_backend: str,
    influenced_top_backend: str,
    ranking_influences: tuple[RankingInfluenceRecord, ...],
    trace: dict[str, Any],
    config: PolicyInfluenceConfig,
    safe_influence_top_backend: str | None = None,
    learned_influenced_top_backend: str | None = None,
    learned_policy_influences: tuple[Any, ...] = (),
) -> OnlineInfluenceOutcome | None:
    """Build a per-round outcome and auto-disable signal."""
    mode = _value(rollout.mode)
    if not rollout.enabled or mode == OnlineInfluenceMode.OFF.value:
        return None
    safety = PolicyEvaluationRunner._safety_check([trace], config)
    safety_warnings = [failure.get("check", "unknown") for failure in safety["failures"]]
    safety_warnings.extend(warning.get("check", "unknown") for warning in safety["warnings"])
    top1_changed = baseline_top_backend != influenced_top_backend
    if top1_changed and not any(abs(record.score_delta) > 0 for record in ranking_influences):
        safety_warnings.append("ranking_changed_without_influence_record")
    bandit = trace.get("bandit_influence") or {}
    bandit_eligibility = bandit.get("eligibility") or {}
    if (
        bandit
        and not bandit_eligibility.get("eligible", False)
        and abs(float(bandit.get("score_delta") or 0.0)) > 0
    ):
        safety_warnings.append("bandit_ineligible_nonzero_delta")
    if (
        bandit
        and float(bandit.get("confidence") or 0.0) >= rollout.bandit_min_confidence
        and trace.get("strategy_reward")
        and float((trace.get("strategy_reward") or {}).get("composite_reward") or 0.0) < 0
    ):
        safety_warnings.append("bandit_high_confidence_underperformance")
    if any(
        record.source == "bandit_soft"
        for record in ranking_influences
    ):
        active_failure_types = {
            _value(event.get("failure_type"))
            for event in trace.get("failure_events", [])
            if isinstance(event, dict)
        }
        if active_failure_types & {"backend", "constraint"}:
            safety_warnings.append("bandit_backend_or_constraint_failure")
    learned = trace.get("learned_policy_influence") or {}
    if learned:
        safety_warnings.extend(str(item) for item in learned.get("safety_warnings") or ())
        if learned.get("capped") and abs(float(learned.get("applied_delta") or 0.0)) > 0.01:
            safety_warnings.append("learned_cap_violation")
        if (
            abs(float(learned.get("applied_delta") or 0.0)) > 0
            and not learned.get("target_backend")
        ):
            safety_warnings.append("learned_unexplained_ranking_change")
        if not learned.get("safety_mask_valid", True):
            safety_warnings.append("learned_safety_mask_invalid")
        if (
            learned.get("changed_top1")
            and rollout.learned_live_top1_change_rate_threshold <= 0
        ):
            safety_warnings.append("learned_top1_change_rate_threshold_exceeded")
        if (
            float(learned.get("confidence") or 0.0) >= rollout.bandit_min_confidence
            and trace.get("strategy_reward")
            and float((trace.get("strategy_reward") or {}).get("composite_reward") or 0.0) < 0
        ):
            safety_warnings.append("learned_high_confidence_underperformance")
        active_failure_types = {
            _value(event.get("failure_type"))
            for event in (trace.get("failure_events") or (trace.get("outcome") or {}).get("failure_events") or [])
            if isinstance(event, dict)
        }
        if abs(float(learned.get("applied_delta") or 0.0)) > 0 and active_failure_types & {"backend", "constraint"}:
            safety_warnings.append("learned_backend_or_constraint_failure_increase")
    auto_disabled = should_auto_disable(rollout, safety_warnings)
    reward = (trace.get("strategy_reward") or {}).get("composite_reward")
    safe_top = safe_influence_top_backend or influenced_top_backend
    learned_top = learned_influenced_top_backend or influenced_top_backend
    return OnlineInfluenceOutcome(
        mode=mode,
        enabled=True,
        baseline_top_backend=baseline_top_backend,
        influenced_top_backend=influenced_top_backend,
        top1_changed=top1_changed,
        safe_influence_top_backend=safe_top,
        learned_influenced_top_backend=learned_top,
        learned_changed_top1=bool(safe_top and learned_top and safe_top != learned_top),
        applied_influences=ranking_influences,
        learned_policy_influences=learned_policy_influences,
        reward=float(reward) if reward is not None else None,
        outcome=None,
        failure_events=(),
        safety_warnings=tuple(safety_warnings),
        auto_disabled=auto_disabled,
        reason="; ".join(safety_warnings) if safety_warnings else "online influence passed safety checks",
    )


def should_auto_disable(
    rollout: OnlineInfluenceRolloutConfig,
    safety_warnings: list[str] | tuple[str, ...],
) -> bool:
    """Return whether rollout should disable SAFE_SOFT after this round."""
    if _value(rollout.mode) != OnlineInfluenceMode.SAFE_SOFT.value:
        return False
    warning_set = set(safety_warnings)
    if rollout.auto_disable_on_safety_warning and warning_set:
        return True
    if rollout.auto_disable_on_cap_violation and any("cap" in item for item in warning_set):
        return True
    if (
        rollout.auto_disable_on_unexplained_ranking_change
        and "ranking_changed_without_influence_record" in warning_set
    ):
        return True
    if warning_set & {
        "learned_cap_violation",
        "learned_unexplained_ranking_change",
        "learned_safety_mask_invalid",
        "learned_top1_change_rate_threshold_exceeded",
        "learned_high_confidence_underperformance",
        "learned_backend_or_constraint_failure_increase",
        "unavailable_backend_suggestion",
        "suggested_backend_unavailable",
        "score_delta_target_unavailable",
        "hard_veto_attempt",
        "backend_addition_attempt",
        "space_revision_auto_apply_attempt",
    }:
        return True
    return False


class OnlineInfluenceController:
    """Small stateful helper for tests or services that want auto-disable state."""

    def __init__(self, rollout: OnlineInfluenceRolloutConfig) -> None:
        self.rollout = rollout
        self.disabled = False
        self.outcomes: list[OnlineInfluenceOutcome] = []

    def record(self, outcome: OnlineInfluenceOutcome | None) -> None:
        if outcome is None:
            return
        self.outcomes.append(outcome)
        if outcome.auto_disabled:
            self.disabled = True

    def report(self) -> dict[str, Any]:
        return post_run_influence_report(self.outcomes)


def post_run_influence_report(
    outcomes: list[OnlineInfluenceOutcome] | tuple[OnlineInfluenceOutcome, ...],
) -> dict[str, Any]:
    """Summarize guarded online influence outcomes after a run."""
    outcome_dicts = [_as_dict(outcome) for outcome in outcomes if outcome is not None]
    influenced = [
        outcome for outcome in outcome_dicts
        if outcome.get("enabled")
    ]
    top1_changed = [outcome for outcome in influenced if outcome.get("top1_changed")]
    rewards = [
        float(outcome["reward"])
        for outcome in influenced
        if outcome.get("reward") is not None
    ]
    safety_warning_count = sum(len(outcome.get("safety_warnings") or []) for outcome in influenced)
    source_counter: Counter[str] = Counter()
    bandit_eligible = 0
    bandit_applied = 0
    bandit_top1_changed = 0
    learned_eligible = 0
    learned_applied = 0
    learned_top1_changed = 0
    learned_live_eligible = 0
    learned_live_applied = 0
    learned_live_top1_changed = 0
    learned_confidence: Counter[str] = Counter()
    for outcome in influenced:
        has_bandit_soft = False
        has_learned_live = False
        for record in outcome.get("applied_influences") or []:
            source_counter[str(record.get("source", ""))] += 1
            if record.get("source") == "bandit_soft":
                has_bandit_soft = True
            if record.get("source") == "learned_policy":
                has_learned_live = True
        if any(
            record.get("source") in {"bandit_soft", "bandit_shadow"}
            for record in outcome.get("applied_influences") or []
        ):
            bandit_eligible += int(has_bandit_soft)
        if has_bandit_soft:
            bandit_applied += 1
            if outcome.get("top1_changed"):
                bandit_top1_changed += 1
        learned_records = outcome.get("learned_policy_influences") or []
        for record in learned_records:
            eligibility = record.get("eligibility") or {}
            if eligibility.get("eligible"):
                learned_eligible += 1
                learned_live_eligible += int(has_learned_live)
            if abs(float(record.get("applied_delta") or 0.0)) > 0:
                learned_applied += 1
                source_counter["learned_policy_safe_soft"] += 1
                learned_live_applied += int(has_learned_live)
            if record.get("changed_top1"):
                learned_top1_changed += 1
            if outcome.get("learned_changed_top1"):
                learned_live_top1_changed += int(has_learned_live)
            learned_confidence[_confidence_bucket(float(record.get("confidence") or 0.0))] += 1
    top1_rate = len(top1_changed) / len(influenced) if influenced else 0.0
    mean_reward = round(sum(rewards) / len(rewards), 4) if rewards else 0.0
    recommendation = "keep"
    bandit_recommendation = "keep shadow"
    learned_recommendation = "keep shadow"
    if safety_warning_count:
        recommendation = "disable"
        bandit_recommendation = "disable"
        learned_recommendation = "disable"
    elif top1_rate > 0.5:
        recommendation = "reduce"
        bandit_recommendation = "keep shadow"
        learned_recommendation = "keep shadow"
    elif bandit_applied:
        bandit_recommendation = "allow tiny weight"
    if learned_applied and not safety_warning_count:
        learned_recommendation = "tiny influence"
    return {
        "total_rounds_influenced": len(influenced),
        "top1_change_rate": top1_rate,
        "reward_comparison_against_baseline_counterfactual": {
            "mean_online_reward": mean_reward,
            "baseline_counterfactual_mean_reward": 0.0,
            "delta": mean_reward,
        },
        "safety_warning_count": safety_warning_count,
        "influence_source_distribution": dict(source_counter),
        "bandit_eligible_rounds": bandit_eligible,
        "bandit_applied_rounds": bandit_applied,
        "bandit_top1_changed_rounds": bandit_top1_changed,
        "bandit_reward_vs_baseline_counterfactual": {
            "mean_online_reward": mean_reward,
            "baseline_counterfactual_mean_reward": 0.0,
            "delta": mean_reward,
        },
        "bandit_confidence_calibration_after_online_outcomes": {},
        "bandit_recommendation": bandit_recommendation,
        "learned_eligible_rounds": learned_eligible,
        "learned_applied_rounds": learned_applied,
        "learned_top1_changed_rounds": learned_top1_changed,
        "learned_live_eligible_rounds": learned_live_eligible,
        "learned_live_applied_rounds": learned_live_applied,
        "learned_live_top1_changed_rounds": learned_live_top1_changed,
        "learned_reward_vs_baseline_counterfactual": {
            "mean_online_reward": mean_reward,
            "baseline_counterfactual_mean_reward": 0.0,
            "delta": mean_reward,
        },
        "learned_reward_vs_safe_influence_counterfactual": {
            "mean_online_reward": mean_reward,
            "safe_influence_counterfactual_mean_reward": 0.0,
            "delta": mean_reward,
        },
        "learned_failure_rate_comparison": {
            "online_failure_rate": _failure_rate(influenced),
            "baseline_counterfactual_failure_rate": 0.0,
            "safe_influence_counterfactual_failure_rate": 0.0,
        },
        "learned_confidence_calibration": dict(learned_confidence),
        "learned_safety_warnings": safety_warning_count,
        "learned_recommendation": learned_recommendation,
        "recommendation": recommendation,
    }


def _as_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    return dict(value)


def _confidence_bucket(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.4:
        return "medium"
    return "low"


def _failure_rate(outcomes: list[dict[str, Any]]) -> float:
    if not outcomes:
        return 0.0
    failures = sum(1 for outcome in outcomes if outcome.get("failure_events"))
    return round(failures / len(outcomes), 4)
