"""Contextual-bandit eligibility and bounded soft influence."""
from __future__ import annotations

from typing import Any

from app.services.strategy_models import (
    BanditEligibilityResult,
    BanditInfluenceRecord,
    CampaignSnapshot,
    ContextualBanditDecision,
    OnlineInfluenceMode,
    OnlineInfluenceRolloutConfig,
    PolicyInfluenceConfig,
    RankingInfluenceRecord,
    StrategyEvidence,
)


def _value(value: Any) -> str:
    return str(getattr(value, "value", value))


def calibration_bucket(score: float) -> str:
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "medium"
    return "low"


class BanditEligibilityGate:
    """Gate contextual-bandit influence before any score delta is applied."""

    def evaluate(
        self,
        *,
        snapshot: CampaignSnapshot,
        decision: ContextualBanditDecision | None,
        backend_pool: tuple[str, ...],
        policy_config: PolicyInfluenceConfig,
        rollout: OnlineInfluenceRolloutConfig,
    ) -> BanditEligibilityResult:
        reasons: list[str] = []
        mode = _value(rollout.mode)
        if not rollout.enabled:
            reasons.append("online rollout disabled")
        if mode not in {OnlineInfluenceMode.SAFE_SOFT.value, OnlineInfluenceMode.EVALUATION.value}:
            reasons.append("bandit soft influence allowed only in SAFE_SOFT or EVALUATION")
        if not rollout.enable_bandit_soft_influence:
            reasons.append("bandit soft influence flag disabled")
        if not policy_config.enable_bandit_rerank:
            reasons.append("policy bandit rerank flag disabled")
        if not policy_config.bandit_offline_eval_passed:
            reasons.append("offline evaluation has not passed")
        if snapshot.n_observations < rollout.bandit_min_observations:
            reasons.append("not enough observations")
        if decision is None:
            reasons.append("no bandit decision available")
            confidence = 0.0
            suggested_backend = ""
        else:
            confidence = decision.confidence
            suggested_backend = decision.suggested_backend
        if confidence < rollout.bandit_min_confidence:
            reasons.append("bandit confidence below threshold")
        if policy_config.bandit_calibration_score < rollout.bandit_min_calibration_score:
            reasons.append("bandit calibration below threshold")
        context = snapshot.campaign_context
        level = _value(context.current_objective_level if context else "performance")
        allowed_levels = tuple(_value(item) for item in rollout.bandit_allowed_objective_levels)
        if allowed_levels and level not in allowed_levels:
            reasons.append("objective level not allow-listed for bandit")
        disallowed = {_value(item) for item in rollout.bandit_disallowed_failure_types}
        active_failures = {_value(event.failure_type) for event in snapshot.failure_events}
        blocked_failures = sorted(active_failures & disallowed)
        if blocked_failures:
            reasons.append(f"active failure type disallows bandit: {','.join(blocked_failures)}")
        if suggested_backend and suggested_backend not in backend_pool:
            reasons.append("bandit suggested backend is outside candidate pool")
        return BanditEligibilityResult(
            eligible=not reasons,
            reasons=tuple(reasons),
            calibration_bucket=calibration_bucket(policy_config.bandit_calibration_score),
        )


def bandit_soft_influence(
    *,
    snapshot: CampaignSnapshot,
    decision: ContextualBanditDecision | None,
    backend_pool: tuple[str, ...],
    policy_config: PolicyInfluenceConfig,
    rollout: OnlineInfluenceRolloutConfig,
) -> tuple[dict[str, float], tuple[RankingInfluenceRecord, ...], tuple[StrategyEvidence, ...], BanditInfluenceRecord]:
    """Return bounded bandit deltas plus trace records."""
    eligibility = BanditEligibilityGate().evaluate(
        snapshot=snapshot,
        decision=decision,
        backend_pool=backend_pool,
        policy_config=policy_config,
        rollout=rollout,
    )
    suggested_action = decision.suggested_action if decision is not None else ""
    suggested_backend = decision.suggested_backend if decision is not None else ""
    confidence = decision.confidence if decision is not None else 0.0
    if not eligibility.eligible:
        record = BanditInfluenceRecord(
            suggested_action=suggested_action,
            suggested_backend=suggested_backend,
            confidence=confidence,
            eligibility=eligibility,
            applied_weight=0.0,
            score_delta=0.0,
            capped=False,
            calibration_bucket=eligibility.calibration_bucket,
            reason="bandit ineligible; shadow-only record",
        )
        ranking = RankingInfluenceRecord(
            source="bandit_shadow",
            target=suggested_backend,
            raw_signal=confidence,
            applied_weight=0.0,
            score_delta=0.0,
            capped=False,
            reason="Contextual bandit ineligible for soft influence",
        )
        return {}, (ranking,), (), record

    raw_signal = max(0.0, confidence)
    max_weight = min(policy_config.max_bandit_weight, rollout.max_bandit_weight)
    proposed = raw_signal * max_weight
    delta = round(min(max_weight, proposed), 4)
    capped = abs(delta - proposed) > 1e-12
    record = BanditInfluenceRecord(
        suggested_action=suggested_action,
        suggested_backend=suggested_backend,
        confidence=confidence,
        eligibility=eligibility,
        applied_weight=max_weight,
        score_delta=delta,
        capped=capped,
        calibration_bucket=eligibility.calibration_bucket,
        reason="eligible contextual bandit applied tiny bounded backend score delta",
    )
    ranking = RankingInfluenceRecord(
        source="bandit_soft",
        target=suggested_backend,
        raw_signal=confidence,
        applied_weight=max_weight,
        score_delta=delta,
        capped=capped,
        reason="Eligible contextual bandit soft influence applied to existing backend",
    )
    evidence = StrategyEvidence(
        source="ranking_influence:bandit_soft",
        target=suggested_backend,
        effect="boost",
        strength=abs(delta),
        reason="Bounded contextual-bandit soft influence applied",
        metadata={
            "confidence": confidence,
            "calibration_bucket": eligibility.calibration_bucket,
        },
    )
    return {suggested_backend: delta}, (ranking,), (evidence,), record
