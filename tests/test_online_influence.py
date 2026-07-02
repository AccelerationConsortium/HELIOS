from __future__ import annotations

import dataclasses

from app.services.backend_memory import BackendPerformanceMemory, problem_context_key
from app.services.online_influence import (
    OnlineInfluenceController,
    post_run_influence_report,
)
from app.services.strategy_models import (
    CampaignContext,
    CampaignSnapshot,
    LearnedPolicyInfluenceRecord,
    LearnedPolicyPromotionGateResult,
    OnlineInfluenceMode,
    OnlineInfluenceOutcome,
    OnlineInfluenceRolloutConfig,
    PolicyInfluenceConfig,
    RankingInfluenceRecord,
)
from app.services.strategy_selector import PhaseConfig, select_strategy, strategy_trace_to_dict

AVAIL = {
    "lhs": True,
    "built_in": True,
    "random_sampling": True,
    "nexus_lhs": True,
    "nexus_sobol": True,
    "optuna_tpe": True,
    "nexus_tpe": True,
    "nexus_gp_bo": True,
    "bomcp": True,
}


def _snapshot(**kwargs):
    params = tuple({"x": i / 12, "y": (i % 4) / 4} for i in range(12))
    kpis = tuple(0.1 + 0.03 * i for i in range(12))
    base = dict(
        round_number=3,
        max_rounds=12,
        n_observations=12,
        n_dimensions=2,
        has_categorical=False,
        has_log_scale=False,
        kpi_history=kpis,
        direction="maximize",
        available_backends=dict(AVAIL),
        last_batch_kpis=kpis[-3:],
        last_batch_params=params[-3:],
        best_kpi_so_far=max(kpis),
        all_params=params,
        all_kpis=kpis,
        campaign_id="online-allowed",
        campaign_context=CampaignContext(current_objective_level="baseline"),
    )
    base.update(kwargs)
    return CampaignSnapshot(**base)


def _trace(snapshot, rollout=None, influence=None):
    config = PhaseConfig(
        policy_influence=influence or PolicyInfluenceConfig(),
        online_influence_rollout=rollout or OnlineInfluenceRolloutConfig(),
    )
    return strategy_trace_to_dict(select_strategy(snapshot, config=config).strategy_trace)


def test_online_default_config_preserves_existing_behavior():
    snap = _snapshot()
    default_trace = _trace(snap)
    disabled_trace = _trace(
        snap,
        OnlineInfluenceRolloutConfig(enabled=False, mode=OnlineInfluenceMode.SAFE_SOFT),
    )

    assert disabled_trace["selected_backend"] == default_trace["selected_backend"]
    assert disabled_trace["ranking_influences"] == []
    assert disabled_trace["online_influence_outcome"] is None


def test_safe_soft_only_applies_action_policy_and_backend_memory_influence():
    snap = _snapshot()
    memory = BackendPerformanceMemory()
    memory.record_failure_event(
        context_key=problem_context_key(snap),
        backend_name="nexus_lhs",
        action_type="explore",
        failure_type="backend",
    )
    snap = dataclasses.replace(
        snap,
        backend_performance_records=memory.to_json(),
    )
    trace = _trace(
        snap,
        OnlineInfluenceRolloutConfig(
            enabled=True,
            mode=OnlineInfluenceMode.SAFE_SOFT,
            allowed_campaign_ids=("online-allowed",),
            allowed_objective_levels=("baseline",),
            max_rounds=5,
        ),
        PolicyInfluenceConfig(
            enable_action_policy_rerank=True,
            enable_backend_memory_rerank=True,
            enable_bandit_rerank=True,
            enable_transition_guard_penalty=True,
        ),
    )

    sources = {record["source"] for record in trace["ranking_influences"]}
    assert "action_policy" in sources
    assert "backend_memory" in sources
    assert "transition_guard" not in sources
    assert all(record["applied_weight"] <= 0.03 for record in trace["ranking_influences"])
    assert trace["online_influence_outcome"]["enabled"] is True


def test_online_bandit_remains_shadow_only():
    trace = _trace(
        _snapshot(),
        OnlineInfluenceRolloutConfig(
            enabled=True,
            mode=OnlineInfluenceMode.SAFE_SOFT,
            allowed_campaign_ids=("online-allowed",),
        ),
        PolicyInfluenceConfig(enable_bandit_rerank=True),
    )

    assert all(
        record["source"] != "bandit_shadow" or record["score_delta"] == 0.0
        for record in trace["ranking_influences"]
    )


def test_online_space_revision_remains_approval_only():
    trace = _trace(
        _snapshot(
            campaign_context=CampaignContext(
                current_objective_level="generalization",
                synthesis_routes=("electrodeposition", "sol-gel"),
            ),
            previous_intent="optimize",
        ),
        OnlineInfluenceRolloutConfig(
            enabled=True,
            mode=OnlineInfluenceMode.SAFE_SOFT,
            allowed_campaign_ids=("online-allowed",),
            allowed_objective_levels=("generalization",),
        ),
    )

    assert trace["space_revision"] is not None
    assert trace["space_revision"]["auto_applied"] is False
    assert trace["online_influence_outcome"]["auto_disabled"] is False


def test_online_auto_disable_and_post_run_report():
    outcome = OnlineInfluenceOutcome(
        mode=OnlineInfluenceMode.SAFE_SOFT,
        enabled=True,
        baseline_top_backend="lhs",
        influenced_top_backend="nexus_lhs",
        top1_changed=True,
        applied_influences=(
            RankingInfluenceRecord(
                source="action_policy",
                target="nexus_lhs",
                raw_signal=1.0,
                applied_weight=0.03,
                score_delta=0.03,
                capped=False,
                reason="test",
            ),
        ),
        reward=0.1,
        safety_warnings=("score_delta_within_total_cap",),
        auto_disabled=True,
    )
    controller = OnlineInfluenceController(
        OnlineInfluenceRolloutConfig(enabled=True, mode=OnlineInfluenceMode.SAFE_SOFT)
    )
    controller.record(outcome)
    report = post_run_influence_report(controller.outcomes)

    assert controller.disabled is True
    assert report["total_rounds_influenced"] == 1
    assert report["top1_change_rate"] == 1.0
    assert report["safety_warning_count"] == 1
    assert report["recommendation"] == "disable"


def test_post_run_report_includes_learned_policy_safe_soft_fields():
    outcome = OnlineInfluenceOutcome(
        mode=OnlineInfluenceMode.SAFE_SOFT,
        enabled=True,
        baseline_top_backend="lhs",
        influenced_top_backend="lhs",
        top1_changed=False,
        learned_policy_influences=(
            LearnedPolicyInfluenceRecord(
                policy_id="policy-safe",
                policy_version="v1",
                eligibility=LearnedPolicyPromotionGateResult(eligible=True),
                suggested_backend="lhs",
                target_backend="lhs",
                raw_delta=0.02,
                applied_delta=0.005,
                capped=True,
                confidence=0.9,
                changed_top1=False,
                safety_mask_valid=True,
            ),
        ),
        reward=0.2,
    )

    report = post_run_influence_report((outcome,))

    assert report["learned_eligible_rounds"] == 1
    assert report["learned_applied_rounds"] == 1
    assert report["learned_top1_changed_rounds"] == 0
    assert report["learned_confidence_calibration"]["high"] == 1
    assert report["learned_recommendation"] == "tiny influence"
