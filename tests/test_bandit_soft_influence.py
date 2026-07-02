from __future__ import annotations

from app.services.backend_memory import problem_context_key
from app.services.bandit_influence import bandit_soft_influence
from app.services.online_influence import OnlineInfluenceController
from app.services.strategy_models import (
    CampaignContext,
    CampaignSnapshot,
    ContextualBanditDecision,
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
}


def _snapshot(**kwargs):
    params = tuple({"x": i / 12, "y": (i % 4) / 4} for i in range(12))
    kpis = tuple(0.1 + 0.03 * i for i in range(12))
    base = dict(
        round_number=4,
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
        campaign_id="bandit-campaign",
        campaign_context=CampaignContext(current_objective_level="baseline"),
    )
    base.update(kwargs)
    return CampaignSnapshot(**base)


def _bandit_stats(snapshot):
    context_key = problem_context_key(snapshot)
    return {
        f"{context_key}|explore:lhs": {"reward": 10.0, "n": 10.0},
        f"{context_key}|explore:nexus_lhs": {"reward": 20.0, "n": 10.0},
        f"{context_key}|explore:nexus_sobol": {"reward": 8.0, "n": 10.0},
    }


def _trace(snapshot, rollout=None, influence=None):
    config = PhaseConfig(
        policy_influence=influence or PolicyInfluenceConfig(),
        online_influence_rollout=rollout or OnlineInfluenceRolloutConfig(),
    )
    return strategy_trace_to_dict(select_strategy(snapshot, config=config).strategy_trace)


def test_default_bandit_remains_shadow_only():
    snap = _snapshot(strategy_bandit_stats=_bandit_stats(_snapshot()))
    trace = _trace(snap)

    assert trace["bandit_influence"] is None
    assert not [
        record for record in trace["ranking_influences"]
        if record["source"] == "bandit_soft"
    ]


def test_ineligible_bandit_has_zero_delta_with_reasons():
    snap = _snapshot(strategy_bandit_stats=_bandit_stats(_snapshot()))
    trace = _trace(
        snap,
        OnlineInfluenceRolloutConfig(
            enabled=True,
            mode=OnlineInfluenceMode.SAFE_SOFT,
            enable_bandit_soft_influence=True,
            allowed_campaign_ids=("bandit-campaign",),
        ),
        PolicyInfluenceConfig(enable_bandit_rerank=True),
    )

    bandit = trace["bandit_influence"]
    assert bandit["eligibility"]["eligible"] is False
    assert bandit["score_delta"] == 0.0
    assert bandit["eligibility"]["reasons"]
    assert any(
        record["source"] == "bandit_shadow" and record["score_delta"] == 0.0
        for record in trace["ranking_influences"]
    )


def test_eligible_bandit_applies_only_bounded_tiny_delta_to_existing_backend():
    base = _snapshot()
    snap = _snapshot(strategy_bandit_stats=_bandit_stats(base))
    trace = _trace(
        snap,
        OnlineInfluenceRolloutConfig(
            enabled=True,
            mode=OnlineInfluenceMode.SAFE_SOFT,
            enable_bandit_soft_influence=True,
            allowed_campaign_ids=("bandit-campaign",),
            allowed_objective_levels=("baseline",),
            bandit_allowed_objective_levels=("baseline",),
            bandit_min_confidence=0.1,
            bandit_min_calibration_score=0.1,
            max_bandit_weight=0.01,
        ),
        PolicyInfluenceConfig(
            enable_bandit_rerank=True,
            bandit_offline_eval_passed=True,
            bandit_calibration_score=0.8,
            max_bandit_weight=0.01,
        ),
    )

    bandit = trace["bandit_influence"]
    assert bandit["eligibility"]["eligible"] is True
    assert bandit["suggested_backend"] in {
        backend["name"] for backend in trace["candidate_backends"]
    }
    assert 0.0 < bandit["score_delta"] <= 0.01
    assert any(
        record["source"] == "bandit_soft"
        and record["target"] == bandit["suggested_backend"]
        and 0.0 < record["score_delta"] <= 0.01
        for record in trace["ranking_influences"]
    )


def test_bandit_cannot_add_backend_or_hard_veto():
    snap = _snapshot()
    decision = ContextualBanditDecision(
        selected_arm="explore:not_a_backend",
        context_key="ctx",
        arm_scores=(),
        reason="test",
        suggested_action="explore",
        suggested_backend="not_a_backend",
        confidence=1.0,
    )

    deltas, records, _evidence, bandit = bandit_soft_influence(
        snapshot=snap,
        decision=decision,
        backend_pool=("lhs", "nexus_lhs"),
        policy_config=PolicyInfluenceConfig(
            enable_bandit_rerank=True,
            bandit_offline_eval_passed=True,
            bandit_calibration_score=1.0,
            max_bandit_weight=0.01,
        ),
        rollout=OnlineInfluenceRolloutConfig(
            enabled=True,
            mode=OnlineInfluenceMode.SAFE_SOFT,
            enable_bandit_soft_influence=True,
            bandit_min_confidence=0.1,
            bandit_min_calibration_score=0.1,
        ),
    )

    assert deltas == {}
    assert bandit.eligibility.eligible is False
    assert records[0].source == "bandit_shadow"
    assert records[0].score_delta == 0.0


def test_bandit_cannot_affect_space_revision():
    trace = _trace(
        _snapshot(
            campaign_context=CampaignContext(
                current_objective_level="generalization",
                synthesis_routes=("electrodeposition", "sol-gel"),
            ),
            strategy_bandit_stats=_bandit_stats(_snapshot()),
        ),
        OnlineInfluenceRolloutConfig(
            enabled=True,
            mode=OnlineInfluenceMode.SAFE_SOFT,
            enable_bandit_soft_influence=True,
            allowed_campaign_ids=("bandit-campaign",),
            allowed_objective_levels=("generalization",),
        ),
        PolicyInfluenceConfig(
            enable_bandit_rerank=True,
            bandit_offline_eval_passed=True,
            bandit_calibration_score=1.0,
        ),
    )

    assert trace["space_revision"] is not None
    assert trace["space_revision"]["auto_applied"] is False


def test_bandit_auto_disable_triggers_on_safety_violation():
    controller = OnlineInfluenceController(
        OnlineInfluenceRolloutConfig(enabled=True, mode=OnlineInfluenceMode.SAFE_SOFT)
    )
    controller.record(OnlineInfluenceOutcome(
        mode=OnlineInfluenceMode.SAFE_SOFT,
        enabled=True,
        baseline_top_backend="lhs",
        influenced_top_backend="nexus_lhs",
        top1_changed=True,
        applied_influences=(
            RankingInfluenceRecord(
                source="bandit_soft",
                target="nexus_lhs",
                raw_signal=1.0,
                applied_weight=0.01,
                score_delta=0.2,
                capped=False,
                reason="invalid",
            ),
        ),
        safety_warnings=("bandit_ineligible_nonzero_delta",),
        auto_disabled=True,
    ))

    report = controller.report()
    assert controller.disabled is True
    assert report["bandit_applied_rounds"] == 1
    assert report["bandit_recommendation"] == "disable"
