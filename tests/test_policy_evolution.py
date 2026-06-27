from __future__ import annotations

import dataclasses

from app.services.policy_evolution import (
    EvolutionGuard,
    PolicyEvolutionManager,
    PolicyEvolutionPlan,
    PolicyEvolutionPlanStatus,
    PolicyEvolutionRecommendation,
    PolicyEvolutionTrigger,
    PolicyEvolutionTriggerType,
    PolicyVersionRegistry,
    PolicyVersionRegistryEntry,
)
from app.services.strategy_selector import PhaseConfig, select_strategy
from tests.fixtures.strategy_replay import all_replay_scenarios


def _trigger(**kwargs):
    return PolicyEvolutionTrigger(
        trigger_type=kwargs.pop(
            "trigger_type",
            PolicyEvolutionTriggerType.NEW_TRACES_AVAILABLE,
        ),
        trigger_reason=kwargs.pop("trigger_reason", "new offline traces"),
        campaign_ids=kwargs.pop("campaign_ids", ("campaign-a", "campaign-b")),
        trace_count=kwargs.pop("trace_count", 42),
        dataset_version=kwargs.pop("dataset_version", "policy_dataset_v1"),
        metadata=kwargs.pop("metadata", {"source": "offline_replay"}),
        **kwargs,
    )


def _registry():
    base = PolicyVersionRegistryEntry(
        policy_id="policy-a",
        policy_version="v1",
        trained_on_dataset_version="dataset-v1",
        feature_schema_version="policy_feature_schema_v1",
        reward_version="strategy_reward_v1",
        approved_for_shadow=True,
    )
    candidate = PolicyVersionRegistryEntry(
        policy_id="policy-a",
        policy_version="v2",
        parent_policy_id="policy-a",
        parent_policy_version="v1",
        trained_on_dataset_version="dataset-v2",
        feature_schema_version="policy_feature_schema_v1",
        reward_version="strategy_reward_v1",
        offline_evaluation_summary={"passed": True},
        shadow_summary={"n_records": 20},
        canary_summary={"passed": True},
        approved_for_shadow=True,
        approved_for_safe_soft=True,
        approved_for_live_canary=True,
        rollback_target=("policy-a", "v1"),
    )
    return PolicyVersionRegistry().register(base).register(candidate)


def _safe_plan(**kwargs):
    reward_version = kwargs.pop("reward_version", "strategy_reward_v1")
    return PolicyEvolutionPlan(
        plan_id="plan-a",
        source_policy_id="policy-a",
        source_policy_version="v1",
        candidate_policy_id="policy-a",
        candidate_policy_version="v2",
        trigger=_trigger(),
        dataset_version="policy_dataset_v2",
        feature_schema_version="policy_feature_schema_v1",
        reward_version=reward_version,
        **kwargs,
    )


def test_evolution_trigger_serialization_and_round_trip():
    trigger = _trigger(trigger_type=PolicyEvolutionTriggerType.MANUAL_REQUEST)

    raw = trigger.to_dict()
    restored = PolicyEvolutionTrigger.from_dict(raw)

    assert raw["trigger_type"] == "manual_request"
    assert restored.trigger_type == "manual_request"
    assert restored.campaign_ids == trigger.campaign_ids
    assert restored.trace_count == 42
    assert restored.dataset_version == "policy_dataset_v1"


def test_evolution_plan_status_transitions_and_round_trip():
    manager = PolicyEvolutionManager()
    plan = _safe_plan()

    dataset_ready = manager.update_plan_status(
        plan,
        PolicyEvolutionPlanStatus.DATASET_READY,
        "dataset audit passed",
    )
    restored = PolicyEvolutionPlan.from_dict(dataset_ready.to_dict())

    assert dataset_ready.status == "dataset_ready"
    assert restored.status == "dataset_ready"
    assert restored.reasons[-1] == "dataset audit passed"
    assert restored.updated_at >= restored.created_at
    assert manager.recommend_next_step(dataset_ready) == PolicyEvolutionRecommendation.RUN_OFFLINE_EVAL


def test_policy_registry_lineage_lookup_and_latest_helpers():
    registry = _registry()

    shadow = registry.get_latest_approved_shadow_policy()
    canary = registry.get_latest_canary_eligible_policy()
    lineage = registry.get_policy_lineage("policy-a", "v2")

    assert shadow.policy_version == "v2"
    assert canary.policy_version == "v2"
    assert [entry.policy_version for entry in lineage] == ["v2", "v1"]


def test_policy_registry_rollback_target_lookup():
    registry = _registry()

    rollback = registry.get_rollback_target("policy-a", "v2")

    assert rollback is not None
    assert rollback.policy_id == "policy-a"
    assert rollback.policy_version == "v1"


def test_evolution_guard_blocks_unsafe_changes():
    plan = _safe_plan(
        proposed_changes={
            "change_safety_constraints": True,
            "lower_approval_required": True,
            "bypass_promotion_gates": True,
            "max_score_delta_cap": 0.5,
        },
    )

    result = EvolutionGuard(max_allowed_score_delta_cap=0.01).evaluate(plan)

    assert result.allowed is False
    checks = {violation["check"] for violation in result.violations}
    assert checks >= {
        "change_safety_constraints",
        "lower_approval_required",
        "bypass_promotion_gates",
        "score_delta_cap_too_high",
    }
    assert result.required_human_approval is True


def test_unknown_counterfactual_cannot_be_treated_as_ground_truth():
    plan = _safe_plan(
        proposed_changes={"unknown_counterfactual_as_ground_truth": True},
    )

    result = EvolutionGuard().evaluate(plan)

    assert result.allowed is False
    assert "unknown_counterfactual_as_ground_truth" in {
        violation["check"] for violation in result.violations
    }


def test_scientific_negative_cannot_penalize_backend():
    plan = _safe_plan(
        proposed_changes={"penalize_scientific_negative_backend": True},
    )

    result = EvolutionGuard().evaluate(plan)

    assert result.allowed is False
    assert "penalize_scientific_negative_backend" in {
        violation["check"] for violation in result.violations
    }


def test_auto_apply_space_revision_is_rejected():
    plan = _safe_plan(
        proposed_changes={"auto_apply_space_revision": True},
    )

    result = EvolutionGuard().evaluate(plan)

    assert result.allowed is False
    assert "auto_apply_space_revision" in {
        violation["check"] for violation in result.violations
    }


def test_live_influence_cannot_be_enabled_through_evolution_plan():
    plan = _safe_plan(
        proposed_changes={"enable_live_influence_directly": True},
    )

    result = EvolutionGuard().evaluate(plan)

    assert result.allowed is False
    assert "enable_live_influence_directly" in {
        violation["check"] for violation in result.violations
    }


def test_reward_version_change_requires_explicit_version_bump():
    blocked = _safe_plan(
        reward_version="strategy_reward_v2",
        proposed_changes={"reward_version": "strategy_reward_v2"},
    )
    allowed = _safe_plan(
        reward_version="strategy_reward_v2",
        proposed_changes={
            "reward_version": "strategy_reward_v2",
            "explicit_reward_version_bump": True,
        },
    )

    blocked_result = EvolutionGuard(current_reward_version="strategy_reward_v1").evaluate(blocked)
    allowed_result = EvolutionGuard(current_reward_version="strategy_reward_v1").evaluate(allowed)

    assert blocked_result.allowed is False
    assert "reward_version_without_explicit_bump" in {
        violation["check"] for violation in blocked_result.violations
    }
    assert allowed_result.allowed is True
    assert allowed_result.required_human_approval is True


def test_manager_creates_plans_but_does_not_train_or_modify_live_selector():
    snapshot = all_replay_scenarios()[0]
    before = select_strategy(snapshot, config=PhaseConfig())
    registry = _registry()
    manager = PolicyEvolutionManager()

    plan = manager.create_evolution_plan(
        _trigger(trigger_type=PolicyEvolutionTriggerType.SHADOW_POLICY_OUTPERFORMED),
        registry,
        dataset_summary={
            "dataset_version": "dataset-v3",
            "feature_schema_version": "policy_feature_schema_v1",
            "reward_version": "strategy_reward_v1",
        },
        audit_summary={"audit_version": "policy_dataset_audit_v1", "record_count": 42},
    )
    guard = manager.evaluate_plan_guard(plan)
    report = manager.build_report(
        plan,
        dataset_summary={"dataset_version": "dataset-v3", "ready": True},
        audit_summary={"passed": True},
        reward_sanity_summary={"passed": True},
        offline_benchmark_summary={"passed": False},
    )
    after = select_strategy(snapshot, config=PhaseConfig())

    assert plan.source_policy_id == "policy-a"
    assert plan.source_policy_version == "v2"
    assert plan.candidate_policy_version == "v2.candidate"
    assert plan.promotion_allowed is False
    assert guard.allowed is True
    assert report.recommendation == PolicyEvolutionRecommendation.PREPARE_DATASET
    assert not hasattr(manager, "train")
    assert after.backend_name == before.backend_name
    assert after.phase == before.phase


def test_default_bo_mcp_nexus_backend_behavior_remains_unchanged():
    snapshot = all_replay_scenarios()[2]
    before = select_strategy(snapshot, config=PhaseConfig())
    manager = PolicyEvolutionManager()
    plan = manager.create_evolution_plan(_trigger(), PolicyVersionRegistry())

    _ = dataclasses.asdict(manager.build_report(plan))
    after = select_strategy(snapshot, config=PhaseConfig())

    assert after.backend_name == before.backend_name
    assert after.strategy_trace.selected_backend == before.strategy_trace.selected_backend
