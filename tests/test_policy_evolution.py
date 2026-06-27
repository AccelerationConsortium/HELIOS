from __future__ import annotations

import dataclasses
import inspect

from app.optimization.backend_selection import rank_backends
import app.services.policy_evolution as policy_evolution
from app.services.policy_evolution import (
    CanaryApprovalGuard,
    CanaryApprovalMode,
    CanaryApprovalRecord,
    CanaryPromotionGuard,
    CanaryPromotionProposal,
    CanaryRunRecommendation,
    CanaryRunResult,
    CanaryRunScheduleStatus,
    CandidatePolicyArtifact,
    CandidatePolicyTrainingJobStatus,
    CandidatePolicyTrainingMode,
    EvolutionGuard,
    PolicyEvolutionManager,
    PolicyEvolutionPlan,
    PolicyEvolutionPlanStatus,
    PolicyEvolutionRecommendation,
    PolicyEvolutionTrigger,
    PolicyEvolutionTriggerType,
    PolicyAutoTrainer,
    PolicyVersionRegistry,
    PolicyVersionRegistryEntry,
    ShadowApprovalGuard,
    ShadowApprovalMode,
    ShadowApprovalRecord,
    ShadowPromotionGuard,
    ShadowPromotionProposal,
    ShadowRunRecommendation,
    ShadowRunResult,
    ShadowRunScheduleStatus,
    TrainingGuard,
)
from app.services.learned_policy import (
    PolicyDataset,
    PolicyDatasetAuditor,
    PolicyDatasetBuilder,
    RewardSanityChecker,
    replay_records_from_traces,
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
    rollback_policy_id = kwargs.pop("rollback_policy_id", "policy-a")
    rollback_policy_version = kwargs.pop("rollback_policy_version", "v1")
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
        rollback_policy_id=rollback_policy_id,
        rollback_policy_version=rollback_policy_version,
        **kwargs,
    )


def _training_dataset():
    traces = [
        select_strategy(snapshot, config=PhaseConfig()).strategy_trace
        for snapshot in all_replay_scenarios()
    ]
    return PolicyDatasetBuilder().build(replay_records_from_traces(traces))


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
    assert report.recommendation == PolicyEvolutionRecommendation.TRAIN_CANDIDATE
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


def test_training_job_lifecycle_transitions():
    manager = PolicyEvolutionManager()
    plan = _safe_plan()

    job = manager.create_training_job(
        plan,
        training_mode=CandidatePolicyTrainingMode.IMITATION,
        training_config={"max_delta": 0.01},
    )
    built = manager.update_training_job_status(job, CandidatePolicyTrainingJobStatus.DATASET_BUILT)
    done = manager.update_training_job_status(built, CandidatePolicyTrainingJobStatus.OFFLINE_EVALUATED)

    assert job.status == "created"
    assert job.training_mode == "imitation"
    assert built.status == "dataset_built"
    assert done.status == "offline_evaluated"
    assert done.completed_at is not None


def test_training_guard_blocks_failed_audit():
    dataset = PolicyDataset((
        {
            "campaign_id": "c",
            "loop_id": "r",
            "state_features": {},
            "context_features": {},
            "available_actions": [],
            "selected_intent": "optimize",
            "selected_mode": "exploit",
            "selected_backend": "nexus_gp_bo",
            "candidate_backends": [],
            "applied_influences": [],
            "reward": None,
            "outcome": None,
            "safety_flags": [],
            "record_version": "policy_training_record_v1",
        },
    ))
    audit = PolicyDatasetAuditor().audit(dataset)
    reward = RewardSanityChecker().check(dataset)

    result = TrainingGuard().evaluate(_safe_plan(), dataset, audit, reward)

    assert result.allowed is False
    assert "dataset_audit_failed" in {violation["check"] for violation in result.violations}


def test_training_guard_blocks_failed_reward_sanity():
    dataset = _training_dataset()
    row = dict(dataset.records[0])
    row["reward"] = {"composite_reward": 0.1, "reward_version": dataset.reward_version}
    broken = PolicyDataset(
        (row,),
        dataset_version=dataset.dataset_version,
        feature_schema_version=dataset.feature_schema_version,
        reward_version=dataset.reward_version,
    )
    audit = PolicyDatasetAuditor().audit(broken)
    reward = RewardSanityChecker().check(broken)

    result = TrainingGuard().evaluate(_safe_plan(), broken, audit, reward)

    assert result.allowed is False
    assert "reward_sanity_failed" in {violation["check"] for violation in result.violations}


def test_training_guard_blocks_unknown_counterfactual_as_ground_truth_reward():
    dataset = _training_dataset()
    audit = PolicyDatasetAuditor().audit(dataset)
    reward = RewardSanityChecker().check(dataset)

    result = TrainingGuard().evaluate(
        _safe_plan(),
        dataset,
        audit,
        reward,
        training_config={"use_unknown_counterfactual_as_ground_truth": True},
    )

    assert result.allowed is False
    assert "unknown_counterfactual_as_ground_truth" in {
        violation["check"] for violation in result.violations
    }


def test_candidate_artifact_is_created_only_after_offline_evaluation():
    plan = _safe_plan()
    job, artifact, _registry = PolicyAutoTrainer().train_candidate(
        plan,
        dataset=_training_dataset(),
        training_mode=CandidatePolicyTrainingMode.IMITATION,
    )

    assert job.status == "offline_evaluated"
    assert job.completed_at is not None
    assert isinstance(artifact, CandidatePolicyArtifact)
    assert artifact.offline_evaluation_summary
    assert artifact.safety_summary["passed"] is True
    assert artifact.eligible_for_shadow_proposal is True
    assert artifact.eligible_for_canary_proposal is False


def test_registry_receives_candidate_entry_unapproved_by_default():
    registry = _registry()
    plan = _safe_plan()

    job, artifact, updated = PolicyAutoTrainer().train_candidate(
        plan,
        dataset=_training_dataset(),
        training_mode=CandidatePolicyTrainingMode.BACKEND_RERANKER,
        registry=registry,
    )
    entry = updated.get(plan.candidate_policy_id, plan.candidate_policy_version)

    assert job.status == "offline_evaluated"
    assert artifact.registry_entry_preview["approved_for_shadow"] is False
    assert entry is not None
    assert entry.approved_for_shadow is False
    assert entry.approved_for_safe_soft is False
    assert entry.approved_for_live_canary is False


def test_manager_attach_training_result_requires_offline_evaluation():
    manager = PolicyEvolutionManager()
    plan = _safe_plan()
    _job, artifact, _registry = PolicyAutoTrainer().train_candidate(
        plan,
        dataset=_training_dataset(),
        training_mode=CandidatePolicyTrainingMode.META_POLICY,
    )

    updated = manager.attach_training_result(plan, artifact)

    assert updated.status == "offline_evaluated"
    assert manager.recommend_next_step(updated) == PolicyEvolutionRecommendation.APPROVE_SHADOW


def test_policy_auto_trainer_does_not_import_or_modify_strategy_selector():
    source = inspect.getsource(policy_evolution)

    assert "from app.services.strategy_selector" not in source
    assert "import app.services.strategy_selector" not in source
    assert "rank_backends(" not in source


def test_live_rank_backends_behavior_remains_unchanged_after_auto_training():
    before = rank_backends(
        "optimize",
        ("nexus_gp_bo", "built_in"),
        {"nexus_gp_bo": True, "built_in": True},
    )

    PolicyAutoTrainer().train_candidate(
        _safe_plan(),
        dataset=_training_dataset(),
        training_mode=CandidatePolicyTrainingMode.IMITATION,
    )
    after = rank_backends(
        "optimize",
        ("nexus_gp_bo", "built_in"),
        {"nexus_gp_bo": True, "built_in": True},
    )

    assert after == before


def _trained_candidate():
    manager = PolicyEvolutionManager()
    plan = _safe_plan()
    job, artifact, registry = PolicyAutoTrainer().train_candidate(
        plan,
        dataset=_training_dataset(),
        training_mode=CandidatePolicyTrainingMode.IMITATION,
        registry=_registry(),
    )
    return manager, plan, job, artifact, registry


def test_shadow_promotion_proposal_round_trip_serialization():
    manager, plan, job, artifact, _registry = _trained_candidate()

    proposal = manager.create_shadow_promotion_proposal(plan, job, artifact)
    restored = ShadowPromotionProposal.from_dict(proposal.to_dict())

    assert proposal.status == "eligible"
    assert restored.proposal_id == proposal.proposal_id
    assert restored.candidate_policy_id == artifact.policy_id
    assert restored.required_approvals == ("human_shadow_approval",)
    assert restored.eligible is True


def test_shadow_guard_blocks_missing_offline_evaluation():
    manager, plan, job, artifact, _registry = _trained_candidate()
    incomplete = dataclasses.replace(artifact, offline_evaluation_summary={})
    proposal = manager.create_shadow_promotion_proposal(plan, job, incomplete)

    result = ShadowPromotionGuard().evaluate(proposal)

    assert result.allowed is False
    assert "missing_offline_evaluation" in {violation["check"] for violation in result.violations}


def test_shadow_guard_blocks_failed_dataset_audit():
    manager, plan, job, artifact, _registry = _trained_candidate()
    offline = dict(artifact.offline_evaluation_summary)
    offline["dataset_audit"] = {"passed": False}
    artifact = dataclasses.replace(artifact, offline_evaluation_summary=offline)
    proposal = manager.create_shadow_promotion_proposal(plan, job, artifact)

    result = ShadowPromotionGuard().evaluate(proposal)

    assert result.allowed is False
    assert "dataset_audit_failed" in {violation["check"] for violation in result.violations}


def test_shadow_guard_blocks_failed_reward_sanity():
    manager, plan, job, artifact, _registry = _trained_candidate()
    offline = dict(artifact.offline_evaluation_summary)
    offline["reward_sanity"] = {"passed": False}
    artifact = dataclasses.replace(artifact, offline_evaluation_summary=offline)
    proposal = manager.create_shadow_promotion_proposal(plan, job, artifact)

    result = ShadowPromotionGuard().evaluate(proposal)

    assert result.allowed is False
    assert "reward_sanity_failed" in {violation["check"] for violation in result.violations}


def test_shadow_guard_blocks_safety_violations():
    manager, plan, job, artifact, _registry = _trained_candidate()
    artifact = dataclasses.replace(
        artifact,
        safety_summary={"passed": False, "failure_count": 1},
    )
    proposal = manager.create_shadow_promotion_proposal(plan, job, artifact)

    result = ShadowPromotionGuard().evaluate(proposal)

    assert result.allowed is False
    assert "safety_violations_present" in {violation["check"] for violation in result.violations}


def test_shadow_guard_blocks_unknown_counterfactual_primary_evidence():
    manager, plan, job, artifact, _registry = _trained_candidate()
    offline = dict(artifact.offline_evaluation_summary)
    offline["counterfactual_uncertainty_summary"] = {
        "primary_improvement_evidence": "unknown_counterfactual",
    }
    artifact = dataclasses.replace(artifact, offline_evaluation_summary=offline)
    proposal = manager.create_shadow_promotion_proposal(plan, job, artifact)

    result = ShadowPromotionGuard().evaluate(proposal)

    assert result.allowed is False
    assert "unknown_counterfactual_primary_evidence" in {
        violation["check"] for violation in result.violations
    }


def test_shadow_guard_blocks_missing_rollback_target():
    manager, plan, job, artifact, _registry = _trained_candidate()
    plan = dataclasses.replace(
        plan,
        rollback_policy_id=None,
        rollback_policy_version=None,
    )
    proposal = manager.create_shadow_promotion_proposal(plan, job, artifact)

    result = ShadowPromotionGuard().evaluate(proposal)

    assert result.allowed is False
    assert "missing_rollback_target" in {violation["check"] for violation in result.violations}


def test_eligible_shadow_proposal_does_not_auto_approve_shadow():
    manager, plan, job, artifact, _registry = _trained_candidate()

    proposal = manager.create_shadow_promotion_proposal(plan, job, artifact)

    assert proposal.status == "eligible"
    assert artifact.registry_entry_preview["approved_for_shadow"] is False
    assert proposal.required_approvals == ("human_shadow_approval",)


def test_registry_stores_shadow_proposal_metadata_without_shadow_approval():
    manager, plan, job, artifact, registry = _trained_candidate()
    proposal = manager.create_shadow_promotion_proposal(plan, job, artifact)

    updated = registry.register_shadow_proposal(
        artifact.policy_id,
        artifact.policy_version,
        proposal,
    )
    entry = updated.get(artifact.policy_id, artifact.policy_version)

    assert entry.shadow_proposed is True
    assert entry.shadow_proposal_id == proposal.proposal_id
    assert entry.shadow_proposal_status == "eligible"
    assert entry.shadow_eligibility_summary["eligible"] is True
    assert entry.approved_for_shadow is False


def test_manager_recommends_approve_shadow_without_auto_approval():
    manager, plan, job, artifact, _registry = _trained_candidate()
    proposal = manager.create_shadow_promotion_proposal(plan, job, artifact)

    updated_plan = manager.attach_shadow_proposal(plan, proposal)

    assert updated_plan.status == "shadow_eligible"
    assert manager.recommend_next_step(updated_plan) == PolicyEvolutionRecommendation.KEEP_CURRENT
    assert artifact.registry_entry_preview["approved_for_shadow"] is False


def test_default_backend_behavior_remains_unchanged_after_shadow_proposal():
    snapshot = all_replay_scenarios()[2]
    before = select_strategy(snapshot, config=PhaseConfig())
    manager, plan, job, artifact, _registry = _trained_candidate()

    _ = manager.create_shadow_promotion_proposal(plan, job, artifact)
    after = select_strategy(snapshot, config=PhaseConfig())

    assert after.backend_name == before.backend_name
    assert after.strategy_trace.selected_backend == before.strategy_trace.selected_backend


def _eligible_shadow_proposal():
    manager, plan, job, artifact, registry = _trained_candidate()
    proposal = manager.create_shadow_promotion_proposal(plan, job, artifact)
    registry = registry.register_shadow_proposal(
        artifact.policy_id,
        artifact.policy_version,
        proposal,
    )
    return manager, plan, job, artifact, proposal, registry


def _approval_record(proposal):
    return ShadowApprovalRecord(
        approval_id="approval-a",
        proposal_id=proposal.proposal_id,
        policy_id=proposal.candidate_policy_id,
        policy_version=proposal.candidate_policy_version,
        approved_by="sissi",
        approval_mode=ShadowApprovalMode.TEST,
        approval_reason="test approval for shadow-only run",
        expires_at="2099-01-01T00:00:00+00:00",
        max_shadow_rounds=12,
        allowed_campaign_ids=("replay",),
        allowed_objective_levels=("performance",),
    )


def test_shadow_approval_record_round_trip():
    _manager, _plan, _job, _artifact, proposal, _registry = _eligible_shadow_proposal()
    approval = _approval_record(proposal)

    restored = ShadowApprovalRecord.from_dict(approval.to_dict())

    assert restored.approval_id == approval.approval_id
    assert restored.approval_mode == "test"
    assert restored.max_shadow_rounds == 12
    assert restored.allowed_campaign_ids == ("replay",)
    assert restored.revoked is False


def test_approval_guard_blocks_ineligible_proposal():
    _manager, _plan, _job, _artifact, proposal, registry = _eligible_shadow_proposal()
    ineligible = dataclasses.replace(proposal, eligible=False)

    result = ShadowApprovalGuard().evaluate(ineligible, _approval_record(proposal), registry)

    assert result.allowed is False
    assert "proposal_not_eligible" in {violation["check"] for violation in result.violations}


def test_approval_guard_blocks_expired_or_rejected_proposal():
    _manager, _plan, _job, _artifact, proposal, registry = _eligible_shadow_proposal()
    rejected = dataclasses.replace(proposal, status="rejected")

    result = ShadowApprovalGuard().evaluate(rejected, _approval_record(proposal), registry)

    assert result.allowed is False
    assert "proposal_status_blocked" in {violation["check"] for violation in result.violations}


def test_registry_sets_approved_for_shadow_only_with_explicit_approval():
    manager, _plan, _job, artifact, proposal, registry = _eligible_shadow_proposal()
    before = registry.get(artifact.policy_id, artifact.policy_version)
    approval = _approval_record(proposal)

    approved_proposal, updated, guard = manager.approve_shadow_proposal(
        proposal,
        approval,
        registry=registry,
    )
    after = updated.get(artifact.policy_id, artifact.policy_version)

    assert guard.allowed is True
    assert approved_proposal.status == "approved"
    assert before.approved_for_shadow is False
    assert after.approved_for_shadow is True
    assert after.approved_for_safe_soft is False
    assert after.approved_for_live_canary is False
    assert after.shadow_approval_metadata["approval_id"] == "approval-a"


def test_shadow_run_schedule_lifecycle_transitions():
    manager, _plan, _job, _artifact, proposal, _registry = _eligible_shadow_proposal()
    approval = _approval_record(proposal)

    schedule = manager.schedule_shadow_run(approval)
    running = manager.update_shadow_run_status(schedule, ShadowRunScheduleStatus.RUNNING)
    completed = manager.update_shadow_run_status(running, ShadowRunScheduleStatus.COMPLETED)

    assert schedule.status == "scheduled"
    assert schedule.max_rounds == 12
    assert running.status == "running"
    assert running.started_at is not None
    assert completed.status == "completed"
    assert completed.completed_at is not None


def test_shadow_run_result_attaches_to_evolution_plan():
    manager, plan, _job, artifact, _proposal, _registry = _eligible_shadow_proposal()
    result = ShadowRunResult(
        run_id="run-a",
        schedule_id="schedule-a",
        policy_id=artifact.policy_id,
        policy_version=artifact.policy_version,
        campaign_ids=("replay",),
        round_count=10,
        intent_agreement_rate=0.9,
        mode_agreement_rate=0.9,
        backend_agreement_rate=0.9,
        would_change_top1_rate=0.1,
        invalid_suggestion_rate=0.0,
        safety_warning_count=0,
        recommendation=ShadowRunRecommendation.CONTINUE_SHADOW,
        reasons=("not enough evidence for canary",),
    )

    updated = manager.attach_shadow_run_result(plan, result)

    assert updated.status == plan.status
    assert "shadow result does not support canary:run-a" in updated.reasons


def test_manager_recommends_propose_canary_only_after_passing_shadow_result():
    manager, plan, _job, artifact, _proposal, _registry = _eligible_shadow_proposal()
    eligible_plan = dataclasses.replace(plan, status="shadow_eligible")
    weak = ShadowRunResult(
        run_id="run-weak",
        schedule_id="schedule-a",
        policy_id=artifact.policy_id,
        policy_version=artifact.policy_version,
        round_count=5,
        backend_agreement_rate=0.4,
        invalid_suggestion_rate=0.0,
        safety_warning_count=0,
        recommendation=ShadowRunRecommendation.PROPOSE_CANARY,
    )
    strong = dataclasses.replace(
        weak,
        run_id="run-strong",
        round_count=20,
        backend_agreement_rate=0.9,
    )

    weak_plan = manager.attach_shadow_run_result(eligible_plan, weak)
    strong_plan = manager.attach_shadow_run_result(eligible_plan, strong)

    assert manager.recommend_next_step(eligible_plan) == PolicyEvolutionRecommendation.KEEP_CURRENT
    assert manager.recommend_next_step(weak_plan) == PolicyEvolutionRecommendation.KEEP_CURRENT
    assert manager.recommend_next_step(strong_plan) == PolicyEvolutionRecommendation.APPROVE_CANARY


def test_no_automatic_canary_approval_after_shadow_approval():
    manager, _plan, _job, artifact, proposal, registry = _eligible_shadow_proposal()
    approval = _approval_record(proposal)

    _proposal, updated, _guard = manager.approve_shadow_proposal(
        proposal,
        approval,
        registry=registry,
    )
    schedule = manager.schedule_shadow_run(approval)
    updated = updated.register_shadow_schedule(artifact.policy_id, artifact.policy_version, schedule)
    result = ShadowRunResult(
        run_id="run-canary",
        schedule_id=schedule.schedule_id,
        policy_id=artifact.policy_id,
        policy_version=artifact.policy_version,
        round_count=20,
        backend_agreement_rate=0.9,
        invalid_suggestion_rate=0.0,
        safety_warning_count=0,
        recommendation=ShadowRunRecommendation.PROPOSE_CANARY,
    )
    updated = updated.register_shadow_result(artifact.policy_id, artifact.policy_version, result)
    entry = updated.get(artifact.policy_id, artifact.policy_version)

    assert entry.approved_for_shadow is True
    assert entry.approved_for_safe_soft is False
    assert entry.approved_for_live_canary is False
    assert entry.shadow_run_schedule_metadata["schedule_id"] == schedule.schedule_id
    assert entry.latest_shadow_run_result_summary["run_id"] == "run-canary"


def test_learned_policy_still_does_not_affect_live_ranking_after_shadow_schedule():
    before = rank_backends(
        "optimize",
        ("nexus_gp_bo", "built_in"),
        {"nexus_gp_bo": True, "built_in": True},
    )
    manager, _plan, _job, _artifact, proposal, _registry = _eligible_shadow_proposal()

    _ = manager.schedule_shadow_run(_approval_record(proposal))
    after = rank_backends(
        "optimize",
        ("nexus_gp_bo", "built_in"),
        {"nexus_gp_bo": True, "built_in": True},
    )

    assert after == before


def test_default_backend_behavior_remains_unchanged_after_shadow_approval():
    snapshot = all_replay_scenarios()[2]
    before = select_strategy(snapshot, config=PhaseConfig())
    manager, _plan, _job, _artifact, proposal, _registry = _eligible_shadow_proposal()

    _ = manager.schedule_shadow_run(_approval_record(proposal))
    after = select_strategy(snapshot, config=PhaseConfig())

    assert after.backend_name == before.backend_name
    assert after.strategy_trace.selected_backend == before.strategy_trace.selected_backend


def _approved_shadow_state():
    manager, plan, _job, artifact, proposal, registry = _eligible_shadow_proposal()
    approval = _approval_record(proposal)
    _approved_proposal, registry, _guard = manager.approve_shadow_proposal(
        proposal,
        approval,
        registry=registry,
    )
    result = ShadowRunResult(
        run_id="shadow-run-a",
        schedule_id="schedule-a",
        policy_id=artifact.policy_id,
        policy_version=artifact.policy_version,
        campaign_ids=("replay",),
        round_count=20,
        intent_agreement_rate=0.9,
        mode_agreement_rate=0.9,
        backend_agreement_rate=0.9,
        would_change_top1_rate=0.1,
        invalid_suggestion_rate=0.0,
        safety_warning_count=0,
        confidence_calibration_summary={"calibration_score": 0.9},
        counterfactual_breakdown={"observed_outcome": 15, "unknown_counterfactual": 5},
        recommendation=ShadowRunRecommendation.PROPOSE_CANARY,
    )
    registry = registry.register_shadow_result(artifact.policy_id, artifact.policy_version, result)
    return manager, plan, artifact, approval, result, registry


def test_canary_promotion_proposal_round_trip_serialization():
    manager, plan, artifact, approval, result, registry = _approved_shadow_state()

    proposal = manager.create_canary_promotion_proposal(
        plan,
        result,
        shadow_approval=approval,
        registry=registry,
    )
    restored = CanaryPromotionProposal.from_dict(proposal.to_dict())

    assert proposal.status == "eligible"
    assert restored.proposal_id == proposal.proposal_id
    assert restored.policy_id == artifact.policy_id
    assert restored.required_approvals == ("human_canary_approval",)
    assert restored.eligible is True


def test_canary_guard_blocks_missing_shadow_result():
    manager, plan, _artifact, approval, _result, registry = _approved_shadow_state()

    proposal = manager.create_canary_promotion_proposal(
        plan,
        None,
        shadow_approval=approval,
        registry=registry,
    )
    guard = CanaryPromotionGuard().evaluate(proposal, registry=registry, shadow_approval=approval)

    assert guard.allowed is False
    assert "missing_shadow_result" in {violation["check"] for violation in guard.violations}


def test_canary_guard_blocks_shadow_recommendation_not_propose_canary():
    manager, plan, _artifact, approval, result, registry = _approved_shadow_state()
    result = dataclasses.replace(result, recommendation=ShadowRunRecommendation.CONTINUE_SHADOW)

    proposal = manager.create_canary_promotion_proposal(
        plan,
        result,
        shadow_approval=approval,
        registry=registry,
    )
    guard = CanaryPromotionGuard().evaluate(proposal, registry=registry, shadow_approval=approval)

    assert guard.allowed is False
    assert "shadow_recommendation_not_propose_canary" in {violation["check"] for violation in guard.violations}


def test_canary_guard_blocks_policy_not_approved_for_shadow():
    manager, plan, artifact, approval, result, registry = _approved_shadow_state()
    entry = registry.get(artifact.policy_id, artifact.policy_version)
    registry = registry._replace_entry(dataclasses.replace(entry, approved_for_shadow=False))

    proposal = manager.create_canary_promotion_proposal(
        plan,
        result,
        shadow_approval=approval,
        registry=registry,
    )
    guard = CanaryPromotionGuard().evaluate(proposal, registry=registry, shadow_approval=approval)

    assert guard.allowed is False
    assert "policy_not_approved_for_shadow" in {violation["check"] for violation in guard.violations}


def test_canary_guard_blocks_expired_or_revoked_shadow_approval():
    manager, plan, _artifact, approval, result, registry = _approved_shadow_state()
    expired = dataclasses.replace(
        approval,
        expires_at="2000-01-01T00:00:00+00:00",
        revoked=True,
    )

    proposal = manager.create_canary_promotion_proposal(
        plan,
        result,
        shadow_approval=expired,
        registry=registry,
    )
    guard = CanaryPromotionGuard().evaluate(proposal, registry=registry, shadow_approval=expired)

    checks = {violation["check"] for violation in guard.violations}
    assert guard.allowed is False
    assert {"shadow_approval_expired", "shadow_approval_revoked"} <= checks


def test_canary_guard_blocks_insufficient_shadow_rounds():
    manager, plan, _artifact, approval, result, registry = _approved_shadow_state()
    result = dataclasses.replace(result, round_count=3)
    proposal = manager.create_canary_promotion_proposal(plan, result, shadow_approval=approval, registry=registry)

    guard = CanaryPromotionGuard().evaluate(proposal, registry=registry, shadow_approval=approval)

    assert guard.allowed is False
    assert "insufficient_shadow_rounds" in {violation["check"] for violation in guard.violations}


def test_canary_guard_blocks_safety_warning_threshold_breach():
    manager, plan, _artifact, approval, result, registry = _approved_shadow_state()
    result = dataclasses.replace(result, safety_warning_count=1)
    proposal = manager.create_canary_promotion_proposal(plan, result, shadow_approval=approval, registry=registry)

    guard = CanaryPromotionGuard().evaluate(proposal, registry=registry, shadow_approval=approval)

    assert guard.allowed is False
    assert "safety_warning_threshold_breached" in {violation["check"] for violation in guard.violations}


def test_canary_guard_blocks_invalid_suggestion_rate_breach():
    manager, plan, _artifact, approval, result, registry = _approved_shadow_state()
    result = dataclasses.replace(result, invalid_suggestion_rate=0.2)
    proposal = manager.create_canary_promotion_proposal(plan, result, shadow_approval=approval, registry=registry)

    guard = CanaryPromotionGuard().evaluate(proposal, registry=registry, shadow_approval=approval)

    assert guard.allowed is False
    assert "invalid_suggestion_rate_too_high" in {violation["check"] for violation in guard.violations}


def test_canary_guard_blocks_poor_confidence_calibration():
    manager, plan, _artifact, approval, result, registry = _approved_shadow_state()
    result = dataclasses.replace(result, confidence_calibration_summary={"calibration_score": 0.2})
    proposal = manager.create_canary_promotion_proposal(plan, result, shadow_approval=approval, registry=registry)

    guard = CanaryPromotionGuard().evaluate(proposal, registry=registry, shadow_approval=approval)

    assert guard.allowed is False
    assert "confidence_calibration_too_low" in {violation["check"] for violation in guard.violations}


def test_canary_guard_blocks_excessive_would_change_top1_rate():
    manager, plan, _artifact, approval, result, registry = _approved_shadow_state()
    result = dataclasses.replace(result, would_change_top1_rate=0.9)
    proposal = manager.create_canary_promotion_proposal(plan, result, shadow_approval=approval, registry=registry)

    guard = CanaryPromotionGuard().evaluate(proposal, registry=registry, shadow_approval=approval)

    assert guard.allowed is False
    assert "top1_change_rate_too_high" in {violation["check"] for violation in guard.violations}


def test_canary_guard_blocks_unknown_counterfactual_used_as_ground_truth():
    manager, plan, _artifact, approval, result, registry = _approved_shadow_state()
    result = dataclasses.replace(result, reasons=("unknown counterfactual",))
    proposal = manager.create_canary_promotion_proposal(plan, result, shadow_approval=approval, registry=registry)
    proposal = dataclasses.replace(
        proposal,
        failure_summary={"unknown_counterfactual_as_ground_truth": True},
    )

    guard = CanaryPromotionGuard().evaluate(proposal, registry=registry, shadow_approval=approval)

    assert guard.allowed is False
    assert "unknown_counterfactual_as_ground_truth" in {violation["check"] for violation in guard.violations}


def test_canary_guard_blocks_missing_rollback_target():
    manager, plan, _artifact, approval, result, registry = _approved_shadow_state()
    plan = dataclasses.replace(plan, rollback_policy_id=None, rollback_policy_version=None)
    proposal = manager.create_canary_promotion_proposal(plan, result, shadow_approval=approval, registry=registry)

    guard = CanaryPromotionGuard().evaluate(proposal, registry=registry, shadow_approval=approval)

    assert guard.allowed is False
    assert "missing_rollback_target" in {violation["check"] for violation in guard.violations}


def test_registry_stores_canary_proposal_metadata_without_live_approval():
    manager, plan, artifact, approval, result, registry = _approved_shadow_state()
    proposal = manager.create_canary_promotion_proposal(plan, result, shadow_approval=approval, registry=registry)

    updated = registry.register_canary_proposal(artifact.policy_id, artifact.policy_version, proposal)
    entry = updated.get(artifact.policy_id, artifact.policy_version)

    assert entry.canary_proposed is True
    assert entry.canary_proposal_id == proposal.proposal_id
    assert entry.canary_proposal_status == "eligible"
    assert entry.canary_eligibility_summary["eligible"] is True
    assert entry.recommended_canary_scope
    assert entry.approved_for_safe_soft is False
    assert entry.approved_for_live_canary is False


def test_manager_recommends_approve_canary_without_auto_approval():
    manager, plan, artifact, approval, result, registry = _approved_shadow_state()
    proposal = manager.create_canary_promotion_proposal(plan, result, shadow_approval=approval, registry=registry)

    updated_plan = manager.attach_canary_proposal(plan, proposal)

    assert updated_plan.status == "canary_eligible"
    assert manager.recommend_next_step(updated_plan) == PolicyEvolutionRecommendation.APPROVE_CANARY
    entry = registry.get(artifact.policy_id, artifact.policy_version)
    assert entry.approved_for_safe_soft is False
    assert entry.approved_for_live_canary is False


def test_learned_policy_still_does_not_affect_live_ranking_after_canary_proposal():
    before = rank_backends(
        "optimize",
        ("nexus_gp_bo", "built_in"),
        {"nexus_gp_bo": True, "built_in": True},
    )
    manager, plan, _artifact, approval, result, registry = _approved_shadow_state()

    _ = manager.create_canary_promotion_proposal(plan, result, shadow_approval=approval, registry=registry)
    after = rank_backends(
        "optimize",
        ("nexus_gp_bo", "built_in"),
        {"nexus_gp_bo": True, "built_in": True},
    )

    assert after == before


def test_default_backend_behavior_remains_unchanged_after_canary_proposal():
    snapshot = all_replay_scenarios()[2]
    before = select_strategy(snapshot, config=PhaseConfig())
    manager, plan, _artifact, approval, result, registry = _approved_shadow_state()

    _ = manager.create_canary_promotion_proposal(plan, result, shadow_approval=approval, registry=registry)
    after = select_strategy(snapshot, config=PhaseConfig())

    assert after.backend_name == before.backend_name
    assert after.strategy_trace.selected_backend == before.strategy_trace.selected_backend


def _eligible_canary_state():
    manager, plan, artifact, shadow_approval, shadow_result, registry = _approved_shadow_state()
    proposal = manager.create_canary_promotion_proposal(
        plan,
        shadow_result,
        shadow_approval=shadow_approval,
        registry=registry,
    )
    registry = registry.register_canary_proposal(artifact.policy_id, artifact.policy_version, proposal)
    return manager, plan, artifact, proposal, registry


def _canary_approval_record(proposal):
    return CanaryApprovalRecord(
        approval_id="canary-approval-a",
        proposal_id=proposal.proposal_id,
        policy_id=proposal.policy_id,
        policy_version=proposal.policy_version,
        approved_by="sissi",
        approval_mode=CanaryApprovalMode.TEST,
        approval_reason="test approval for bounded canary run",
        expires_at="2099-01-01T00:00:00+00:00",
        allowed_campaign_ids=("replay",),
        allowed_objective_levels=("performance",),
        max_canary_rounds=3,
        max_learned_policy_weight=0.003,
        max_top1_change_rate=0.2,
    )


def test_canary_approval_record_round_trip():
    _manager, _plan, _artifact, proposal, _registry = _eligible_canary_state()
    approval = _canary_approval_record(proposal)

    restored = CanaryApprovalRecord.from_dict(approval.to_dict())

    assert restored.approval_id == approval.approval_id
    assert restored.approval_mode == "test"
    assert restored.allowed_campaign_ids == ("replay",)
    assert restored.max_learned_policy_weight == 0.003
    assert restored.auto_disable_enabled is True


def test_canary_approval_guard_blocks_ineligible_proposal():
    _manager, _plan, _artifact, proposal, registry = _eligible_canary_state()
    ineligible = dataclasses.replace(proposal, eligible=False)

    result = CanaryApprovalGuard().evaluate(ineligible, _canary_approval_record(proposal), registry)

    assert result.allowed is False
    assert "proposal_not_eligible" in {violation["check"] for violation in result.violations}


def test_canary_approval_guard_blocks_missing_proposal():
    _manager, _plan, _artifact, proposal, registry = _eligible_canary_state()

    result = CanaryApprovalGuard().evaluate(None, _canary_approval_record(proposal), registry)

    assert result.allowed is False
    assert "missing_canary_proposal" in {violation["check"] for violation in result.violations}


def test_canary_approval_guard_blocks_expired_or_rejected_proposal():
    _manager, _plan, _artifact, proposal, registry = _eligible_canary_state()
    rejected = dataclasses.replace(proposal, status="rejected")

    result = CanaryApprovalGuard().evaluate(rejected, _canary_approval_record(proposal), registry)

    assert result.allowed is False
    assert "proposal_status_blocked" in {violation["check"] for violation in result.violations}


def test_canary_approval_guard_blocks_policy_not_approved_for_shadow():
    _manager, _plan, artifact, proposal, registry = _eligible_canary_state()
    entry = registry.get(artifact.policy_id, artifact.policy_version)
    registry = registry._replace_entry(dataclasses.replace(entry, approved_for_shadow=False))

    result = CanaryApprovalGuard().evaluate(proposal, _canary_approval_record(proposal), registry)

    assert result.allowed is False
    assert "policy_not_approved_for_shadow" in {violation["check"] for violation in result.violations}


def test_canary_approval_guard_blocks_missing_rollback_target():
    _manager, _plan, _artifact, proposal, registry = _eligible_canary_state()
    proposal = dataclasses.replace(proposal, rollback_policy_id=None, rollback_policy_version=None)

    result = CanaryApprovalGuard().evaluate(proposal, _canary_approval_record(proposal), registry)

    assert result.allowed is False
    assert "missing_rollback_target" in {violation["check"] for violation in result.violations}


def test_canary_approval_guard_blocks_required_approval_missing():
    _manager, _plan, _artifact, proposal, registry = _eligible_canary_state()
    approval = dataclasses.replace(_canary_approval_record(proposal), approved_by="")

    result = CanaryApprovalGuard().evaluate(proposal, approval, registry)

    assert result.allowed is False
    assert "required_approval_missing" in {violation["check"] for violation in result.violations}


def test_canary_approval_guard_blocks_scope_broader_than_proposal():
    _manager, _plan, _artifact, proposal, registry = _eligible_canary_state()
    approval = dataclasses.replace(
        _canary_approval_record(proposal),
        allowed_campaign_ids=("replay", "unapproved-campaign"),
    )

    result = CanaryApprovalGuard().evaluate(proposal, approval, registry)

    assert result.allowed is False
    assert "scope_exceeds_proposal" in {violation["check"] for violation in result.violations}


def test_canary_approval_guard_blocks_weight_above_proposal_cap():
    _manager, _plan, _artifact, proposal, registry = _eligible_canary_state()
    approval = dataclasses.replace(
        _canary_approval_record(proposal),
        max_learned_policy_weight=proposal.max_learned_policy_weight + 0.001,
    )

    result = CanaryApprovalGuard().evaluate(proposal, approval, registry)

    assert result.allowed is False
    assert "weight_above_proposal_cap" in {violation["check"] for violation in result.violations}


def test_canary_approval_guard_blocks_rounds_above_proposal_cap():
    _manager, _plan, _artifact, proposal, registry = _eligible_canary_state()
    approval = dataclasses.replace(
        _canary_approval_record(proposal),
        max_canary_rounds=proposal.max_canary_rounds + 1,
    )

    result = CanaryApprovalGuard().evaluate(proposal, approval, registry)

    assert result.allowed is False
    assert "rounds_above_proposal_cap" in {violation["check"] for violation in result.violations}


def test_registry_sets_approved_for_live_canary_only_with_explicit_approval():
    manager, _plan, artifact, proposal, registry = _eligible_canary_state()
    before = registry.get(artifact.policy_id, artifact.policy_version)
    approval = _canary_approval_record(proposal)

    approved_proposal, updated, guard = manager.approve_canary_proposal(
        proposal,
        approval,
        registry=registry,
    )
    after = updated.get(artifact.policy_id, artifact.policy_version)

    assert guard.allowed is True
    assert approved_proposal.status == "approved"
    assert before.approved_for_live_canary is False
    assert after.approved_for_live_canary is True
    assert after.approved_for_safe_soft is False
    assert after.canary_approval_metadata["approval_id"] == "canary-approval-a"


def test_canary_schedule_lifecycle_transitions():
    manager, _plan, _artifact, proposal, _registry = _eligible_canary_state()
    approval = _canary_approval_record(proposal)

    schedule = manager.schedule_canary_run(approval)
    running = manager.update_canary_run_status(schedule, CanaryRunScheduleStatus.RUNNING)
    disabled = manager.update_canary_run_status(
        running,
        CanaryRunScheduleStatus.AUTO_DISABLED,
        "safety warning",
    )

    assert schedule.status == "scheduled"
    assert schedule.max_rounds == 3
    assert schedule.max_learned_policy_weight == 0.003
    assert running.status == "running"
    assert running.started_at is not None
    assert disabled.status == "auto_disabled"
    assert disabled.completed_at is not None
    assert disabled.cancellation_reason == "safety warning"


def test_canary_run_result_attaches_to_evolution_plan():
    manager, plan, artifact, _proposal, _registry = _eligible_canary_state()
    result = CanaryRunResult(
        run_id="canary-run-a",
        schedule_id="schedule-a",
        policy_id=artifact.policy_id,
        policy_version=artifact.policy_version,
        campaign_ids=("replay",),
        round_count=5,
        applied_round_count=5,
        top1_changed_count=0,
        top1_change_rate=0.0,
        reward_vs_baseline=0.1,
        reward_vs_safe_influence=0.01,
        backend_failure_rate=0.0,
        constraint_failure_rate=0.0,
        safety_warning_count=0,
        recommendation=CanaryRunRecommendation.PROPOSE_PROMOTION,
    )

    updated = manager.attach_canary_run_result(dataclasses.replace(plan, status="canary_eligible"), result)

    assert updated.status == "canary_eligible"
    assert "canary result supports promotion proposal:canary-run-a" in updated.reasons


def test_manager_recommends_propose_promotion_only_after_passing_canary_result():
    manager, plan, artifact, _proposal, _registry = _eligible_canary_state()
    eligible_plan = dataclasses.replace(plan, status="canary_eligible")
    weak = CanaryRunResult(
        run_id="canary-weak",
        schedule_id="schedule-a",
        policy_id=artifact.policy_id,
        policy_version=artifact.policy_version,
        round_count=5,
        applied_round_count=5,
        reward_vs_baseline=-0.1,
        reward_vs_safe_influence=0.0,
        backend_failure_rate=0.0,
        constraint_failure_rate=0.0,
        safety_warning_count=0,
        recommendation=CanaryRunRecommendation.PROPOSE_PROMOTION,
    )
    strong = dataclasses.replace(weak, run_id="canary-strong", reward_vs_baseline=0.1)

    weak_plan = manager.attach_canary_run_result(eligible_plan, weak)
    strong_plan = manager.attach_canary_run_result(eligible_plan, strong)

    assert manager.recommend_next_step(eligible_plan) == PolicyEvolutionRecommendation.APPROVE_CANARY
    assert manager.recommend_next_step(weak_plan) == PolicyEvolutionRecommendation.APPROVE_CANARY
    assert manager.recommend_next_step(strong_plan) == PolicyEvolutionRecommendation.PROPOSE_PROMOTION


def test_no_automatic_final_promotion_after_canary_result():
    manager, plan, artifact, proposal, registry = _eligible_canary_state()
    approval = _canary_approval_record(proposal)

    _proposal, registry, _guard = manager.approve_canary_proposal(proposal, approval, registry=registry)
    schedule = manager.schedule_canary_run(approval)
    registry = registry.register_canary_schedule(artifact.policy_id, artifact.policy_version, schedule)
    result = CanaryRunResult(
        run_id="canary-passing",
        schedule_id=schedule.schedule_id,
        policy_id=artifact.policy_id,
        policy_version=artifact.policy_version,
        round_count=5,
        applied_round_count=5,
        reward_vs_baseline=0.1,
        reward_vs_safe_influence=0.01,
        backend_failure_rate=0.0,
        constraint_failure_rate=0.0,
        safety_warning_count=0,
        recommendation=CanaryRunRecommendation.PROPOSE_PROMOTION,
    )
    registry = registry.register_canary_result(artifact.policy_id, artifact.policy_version, result)
    updated_plan = manager.attach_canary_run_result(dataclasses.replace(plan, status="canary_eligible"), result)
    entry = registry.get(artifact.policy_id, artifact.policy_version)

    assert updated_plan.status == "canary_eligible"
    assert manager.recommend_next_step(updated_plan) == PolicyEvolutionRecommendation.PROPOSE_PROMOTION
    assert entry.approved_for_live_canary is True
    assert entry.approved_for_safe_soft is False
    assert entry.latest_canary_run_result_summary["run_id"] == "canary-passing"


def test_learned_policy_canary_constraints_are_guarded():
    plan = _safe_plan(
        proposed_changes={
            "learned_policy_hard_veto": True,
            "learned_policy_add_backend": True,
            "learned_policy_override_action": True,
            "learned_policy_override_objective": True,
            "auto_apply_space_revision": True,
        },
    )

    result = EvolutionGuard().evaluate(plan)

    checks = {violation["check"] for violation in result.violations}
    assert result.allowed is False
    assert {
        "learned_policy_hard_veto",
        "learned_policy_add_backend",
        "learned_policy_override_action_objective",
        "auto_apply_space_revision",
    } <= checks


def test_default_backend_behavior_remains_unchanged_without_explicit_canary_approval():
    snapshot = all_replay_scenarios()[2]
    before = select_strategy(snapshot, config=PhaseConfig())
    manager, plan, _artifact, proposal, _registry = _eligible_canary_state()

    _ = manager.attach_canary_proposal(plan, proposal)
    after = select_strategy(snapshot, config=PhaseConfig())

    assert after.backend_name == before.backend_name
    assert after.strategy_trace.selected_backend == before.strategy_trace.selected_backend
