from __future__ import annotations

import dataclasses
import inspect

import app.services.policy_evolution as policy_evolution
from app.optimization.backend_selection import rank_backends
from app.services.learned_policy import (
    PolicyDatasetAuditor,
    PolicyDatasetBuilder,
    RewardSanityChecker,
    replay_records_from_traces,
)
from app.services.policy_evolution import (
    CanaryApprovalMode,
    CanaryApprovalRecord,
    CanaryPromotionGuard,
    CanaryRunRecommendation,
    CanaryRunResult,
    CandidatePolicyArtifact,
    CandidatePolicyTrainingJob,
    CandidatePolicyTrainingJobStatus,
    CandidatePolicyTrainingMode,
    EvolutionGuard,
    FinalApprovalMode,
    FinalApprovalRecord,
    FinalPromotionGuard,
    PolicyAutoTrainer,
    PolicyEvolutionAuditActorType,
    PolicyEvolutionManager,
    PolicyEvolutionPlan,
    PolicyEvolutionStage,
    PolicyEvolutionTrigger,
    PolicyEvolutionTriggerType,
    PolicyEvolutionWorkflowGuard,
    PolicyEvolutionWorkflowManager,
    PolicyStructureEvidence,
    PolicyStructureEvidenceSource,
    PolicyStructureProposalGuard,
    PolicyStructureProposalType,
    PolicyVersionRegistry,
    PolicyVersionRegistryEntry,
    PolicyWeightTuningGuard,
    PolicyWeightTuningTarget,
    ShadowApprovalMode,
    ShadowApprovalRecord,
    ShadowPromotionGuard,
    ShadowRunRecommendation,
    ShadowRunResult,
    WeightTuningEvidence,
    WeightTuningEvidenceSource,
)
from app.services.strategy_selector import PhaseConfig, select_strategy
from tests.fixtures.strategy_replay import all_replay_scenarios


def _trigger() -> PolicyEvolutionTrigger:
    return PolicyEvolutionTrigger(
        trigger_type=PolicyEvolutionTriggerType.NEW_TRACES_AVAILABLE,
        trigger_reason="e2e replay traces ready",
        campaign_ids=("replay",),
        trace_count=12,
        dataset_version="policy_dataset_e2e",
        metadata={"source": "e2e_test"},
    )


def _plan(trigger: PolicyEvolutionTrigger | None = None, **kwargs) -> PolicyEvolutionPlan:
    return PolicyEvolutionPlan(
        plan_id="plan-e2e",
        source_policy_id="policy-e2e",
        source_policy_version="v1",
        candidate_policy_id="policy-e2e",
        candidate_policy_version="v2",
        trigger=trigger or _trigger(),
        dataset_version="policy_dataset_e2e",
        feature_schema_version="policy_feature_schema_v1",
        reward_version="strategy_reward_v1",
        rollback_policy_id="policy-e2e",
        rollback_policy_version="v1",
        **kwargs,
    )


def _registry() -> PolicyVersionRegistry:
    return PolicyVersionRegistry().register(
        PolicyVersionRegistryEntry(
            policy_id="policy-e2e",
            policy_version="v1",
            trained_on_dataset_version="policy_dataset_e2e_previous",
            feature_schema_version="policy_feature_schema_v1",
            reward_version="strategy_reward_v1",
        )
    ).register(
        PolicyVersionRegistryEntry(
            policy_id="policy-e2e",
            policy_version="v2",
            parent_policy_id="policy-e2e",
            parent_policy_version="v1",
            trained_on_dataset_version="policy_dataset_e2e",
            feature_schema_version="policy_feature_schema_v1",
            reward_version="strategy_reward_v1",
            rollback_target=("policy-e2e", "v1"),
        )
    )


def _job(plan: PolicyEvolutionPlan) -> CandidatePolicyTrainingJob:
    return CandidatePolicyTrainingJob(
        job_id="train-e2e",
        plan_id=plan.plan_id,
        source_policy_id=plan.source_policy_id,
        source_policy_version=plan.source_policy_version,
        candidate_policy_id=plan.candidate_policy_id,
        candidate_policy_version=plan.candidate_policy_version,
        dataset_version=plan.dataset_version,
        feature_schema_version=plan.feature_schema_version,
        reward_version=plan.reward_version,
        training_mode=CandidatePolicyTrainingMode.IMITATION,
        status=CandidatePolicyTrainingJobStatus.OFFLINE_EVALUATED,
        completed_at="2099-01-01T00:00:00+00:00",
    )


def _artifact(plan: PolicyEvolutionPlan) -> CandidatePolicyArtifact:
    offline = {
        "dataset_audit": {
            "passed": True,
            "record_count": 12,
            "dataset_version": plan.dataset_version,
            "feature_schema_version": plan.feature_schema_version,
        },
        "reward_sanity": {
            "passed": True,
            "reward_version_distribution": {plan.reward_version: 12},
        },
        "feature_schema_version": plan.feature_schema_version,
        "reward_version": plan.reward_version,
        "counterfactual_uncertainty_summary": {
            "label_distribution": {"replay_outcome": 12},
            "primary_improvement_evidence": "observed_or_replay_reward",
        },
    }
    return CandidatePolicyArtifact(
        policy_id=plan.candidate_policy_id,
        policy_version=plan.candidate_policy_version,
        parent_policy_id=plan.source_policy_id,
        parent_policy_version=plan.source_policy_version,
        artifact_type="imitation_policy",
        training_mode=CandidatePolicyTrainingMode.IMITATION,
        dataset_version=plan.dataset_version,
        feature_schema_version=plan.feature_schema_version,
        reward_version=plan.reward_version,
        training_summary={"online_enabled": False},
        offline_evaluation_summary=offline,
        safety_summary={"passed": True, "failure_count": 0},
        eligible_for_shadow_proposal=True,
        eligible_for_canary_proposal=False,
        shadow_promotion_eligible=True,
        shadow_promotion_reason="offline evaluation passed; explicit approval required",
    )


def _shadow_approval(proposal) -> ShadowApprovalRecord:
    return ShadowApprovalRecord(
        approval_id="shadow-approval-e2e",
        proposal_id=proposal.proposal_id,
        policy_id=proposal.candidate_policy_id,
        policy_version=proposal.candidate_policy_version,
        approved_by="test-reviewer",
        approval_mode=ShadowApprovalMode.TEST,
        approval_reason="explicit e2e shadow approval",
        expires_at="2099-01-01T00:00:00+00:00",
        max_shadow_rounds=12,
        allowed_campaign_ids=("replay",),
        allowed_objective_levels=("performance",),
    )


def _shadow_result(schedule, artifact: CandidatePolicyArtifact) -> ShadowRunResult:
    return ShadowRunResult(
        run_id="shadow-result-e2e",
        schedule_id=schedule.schedule_id,
        policy_id=artifact.policy_id,
        policy_version=artifact.policy_version,
        campaign_ids=("replay",),
        round_count=12,
        intent_agreement_rate=0.95,
        mode_agreement_rate=0.95,
        backend_agreement_rate=0.9,
        would_change_top1_rate=0.1,
        invalid_suggestion_rate=0.0,
        safety_warning_count=0,
        confidence_calibration_summary={"calibration_score": 0.9},
        counterfactual_breakdown={"replay_outcome": 12},
        recommendation=ShadowRunRecommendation.PROPOSE_CANARY,
        reasons=("shadow thresholds passed",),
    )


def _canary_approval(proposal) -> CanaryApprovalRecord:
    return CanaryApprovalRecord(
        approval_id="canary-approval-e2e",
        proposal_id=proposal.proposal_id,
        policy_id=proposal.policy_id,
        policy_version=proposal.policy_version,
        approved_by="test-reviewer",
        approval_mode=CanaryApprovalMode.TEST,
        approval_reason="explicit e2e canary approval",
        expires_at="2099-01-01T00:00:00+00:00",
        allowed_campaign_ids=("replay",),
        allowed_objective_levels=("performance",),
        max_canary_rounds=5,
        max_learned_policy_weight=proposal.max_learned_policy_weight,
        max_top1_change_rate=0.2,
    )


def _canary_result(schedule, artifact: CandidatePolicyArtifact) -> CanaryRunResult:
    return CanaryRunResult(
        run_id="canary-result-e2e",
        schedule_id=schedule.schedule_id,
        policy_id=artifact.policy_id,
        policy_version=artifact.policy_version,
        campaign_ids=("replay",),
        round_count=5,
        applied_round_count=3,
        top1_changed_count=1,
        top1_change_rate=0.2,
        reward_vs_baseline=0.05,
        reward_vs_safe_influence=0.01,
        backend_failure_rate=0.0,
        constraint_failure_rate=0.0,
        safety_warning_count=0,
        auto_disable_triggered=False,
        recommendation=CanaryRunRecommendation.PROPOSE_PROMOTION,
        reasons=("canary thresholds passed",),
    )


def _final_approval(proposal) -> FinalApprovalRecord:
    return FinalApprovalRecord(
        approval_id="final-approval-e2e",
        proposal_id=proposal.proposal_id,
        policy_id=proposal.policy_id,
        policy_version=proposal.policy_version,
        approved_by="test-reviewer",
        approval_mode=FinalApprovalMode.TEST,
        approval_reason="explicit e2e final approval",
        expires_at="2099-01-01T00:00:00+00:00",
        allowed_campaign_ids=("replay",),
        allowed_objective_levels=("performance",),
        max_live_weight=proposal.max_live_weight,
        max_top1_change_rate=proposal.max_top1_change_rate,
        rollback_policy_id=proposal.rollback_policy_id,
        rollback_policy_version=proposal.rollback_policy_version,
    )


def _run_full_workflow():
    lifecycle = PolicyEvolutionManager()
    workflow_manager = PolicyEvolutionWorkflowManager()
    trigger = _trigger()
    plan = _plan(trigger)
    registry = _registry()
    job = _job(plan)
    artifact = _artifact(plan)
    workflow = workflow_manager.create_workflow(trigger, plan)

    workflow = workflow_manager.attach_training_job(workflow, job)
    workflow = workflow_manager.attach_candidate_artifact(workflow, artifact)

    shadow_proposal = lifecycle.create_shadow_promotion_proposal(plan, job, artifact)
    registry = registry.register_shadow_proposal(artifact.policy_id, artifact.policy_version, shadow_proposal)
    assert registry.get(artifact.policy_id, artifact.policy_version).approved_for_shadow is False
    workflow = workflow_manager.attach_shadow_proposal(workflow, shadow_proposal)

    shadow_approval = _shadow_approval(shadow_proposal)
    shadow_proposal, registry, shadow_guard = lifecycle.approve_shadow_proposal(
        shadow_proposal,
        shadow_approval,
        registry=registry,
    )
    assert shadow_guard.allowed is True
    assert registry.get(artifact.policy_id, artifact.policy_version).approved_for_shadow is True
    assert registry.get(artifact.policy_id, artifact.policy_version).approved_for_live_canary is False
    workflow = workflow_manager.attach_shadow_approval(workflow, shadow_approval)

    shadow_schedule = lifecycle.schedule_shadow_run(shadow_approval)
    registry = registry.register_shadow_schedule(artifact.policy_id, artifact.policy_version, shadow_schedule)
    workflow = workflow_manager.attach_shadow_schedule(workflow, shadow_schedule)
    shadow_result = _shadow_result(shadow_schedule, artifact)
    registry = registry.register_shadow_result(artifact.policy_id, artifact.policy_version, shadow_result)
    workflow = workflow_manager.attach_shadow_result(workflow, shadow_result)

    canary_proposal = lifecycle.create_canary_promotion_proposal(
        plan,
        shadow_result,
        shadow_approval=shadow_approval,
        registry=registry,
    )
    registry = registry.register_canary_proposal(artifact.policy_id, artifact.policy_version, canary_proposal)
    assert registry.get(artifact.policy_id, artifact.policy_version).approved_for_live_canary is False
    workflow = workflow_manager.attach_canary_proposal(workflow, canary_proposal)

    canary_approval = _canary_approval(canary_proposal)
    canary_proposal, registry, canary_guard = lifecycle.approve_canary_proposal(
        canary_proposal,
        canary_approval,
        registry=registry,
    )
    assert canary_guard.allowed is True
    assert registry.get(artifact.policy_id, artifact.policy_version).approved_for_live_canary is True
    assert registry.get(artifact.policy_id, artifact.policy_version).approved_for_safe_soft is False
    workflow = workflow_manager.attach_canary_approval(workflow, canary_approval)

    canary_schedule = lifecycle.schedule_canary_run(canary_approval)
    registry = registry.register_canary_schedule(artifact.policy_id, artifact.policy_version, canary_schedule)
    workflow = workflow_manager.attach_canary_schedule(workflow, canary_schedule)
    canary_result = _canary_result(canary_schedule, artifact)
    registry = registry.register_canary_result(artifact.policy_id, artifact.policy_version, canary_result)
    workflow = workflow_manager.attach_canary_result(workflow, canary_result)

    final_proposal = lifecycle.create_final_promotion_proposal(
        plan,
        canary_result,
        canary_approval=canary_approval,
        registry=registry,
    )
    final_proposal = dataclasses.replace(
        final_proposal,
        counterfactual_breakdown={
            "replay_outcome": 5,
            "primary_improvement_evidence": "observed_or_replay_reward",
        },
    )
    registry = registry.register_final_promotion_proposal(artifact.policy_id, artifact.policy_version, final_proposal)
    assert registry.get(artifact.policy_id, artifact.policy_version).approved_for_safe_soft is False
    workflow = workflow_manager.attach_final_promotion_proposal(workflow, final_proposal)

    final_approval = _final_approval(final_proposal)
    final_proposal, registry, final_guard = lifecycle.approve_final_promotion(
        final_proposal,
        final_approval,
        registry=registry,
    )
    assert final_guard.allowed is True
    assert registry.get(artifact.policy_id, artifact.policy_version).approved_for_safe_soft is True
    workflow = workflow_manager.attach_final_approval(workflow, final_approval)
    workflow = workflow_manager.transition_stage(
        workflow,
        PolicyEvolutionStage.COMPLETED,
        PolicyEvolutionAuditActorType.TEST,
        "all explicit approvals complete",
    )
    report = workflow_manager.build_report(workflow)
    registry = registry.register_workflow_metadata(artifact.policy_id, artifact.policy_version, workflow, report)
    return {
        "lifecycle": lifecycle,
        "workflow_manager": workflow_manager,
        "trigger": trigger,
        "plan": plan,
        "job": job,
        "artifact": artifact,
        "workflow": workflow,
        "registry": registry,
        "shadow_proposal": shadow_proposal,
        "shadow_approval": shadow_approval,
        "shadow_result": shadow_result,
        "canary_proposal": canary_proposal,
        "canary_approval": canary_approval,
        "canary_result": canary_result,
        "final_proposal": final_proposal,
        "final_approval": final_approval,
        "report": report,
    }


def test_human_approved_policy_evolution_workflow_runs_trigger_to_final_approval():
    state = _run_full_workflow()
    registry = state["registry"]
    artifact = state["artifact"]
    workflow = state["workflow"]
    report = state["report"]
    audit_log = state["workflow_manager"].audit_log
    entry = registry.get(artifact.policy_id, artifact.policy_version)

    assert entry.approved_for_shadow is True
    assert entry.approved_for_live_canary is True
    assert entry.approved_for_safe_soft is True
    assert workflow.current_stage == "completed"
    assert workflow.status == "completed"
    assert report.status == "completed"
    assert report.recommendation == "complete"
    assert report.audit_log_count == len(audit_log)
    assert report.guard_violations == ()
    assert {
        "planned",
        "training_requested",
        "offline_evaluated",
        "shadow_proposed",
        "shadow_approved",
        "shadow_running",
        "shadow_completed",
        "canary_proposed",
        "canary_approved",
        "canary_running",
        "canary_completed",
        "promotion_proposed",
        "final_approved",
        "completed",
    } <= set(report.completed_stages)
    assert all(item.guard_allowed for item in audit_log)
    assert {item.to_stage for item in audit_log} >= set(report.completed_stages)


def test_invalid_workflow_transitions_are_blocked_by_stage_guard():
    trigger = _trigger()
    plan = _plan(trigger)
    manager = PolicyEvolutionWorkflowManager()
    workflow = manager.create_workflow(trigger, plan)
    guard = PolicyEvolutionWorkflowGuard()

    cases = (
        (workflow, PolicyEvolutionStage.SHADOW_PROPOSED, "shadow_before_offline_evaluated"),
        (
            dataclasses.replace(workflow, candidate_artifact_id="policy-e2e:v2", shadow_proposal_id="shadow-e2e"),
            PolicyEvolutionStage.SHADOW_RUNNING,
            "missing_shadow_schedule",
        ),
        (
            dataclasses.replace(workflow, candidate_artifact_id="policy-e2e:v2", shadow_proposal_id="shadow-e2e"),
            PolicyEvolutionStage.CANARY_PROPOSED,
            "canary_before_shadow_completed",
        ),
        (
            dataclasses.replace(workflow, shadow_result_id="shadow-result-e2e", canary_proposal_id="canary-e2e"),
            PolicyEvolutionStage.CANARY_RUNNING,
            "missing_canary_schedule",
        ),
        (
            dataclasses.replace(workflow, shadow_result_id="shadow-result-e2e"),
            PolicyEvolutionStage.PROMOTION_PROPOSED,
            "promotion_before_canary_completed",
        ),
        (
            dataclasses.replace(workflow, canary_result_id="canary-result-e2e"),
            PolicyEvolutionStage.FINAL_APPROVED,
            "missing_final_approval",
        ),
        (
            dataclasses.replace(workflow, shadow_approval_id="shadow-approval-e2e"),
            PolicyEvolutionStage.COMPLETED,
            "missing_required_approval_stages",
        ),
    )

    for wf, target, expected_check in cases:
        result = guard.evaluate(wf, target)
        assert result.allowed is False
        assert expected_check in {violation["check"] for violation in result.violations}


def test_registry_flags_change_only_after_explicit_approval_records():
    lifecycle = PolicyEvolutionManager()
    trigger = _trigger()
    plan = _plan(trigger)
    registry = _registry()
    job = _job(plan)
    artifact = _artifact(plan)

    shadow_proposal = lifecycle.create_shadow_promotion_proposal(plan, job, artifact)
    registry = registry.register_shadow_proposal(artifact.policy_id, artifact.policy_version, shadow_proposal)
    entry = registry.get(artifact.policy_id, artifact.policy_version)
    assert (entry.approved_for_shadow, entry.approved_for_live_canary, entry.approved_for_safe_soft) == (False, False, False)

    shadow_approval = _shadow_approval(shadow_proposal)
    shadow_proposal, registry, _guard = lifecycle.approve_shadow_proposal(shadow_proposal, shadow_approval, registry=registry)
    entry = registry.get(artifact.policy_id, artifact.policy_version)
    assert (entry.approved_for_shadow, entry.approved_for_live_canary, entry.approved_for_safe_soft) == (True, False, False)

    shadow_schedule = lifecycle.schedule_shadow_run(shadow_approval)
    shadow_result = _shadow_result(shadow_schedule, artifact)
    canary_proposal = lifecycle.create_canary_promotion_proposal(
        plan,
        shadow_result,
        shadow_approval=shadow_approval,
        registry=registry,
    )
    registry = registry.register_canary_proposal(artifact.policy_id, artifact.policy_version, canary_proposal)
    entry = registry.get(artifact.policy_id, artifact.policy_version)
    assert entry.approved_for_live_canary is False
    canary_approval = _canary_approval(canary_proposal)
    canary_proposal, registry, _guard = lifecycle.approve_canary_proposal(canary_proposal, canary_approval, registry=registry)
    entry = registry.get(artifact.policy_id, artifact.policy_version)
    assert (entry.approved_for_shadow, entry.approved_for_live_canary, entry.approved_for_safe_soft) == (True, True, False)

    canary_schedule = lifecycle.schedule_canary_run(canary_approval)
    canary_result = _canary_result(canary_schedule, artifact)
    final_proposal = lifecycle.create_final_promotion_proposal(
        plan,
        canary_result,
        canary_approval=canary_approval,
        registry=registry,
    )
    registry = registry.register_final_promotion_proposal(artifact.policy_id, artifact.policy_version, final_proposal)
    assert registry.get(artifact.policy_id, artifact.policy_version).approved_for_safe_soft is False
    final_approval = _final_approval(final_proposal)
    _final_proposal, registry, _guard = lifecycle.approve_final_promotion(final_proposal, final_approval, registry=registry)
    assert registry.get(artifact.policy_id, artifact.policy_version).approved_for_safe_soft is True


def test_policy_evolution_guardrail_invariants_block_unsafe_semantics():
    plan = _plan(proposed_changes={
        "unknown_counterfactual_as_ground_truth": True,
        "penalize_scientific_negative_backend": True,
        "auto_apply_space_revision": True,
        "learned_policy_hard_veto": True,
        "learned_policy_add_backend": True,
        "learned_policy_override_action": True,
        "learned_policy_override_objective": True,
        "change_safety_constraints": True,
        "lower_approval_required": True,
    })
    evolution_checks = {violation["check"] for violation in EvolutionGuard().evaluate(plan).violations}
    assert {
        "unknown_counterfactual_as_ground_truth",
        "penalize_scientific_negative_backend",
        "auto_apply_space_revision",
        "learned_policy_hard_veto",
        "learned_policy_add_backend",
        "learned_policy_override_action_objective",
        "change_safety_constraints",
        "lower_approval_required",
    } <= evolution_checks

    trigger = _trigger()
    workflow = PolicyEvolutionWorkflowManager().create_workflow(trigger, _plan(trigger))
    workflow_result = PolicyEvolutionWorkflowGuard().evaluate(
        workflow,
        PolicyEvolutionStage.STRUCTURE_REVIEW_PROPOSED,
        metadata={
            "auto_apply_space_revision": True,
            "modify_safety_gates": True,
            "lower_approval_requirements": True,
            "enable_learned_online_influence": True,
            "apply_structure_proposal": True,
        },
    )
    workflow_checks = {violation["check"] for violation in workflow_result.violations}
    assert {
        "auto_apply_space_revision",
        "safety_or_approval_gate_change",
        "auto_enable_live_influence",
        "auto_apply_structure_proposal",
    } <= workflow_checks


def test_policy_proposal_guards_block_unknown_counterfactual_scientific_negative_and_live_overrides():
    state = _run_full_workflow()
    plan = state["plan"]
    job = state["job"]
    artifact = state["artifact"]
    shadow_proposal = PolicyEvolutionManager().create_shadow_promotion_proposal(
        plan,
        job,
        dataclasses.replace(
            artifact,
            offline_evaluation_summary={
                **artifact.offline_evaluation_summary,
                "counterfactual_uncertainty_summary": {
                    "primary_improvement_evidence": "unknown_counterfactual",
                },
            },
        ),
    )
    assert "unknown_counterfactual_primary_evidence" in {
        violation["check"] for violation in ShadowPromotionGuard().evaluate(shadow_proposal).violations
    }

    canary_proposal = dataclasses.replace(
        state["canary_proposal"],
        failure_summary={"unknown_counterfactual_as_ground_truth": True},
    )
    assert "unknown_counterfactual_as_ground_truth" in {
        violation["check"] for violation in CanaryPromotionGuard().evaluate(canary_proposal).violations
    }

    final_proposal = dataclasses.replace(
        state["final_proposal"],
        counterfactual_breakdown={"primary_improvement_evidence": "unknown_counterfactual"},
    )
    assert "unknown_counterfactual_primary_evidence" in {
        violation["check"] for violation in FinalPromotionGuard().evaluate(final_proposal).violations
    }

    weight_evidence = (
        WeightTuningEvidence(
            evidence_id="unsafe-weight-evidence",
            source_type=WeightTuningEvidenceSource.SAFETY_REPORT,
            metric_name="backend_failure_rate",
            baseline_value=0.0,
            candidate_value=0.1,
            delta=0.1,
            confidence=0.9,
        ),
    )
    weight_proposal = PolicyEvolutionManager().create_weight_tuning_proposal(
        state["registry"].get(artifact.policy_id, artifact.policy_version),
        PolicyWeightTuningTarget.LEARNED_POLICY_MAX_WEIGHT,
        0.001,
        0.002,
        weight_evidence,
        evidence_summary={
            "primary_improvement_evidence": "unknown_counterfactual",
            "alter_hard_safety_gates": True,
            "lower_approval_requirements": True,
            "auto_apply_space_revision": True,
        },
        registry=state["registry"],
    )
    weight_checks = {violation["check"] for violation in PolicyWeightTuningGuard().evaluate(weight_proposal, registry=state["registry"]).violations}
    assert {"backend_failure_rate_increased", "unknown_counterfactual_primary_evidence", "safety_or_approval_gate_change", "auto_apply_space_revision"} <= weight_checks

    structure_proposal = PolicyEvolutionManager().create_policy_structure_proposal(
        PolicyStructureProposalType.NEW_POLICY_RULE,
        "Unsafe policy structure change",
        "Should be blocked by guard",
        "current",
        "proposed",
        (
            PolicyStructureEvidence(
                evidence_id="structure-evidence-unsafe",
                source_type=PolicyStructureEvidenceSource.FAILURE_REPORT,
                metric_name="failure_rate",
                confidence=0.9,
                counterfactual_label="unknown_counterfactual",
            ),
        ),
        affected_components=("safety", "reward", "backend_prior"),
        evidence_summary={
            "lower_safety_gates": True,
            "lower_approval_requirements": True,
            "auto_apply_space_revision": True,
            "penalize_scientific_negative_backend": True,
            "enable_live_hard_veto": True,
            "added_backends": ("unknown_backend",),
            "changes_reward_semantics": True,
            "bypass_shadow_canary_promotion_lifecycle": True,
        },
    )
    structure_checks = {violation["check"] for violation in PolicyStructureProposalGuard().evaluate(structure_proposal).violations}
    assert {
        "lower_safety_gates",
        "lower_approval_requirements",
        "auto_apply_space_revision",
        "unknown_counterfactual_as_ground_truth",
        "penalize_scientific_negative_backend",
        "enable_live_hard_veto",
        "backend_outside_registry",
        "reward_semantics_without_version_bump",
        "bypass_lifecycle",
    } <= structure_checks


def test_self_evolution_metadata_does_not_change_default_selector_or_backend_ranking():
    snapshot = all_replay_scenarios()[2]
    rank_before = rank_backends(
        "optimize",
        ("nexus_gp_bo", "built_in"),
        {"nexus_gp_bo": True, "built_in": True},
    )
    selector_before = select_strategy(snapshot, config=PhaseConfig())

    state = _run_full_workflow()

    rank_after = rank_backends(
        "optimize",
        ("nexus_gp_bo", "built_in"),
        {"nexus_gp_bo": True, "built_in": True},
    )
    selector_after = select_strategy(snapshot, config=PhaseConfig())
    active = state["lifecycle"].get_active_safe_soft_policy(
        state["registry"],
        campaign_id="replay",
        objective_level="performance",
    )
    source = inspect.getsource(policy_evolution)

    assert rank_after == rank_before
    assert selector_after.backend_name == selector_before.backend_name
    assert selector_after.strategy_trace.selected_backend == selector_before.strategy_trace.selected_backend
    assert active["approved_for_safe_soft"] is True
    assert active["live_selector_activation"] is False
    assert "from app.services.strategy_selector" not in source
    assert "import app.services.strategy_selector" not in source
    assert "rank_backends(" not in source


def test_replay_records_drive_offline_candidate_to_final_approval_lifecycle():
    traces = [
        select_strategy(snapshot, config=PhaseConfig()).strategy_trace
        for snapshot in all_replay_scenarios()
    ]
    replay_records = replay_records_from_traces(traces)
    dataset = PolicyDatasetBuilder().build(replay_records)
    audit = PolicyDatasetAuditor().audit(dataset)
    reward_sanity = RewardSanityChecker().check(dataset)
    trigger = _trigger()
    plan = dataclasses.replace(
        _plan(trigger),
        dataset_version=dataset.dataset_version,
        feature_schema_version=dataset.feature_schema_version,
        reward_version=dataset.reward_version,
    )
    registry = _registry()
    trainer = PolicyAutoTrainer()

    job, artifact, registry = trainer.train_candidate(
        plan,
        dataset=dataset,
        training_mode=CandidatePolicyTrainingMode.IMITATION,
        registry=registry,
    )
    assert artifact is not None
    assert audit.record_count == len(replay_records)
    assert reward_sanity.passed is True
    assert job.status == "offline_evaluated"
    assert artifact.dataset_version == dataset.dataset_version
    assert artifact.feature_schema_version == dataset.feature_schema_version
    assert artifact.reward_version == dataset.reward_version
    assert set(artifact.offline_evaluation_summary["counterfactual_uncertainty_summary"]["label_distribution"])

    workflow_manager = PolicyEvolutionWorkflowManager()
    lifecycle = PolicyEvolutionManager()
    workflow = workflow_manager.create_workflow(trigger, plan)
    workflow = workflow_manager.attach_training_job(workflow, job)
    workflow = workflow_manager.attach_candidate_artifact(workflow, artifact)
    shadow_proposal = lifecycle.create_shadow_promotion_proposal(plan, job, artifact)
    registry = registry.register_shadow_proposal(artifact.policy_id, artifact.policy_version, shadow_proposal)
    workflow = workflow_manager.attach_shadow_proposal(workflow, shadow_proposal)
    shadow_approval = _shadow_approval(shadow_proposal)
    shadow_proposal, registry, _guard = lifecycle.approve_shadow_proposal(shadow_proposal, shadow_approval, registry=registry)
    workflow = workflow_manager.attach_shadow_approval(workflow, shadow_approval)
    shadow_schedule = lifecycle.schedule_shadow_run(shadow_approval)
    workflow = workflow_manager.attach_shadow_schedule(workflow, shadow_schedule)
    shadow_result = _shadow_result(shadow_schedule, artifact)
    workflow = workflow_manager.attach_shadow_result(workflow, shadow_result)
    canary_proposal = lifecycle.create_canary_promotion_proposal(
        plan,
        shadow_result,
        shadow_approval=shadow_approval,
        registry=registry,
    )
    registry = registry.register_canary_proposal(artifact.policy_id, artifact.policy_version, canary_proposal)
    workflow = workflow_manager.attach_canary_proposal(workflow, canary_proposal)
    canary_approval = _canary_approval(canary_proposal)
    canary_proposal, registry, _guard = lifecycle.approve_canary_proposal(canary_proposal, canary_approval, registry=registry)
    workflow = workflow_manager.attach_canary_approval(workflow, canary_approval)
    canary_schedule = lifecycle.schedule_canary_run(canary_approval)
    workflow = workflow_manager.attach_canary_schedule(workflow, canary_schedule)
    canary_result = _canary_result(canary_schedule, artifact)
    workflow = workflow_manager.attach_canary_result(workflow, canary_result)
    final_proposal = lifecycle.create_final_promotion_proposal(
        plan,
        canary_result,
        canary_approval=canary_approval,
        registry=registry,
    )
    registry = registry.register_final_promotion_proposal(artifact.policy_id, artifact.policy_version, final_proposal)
    workflow = workflow_manager.attach_final_promotion_proposal(workflow, final_proposal)

    report = workflow_manager.build_report(workflow)
    assert report.current_stage == "promotion_proposed"
    assert artifact.offline_evaluation_summary["feature_schema_version"] == dataset.feature_schema_version
    assert artifact.offline_evaluation_summary["reward_version"] == dataset.reward_version
    assert shadow_proposal.dataset_version == dataset.dataset_version
    assert shadow_proposal.feature_schema_version == dataset.feature_schema_version
    assert shadow_proposal.reward_version == dataset.reward_version
    assert "label_distribution" in shadow_proposal.counterfactual_uncertainty_summary
    assert canary_proposal.counterfactual_breakdown
    assert final_proposal.canary_result_summary["recommendation"] == "propose_promotion"


def test_workflow_report_and_audit_log_cover_artifacts_and_transition_details():
    state = _run_full_workflow()
    report = state["report"]
    workflow = state["workflow"]
    audit_log = state["workflow_manager"].audit_log

    assert workflow.trigger_id.startswith("new_traces_available:")
    assert workflow.plan_id == "plan-e2e"
    assert report.training_summary["training_job_id"] == "train-e2e"
    assert report.offline_evaluation_summary["candidate_artifact_id"] == "policy-e2e:v2"
    assert report.shadow_summary["proposal_id"] == state["shadow_proposal"].proposal_id
    assert report.shadow_summary["result_id"] == state["shadow_result"].run_id
    assert report.canary_summary["proposal_id"] == state["canary_proposal"].proposal_id
    assert report.canary_summary["result_id"] == state["canary_result"].run_id
    assert report.final_approval_summary["promotion_proposal_id"] == state["final_proposal"].proposal_id
    assert report.final_approval_summary["final_approval_id"] == state["final_approval"].approval_id
    assert report.rollback_target == ("policy-e2e", "v1")

    weight_proposal = state["lifecycle"].create_weight_tuning_proposal(
        state["registry"].get("policy-e2e", "v2"),
        PolicyWeightTuningTarget.LEARNED_POLICY_MAX_WEIGHT,
        0.001,
        0.002,
        (
            WeightTuningEvidence(
                evidence_id="weight-evidence-e2e",
                source_type=WeightTuningEvidenceSource.REWARD_REPORT,
                metric_name="reward_delta",
                baseline_value=0.0,
                candidate_value=0.01,
                delta=0.01,
                confidence=0.8,
            ),
        ),
        registry=state["registry"],
    )
    workflow = state["workflow_manager"].attach_weight_tuning_proposal(workflow, weight_proposal)
    structure_proposal = state["lifecycle"].create_policy_structure_proposal(
        PolicyStructureProposalType.NEW_POLICY_RULE,
        "Review-only rule proposal",
        "Proposal metadata only",
        "current",
        "proposed",
        (
            PolicyStructureEvidence(
                evidence_id="structure-evidence-e2e",
                source_type=PolicyStructureEvidenceSource.TRACE_ANALYSIS,
                metric_name="agreement_rate",
                baseline_value=0.8,
                candidate_value=0.85,
                delta=0.05,
                confidence=0.8,
            ),
        ),
        affected_components=("policy_rule",),
    )
    workflow = state["workflow_manager"].attach_structure_proposal(workflow, structure_proposal)
    report = state["workflow_manager"].build_report(workflow)
    assert report.weight_tuning_summary["proposal_ids"] == (weight_proposal.proposal_id,)
    assert report.structure_proposal_summary["proposal_ids"] == (structure_proposal.proposal_id,)

    assert len(audit_log) >= 14
    for entry in audit_log:
        assert entry.workflow_id == state["workflow"].workflow_id
        assert entry.actor_type in {"system", "test"}
        assert entry.action == "transition_stage"
        assert entry.from_stage
        assert entry.to_stage
        assert isinstance(entry.guard_allowed, bool)
        assert isinstance(entry.guard_violations, tuple)
        assert entry.reason
