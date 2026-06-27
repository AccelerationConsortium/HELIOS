from __future__ import annotations

import dataclasses

from app.services.learned_policy import (
    CounterfactualOutcomeLabel,
    ImitationPolicy,
    LearnedBackendReranker,
    LearnedMetaPolicy,
    LearnedPolicyPromotionGate,
    LearnedPolicyShadowAnalyzer,
    LearnedPolicyShadowRunner,
    LearnedPolicyTrace,
    OfflineMetaPolicyEvaluator,
    OfflineMetaPolicyTrainer,
    OfflinePolicyEvaluator,
    PolicyDataset,
    PolicyDatasetAuditor,
    PolicyDatasetBuilder,
    PolicyOfflineCompletenessChecker,
    PolicySimulationEnvironment,
    RewardModel,
    RewardSanityChecker,
    baseline_traces_from_snapshots,
    replay_records_from_traces,
)
from app.services.policy_evaluation import PolicyEvaluationRunner
from app.services.strategy_models import (
    LearnedPolicyDeploymentMode,
    LearnedPolicyRegistryEntry,
    OnlineInfluenceMode,
    OnlineInfluenceOutcome,
    PolicyInfluenceConfig,
    policy_training_record_from_trace,
)
from app.services.strategy_selector import PhaseConfig, select_strategy
from tests.fixtures.strategy_replay import all_replay_scenarios


def _baseline_traces():
    return baseline_traces_from_snapshots(all_replay_scenarios())


def test_policy_dataset_builder_is_stable_and_versioned():
    traces = _baseline_traces()
    records = replay_records_from_traces(traces)
    dataset = PolicyDatasetBuilder().build(records)

    assert dataset.dataset_version == "policy_dataset_v1"
    assert dataset.feature_schema_version == "policy_feature_schema_v1"
    assert dataset.reward_version == "strategy_reward_v1"
    assert dataset.metadata["n_records"] == len(records)
    first = dataset.records[0]
    assert first["state_features"]
    assert first["context_features"]
    assert first["available_actions"]
    assert first["candidate_backends"]
    assert all("rank" in row and "total" in row for row in first["candidate_backends"])
    assert all("intent" in row and "mode" in row for row in first["available_actions"])
    assert "record_version" in first
    assert first["outcome"]["counterfactual_label"] in {
        "unknown_counterfactual",
        "observed_outcome",
        "replay_outcome",
        "synthetic_outcome",
    }


def test_dataset_audit_catches_missing_features_and_class_imbalance():
    rows = [
        {
            "campaign_id": "c1",
            "loop_id": "r1",
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
        {
            "campaign_id": "c2",
            "loop_id": "r2",
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
    ]
    audit = PolicyDatasetAuditor().audit(PolicyDataset(tuple(rows)))

    assert audit.record_count == 2
    assert audit.missing_feature_rates["state_features"] == 1.0
    assert audit.reward_coverage == 0.0
    assert audit.candidate_backend_coverage == {}
    assert audit.candidate_score_coverage == 0.0
    assert audit.candidate_rank_coverage == 0.0
    assert audit.offline_readiness_warnings
    assert any("single observed class" in warning for warning in audit.class_imbalance_warnings)


def test_policy_training_record_conversion_supports_required_fields():
    trace = select_strategy(all_replay_scenarios()[0]).strategy_trace
    record = policy_training_record_from_trace(trace, loop_id="loop-a")

    assert record.record_version == "policy_training_record_v1"
    assert record.campaign_id
    assert record.loop_id == "loop-a"
    assert record.state_features
    assert record.context_features
    assert record.available_actions
    assert record.selected_intent
    assert record.selected_mode
    assert record.selected_backend
    assert record.candidate_backends
    assert record.reward["reward_version"] == "strategy_reward_v1"
    assert record.outcome["observed"] is False


def test_imitation_policy_can_fit_and_evaluate_replay_fixtures():
    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    policy = ImitationPolicy().fit(dataset)
    summary = policy.evaluate(dataset)

    assert summary["n_records"] == len(dataset.records)
    assert 0.0 <= summary["intent_accuracy"] <= 1.0
    assert 0.0 <= summary["mode_accuracy"] <= 1.0
    assert 0.0 <= summary["backend_top1_accuracy"] <= 1.0
    assert policy.predict(dataset.records[0])["backend"]


def test_learned_reranker_is_offline_only_and_caps_deltas():
    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    reranker = LearnedBackendReranker(max_delta=0.01).fit(dataset)
    trace = reranker.trace_for(dataset.records[0], cap=0.01)

    assert isinstance(trace, LearnedPolicyTrace)
    assert trace.trace_version == "learned_policy_trace_v1"
    assert trace.reasons == ("offline learned reranker suggestion only",)
    assert all(abs(delta["score_delta"]) <= 0.01 for delta in trace.score_deltas)


def test_offline_policy_evaluator_compares_variants_and_learned_reranker():
    report = OfflinePolicyEvaluator(learned_delta_cap=0.01).evaluate_snapshots(
        all_replay_scenarios()
    )

    assert report["baseline_summary"]["n_traces"] >= 10
    assert report["dataset_summary"]["dataset_version"] == "policy_dataset_v1"
    assert "combined_safe_influence" in report["policy_variants"]
    assert report["imitation_policy_summary"]["n_records"] >= 10
    assert report["learned_reranker_summary"]["top1_change_rate"] >= 0.0
    assert report["learned_policy_safety"]["passed"] is True
    assert report["learned_policy_traces"]
    assert report["learned_policy_benchmark_report"]["report_version"] == "policy_benchmark_report_v1"
    assert "rule_plus_learned_correction" in report["learned_policy_benchmark_report"]
    assert report["reward_sanity"]["passed"] is True
    assert report["offline_completeness"]["passed"] is True
    assert report["offline_completeness"]["required_sections"]["candidate_backend_rankings"] is True
    assert report["dataset_audit"]["record_count"] >= 10
    assert report["dataset_audit"]["candidate_score_coverage"] == 1.0
    assert report["dataset_audit"]["candidate_rank_coverage"] == 1.0


def test_feature_ablation_runs_without_touching_live_selector():
    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    before = select_strategy(all_replay_scenarios()[0], config=PhaseConfig())

    ablation = OfflinePolicyEvaluator(learned_delta_cap=0.01).evaluate_feature_ablation(dataset)

    after = select_strategy(all_replay_scenarios()[0], config=PhaseConfig())
    assert set(ablation) >= {
        "full_features",
        "without_objective_hierarchy",
        "without_failure_taxonomy",
        "without_backend_memory",
        "without_nexus_recommendation",
        "without_route_budget_data_quality_prior_campaign",
    }
    assert ablation["full_features"]["backend_top3_accuracy"] >= 0.0
    assert "safety_metrics" in ablation["without_backend_memory"]
    assert after.backend_name == before.backend_name


def test_reward_sanity_checks_catch_contaminated_failure_reward_attribution():
    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    contaminated = dict(dataset.records[0])
    contaminated["reward"] = {
        **dict(contaminated["reward"]),
        "failure_penalty": 0.5,
    }
    contaminated["outcome"] = {
        "counterfactual_label": CounterfactualOutcomeLabel.OBSERVED_OUTCOME.value,
        "failure_events": [
            {
                "failure_type": "scientific_negative",
                "reason": "clean negative",
            },
            {
                "failure_type": "hardware",
                "reason": "pump failure",
            },
        ],
    }

    report = RewardSanityChecker().check(PolicyDataset((contaminated,)))

    assert report.passed is False
    checks = {failure["check"] for failure in report.failures}
    assert "scientific_negative_penalized_as_failure" in checks
    assert "execution_failure_contaminates_backend_reward" in checks


def test_reward_sanity_checks_missing_and_inconsistent_reward_fields():
    row = dict(PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces())).records[0])
    row["reward"] = {"composite_reward": 0.1, "reward_version": "other"}
    dataset = PolicyDataset((row,), reward_version="strategy_reward_v1")

    report = RewardSanityChecker().check(dataset)

    assert report.passed is False
    checks = {failure["check"] for failure in report.failures}
    assert "missing_reward_fields" in checks
    assert "reward_version_inconsistent" in checks


def test_safety_constraints_reject_unsafe_learned_outputs():
    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    unsafe = LearnedPolicyTrace(
        suggested_backend="not_in_pool",
        score_deltas=(
            {
                "source": "learned_backend_reranker",
                "target": "not_in_pool",
                "score_delta": 1.0,
                "cap": 0.01,
            },
        ),
    )

    safety = OfflinePolicyEvaluator(learned_delta_cap=0.01)._safety_check(
        (unsafe,),
        dataset,
    )

    assert safety["passed"] is False
    assert {failure["check"] for failure in safety["failures"]} >= {
        "learned_delta_cap",
        "learned_backend_must_exist",
        "learned_suggestion_must_exist",
    }


def test_counterfactual_labels_are_present_and_unknown_is_not_overstated():
    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    report = OfflinePolicyEvaluator(learned_delta_cap=0.01).evaluate_dataset(dataset)

    assert all("counterfactual_label" in row["outcome"] for row in dataset.records)
    assert {
        row["outcome"]["counterfactual_label"]
        for row in dataset.records
    } <= {item.value for item in CounterfactualOutcomeLabel}
    assert report["learned_reranker_summary"]["reward_delta_vs_baseline"] == 0.0


def test_simulation_environment_respects_available_actions_and_safety_masks():
    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    env = PolicySimulationEnvironment(dataset)
    obs = env.reset()

    assert obs["available_actions"]
    assert env.action_space()["backends"]
    assert "add_backend" in env.safety_mask()["blocked_operations"]

    step = env.step({
        "backend": "not_in_pool",
        "intent": "not_available",
        "mode": "not_available",
        "hard_veto": True,
        "auto_apply_space_revision": True,
        "score_deltas": [{"target": "not_in_pool", "score_delta": 0.1}],
    })

    assert step.info["safety"]["valid"] is False
    assert set(step.info["safety"]["violations"]) >= {
        "backend_not_available",
        "intent_not_available",
        "mode_not_available",
        "hard_veto_not_allowed",
        "space_revision_auto_apply_not_allowed",
        "score_delta_target_not_available",
    }
    assert step.info["space_revision_auto_applied"] is False


def test_policy_offline_completeness_checker_requires_full_offline_contract():
    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    report = OfflinePolicyEvaluator(learned_delta_cap=0.01).evaluate_dataset(dataset)
    benchmark = OfflinePolicyEvaluator(learned_delta_cap=0.01).benchmark_report(
        PolicyEvaluationRunner().evaluate_snapshots(all_replay_scenarios()),
        dataset,
        report,
    )

    completeness = PolicyOfflineCompletenessChecker().check(
        dataset,
        learned_safety=report["learned_policy_safety"],
        benchmark_report=benchmark,
    )

    assert completeness.passed is True
    assert completeness.required_sections["state_context_features"] is True
    assert completeness.required_sections["candidate_backend_rankings"] is True

    incomplete = PolicyDataset((
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
    failed = PolicyOfflineCompletenessChecker().check(incomplete)
    assert failed.passed is False
    assert failed.failure_count > 0


def test_unknown_counterfactual_is_not_ground_truth_reward():
    row = dict(PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces())).records[0])
    row["outcome"] = {
        "counterfactual_label": CounterfactualOutcomeLabel.UNKNOWN_COUNTERFACTUAL.value
    }

    result = RewardModel().compute(row)

    assert result["reward"] is None
    assert result["ground_truth"] is False
    assert result["counterfactual_label"] == "unknown_counterfactual"


def test_reward_model_uses_observed_replay_and_synthetic_labels_only():
    row = dict(PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces())).records[0])
    for label in (
        CounterfactualOutcomeLabel.OBSERVED_OUTCOME.value,
        CounterfactualOutcomeLabel.REPLAY_OUTCOME.value,
        CounterfactualOutcomeLabel.SYNTHETIC_OUTCOME.value,
    ):
        labeled = dict(row)
        labeled["outcome"] = {"counterfactual_label": label}
        result = RewardModel().compute(labeled)
        assert result["reward"] is not None
        assert result["reward_version"] == "strategy_reward_v1"


def test_learned_meta_policy_remains_offline_only_and_safety_masked():
    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    env = PolicySimulationEnvironment(dataset)
    policy = LearnedMetaPolicy(max_delta=0.01).fit_imitation(dataset)
    trace = policy.predict(env.reset())

    assert trace.reasons == ("offline simulation meta-policy suggestion only",)
    assert trace.suggested_backend in env.safety_mask()["allowed_backends"]
    assert all(
        delta["target"] in env.safety_mask()["allowed_backends"]
        and abs(delta["score_delta"]) <= 0.01
        for delta in trace.score_deltas
    )


def test_learned_policy_cannot_add_backend_hard_veto_or_alter_space_revision():
    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    env = PolicySimulationEnvironment(dataset)
    env.reset()
    unsafe_step = env.step({
        "backend": "brand_new_backend",
        "hard_veto": True,
        "auto_apply_space_revision": True,
    })

    assert unsafe_step.info["safety"]["valid"] is False
    assert "backend_not_available" in unsafe_step.info["safety"]["violations"]
    assert "hard_veto_not_allowed" in unsafe_step.info["safety"]["violations"]
    assert "space_revision_auto_apply_not_allowed" in unsafe_step.info["safety"]["violations"]
    assert unsafe_step.info["space_revision_auto_applied"] is False


def test_offline_meta_policy_trainer_and_evaluator_run_on_replay_fixtures():
    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    trainer = OfflineMetaPolicyTrainer()
    trained = trainer.train_imitation(dataset)
    sim_result = trainer.train_policy_gradient_style(
        PolicySimulationEnvironment(dataset),
        episodes=1,
    )
    report = OfflineMetaPolicyEvaluator(learned_delta_cap=0.01).evaluate_snapshots(
        all_replay_scenarios()
    )

    assert trained["training_mode"] == "supervised_imitation"
    assert trained["online_enabled"] is False
    assert sim_result["training_mode"] == "simulation_policy_gradient_placeholder"
    assert sim_result["online_enabled"] is False
    assert report["trained_meta_policy"]["online_enabled"] is False
    assert report["trained_meta_policy_summary"]["n_records"] >= 10
    assert report["counterfactual_uncertainty_breakdown"]


def test_default_backend_execution_remains_unchanged_with_learned_scaffold():
    snapshot = all_replay_scenarios()[2]
    before = select_strategy(snapshot, config=PhaseConfig())

    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    LearnedBackendReranker(max_delta=0.01).fit(dataset).trace_for(dataset.records[0])
    OfflinePolicyEvaluator(learned_delta_cap=0.01).evaluate_dataset(dataset)
    OfflineMetaPolicyEvaluator(learned_delta_cap=0.01).evaluate_snapshots(
        all_replay_scenarios()
    )

    after = select_strategy(snapshot, config=PhaseConfig())
    safe_variant = select_strategy(
        snapshot,
        config=PhaseConfig(policy_influence=PolicyInfluenceConfig()),
    )

    assert after.backend_name == before.backend_name
    assert after.phase == before.phase
    assert safe_variant.backend_name == before.backend_name


def test_learned_policy_shadow_default_off_preserves_trace_and_behavior():
    snapshot = all_replay_scenarios()[0]
    decision = select_strategy(snapshot, config=PhaseConfig())
    trace = decision.strategy_trace

    out = LearnedPolicyShadowRunner().run(trace)
    after = select_strategy(snapshot, config=PhaseConfig())

    assert out is trace
    assert out.learned_policy_shadow is None
    assert after.backend_name == decision.backend_name


def test_learned_policy_shadow_records_suggestion_without_altering_ranking():
    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    policy = LearnedMetaPolicy(max_delta=0.01).fit_imitation(dataset)
    registry = LearnedPolicyRegistryEntry(
        policy_id="policy-a",
        policy_version="v1",
        trained_on_dataset_version=dataset.dataset_version,
        feature_schema_version=dataset.feature_schema_version,
        reward_version=dataset.reward_version,
        evaluation_summary={"backend_top1_accuracy": 1.0},
        approved_for_shadow=True,
    )
    trace = select_strategy(all_replay_scenarios()[0], config=PhaseConfig()).strategy_trace
    before_ranking = trace.candidate_backends
    before_influences = trace.ranking_influences

    shadowed = LearnedPolicyShadowRunner(
        registry_entry=registry,
        policy=policy,
        mode=LearnedPolicyDeploymentMode.SHADOW,
    ).run(trace)

    assert shadowed.selected_backend == trace.selected_backend
    assert shadowed.candidate_backends == before_ranking
    assert shadowed.ranking_influences == before_influences
    assert shadowed.learned_policy_shadow is not None
    assert shadowed.learned_policy_shadow.policy_id == "policy-a"
    assert shadowed.learned_policy_shadow.deployment_mode == "shadow"
    assert all(
        record.source != "learned_backend_reranker"
        for record in shadowed.ranking_influences
    )


def test_unavailable_backend_shadow_suggestion_is_masked_and_flagged():
    class BadPolicy:
        def predict(self, _observation):
            return LearnedPolicyTrace(
                suggested_intent="optimize",
                suggested_mode="exploit",
                suggested_backend="not_available",
                confidence=0.9,
                score_deltas=(
                    {
                        "target": "not_available",
                        "score_delta": 0.01,
                    },
                ),
            )

    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    registry = LearnedPolicyRegistryEntry(
        policy_id="bad-policy",
        policy_version="v1",
        trained_on_dataset_version=dataset.dataset_version,
        feature_schema_version=dataset.feature_schema_version,
        reward_version=dataset.reward_version,
        approved_for_shadow=True,
    )
    trace = select_strategy(all_replay_scenarios()[0], config=PhaseConfig()).strategy_trace

    shadowed = LearnedPolicyShadowRunner(
        registry_entry=registry,
        policy=BadPolicy(),
        mode=LearnedPolicyDeploymentMode.SHADOW,
    ).run(trace)
    record = shadowed.learned_policy_shadow

    assert record.suggested_backend is None
    assert record.score_deltas == ()
    assert record.safety_mask_valid is False
    assert set(record.invalid_suggestion_reasons) >= {
        "suggested_backend_unavailable",
        "score_delta_target_unavailable",
    }
    assert record.safety_warnings


def test_only_approved_for_shadow_policy_can_run():
    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    policy = LearnedMetaPolicy(max_delta=0.01).fit_imitation(dataset)
    registry = LearnedPolicyRegistryEntry(
        policy_id="not-approved",
        policy_version="v1",
        trained_on_dataset_version=dataset.dataset_version,
        feature_schema_version=dataset.feature_schema_version,
        reward_version=dataset.reward_version,
        approved_for_shadow=False,
    )
    trace = select_strategy(all_replay_scenarios()[0], config=PhaseConfig()).strategy_trace

    shadowed = LearnedPolicyShadowRunner(
        registry_entry=registry,
        policy=policy,
        mode=LearnedPolicyDeploymentMode.SHADOW,
    ).run(trace)

    assert shadowed.learned_policy_shadow is not None
    assert shadowed.learned_policy_shadow.safety_mask_valid is False
    assert shadowed.learned_policy_shadow.invalid_suggestion_reasons == (
        "policy_not_approved_for_shadow",
    )
    assert shadowed.selected_backend == trace.selected_backend


def _approved_safe_soft_registry(dataset):
    return LearnedPolicyRegistryEntry(
        policy_id="policy-safe",
        policy_version="v1",
        trained_on_dataset_version=dataset.dataset_version,
        feature_schema_version=dataset.feature_schema_version,
        reward_version=dataset.reward_version,
        evaluation_summary={
            "shadow_rounds": 20,
            "confidence_calibration": 0.9,
            "top_k_agreement": 0.9,
            "offline_benchmark_pass": True,
            "reward_sanity_pass": True,
        },
        approved_for_shadow=True,
        approved_for_safe_soft=True,
    )


class _FixedDeltaPolicy:
    def __init__(self, target: str, delta: float = 0.02, confidence: float = 0.9):
        self.target = target
        self.delta = delta
        self.confidence = confidence

    def predict(self, _observation):
        return LearnedPolicyTrace(
            suggested_intent="optimize",
            suggested_mode="exploit",
            suggested_backend=self.target,
            confidence=self.confidence,
            score_deltas=({"target": self.target, "score_delta": self.delta},),
        )


def _live_canary_config(dataset, policy, *, flag=True, max_total=0.04):
    return PhaseConfig(
        online_influence_rollout=dataclasses.replace(
            PhaseConfig().online_influence_rollout,
            enabled=True,
            mode=OnlineInfluenceMode.SAFE_SOFT,
            allowed_campaign_ids=("replay",),
            allowed_objective_levels=("performance",),
            enable_learned_safe_soft_live=flag,
            max_total_score_delta=max_total,
        ),
        learned_policy_registry_entry=_approved_safe_soft_registry(dataset),
        learned_policy=policy,
        learned_policy_shadow_summary={"n_records": 20, "safety_warning_count": 0},
        learned_policy_min_shadow_rounds=5,
        learned_policy_max_safe_soft_delta=0.005,
    )


def test_learned_live_default_flag_preserves_backend_behavior():
    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    snapshot = all_replay_scenarios()[2]
    baseline = select_strategy(snapshot, config=PhaseConfig())
    target = baseline.strategy_trace.candidate_backends[0]["name"]

    configured_but_disabled = select_strategy(
        snapshot,
        config=_live_canary_config(dataset, _FixedDeltaPolicy(target), flag=False),
    )

    assert configured_but_disabled.backend_name == baseline.backend_name
    assert configured_but_disabled.strategy_trace.learned_policy_influence is None
    assert all(
        record.source != "learned_policy"
        for record in configured_but_disabled.strategy_trace.ranking_influences
    )


def test_learned_live_influence_requires_all_gates_to_pass():
    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    snapshot = all_replay_scenarios()[2]
    trace = select_strategy(snapshot, config=PhaseConfig()).strategy_trace
    target = trace.candidate_backends[0]["name"]
    config = _live_canary_config(dataset, _FixedDeltaPolicy(target))
    config = dataclasses.replace(
        config,
        learned_policy_shadow_summary={"n_records": 0},
    )

    decision = select_strategy(snapshot, config=config)
    influence = decision.strategy_trace.learned_policy_influence

    assert influence is not None
    assert influence.eligibility.eligible is False
    assert influence.applied_delta == 0.0
    assert all(record.source != "learned_policy" for record in decision.strategy_trace.ranking_influences)


def test_learned_live_delta_becomes_ranking_record_only_in_canary_mode():
    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    snapshot = all_replay_scenarios()[2]
    trace = select_strategy(snapshot, config=PhaseConfig()).strategy_trace
    target = trace.candidate_backends[0]["name"]

    decision = select_strategy(
        snapshot,
        config=_live_canary_config(dataset, _FixedDeltaPolicy(target)),
    )

    learned_records = [
        record for record in decision.strategy_trace.ranking_influences
        if record.source == "learned_policy"
    ]
    assert len(learned_records) == 1
    assert learned_records[0].target == target
    assert abs(learned_records[0].score_delta) <= 0.005
    assert decision.strategy_trace.learned_policy_influence.applied_delta == learned_records[0].score_delta
    assert decision.strategy_trace.online_influence_outcome.learned_policy_influences


def test_learned_live_delta_respects_total_cap():
    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    snapshot = all_replay_scenarios()[2]
    trace = select_strategy(snapshot, config=PhaseConfig()).strategy_trace
    target = trace.candidate_backends[0]["name"]

    decision = select_strategy(
        snapshot,
        config=_live_canary_config(dataset, _FixedDeltaPolicy(target), max_total=0.001),
    )
    record = next(
        record for record in decision.strategy_trace.ranking_influences
        if record.source == "learned_policy"
    )

    assert abs(record.score_delta) <= 0.001
    assert record.capped is True


def test_learned_live_cannot_add_backend_or_hard_veto():
    class UnsafePolicy:
        hard_veto = True

        def predict(self, _observation):
            return LearnedPolicyTrace(
                suggested_backend="new_backend",
                confidence=0.9,
                score_deltas=({"target": "new_backend", "score_delta": 0.02},),
            )

    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    snapshot = all_replay_scenarios()[2]

    decision = select_strategy(snapshot, config=_live_canary_config(dataset, UnsafePolicy()))
    shadow = decision.strategy_trace.learned_policy_shadow
    influence = decision.strategy_trace.learned_policy_influence

    assert set(shadow.invalid_suggestion_reasons) >= {
        "suggested_backend_unavailable",
        "backend_addition_attempt",
        "hard_veto_attempt",
    }
    assert influence.applied_delta == 0.0
    assert all(record.source != "learned_policy" for record in decision.strategy_trace.ranking_influences)
    assert decision.strategy_trace.online_influence_outcome.auto_disabled is True


def test_learned_live_cannot_affect_action_objective_or_space_revision():
    class UnsafeSpacePolicy(_FixedDeltaPolicy):
        action_override = True
        objective_override = True
        auto_apply_space_revision = True

    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    snapshot = all_replay_scenarios()[5]
    base_trace = select_strategy(snapshot, config=PhaseConfig()).strategy_trace
    target = base_trace.candidate_backends[0]["name"]
    config = _live_canary_config(dataset, UnsafeSpacePolicy(target))
    config = dataclasses.replace(
        config,
        online_influence_rollout=dataclasses.replace(
            config.online_influence_rollout,
            allowed_objective_levels=("generalization",),
        ),
    )

    decision = select_strategy(snapshot, config=config)
    trace = decision.strategy_trace

    assert trace.selected_intent == base_trace.selected_intent
    assert trace.selected_mode == base_trace.selected_mode
    assert trace.space_revision == base_trace.space_revision
    assert trace.space_revision.auto_applied is False
    assert set(trace.learned_policy_shadow.invalid_suggestion_reasons) >= {
        "action_override_attempt",
        "objective_override_attempt",
        "space_revision_auto_apply_attempt",
    }
    assert trace.learned_policy_influence.applied_delta == 0.0


def test_safe_soft_mode_blocks_influence_when_promotion_gate_fails():
    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    policy = LearnedMetaPolicy(max_delta=0.01).fit_imitation(dataset)
    registry = LearnedPolicyRegistryEntry(
        policy_id="policy-gated",
        policy_version="v1",
        trained_on_dataset_version=dataset.dataset_version,
        feature_schema_version=dataset.feature_schema_version,
        reward_version=dataset.reward_version,
        approved_for_shadow=True,
        approved_for_safe_soft=True,
    )
    trace = select_strategy(all_replay_scenarios()[0], config=PhaseConfig()).strategy_trace
    shadowed = LearnedPolicyShadowRunner(
        registry_entry=registry,
        policy=policy,
        mode=LearnedPolicyDeploymentMode.SAFE_SOFT,
    ).run(trace)

    assert shadowed.selected_backend == trace.selected_backend
    assert shadowed.learned_policy_shadow.safety_mask_valid is True
    assert shadowed.learned_policy_influence is not None
    assert shadowed.learned_policy_influence.eligibility.eligible is False
    assert "insufficient_shadow_rounds" in shadowed.learned_policy_influence.eligibility.reasons
    assert shadowed.learned_policy_influence.applied_delta == 0.0
    assert all(
        record.source != "learned_backend_reranker"
        for record in shadowed.ranking_influences
    )


def test_safe_soft_eligible_policy_applies_only_tiny_bounded_trace_delta():
    class DeltaPolicy:
        def __init__(self, target):
            self.target = target

        def predict(self, _observation):
            return LearnedPolicyTrace(
                suggested_intent="optimize",
                suggested_mode="exploit",
                suggested_backend=self.target,
                confidence=0.9,
                score_deltas=({"target": self.target, "score_delta": 0.02},),
            )

    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    trace = select_strategy(all_replay_scenarios()[2], config=PhaseConfig()).strategy_trace
    target = trace.candidate_backends[0]["name"]

    influenced = LearnedPolicyShadowRunner(
        registry_entry=_approved_safe_soft_registry(dataset),
        policy=DeltaPolicy(target),
        mode=LearnedPolicyDeploymentMode.SAFE_SOFT,
        promotion_gate=LearnedPolicyPromotionGate(min_shadow_rounds=5),
        shadow_summary={"n_records": 20, "safety_warning_count": 0},
        max_safe_soft_delta=0.005,
    ).run(trace)

    record = influenced.learned_policy_influence
    assert influenced.selected_backend == trace.selected_backend
    assert influenced.candidate_backends == trace.candidate_backends
    assert influenced.ranking_influences == trace.ranking_influences
    assert record.eligibility.eligible is True
    assert record.raw_delta == 0.02
    assert record.applied_delta == 0.005
    assert record.capped is True
    assert record.target_backend == target
    assert all(item.source != "learned_policy_safe_soft" for item in influenced.ranking_influences)


def test_safe_soft_learned_policy_cannot_add_backend_or_hard_veto():
    class UnsafePolicy:
        hard_veto = True

        def predict(self, _observation):
            return LearnedPolicyTrace(
                suggested_backend="new_backend",
                confidence=0.95,
                score_deltas=({"target": "new_backend", "score_delta": 0.02},),
            )

    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    trace = select_strategy(all_replay_scenarios()[0], config=PhaseConfig()).strategy_trace

    influenced = LearnedPolicyShadowRunner(
        registry_entry=_approved_safe_soft_registry(dataset),
        policy=UnsafePolicy(),
        mode=LearnedPolicyDeploymentMode.SAFE_SOFT,
        promotion_gate=LearnedPolicyPromotionGate(min_shadow_rounds=1),
        shadow_summary={"n_records": 20},
    ).run(trace)

    shadow = influenced.learned_policy_shadow
    record = influenced.learned_policy_influence
    assert shadow.suggested_backend is None
    assert shadow.score_deltas == ()
    assert set(shadow.invalid_suggestion_reasons) >= {
        "suggested_backend_unavailable",
        "score_delta_target_unavailable",
        "hard_veto_attempt",
    }
    assert record.applied_delta == 0.0
    assert record.safety_mask_valid is False
    assert influenced.selected_backend == trace.selected_backend


def test_safe_soft_learned_policy_cannot_affect_space_revision():
    class SpacePolicy:
        auto_apply_space_revision = True

        def __init__(self, target):
            self.target = target

        def predict(self, _observation):
            return LearnedPolicyTrace(
                suggested_backend=self.target,
                confidence=0.9,
                score_deltas=({"target": self.target, "score_delta": 0.003},),
            )

    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    trace = select_strategy(all_replay_scenarios()[5], config=PhaseConfig()).strategy_trace
    assert trace.space_revision is not None
    target = trace.candidate_backends[0]["name"]

    influenced = LearnedPolicyShadowRunner(
        registry_entry=_approved_safe_soft_registry(dataset),
        policy=SpacePolicy(target),
        mode=LearnedPolicyDeploymentMode.SAFE_SOFT,
        promotion_gate=LearnedPolicyPromotionGate(min_shadow_rounds=1),
        shadow_summary={"n_records": 20},
    ).run(trace)

    assert influenced.space_revision == trace.space_revision
    assert influenced.space_revision.auto_applied is False
    assert "space_revision_auto_apply_attempt" in influenced.learned_policy_shadow.invalid_suggestion_reasons
    assert influenced.learned_policy_influence.applied_delta == 0.0


def test_safe_soft_learned_auto_disable_attaches_to_online_outcome():
    class UnsafePolicy:
        def predict(self, _observation):
            return LearnedPolicyTrace(
                suggested_backend="missing_backend",
                confidence=0.99,
                score_deltas=({"target": "missing_backend", "score_delta": 0.02},),
            )

    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    trace = select_strategy(all_replay_scenarios()[0], config=PhaseConfig()).strategy_trace
    trace = dataclasses.replace(
        trace,
        online_influence_outcome=OnlineInfluenceOutcome(
            mode=OnlineInfluenceMode.SAFE_SOFT,
            enabled=True,
            baseline_top_backend=trace.selected_backend,
            influenced_top_backend=trace.selected_backend,
            top1_changed=False,
        ),
    )

    influenced = LearnedPolicyShadowRunner(
        registry_entry=_approved_safe_soft_registry(dataset),
        policy=UnsafePolicy(),
        mode=LearnedPolicyDeploymentMode.SAFE_SOFT,
        promotion_gate=LearnedPolicyPromotionGate(min_shadow_rounds=1),
        shadow_summary={"n_records": 20},
    ).run(trace)

    assert influenced.online_influence_outcome.auto_disabled is True
    assert influenced.online_influence_outcome.learned_policy_influences
    assert "suggested_backend_unavailable" in influenced.online_influence_outcome.safety_warnings


def test_learned_policy_shadow_analyzer_summarizes_records():
    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(_baseline_traces()))
    policy = LearnedMetaPolicy(max_delta=0.01).fit_imitation(dataset)
    registry = LearnedPolicyRegistryEntry(
        policy_id="policy-a",
        policy_version="v1",
        trained_on_dataset_version=dataset.dataset_version,
        feature_schema_version=dataset.feature_schema_version,
        reward_version=dataset.reward_version,
        approved_for_shadow=True,
    )
    runner = LearnedPolicyShadowRunner(
        registry_entry=registry,
        policy=policy,
        mode=LearnedPolicyDeploymentMode.SHADOW,
    )
    traces = [
        runner.run(select_strategy(snapshot, config=PhaseConfig()).strategy_trace)
        for snapshot in all_replay_scenarios()[:3]
    ]

    summary = LearnedPolicyShadowAnalyzer().summarize(traces)

    assert summary["n_records"] == 3
    assert 0.0 <= summary["backend_agreement_rate"] <= 1.0
    assert 0.0 <= summary["top1_would_change_rate"] <= 1.0
    assert "counterfactual_uncertainty_breakdown" in summary


def test_policy_evaluation_baseline_unchanged_by_learned_module():
    before = PolicyEvaluationRunner().evaluate_snapshots(all_replay_scenarios())
    _ = OfflinePolicyEvaluator().evaluate_snapshots(all_replay_scenarios())
    after = PolicyEvaluationRunner().evaluate_snapshots(all_replay_scenarios())

    assert dataclasses.asdict(
        policy_training_record_from_trace(
            select_strategy(all_replay_scenarios()[0]).strategy_trace
        )
    )["record_version"] == "policy_training_record_v1"
    assert after["baseline_summary"]["backend_changed_rate"] == before["baseline_summary"][
        "backend_changed_rate"
    ]
