from __future__ import annotations

import dataclasses
from dataclasses import asdict

from app.services.strategy_models import (
    BudgetContext,
    CampaignContext,
    CampaignSnapshot,
    DataQualityContext,
    FailureEvent,
    ObjectiveHierarchy,
    ParameterSpaceHealth,
    PriorCampaignContext,
    RouteContext,
    policy_replay_record_from_trace,
    policy_training_record_from_trace,
)
from app.services.strategy_selector import select_strategy, strategy_trace_to_dict

_AVAIL = {
    "lhs": True,
    "built_in": True,
    "random_sampling": True,
    "optuna_tpe": True,
    "optuna_cmaes": False,
    "scipy_de": False,
    "pymoo_nsga2": False,
    "nexus_lhs": True,
    "nexus_sobol": True,
    "nexus_tpe": True,
    "nexus_gp_bo": True,
    "nexus_turbo": True,
}


def _base_snapshot(**kwargs):
    params = tuple({"x": i / 12, "y": (i % 4) / 4} for i in range(12))
    kpis = tuple(0.1 + 0.03 * i for i in range(12))
    base = dict(
        round_number=6,
        max_rounds=12,
        n_observations=12,
        n_dimensions=2,
        has_categorical=False,
        has_log_scale=False,
        kpi_history=kpis,
        direction="maximize",
        available_backends=dict(_AVAIL),
        last_batch_kpis=kpis[-3:],
        last_batch_params=params[-3:],
        best_kpi_so_far=max(kpis),
        all_params=params,
        all_kpis=kpis,
    )
    base.update(kwargs)
    return CampaignSnapshot(**base)


def _trace(snapshot):
    return strategy_trace_to_dict(select_strategy(snapshot).strategy_trace)


def test_golden_high_noise_trace_favors_diagnosis_context():
    snap = _base_snapshot(
        last_batch_kpis=(1.0, 2.0, 0.2),
        failure_events=(
            FailureEvent(failure_type="measurement", reason="blank drift"),
        ),
    )

    trace = _trace(snap)

    assert trace["selected_intent"] == "diagnose"
    assert trace["context_gate"]["requires_calibration"] is True
    assert any(e["source"] == "failure:measurement" for e in trace["evidence"])


def test_golden_tiny_data_trace_favors_baseline_exploration():
    snap = _base_snapshot(
        round_number=1,
        n_observations=1,
        kpi_history=(0.1,),
        all_kpis=(0.1,),
        all_params=({"x": 0.1, "y": 0.1},),
        last_batch_kpis=(0.1,),
        last_batch_params=({"x": 0.1, "y": 0.1},),
        campaign_context=CampaignContext(current_objective_level="baseline"),
    )

    trace = _trace(snap)

    assert trace["selected_intent"] == "discover"
    assert trace["selected_mode"] == "explore"
    assert trace["context_gate"]["recommended_intent"] == "discover"
    assert trace["action_policy"]["intent_priors"]["discover"] > 0
    assert trace["action_policy"]["backend_priors"]["lhs"] > 0


def test_golden_hardware_failure_trace_recovers_without_backend_penalty():
    snap = _base_snapshot(
        failure_events=(
            FailureEvent(
                failure_type="hardware",
                reason="pump stalled",
                backend_name="nexus_gp_bo",
                penalize_backend=False,
            ),
        )
    )

    trace = _trace(snap)

    assert trace["selected_intent"] == "recover"
    assert trace["context_gate"]["requires_human_review"] is True
    hardware = [e for e in trace["evidence"] if e["source"] == "failure:hardware"]
    assert hardware and hardware[0]["effect"] == "context"


def test_golden_constraint_violation_trace_reviews_space_revision():
    snap = _base_snapshot(
        failure_events=(
            FailureEvent(
                failure_type="constraint",
                reason="voltage exceeded window",
                backend_name="nexus_gp_bo",
                params={"voltage": 3.4},
                penalize_backend=True,
            ),
        )
    )

    trace = _trace(snap)

    assert trace["selected_intent"] == "recover"
    assert trace["space_revision"]["revision_type"] == "constraint_update"
    assert trace["space_revision"]["approval_required"] is True
    assert trace["space_revision"]["affected_parameters"] == ["voltage"]
    assert any(e["effect"] == "penalize" for e in trace["evidence"])
    assert trace["action_policy"]["mode_priors"]["constrained_search"] > 0
    assert trace["action_policy"]["intent_priors"]["revise_space"] > 0
    assert trace["action_policy"]["mode_priors"]["revise_space"] > 0
    assert trace["context_summary"]["parameter_space_health"]["requires_revision"] is True


def test_golden_scientific_negative_trace_is_evidence_not_failure():
    snap = _base_snapshot(
        failure_events=(
            FailureEvent(
                failure_type="scientific_negative",
                reason="active phase absent despite clean execution",
                backend_name="nexus_tpe",
                penalize_backend=False,
            ),
        )
    )

    trace = _trace(snap)

    assert trace["selected_intent"] == "validate"
    negative = [
        e for e in trace["evidence"]
        if e["source"] == "failure:scientific_negative"
    ]
    assert negative and negative[0]["effect"] == "context"
    assert any(
        e["source"] == "hypothesis_policy" and e["target"] == "hypothesis_update"
        for e in trace["evidence"]
    )
    assert trace["action_policy"]["intent_priors"]["hypothesis_generate"] > 0
    assert trace["action_policy"]["intent_priors"]["hypothesis_test"] > 0
    assert trace["action_policy"]["mode_priors"]["hypothesis_generate"] > 0
    assert trace["action_policy"]["mode_priors"]["hypothesis_test"] > 0


def test_golden_plateau_with_route_switch_suggestion():
    kpis = tuple([0.1, 0.3, 0.5, 0.55] + [0.56] * 8)
    snap = _base_snapshot(
        kpi_history=kpis,
        all_kpis=kpis,
        last_batch_kpis=kpis[-3:],
        campaign_context=CampaignContext(
            current_objective_level="generalization",
            synthesis_routes=("electrodeposition", "gel"),
        ),
    )

    trace = _trace(snap)

    assert trace["selected_intent"] == "pivot"
    assert trace["context_gate"]["requires_route_switch"] is True
    assert trace["space_revision"]["revision_type"] == "route_switch"
    assert trace["space_revision"]["switch_route"] == "gel"
    assert trace["space_revision"]["risk_level"] == "high"


def test_golden_promising_best_requires_mechanism_validation():
    snap = _base_snapshot(
        campaign_context=CampaignContext(
            current_objective_level="mechanism",
            domain_hypotheses=("Fe stabilizes NiOOH",),
        ),
    )

    trace = _trace(snap)

    assert trace["selected_intent"] == "validate"
    assert trace["context_gate"]["requires_hypothesis_update"] is True
    assert trace["context_summary"]["n_hypotheses"] == 1
    assert trace["action_policy"]["intent_priors"]["validate"] > 0
    assert trace["action_policy"]["mode_priors"]["mechanism_validation"] > 0
    assert trace["action_policy"]["intent_priors"]["hypothesis_test"] > 0
    assert trace["action_policy"]["mode_priors"]["hypothesis_test"] > 0
    assert any(e["source"] == "hypothesis_policy" for e in trace["evidence"])


def test_strategy_trace_promotes_available_actions_nexus_and_outcome_fields():
    decision = select_strategy(_base_snapshot())
    trace = strategy_trace_to_dict(decision.strategy_trace)

    assert trace["available_actions"]
    assert {action["name"] for action in trace["available_actions"]} >= {
        "explore",
        "exploit",
        "refine",
        "stabilize",
    }
    assert "nexus_recommendation" in trace
    if trace["nexus_recommendation"] is not None:
        assert "recommended_backends" in trace["nexus_recommendation"]
        assert trace["nexus_recommendation"]["applied_as_evidence"] is True
    assert trace["outcome"]["reward"]["reward_version"] == "strategy_reward_v1"
    assert trace["outcome"]["observed"] is False


def test_campaign_context_summary_exposes_first_class_context_sections():
    context = CampaignContext(
        scientific_goal="maximize device efficiency",
        objective_context=ObjectiveHierarchy(
            current_level="mechanism",
            active_objective="phase mechanism",
            rationale="mechanism validation stage",
        ),
        parameter_space_health=ParameterSpaceHealth(
            n_failed_params=1,
            infeasible_region_count=2,
            requires_revision=True,
            reason="constraint cluster observed",
        ),
        route_context=RouteContext(
            routes=("electrodeposition", "gel"),
            active_route="electrodeposition",
            suggested_route="gel",
            requires_route_switch=True,
            reason="plateau across current route",
        ),
        budget_context=BudgetContext(
            remaining={"rounds": 2},
            max_rounds=10,
            current_round=8,
            pressure="high",
        ),
        data_quality_context=DataQualityContext(
            qc_fail_rate=0.2,
            noise_ratio=0.8,
            requires_calibration=True,
            reason="blank drift",
        ),
        prior_campaign_context=PriorCampaignContext(
            prior_campaigns=({"campaign_id": "old"},),
            literature_priors=({"doi": "10.1/example"},),
            warm_start_available=True,
            transfer_reason="same chemistry",
        ),
    )

    summary = context.summary()

    assert summary["objective_hierarchy"]["active_objective"] == "phase mechanism"
    assert summary["parameter_space_health"]["requires_revision"] is True
    assert summary["route_context"]["suggested_route"] == "gel"
    assert summary["budget_context"]["pressure"] == "high"
    assert summary["data_quality_context"]["requires_calibration"] is True
    assert summary["prior_campaign_context"]["warm_start_available"] is True

    restored = CampaignContext(
        **{
            **asdict(context),
            "objective_context": ObjectiveHierarchy(
                **asdict(context.objective_context)
            ),
            "parameter_space_health": ParameterSpaceHealth(
                **asdict(context.parameter_space_health)
            ),
            "route_context": RouteContext(**asdict(context.route_context)),
            "budget_context": BudgetContext(**asdict(context.budget_context)),
            "data_quality_context": DataQualityContext(
                **asdict(context.data_quality_context)
            ),
            "prior_campaign_context": PriorCampaignContext(
                **asdict(context.prior_campaign_context)
            ),
        }
    )
    assert restored.summary()["route_context"]["suggested_route"] == "gel"


def test_campaign_context_backward_compatible_summary_fields_remain():
    context = CampaignContext(
        current_objective_level="baseline",
        synthesis_routes=("route-a",),
        budget_remaining={"rounds": 3},
        prior_campaigns=({"id": "prior"},),
        literature_priors=({"doi": "10/example"},),
    )

    summary = context.summary()

    assert summary["current_objective_level"] == "baseline"
    assert summary["synthesis_routes"] == ["route-a"]
    assert summary["budget_remaining"] == {"rounds": 3}
    assert summary["n_literature_priors"] == 1
    assert summary["prior_campaign_context"]["warm_start_available"] is True


def test_policy_training_and_replay_records_are_stable_and_versioned():
    decision = select_strategy(_base_snapshot(campaign_id="campaign-1"))
    trace = decision.strategy_trace

    training = policy_training_record_from_trace(trace, loop_id="loop-7")
    replay = policy_replay_record_from_trace(trace, loop_id="loop-7")

    assert training.record_version == "policy_training_record_v1"
    assert training.campaign_id == "campaign-1"
    assert training.loop_id == "loop-7"
    assert training.available_actions
    assert training.selected_backend == trace.selected_backend
    assert training.reward["reward_version"] == "strategy_reward_v1"
    assert replay.record_version == "policy_replay_record_v1"
    assert replay.candidate_backends == training.candidate_backends
    assert replay.applied_influences == training.applied_influences


def test_action_transition_guard_records_unstable_transition_without_blocking():
    snap = _base_snapshot(
        campaign_context=CampaignContext(current_objective_level="baseline"),
        previous_intent="pivot",
    )

    trace = _trace(snap)

    assert trace["selected_intent"] == "discover"
    guard = trace["transition_guard"]
    assert guard["from_intent"] == "pivot"
    assert guard["to_intent"] == "discover"
    assert guard["allowed"] is True
    assert guard["unstable"] is False


def test_action_transition_guard_warns_on_unjustified_jump():
    snap = _base_snapshot(
        campaign_context=CampaignContext(current_objective_level="baseline"),
        previous_intent="optimize",
    )

    trace = _trace(snap)

    guard = trace["transition_guard"]
    assert guard["from_intent"] == "optimize"
    assert guard["to_intent"] == "discover"
    assert guard["unstable"] is True
    assert any(e["source"] == "transition_guard" for e in trace["evidence"])


def test_competing_hypotheses_propose_discriminative_validation_evidence():
    snap = _base_snapshot(
        campaign_context=CampaignContext(
            current_objective_level="mechanism",
            domain_hypotheses=("Fe stabilizes NiOOH", "Ni vacancy reduces decay"),
        ),
    )

    trace = _trace(snap)

    assert trace["selected_intent"] == "validate"
    assert any(
        e["target"] == "discriminative_validation"
        and e["source"] == "hypothesis_policy"
        for e in trace["evidence"]
    )


def test_shadow_bandit_trace_reports_actual_suggested_and_agreement():
    trace = _trace(_base_snapshot())
    bandit = trace["bandit_decision"]
    shadow = trace["shadow_bandit_record"]

    assert bandit["actual_action"]
    assert bandit["actual_backend"] == trace["selected_backend"]
    assert bandit["suggested_action"]
    assert bandit["suggested_backend"]
    assert isinstance(bandit["agrees_with_actual"], bool)
    assert 0.0 <= bandit["confidence"] <= 1.0
    assert shadow["actual_backend"] == bandit["actual_backend"]
    assert shadow["suggested_backend"] == bandit["suggested_backend"]
    assert shadow["actual_reward"] == trace["strategy_reward"]["composite_reward"]


def test_objective_level_changes_action_priors_on_same_state():
    base = _base_snapshot()
    feasibility = select_strategy(dataclasses.replace(
        base,
        campaign_context=CampaignContext(current_objective_level="feasibility"),
    ))
    baseline = select_strategy(dataclasses.replace(
        base,
        campaign_context=CampaignContext(current_objective_level="baseline"),
    ))
    performance = select_strategy(dataclasses.replace(
        base,
        campaign_context=CampaignContext(current_objective_level="performance"),
    ))
    generalization = select_strategy(dataclasses.replace(
        base,
        campaign_context=CampaignContext(current_objective_level="generalization"),
    ))

    assert feasibility.strategy_trace.selected_intent in {"recover", "stabilize"}
    assert baseline.strategy_trace.selected_intent == "discover"
    assert performance.strategy_trace.selected_intent == "optimize"
    assert generalization.strategy_trace.selected_intent == "pivot"


def test_objective_transition_proposal_is_not_auto_applied():
    snap = _base_snapshot(
        campaign_context=CampaignContext(current_objective_level="performance"),
        failure_events=(
            FailureEvent(
                failure_type="scientific_negative",
                reason="clean negative result needs hypothesis validation",
            ),
        ),
    )

    trace = _trace(snap)

    transition = trace["objective_transition"]
    assert transition is not None
    assert transition["from_level"] == "performance"
    assert transition["to_level"] == "mechanism"
    assert transition["auto_applied"] is False
