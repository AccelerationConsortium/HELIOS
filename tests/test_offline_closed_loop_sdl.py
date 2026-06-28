from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass
from typing import Any

import app.services.policy_evolution as policy_evolution
from app.optimization.backend_selection import rank_backends
from app.services.backend_memory import (
    BackendPerformanceMemory,
    audit_failure_attribution,
    problem_context_key,
)
from app.services.candidate_gen import (
    Candidate,
    LinearConstraint,
    ParameterSpace,
    SearchDimension,
    is_feasible,
    sample_lhs,
    sample_random,
)
from app.services.learned_policy import PolicyDatasetBuilder, replay_records_from_traces
from app.services.policy_evolution import (
    FinalApprovalMode,
    FinalApprovalRecord,
    PolicyVersionRegistry,
    PolicyVersionRegistryEntry,
)
from app.services.strategy_models import (
    BudgetContext,
    CampaignContext,
    CampaignIntent,
    CampaignSnapshot,
    DataQualityContext,
    FailureEvent,
    FailureType,
    ObjectiveHierarchy,
    ObjectiveLevel,
    ObjectiveSpec,
    OnlineInfluenceMode,
    OptimizationMode,
    ParameterSpaceHealth,
    PriorCampaignContext,
    RouteContext,
    StrategyOutcome,
    StrategyReward,
    policy_replay_record_from_trace,
)
from app.services.strategy_selector import PhaseConfig, select_strategy


@dataclass(frozen=True)
class SimulatedWorkflow:
    workflow_id: str
    backend: str
    candidate: dict[str, Any]
    steps: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SimulatedExecutionResult:
    candidate: dict[str, Any]
    workflow: SimulatedWorkflow | None
    kpi: float | None
    executed: bool
    blocked: bool = False
    failure_type: FailureType | str | None = None
    reason: str = ""


@dataclass(frozen=True)
class OfflineRoundRecord:
    snapshot: CampaignSnapshot
    decision: Any
    outcome: StrategyOutcome
    result: SimulatedExecutionResult


def _space() -> ParameterSpace:
    return ParameterSpace(
        dimensions=(
            SearchDimension("temperature", "number", min_value=20.0, max_value=90.0),
            SearchDimension("concentration", "number", min_value=0.0, max_value=1.0),
        ),
        protocol_template={"steps": [{"op": "mix"}, {"op": "measure"}]},
        linear_constraints=(
            LinearConstraint(("temperature", "concentration"), (1.0, 60.0), "<=", 120.0),
        ),
    )


def _context(
    *,
    level: ObjectiveLevel | str = ObjectiveLevel.PERFORMANCE,
    requires_route_switch: bool = False,
    requires_calibration: bool = False,
    requires_revision: bool = False,
) -> CampaignContext:
    return CampaignContext(
        scientific_goal="Optimize offline SDL material response without human intervention",
        objective_hierarchy=(
            ObjectiveSpec(ObjectiveLevel.BASELINE, "baseline coverage", "coverage", "maximize"),
            ObjectiveSpec(ObjectiveLevel.PERFORMANCE, "device proxy", "efficiency", "maximize", target=0.85),
            ObjectiveSpec(ObjectiveLevel.MECHANISM, "mechanism validation", "hypothesis_score", "maximize"),
        ),
        objective_context=ObjectiveHierarchy(
            objectives=(
                ObjectiveSpec(ObjectiveLevel.BASELINE, "baseline coverage", "coverage"),
                ObjectiveSpec(ObjectiveLevel.PERFORMANCE, "device proxy", "efficiency"),
            ),
            current_level=level,
            active_objective="device proxy",
            rationale="offline closed-loop SDL fixture",
        ),
        current_objective_level=level,
        domain_hypotheses=("higher temperature improves conversion", "overheating damages film"),
        known_constraints=({"name": "thermal_load", "expr": "temperature + 60*concentration <= 120"},),
        parameter_space_health=ParameterSpaceHealth(
            known_constraints=({"name": "thermal_load"},),
            requires_revision=requires_revision,
            reason="constraint review requested" if requires_revision else "",
        ),
        synthesis_routes=("route_a", "route_b"),
        route_context=RouteContext(
            routes=("route_a", "route_b"),
            active_route="route_a",
            suggested_route="route_b" if requires_route_switch else None,
            requires_route_switch=requires_route_switch,
            reason="generalization requires route comparison" if requires_route_switch else "",
        ),
        measurement_protocols=({"name": "uv_vis_qc", "replicates": 2},),
        instrument_state={"mock_hplc": "ready", "mock_robot": "ready"},
        data_quality_context=DataQualityContext(
            measurement_protocols=({"name": "uv_vis_qc", "replicates": 2},),
            instrument_state={"mock_hplc": "ready"},
            qc_fail_rate=0.0,
            noise_ratio=0.1,
            requires_calibration=requires_calibration,
        ),
        material_family="offline_polymer",
        prior_campaigns=({"campaign_id": "prior-a", "best_kpi": 0.62},),
        literature_priors=({"doi": "mock-literature", "claim": "temperature improves response"},),
        prior_campaign_context=PriorCampaignContext(
            prior_campaigns=({"campaign_id": "prior-a", "best_kpi": 0.62},),
            literature_priors=({"doi": "mock-literature"},),
            warm_start_available=True,
            transfer_reason="same material family",
        ),
        budget_remaining={"rounds": 5, "samples": 16},
        budget_context=BudgetContext(
            remaining={"rounds": 5, "samples": 16},
            max_rounds=5,
            current_round=1,
            pressure="low",
        ),
    )


def _snapshot(
    *,
    round_number: int = 1,
    all_params: tuple[dict[str, Any], ...] = (),
    all_kpis: tuple[float, ...] = (),
    last_batch_params: tuple[dict[str, Any], ...] = (),
    last_batch_kpis: tuple[float, ...] = (),
    context: CampaignContext | None = None,
    failures: tuple[FailureEvent, ...] = (),
    failed_params: tuple[dict[str, Any], ...] = (),
    backend_failures: dict[str, int] | None = None,
    available_backends: dict[str, bool] | None = None,
    previous_intent: CampaignIntent | str | None = None,
) -> CampaignSnapshot:
    return CampaignSnapshot(
        round_number=round_number,
        max_rounds=5,
        n_observations=len(all_kpis),
        n_dimensions=2,
        has_categorical=False,
        has_log_scale=False,
        kpi_history=all_kpis,
        direction="maximize",
        available_backends=available_backends
        or {
            "lhs": True,
            "nexus_lhs": True,
            "nexus_sobol": True,
            "optuna_tpe": False,
            "nexus_tpe": True,
            "nexus_gp_bo": True,
            "bomcp": True,
            "built_in": True,
        },
        last_batch_kpis=last_batch_kpis,
        last_batch_params=last_batch_params,
        best_kpi_so_far=max(all_kpis) if all_kpis else None,
        all_params=all_params,
        all_kpis=all_kpis,
        qc_fail_rate=0.0,
        backend_failure_counts=backend_failures or {},
        failed_params=failed_params,
        campaign_context=context or _context(),
        failure_events=failures,
        campaign_id="offline-sdl",
        previous_intent=previous_intent,
    )


def _candidate_from_backend(backend: str, space: ParameterSpace, round_number: int) -> Candidate:
    if backend.startswith("nexus"):
        params = sample_lhs(space, 1, seed=100 + round_number)[0]
        return Candidate(0, params, origin=f"mock_{backend}", score=0.7)
    if backend == "bomcp":
        params = {
            "temperature": 58.0 + round_number,
            "concentration": 0.25,
        }
        return Candidate(0, params, origin="mock_bomcp", score=0.8)
    params = sample_random(space, 1, seed=200 + round_number)[0]
    return Candidate(0, params, origin=f"mock_{backend}", score=0.4)


def _compile(candidate: Candidate, backend: str) -> SimulatedWorkflow:
    return SimulatedWorkflow(
        workflow_id=f"workflow-{backend}-{candidate.index}",
        backend=backend,
        candidate=dict(candidate.params),
        steps=(
            {"op": "set_temperature", "value": candidate.params["temperature"]},
            {"op": "dose", "value": candidate.params["concentration"]},
            {"op": "measure", "metric": "efficiency"},
        ),
    )


def _safety_check(candidate: Candidate, space: ParameterSpace) -> tuple[bool, str]:
    if not is_feasible(candidate.params, space):
        return False, "hard constraint violated"
    return True, "safe"


def _execute(workflow: SimulatedWorkflow) -> SimulatedExecutionResult:
    temperature = float(workflow.candidate["temperature"])
    concentration = float(workflow.candidate["concentration"])
    kpi = 0.35 + 0.004 * temperature + 0.18 * concentration - 0.00008 * (temperature - 65.0) ** 2
    return SimulatedExecutionResult(
        candidate=dict(workflow.candidate),
        workflow=workflow,
        kpi=round(kpi, 5),
        executed=True,
    )


def _outcome(
    result: SimulatedExecutionResult,
    *,
    failure_events: tuple[FailureEvent, ...] = (),
    notes: str = "",
) -> StrategyOutcome:
    reward = StrategyReward(
        objective_improvement=float(result.kpi or 0.0),
        information_gain=0.1,
        constraint_satisfaction=0.0 if result.blocked else 1.0,
        data_quality_gain=0.0 if failure_events else 0.1,
        failure_penalty=1.0 if result.blocked else 0.0,
        composite_reward=0.0 if result.kpi is None else min(1.0, float(result.kpi)),
    )
    return StrategyOutcome(
        outcome="blocked" if result.blocked else "observed",
        reward=reward,
        failure_events=failure_events,
        safety_flags=("blocked_by_safety",) if result.blocked else (),
        notes=notes,
        observed=result.executed,
    )


def _run_round(snapshot: CampaignSnapshot, space: ParameterSpace) -> OfflineRoundRecord:
    decision = select_strategy(snapshot, config=PhaseConfig(enable_nexus=False, enable_method_advisor=False))
    candidate = _candidate_from_backend(decision.backend_name, space, snapshot.round_number)
    safe, reason = _safety_check(candidate, space)
    if not safe:
        failure = FailureEvent(
            FailureType.CONSTRAINT,
            reason,
            backend_name=decision.backend_name,
            round_number=snapshot.round_number,
            candidate_index=candidate.index,
            params=dict(candidate.params),
            penalize_backend=True,
        )
        result = SimulatedExecutionResult(
            candidate=dict(candidate.params),
            workflow=None,
            kpi=None,
            executed=False,
            blocked=True,
            failure_type=FailureType.CONSTRAINT,
            reason=reason,
        )
        outcome = _outcome(
            result,
            failure_events=tuple((*snapshot.failure_events, failure)),
            notes=reason,
        )
    else:
        workflow = _compile(candidate, decision.backend_name)
        result = _execute(workflow)
        outcome = _outcome(
            result,
            failure_events=snapshot.failure_events,
            notes=f"candidate generated by {decision.backend_name}",
        )
    trace = dataclasses.replace(decision.strategy_trace, outcome=outcome, strategy_reward=outcome.reward)
    decision = dataclasses.replace(decision, strategy_trace=trace)
    return OfflineRoundRecord(snapshot, decision, outcome, result)


def _advance_snapshot(record: OfflineRoundRecord) -> CampaignSnapshot:
    snapshot = record.snapshot
    all_params = tuple((*snapshot.all_params, record.result.candidate))
    all_kpis = snapshot.all_kpis if record.result.kpi is None else tuple((*snapshot.all_kpis, record.result.kpi))
    failures = tuple((*snapshot.failure_events, *record.outcome.failure_events))
    failed_params = snapshot.failed_params
    if record.result.blocked:
        failed_params = tuple((*failed_params, record.result.candidate))
    return _snapshot(
        round_number=snapshot.round_number + 1,
        all_params=all_params,
        all_kpis=all_kpis,
        last_batch_params=(record.result.candidate,),
        last_batch_kpis=() if record.result.kpi is None else (record.result.kpi,),
        context=snapshot.campaign_context,
        failures=failures,
        failed_params=failed_params,
        previous_intent=record.decision.strategy_trace.selected_intent,
    )


def test_offline_campaign_runs_multiple_rounds_without_human_intervention():
    space = _space()
    snapshot = _snapshot(all_params=({"temperature": 35.0, "concentration": 0.2},), all_kpis=(0.48,))
    records: list[OfflineRoundRecord] = []

    for _ in range(4):
        record = _run_round(snapshot, space)
        records.append(record)
        snapshot = _advance_snapshot(record)

    assert len(records) == 4
    assert snapshot.n_observations >= 4
    for record in records:
        trace = record.decision.strategy_trace
        assert trace is not None
        assert trace.outcome is not None
        assert isinstance(trace.outcome.reward, StrategyReward)
        assert trace.round_number == record.snapshot.round_number
        if trace.space_revision is not None:
            assert trace.space_revision.approval_required is True
            assert trace.space_revision.auto_applied is False
        assert trace.learned_policy_shadow is None
        assert trace.learned_policy_influence is None
        assert trace.ranking_influences == ()


def test_dynamic_strategy_routes_to_valid_downstream_actions():
    cases = (
        (_context(level=ObjectiveLevel.BASELINE), (), {CampaignIntent.DISCOVER}, {OptimizationMode.EXPLORE}),
        (_context(level=ObjectiveLevel.PERFORMANCE), (), {CampaignIntent.OPTIMIZE}, {OptimizationMode.EXPLOIT, OptimizationMode.EXPLORE}),
        (_context(level=ObjectiveLevel.MECHANISM), (), {CampaignIntent.VALIDATE, CampaignIntent.STABILIZE}, {OptimizationMode.EXPLOIT, OptimizationMode.EXPLORE, OptimizationMode.REPLICATE}),
        (_context(level=ObjectiveLevel.DATA_QUALITY, requires_calibration=True), (), {CampaignIntent.STABILIZE}, {OptimizationMode.REPLICATE}),
        (_context(level=ObjectiveLevel.PERFORMANCE), (FailureEvent(FailureType.MEASUREMENT, "qc drift"),), {CampaignIntent.DIAGNOSE}, {OptimizationMode.REPLICATE, OptimizationMode.EXPLORE, OptimizationMode.EXPLOIT}),
        (_context(level=ObjectiveLevel.PERFORMANCE), (FailureEvent(FailureType.HARDWARE, "pump stalled"),), {CampaignIntent.RECOVER}, {OptimizationMode.REPLICATE, OptimizationMode.EXPLORE, OptimizationMode.EXPLOIT}),
        (_context(level=ObjectiveLevel.GENERALIZATION, requires_route_switch=True), (), {CampaignIntent.PIVOT}, {OptimizationMode.EXPLOIT, OptimizationMode.EXPLORE}),
    )

    for context, failures, expected_intents, allowed_modes in cases:
        decision = select_strategy(
            _snapshot(
                all_params=({"temperature": 40.0, "concentration": 0.2},) * 6,
                all_kpis=(0.4, 0.42, 0.45, 0.47, 0.5, 0.53),
                last_batch_params=({"temperature": 50.0, "concentration": 0.2},),
                last_batch_kpis=(0.53,),
                context=context,
                failures=failures,
            ),
            config=PhaseConfig(enable_nexus=False, enable_method_advisor=False),
        )
        trace = decision.strategy_trace
        assert trace.selected_intent in {item.value for item in expected_intents}
        assert trace.selected_mode in {item.value for item in allowed_modes}
        if context.current_objective_level == ObjectiveLevel.MECHANISM:
            assert trace.action_policy.intent_priors.get("validate", 0.0) > 0
            assert trace.action_policy.mode_priors.get("mechanism_validation", 0.0) > 0
        if trace.space_revision is not None:
            assert trace.space_revision.auto_applied is False

    transfer_decision = select_strategy(
        _snapshot(
            context=_context(level=ObjectiveLevel.GENERALIZATION),
            all_params=({"temperature": 40.0, "concentration": 0.2},) * 6,
            all_kpis=(0.4, 0.42, 0.45, 0.47, 0.5, 0.53),
        ),
        config=PhaseConfig(enable_nexus=False, enable_method_advisor=False),
    )
    assert "transfer" in transfer_decision.strategy_trace.action_policy.intent_priors
    assert any(
        evidence.target in {"route_switch", "space_revision", "warm_start"}
        for evidence in transfer_decision.strategy_trace.action_policy.evidence
    )


def test_backend_failure_fallbacks_preserve_closed_loop():
    space = _space()
    failure = FailureEvent(
        FailureType.BACKEND,
        "mock nexus unavailable and BO MCP timed out",
        backend_name="nexus_gp_bo",
        round_number=2,
        penalize_backend=True,
    )
    snapshot = _snapshot(
        round_number=2,
        all_params=({"temperature": 40.0, "concentration": 0.2},) * 5,
        all_kpis=(0.4, 0.43, 0.45, 0.48, 0.5),
        failures=(failure,),
        backend_failures={"nexus_gp_bo": 3, "bomcp": 3},
        available_backends={"nexus_gp_bo": False, "bomcp": False, "built_in": True, "lhs": True},
    )

    decision = select_strategy(snapshot, config=PhaseConfig(enable_nexus=False, enable_method_advisor=False))
    record = _run_round(dataclasses.replace(snapshot, available_backends={"built_in": True, "lhs": True}), space)

    assert audit_failure_attribution(FailureType.BACKEND)["penalizes_backend"] is True
    assert decision.strategy_trace.context_gate.ready_for_optimization is False
    assert decision.strategy_trace.selected_intent == "recover"
    assert record.decision.backend_name in {"built_in", "lhs"}
    assert record.result.executed or record.result.blocked
    assert any(e.source == "failure:backend" for e in decision.strategy_trace.evidence)


def test_candidate_compile_safety_execution_pipeline():
    space = _space()
    provenance = []
    for backend in ("nexus_gp_bo", "bomcp"):
        candidate = _candidate_from_backend(backend, space, 1)
        safe, reason = _safety_check(candidate, space)
        assert safe, reason
        workflow = _compile(candidate, backend)
        result = _execute(workflow)
        outcome = _outcome(result, notes=f"{candidate.origin}->{workflow.workflow_id}")
        provenance.append((candidate, workflow, result, outcome))

    for candidate, workflow, result, outcome in provenance:
        assert candidate.origin.startswith("mock_")
        assert workflow.backend in {"nexus_gp_bo", "bomcp"}
        assert workflow.candidate == candidate.params
        assert result.workflow == workflow
        assert outcome.observed is True
        assert workflow.workflow_id in outcome.notes


def test_safety_violation_blocks_execution_and_triggers_recovery():
    space = _space()
    bad_candidate = Candidate(0, {"temperature": 90.0, "concentration": 0.8}, "mock_bomcp", score=0.9)
    safe, reason = _safety_check(bad_candidate, space)
    failure = FailureEvent(
        FailureType.CONSTRAINT,
        reason,
        backend_name="bomcp",
        round_number=1,
        candidate_index=0,
        params=bad_candidate.params,
        penalize_backend=True,
    )
    result = SimulatedExecutionResult(
        candidate=bad_candidate.params,
        workflow=None,
        kpi=None,
        executed=False,
        blocked=True,
        failure_type=FailureType.CONSTRAINT,
        reason=reason,
    )
    next_decision = select_strategy(
        _snapshot(failures=(failure,), failed_params=(bad_candidate.params,)),
        config=PhaseConfig(enable_nexus=False, enable_method_advisor=False),
    )

    assert safe is False
    assert result.executed is False
    assert failure.failure_type == FailureType.CONSTRAINT
    assert next_decision.strategy_trace.selected_intent in {"recover", "diagnose"}
    assert next_decision.strategy_trace.context_gate.requires_space_revision is True
    assert next_decision.strategy_trace.space_revision.auto_applied is False


def test_measurement_failure_triggers_stabilize_not_backend_penalty():
    memory = BackendPerformanceMemory()
    snapshot = _snapshot(
        failures=(FailureEvent(FailureType.MEASUREMENT, "calibration drift", backend_name="nexus_gp_bo"),),
        context=_context(level=ObjectiveLevel.DATA_QUALITY, requires_calibration=True),
    )
    context_key = problem_context_key(snapshot)
    perf = memory.record_failure_event(
        context_key=context_key,
        backend_name="nexus_gp_bo",
        action_type="optimize",
        failure_type=FailureType.MEASUREMENT,
    )
    decision = select_strategy(snapshot, config=PhaseConfig(enable_nexus=False, enable_method_advisor=False))

    assert audit_failure_attribution(FailureType.MEASUREMENT)["penalizes_backend"] is False
    assert perf.success_rate == 1.0
    assert perf.failure_rate == 0.0
    assert decision.strategy_trace.selected_intent in {"diagnose", "stabilize"}
    priors = decision.strategy_trace.action_policy.intent_priors
    assert priors.get("stabilize", 0.0) > 0
    assert decision.strategy_trace.action_policy.mode_priors.get("baseline_calibration", 0.0) > 0


def test_scientific_negative_is_evidence_not_failure():
    memory = BackendPerformanceMemory()
    event = FailureEvent(
        FailureType.SCIENTIFIC_NEGATIVE,
        "valid film produced but hypothesis failed",
        backend_name="nexus_gp_bo",
        penalize_backend=False,
    )
    snapshot = _snapshot(
        failures=(event,),
        context=_context(level=ObjectiveLevel.MECHANISM),
        all_params=({"temperature": 62.0, "concentration": 0.25},),
        all_kpis=(0.22,),
    )
    perf = memory.record_failure_event(
        context_key=problem_context_key(snapshot),
        backend_name="nexus_gp_bo",
        action_type="validate",
        failure_type=FailureType.SCIENTIFIC_NEGATIVE,
    )
    decision = select_strategy(snapshot, config=PhaseConfig(enable_nexus=False, enable_method_advisor=False))

    assert audit_failure_attribution(FailureType.SCIENTIFIC_NEGATIVE)["evidence_only"] is True
    assert perf.success_rate == 1.0
    assert perf.failure_rate == 0.0
    assert decision.strategy_trace.selected_intent == "validate"
    assert decision.strategy_trace.outcome.failure_events == (event,)
    assert any(e.source == "hypothesis_policy" for e in decision.strategy_trace.evidence)


def test_campaign_state_persists_across_rounds():
    space = _space()
    snapshot = _snapshot(
        all_params=({"temperature": 35.0, "concentration": 0.2},),
        all_kpis=(0.45,),
    )
    records = []
    memory = BackendPerformanceMemory()
    constraint_failure = FailureEvent(
        FailureType.CONSTRAINT,
        "manual simulated constraint event",
        backend_name="bomcp",
        round_number=2,
        params={"temperature": 90.0, "concentration": 0.8},
        penalize_backend=True,
    )
    for idx in range(3):
        if idx == 1:
            snapshot = dataclasses.replace(
                snapshot,
                failure_events=(*snapshot.failure_events, constraint_failure),
                failed_params=(*snapshot.failed_params, constraint_failure.params),
            )
        record = _run_round(snapshot, space)
        records.append(record)
        memory.record_failure_event(
            context_key=problem_context_key(snapshot),
            backend_name=record.decision.backend_name,
            action_type=record.decision.strategy_trace.selected_intent,
            failure_type=FailureType.SCIENTIFIC_NEGATIVE if idx == 2 else FailureType.MEASUREMENT,
        )
        snapshot = _advance_snapshot(record)

    traces = tuple(record.decision.strategy_trace for record in records)
    dataset = PolicyDatasetBuilder().build(replay_records_from_traces(traces))

    assert snapshot.campaign_context.summary()["objective_hierarchy"]["current_level"] == "performance"
    assert any(trace.space_revision is not None for trace in traces)
    assert any(trace.outcome.failure_events for trace in traces)
    assert memory.to_json()
    assert dataset.records
    assert all(record["record_version"] == "policy_replay_record_v1" for record in dataset.records)


def test_legacy_closed_loop_regression_with_new_features_disabled():
    config = PhaseConfig(
        enable_nexus=False,
        enable_optimization_intelligence=False,
        enable_method_advisor=False,
    )
    snapshot = _snapshot(
        all_params=({"temperature": 40.0, "concentration": 0.2},) * 6,
        all_kpis=(0.4, 0.43, 0.45, 0.48, 0.5, 0.53),
        last_batch_params=({"temperature": 50.0, "concentration": 0.2},),
        last_batch_kpis=(0.53,),
        available_backends={"built_in": True},
        context=CampaignContext(),
    )

    decision = select_strategy(snapshot, config=config)

    assert decision.backend_name == "built_in"
    assert decision.strategy_trace.learned_policy_shadow is None
    assert decision.strategy_trace.learned_policy_influence is None
    assert decision.strategy_trace.online_influence_outcome is None
    assert decision.strategy_trace.space_revision is None
    assert decision.strategy_trace.outcome.observed is False


def test_self_evolution_metadata_does_not_affect_live_campaign():
    snapshot = _snapshot(
        all_params=({"temperature": 40.0, "concentration": 0.2},) * 5,
        all_kpis=(0.4, 0.43, 0.45, 0.48, 0.5),
    )
    baseline_rank = rank_backends(
        "optimize",
        ("nexus_gp_bo", "bomcp", "built_in"),
        {"nexus_gp_bo": True, "bomcp": True, "built_in": True},
    )
    baseline_decision = select_strategy(snapshot, config=PhaseConfig(enable_nexus=False, enable_method_advisor=False))
    registry = PolicyVersionRegistry().register(
        PolicyVersionRegistryEntry(
            policy_id="learned-sdl",
            policy_version="v1",
            trained_on_dataset_version="policy_dataset_sdl",
            feature_schema_version="policy_feature_schema_v1",
            reward_version="strategy_reward_v1",
            approved_for_shadow=True,
            approved_for_live_canary=True,
            rollback_target=("learned-sdl", "v0"),
        )
    )
    approval = FinalApprovalRecord(
        approval_id="final-sdl",
        proposal_id="promotion-sdl",
        policy_id="learned-sdl",
        policy_version="v1",
        approved_by="test-reviewer",
        approval_mode=FinalApprovalMode.TEST,
        approval_reason="metadata-only approval",
        allowed_campaign_ids=("offline-sdl",),
        allowed_objective_levels=("performance",),
        max_live_weight=0.005,
        max_top1_change_rate=0.2,
        rollback_policy_id="learned-sdl",
        rollback_policy_version="v0",
    )
    registry = registry.mark_final_approved("learned-sdl", "v1", approval)
    config = PhaseConfig(
        enable_nexus=False,
        enable_method_advisor=False,
        learned_policy_registry_entry=registry.get("learned-sdl", "v1"),
        learned_policy=object(),
    )

    after_rank = rank_backends(
        "optimize",
        ("nexus_gp_bo", "bomcp", "built_in"),
        {"nexus_gp_bo": True, "bomcp": True, "built_in": True},
    )
    after_decision = select_strategy(snapshot, config=config)
    source = inspect.getsource(policy_evolution)

    assert after_rank == baseline_rank
    assert after_decision.backend_name == baseline_decision.backend_name
    assert after_decision.strategy_trace.learned_policy_shadow is None
    assert after_decision.strategy_trace.learned_policy_influence is None
    assert config.online_influence_rollout.mode == OnlineInfluenceMode.OFF
    assert "from app.services.strategy_selector" not in source
    assert "rank_backends(" not in source


def test_replay_record_contains_closed_loop_outcome_links():
    space = _space()
    record = _run_round(_snapshot(all_params=({"temperature": 40.0, "concentration": 0.2},), all_kpis=(0.4,)), space)
    replay = policy_replay_record_from_trace(record.decision.strategy_trace, loop_id="offline-loop-1")

    assert replay.campaign_id == "offline-sdl"
    assert replay.outcome["outcome"] == "observed"
    assert replay.reward["reward_version"] == "strategy_reward_v1"
    assert replay.selected_backend == record.decision.backend_name
    assert replay.candidate_backends
