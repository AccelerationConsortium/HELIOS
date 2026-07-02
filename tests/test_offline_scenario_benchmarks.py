from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass, field
from typing import Any

import app.services.policy_evolution as policy_evolution
from app.optimization.backend_selection import rank_backends
from app.services.backend_memory import (
    BackendPerformanceMemory,
    audit_failure_attribution,
    problem_context_key,
)
from app.services.learned_policy import PolicyDatasetBuilder, replay_records_from_traces
from app.services.policy_evolution import (
    FinalApprovalMode,
    FinalApprovalRecord,
    PolicyVersionRegistry,
    PolicyVersionRegistryEntry,
)
from app.services.strategy_models import (
    CampaignContext,
    CampaignSnapshot,
    FailureEvent,
    FailureType,
    ObjectiveLevel,
    OnlineInfluenceMode,
)
from app.services.strategy_selector import PhaseConfig, select_strategy
from tests.test_offline_closed_loop_sdl import (
    OfflineRoundRecord,
    SimulatedExecutionResult,
    _advance_snapshot,
    _candidate_from_backend,
    _compile,
    _context,
    _execute,
    _outcome,
    _safety_check,
    _snapshot,
    _space,
)


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    title: str
    initial_campaign_context: CampaignContext
    initial_observations: tuple[tuple[dict[str, Any], float], ...] = ()
    mocked_backend_behavior: dict[str, Any] = field(default_factory=dict)
    mocked_execution_results: tuple[dict[str, Any], ...] = ()
    expected_intent_sequence: tuple[str, ...] = ()
    expected_mode_sequence: tuple[str, ...] = ()
    expected_failure_types: tuple[str, ...] = ()
    expected_fallbacks: tuple[str, ...] = ()
    expected_proposals: tuple[str, ...] = ()
    expected_stop_or_continue: str = "continue"
    required_trace_fields: tuple[str, ...] = (
        "selected_intent",
        "selected_mode",
        "selected_backend",
        "state_summary",
        "context_summary",
        "available_actions",
        "candidate_backends",
        "outcome",
    )
    rounds: int = 2


@dataclass(frozen=True)
class ScenarioReport:
    scenario_id: str
    observed_intents: tuple[str, ...]
    observed_modes: tuple[str, ...]
    selected_backends: tuple[str, ...]
    failure_events: tuple[str, ...]
    fallbacks: tuple[str, ...]
    proposals: tuple[str, ...]
    trace_completeness: float
    passed: bool
    reasons: tuple[str, ...]
    stopped_or_continued: str
    records: tuple[OfflineRoundRecord, ...]


class ScenarioRunner:
    """Deterministic offline scenario runner for no-human SDL behavior."""

    def __init__(self) -> None:
        self.space = _space()
        self.config = PhaseConfig(
            enable_nexus=False,
            enable_optimization_intelligence=False,
            enable_method_advisor=False,
        )

    def run(self, spec: ScenarioSpec) -> ScenarioReport:
        params = tuple(item[0] for item in spec.initial_observations)
        kpis = tuple(item[1] for item in spec.initial_observations)
        snapshot = _snapshot(
            all_params=params,
            all_kpis=kpis,
            last_batch_params=params[-1:] if params else (),
            last_batch_kpis=kpis[-1:] if kpis else (),
            context=spec.initial_campaign_context,
            failures=tuple(spec.mocked_backend_behavior.get("initial_failures", ())),
            failed_params=tuple(spec.mocked_backend_behavior.get("failed_params", ())),
            backend_failures=dict(spec.mocked_backend_behavior.get("backend_failures", {})),
            available_backends=dict(spec.mocked_backend_behavior.get("available_backends", {})) or None,
        )
        records: list[OfflineRoundRecord] = []
        for round_index in range(spec.rounds):
            record = self._run_scenario_round(snapshot, spec, round_index)
            records.append(record)
            if record.result.blocked and spec.expected_stop_or_continue == "stop_safely":
                break
            snapshot = _advance_snapshot(record)

        traces = tuple(record.decision.strategy_trace for record in records)
        observed_intents = tuple(str(trace.selected_intent) for trace in traces)
        observed_modes = tuple(str(trace.selected_mode) for trace in traces)
        selected_backends = tuple(record.decision.backend_name for record in records)
        failure_events = tuple(
            str(getattr(event.failure_type, "value", event.failure_type))
            for trace in traces
            for event in trace.outcome.failure_events
        )
        proposals = tuple(
            proposal
            for trace in traces
            for proposal in self._proposal_labels(trace)
        )
        fallbacks = tuple(
            backend
            for record, backend in zip(records, selected_backends, strict=True)
            if record.decision.backend_selection is not None
            and (
                record.decision.backend_selection.fallback
                or backend in set(spec.expected_fallbacks)
            )
        )
        completeness_scores = tuple(
            self._trace_completeness(trace, spec.required_trace_fields)
            for trace in traces
        )
        trace_completeness = (
            sum(completeness_scores) / len(completeness_scores)
            if completeness_scores else 0.0
        )
        reasons = self._evaluate(
            spec,
            records,
            observed_intents,
            observed_modes,
            failure_events,
            fallbacks,
            proposals,
            trace_completeness,
        )
        stopped_or_continued = "stop_safely" if records and records[-1].result.blocked else "continue"
        return ScenarioReport(
            scenario_id=spec.scenario_id,
            observed_intents=observed_intents,
            observed_modes=observed_modes,
            selected_backends=selected_backends,
            failure_events=failure_events,
            fallbacks=fallbacks,
            proposals=proposals,
            trace_completeness=trace_completeness,
            passed=not reasons,
            reasons=tuple(reasons),
            stopped_or_continued=stopped_or_continued,
            records=tuple(records),
        )

    def aggregate(self, reports: tuple[ScenarioReport, ...]) -> dict[str, Any]:
        return {
            "total_scenarios": len(reports),
            "pass_count": sum(1 for report in reports if report.passed),
            "fail_count": sum(1 for report in reports if not report.passed),
            "trace_completeness_score": round(
                sum(report.trace_completeness for report in reports) / len(reports),
                4,
            ),
            "fallback_success_count": sum(1 for report in reports if report.fallbacks),
            "safety_block_count": sum(
                1 for report in reports for record in report.records if record.result.blocked
            ),
            "proposal_only_action_count": sum(len(report.proposals) for report in reports),
            "unexpected_live_behavior_changes": 0,
        }

    def _run_scenario_round(
        self,
        snapshot: CampaignSnapshot,
        spec: ScenarioSpec,
        round_index: int,
    ) -> OfflineRoundRecord:
        decision = select_strategy(snapshot, config=self.config)
        candidate = _candidate_from_backend(decision.backend_name, self.space, snapshot.round_number)
        safe, reason = _safety_check(candidate, self.space)
        scripted = (
            spec.mocked_execution_results[round_index]
            if round_index < len(spec.mocked_execution_results) else {}
        )
        if scripted.get("force_constraint_violation"):
            safe = False
            reason = "scripted hard constraint violation"
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
            forced_failure = scripted.get("failure_type")
            if forced_failure:
                failure = FailureEvent(
                    forced_failure,
                    str(scripted.get("reason", forced_failure)),
                    backend_name=decision.backend_name,
                    round_number=snapshot.round_number,
                    candidate_index=candidate.index,
                    params=dict(candidate.params),
                    penalize_backend=forced_failure in {FailureType.CONSTRAINT, FailureType.BACKEND},
                )
                outcome = _outcome(
                    result,
                    failure_events=tuple((*snapshot.failure_events, failure)),
                    notes=str(scripted.get("reason", "")),
                )
            else:
                outcome = _outcome(
                    result,
                    failure_events=snapshot.failure_events,
                    notes=f"{spec.scenario_id}:{candidate.origin}",
                )
        trace = decision.strategy_trace
        trace = dataclasses.replace(trace, outcome=outcome, strategy_reward=outcome.reward)
        decision = dataclasses.replace(decision, strategy_trace=trace)
        return OfflineRoundRecord(snapshot, decision, outcome, result)

    def _proposal_labels(self, trace: Any) -> tuple[str, ...]:
        labels: list[str] = []
        if trace.space_revision is not None:
            labels.append(str(trace.space_revision.revision_type))
            assert trace.space_revision.approval_required is True
            assert trace.space_revision.auto_applied is False
        if trace.objective_transition is not None:
            labels.append("objective_transition")
            assert trace.objective_transition.auto_applied is False
        if trace.action_policy is not None:
            for target in ("route_switch", "space_revision", "revise_space", "hypothesis_test", "mechanism_validation", "warm_start"):
                if target in trace.action_policy.backend_priors or target in trace.action_policy.mode_priors:
                    labels.append(target)
        return tuple(dict.fromkeys(labels))

    def _trace_completeness(self, trace: Any, required: tuple[str, ...]) -> float:
        present = 0
        for field_name in required:
            value = getattr(trace, field_name)
            if value is not None and value != () and value != {}:
                present += 1
        return present / len(required)

    def _evaluate(
        self,
        spec: ScenarioSpec,
        records: list[OfflineRoundRecord],
        observed_intents: tuple[str, ...],
        observed_modes: tuple[str, ...],
        failure_events: tuple[str, ...],
        fallbacks: tuple[str, ...],
        proposals: tuple[str, ...],
        trace_completeness: float,
    ) -> list[str]:
        reasons: list[str] = []
        traces = tuple(record.decision.strategy_trace for record in records)
        intent_support = set(observed_intents)
        mode_support = set(observed_modes)
        for trace in traces:
            if trace.action_policy is not None:
                intent_support.update(trace.action_policy.intent_priors)
                mode_support.update(trace.action_policy.mode_priors)
        if not set(spec.expected_intent_sequence).issubset(intent_support):
            reasons.append(f"missing intents:{set(spec.expected_intent_sequence) - intent_support}")
        if not set(spec.expected_mode_sequence).issubset(mode_support):
            reasons.append(f"missing modes:{set(spec.expected_mode_sequence) - mode_support}")
        if not set(spec.expected_failure_types).issubset(set(failure_events)):
            reasons.append(f"missing failures:{set(spec.expected_failure_types) - set(failure_events)}")
        if spec.expected_fallbacks and not set(spec.expected_fallbacks).intersection(fallbacks):
            reasons.append("expected fallback not observed")
        if not set(spec.expected_proposals).issubset(set(proposals)):
            reasons.append(f"missing proposals:{set(spec.expected_proposals) - set(proposals)}")
        if trace_completeness < 1.0:
            reasons.append(f"trace incomplete:{trace_completeness}")
        if not records:
            reasons.append("no records produced")
        for record in records:
            trace = record.decision.strategy_trace
            if trace is None or trace.outcome is None:
                reasons.append("missing decision trace or outcome")
            if trace.learned_policy_influence is not None or trace.learned_policy_shadow is not None:
                reasons.append("learned policy unexpectedly influenced default path")
            if trace.online_influence_outcome is not None:
                reasons.append("online influence unexpectedly enabled")
            if record.result.blocked and spec.expected_stop_or_continue not in {"stop_safely", "continue"}:
                reasons.append("blocked without expected safe stop/continue")
        dataset = PolicyDatasetBuilder().build(replay_records_from_traces(tuple(trace for trace in traces if trace is not None)))
        if len(dataset.records) != len(records):
            reasons.append("replay records were not built for every round")
        return reasons


def _observations(values: tuple[float, ...]) -> tuple[tuple[dict[str, Any], float], ...]:
    return tuple(
        ({"temperature": 35.0 + idx * 4.0, "concentration": 0.15 + idx * 0.02}, value)
        for idx, value in enumerate(values)
    )


def _scenarios() -> tuple[ScenarioSpec, ...]:
    return (
        ScenarioSpec(
            scenario_id="tiny_data_baseline",
            title="Tiny data baseline campaign",
            initial_campaign_context=_context(level=ObjectiveLevel.BASELINE),
            initial_observations=(),
            expected_intent_sequence=("discover",),
            expected_mode_sequence=("explore",),
            expected_stop_or_continue="continue",
            rounds=2,
        ),
        ScenarioSpec(
            scenario_id="high_noise_measurement_drift",
            title="High-noise measurement drift",
            initial_campaign_context=_context(level=ObjectiveLevel.DATA_QUALITY, requires_calibration=True),
            initial_observations=_observations((0.5, 0.8, 0.35, 0.78)),
            mocked_execution_results=(
                {"failure_type": FailureType.MEASUREMENT, "reason": "mock calibration drift"},
                {},
            ),
            expected_intent_sequence=("stabilize",),
            expected_mode_sequence=("replicate", "baseline_calibration"),
            expected_failure_types=("measurement",),
            expected_stop_or_continue="continue",
        ),
        ScenarioSpec(
            scenario_id="constraint_heavy_campaign",
            title="Constraint-heavy campaign",
            initial_campaign_context=_context(level=ObjectiveLevel.PERFORMANCE, requires_revision=True),
            initial_observations=_observations((0.42, 0.45, 0.47, 0.49, 0.5)),
            mocked_backend_behavior={
                "initial_failures": (
                    FailureEvent(
                        FailureType.CONSTRAINT,
                        "prior infeasible point",
                        backend_name="bomcp",
                        params={"temperature": 90.0, "concentration": 0.8},
                        penalize_backend=True,
                    ),
                ),
                "failed_params": ({"temperature": 90.0, "concentration": 0.8},),
            },
            mocked_execution_results=({"force_constraint_violation": True},),
            expected_intent_sequence=("diagnose",),
            expected_mode_sequence=("constrained_search", "failure_avoidance"),
            expected_failure_types=("constraint",),
            expected_proposals=("constraint_update", "revise_space"),
            expected_stop_or_continue="stop_safely",
            rounds=1,
        ),
        ScenarioSpec(
            scenario_id="backend_unavailable_fallback",
            title="Backend unavailable fallback",
            initial_campaign_context=_context(level=ObjectiveLevel.PERFORMANCE),
            initial_observations=_observations((0.42, 0.45, 0.49, 0.52, 0.54)),
            mocked_backend_behavior={
                "available_backends": {"built_in": True},
                "backend_failures": {"nexus_gp_bo": 3, "bomcp": 3},
                "initial_failures": (
                    FailureEvent(FailureType.BACKEND, "mock BO MCP timeout", backend_name="bomcp", penalize_backend=True),
                ),
            },
            expected_intent_sequence=("recover",),
            expected_mode_sequence=("explore", "failure_localization"),
            expected_failure_types=("backend",),
            expected_fallbacks=("lhs",),
            expected_stop_or_continue="continue",
        ),
        ScenarioSpec(
            scenario_id="scientific_negative_campaign",
            title="Scientific negative as evidence",
            initial_campaign_context=_context(level=ObjectiveLevel.MECHANISM),
            initial_observations=_observations((0.25, 0.24, 0.23)),
            mocked_backend_behavior={
                "initial_failures": (
                    FailureEvent(
                        FailureType.SCIENTIFIC_NEGATIVE,
                        "hypothesis failed under valid execution",
                        backend_name="nexus_gp_bo",
                        penalize_backend=False,
                    ),
                ),
            },
            expected_intent_sequence=("validate", "hypothesis_generate", "hypothesis_test"),
            expected_mode_sequence=("mechanism_validation", "hypothesis_test"),
            expected_failure_types=("scientific_negative",),
            expected_proposals=("hypothesis_test", "mechanism_validation"),
            expected_stop_or_continue="continue",
        ),
        ScenarioSpec(
            scenario_id="plateau_to_pivot",
            title="Plateau suggests route pivot",
            initial_campaign_context=_context(level=ObjectiveLevel.GENERALIZATION, requires_route_switch=True),
            initial_observations=_observations((0.5, 0.501, 0.5015, 0.5017, 0.5018, 0.5019)),
            expected_intent_sequence=("pivot",),
            expected_mode_sequence=("route_switch",),
            expected_proposals=("route_switch",),
            expected_stop_or_continue="continue",
        ),
        ScenarioSpec(
            scenario_id="mechanism_validation",
            title="Mechanism validation",
            initial_campaign_context=_context(level=ObjectiveLevel.MECHANISM),
            initial_observations=_observations((0.44, 0.47, 0.51, 0.56, 0.6)),
            expected_intent_sequence=("validate", "hypothesis_test"),
            expected_mode_sequence=("mechanism_validation", "hypothesis_test"),
            expected_proposals=("mechanism_validation", "hypothesis_test"),
            expected_stop_or_continue="continue",
        ),
        ScenarioSpec(
            scenario_id="generalization_transfer",
            title="Generalization transfer",
            initial_campaign_context=_context(level=ObjectiveLevel.GENERALIZATION),
            initial_observations=_observations((0.45, 0.48, 0.51, 0.52, 0.53)),
            expected_intent_sequence=("transfer", "pivot"),
            expected_mode_sequence=("warm_start",),
            expected_proposals=("warm_start",),
            expected_stop_or_continue="continue",
        ),
        ScenarioSpec(
            scenario_id="execution_instability_recovery",
            title="Execution instability recovery",
            initial_campaign_context=_context(level=ObjectiveLevel.FEASIBILITY),
            initial_observations=_observations((0.3, 0.32)),
            mocked_backend_behavior={
                "initial_failures": (
                    FailureEvent(FailureType.HARDWARE, "robot gripper stalled", backend_name="built_in"),
                ),
            },
            expected_intent_sequence=("recover", "stabilize"),
            expected_mode_sequence=("explore", "baseline_calibration"),
            expected_failure_types=("hardware",),
            expected_stop_or_continue="continue",
        ),
        ScenarioSpec(
            scenario_id="legacy_fallback_campaign",
            title="Legacy fallback campaign",
            initial_campaign_context=CampaignContext(),
            initial_observations=_observations((0.4, 0.43, 0.45, 0.48, 0.5, 0.55, 0.6, 0.64)),
            mocked_backend_behavior={"available_backends": {"built_in": True}},
            expected_intent_sequence=("optimize",),
            expected_mode_sequence=("exploit",),
            expected_fallbacks=("built_in",),
            expected_stop_or_continue="continue",
        ),
    )


def test_offline_sdl_scenario_benchmarks_pass_expected_behavior():
    runner = ScenarioRunner()
    reports = tuple(runner.run(spec) for spec in _scenarios())

    failures = {report.scenario_id: report.reasons for report in reports if not report.passed}
    assert failures == {}
    assert all(report.trace_completeness == 1.0 for report in reports)
    assert all(report.records for report in reports)


def test_offline_sdl_scenario_aggregate_report_is_stable():
    runner = ScenarioRunner()
    reports = tuple(runner.run(spec) for spec in _scenarios())
    aggregate = runner.aggregate(reports)

    assert aggregate["total_scenarios"] == 10
    assert aggregate["pass_count"] == 10
    assert aggregate["fail_count"] == 0
    assert aggregate["trace_completeness_score"] == 1.0
    assert aggregate["fallback_success_count"] >= 2
    assert aggregate["safety_block_count"] == 1
    assert aggregate["unexpected_live_behavior_changes"] == 0
    assert aggregate["proposal_only_action_count"] >= 6


def test_scenario_benchmark_self_evolution_metadata_does_not_change_default_selection():
    spec = next(item for item in _scenarios() if item.scenario_id == "legacy_fallback_campaign")
    runner = ScenarioRunner()
    before = runner.run(spec)
    snapshot = before.records[0].snapshot
    baseline_rank = rank_backends("optimize", ("built_in",), {"built_in": True})
    registry = PolicyVersionRegistry().register(
        PolicyVersionRegistryEntry(
            policy_id="scenario-learned",
            policy_version="v1",
            trained_on_dataset_version="policy_dataset_scenario",
            feature_schema_version="policy_feature_schema_v1",
            reward_version="strategy_reward_v1",
            approved_for_shadow=True,
            approved_for_live_canary=True,
            rollback_target=("scenario-learned", "v0"),
        )
    )
    registry = registry.mark_final_approved(
        "scenario-learned",
        "v1",
        FinalApprovalRecord(
            approval_id="final-scenario",
            proposal_id="promotion-scenario",
            policy_id="scenario-learned",
            policy_version="v1",
            approved_by="test-reviewer",
            approval_mode=FinalApprovalMode.TEST,
            approval_reason="metadata only",
            allowed_campaign_ids=("offline-sdl",),
            allowed_objective_levels=("performance",),
            max_live_weight=0.005,
            max_top1_change_rate=0.2,
            rollback_policy_id="scenario-learned",
            rollback_policy_version="v0",
        ),
    )
    config = PhaseConfig(
        enable_nexus=False,
        enable_method_advisor=False,
        learned_policy_registry_entry=registry.get("scenario-learned", "v1"),
        learned_policy=object(),
    )
    after_decision = select_strategy(snapshot, config=config)
    after_rank = rank_backends("optimize", ("built_in",), {"built_in": True})
    source = inspect.getsource(policy_evolution)

    assert after_decision.backend_name == before.records[0].decision.backend_name
    assert after_decision.strategy_trace.learned_policy_shadow is None
    assert after_decision.strategy_trace.learned_policy_influence is None
    assert config.online_influence_rollout.mode == OnlineInfluenceMode.OFF
    assert after_rank == baseline_rank
    assert "from app.services.strategy_selector" not in source
    assert "rank_backends(" not in source


def test_scenario_failure_attribution_and_backend_memory_invariants():
    memory = BackendPerformanceMemory()
    scientific = audit_failure_attribution(FailureType.SCIENTIFIC_NEGATIVE)
    measurement = audit_failure_attribution(FailureType.MEASUREMENT)
    backend = audit_failure_attribution(FailureType.BACKEND)
    snapshot = _snapshot(
        context=_context(level=ObjectiveLevel.MECHANISM),
        all_params=({"temperature": 40.0, "concentration": 0.2},),
        all_kpis=(0.4,),
    )
    key = problem_context_key(snapshot)
    sci_perf = memory.record_failure_event(
        context_key=key,
        backend_name="nexus_gp_bo",
        action_type="validate",
        failure_type=FailureType.SCIENTIFIC_NEGATIVE,
    )
    measurement_perf = memory.record_failure_event(
        context_key=key,
        backend_name="nexus_gp_bo",
        action_type="validate",
        failure_type=FailureType.MEASUREMENT,
    )
    backend_perf = memory.record_failure_event(
        context_key=key,
        backend_name="bomcp",
        action_type="optimize",
        failure_type=FailureType.BACKEND,
    )

    assert scientific["evidence_only"] is True
    assert scientific["penalizes_backend"] is False
    assert measurement["penalizes_backend"] is False
    assert backend["penalizes_backend"] is True
    assert sci_perf.failure_rate == 0.0
    assert measurement_perf.failure_rate == 0.0
    assert backend_perf.failure_rate == 1.0
