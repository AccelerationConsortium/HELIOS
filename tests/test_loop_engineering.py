from __future__ import annotations

from datetime import datetime

import pytest

from app.services.loop_engineering import (
    LoopDecision,
    LoopIterationBuilder,
    LoopOutcome,
    LoopReplayAnalyzer,
    LoopRewardCalculator,
    LoopSignal,
    LoopSpec,
    build_loop_iteration,
    calculate_loop_reward,
    summarize_loop_replay,
)


def _spec() -> LoopSpec:
    return LoopSpec(
        objective_name="delta_e",
        direction="minimize",
        budget=15,
        target_value=8.0,
        allowed_actions=("explore", "exploit", "recover"),
        stop_conditions=("target_reached", "budget_exhausted"),
    )


def _decision(
    *,
    action: str = "exploit",
    backend: str | None = "gp_backend",
) -> LoopDecision:
    return LoopDecision(
        action=action,
        authority="shadow",
        selected_backend=backend,
        candidate_count=3,
        confidence=0.72,
        rationale="best expected improvement under current constraints",
        evidence=[{"kind": "strategy_trace", "weight": 0.7}],
    )


def _iteration(
    *,
    iteration_id: str,
    campaign_id: str = "campaign-1",
    round_index: int = 0,
    action: str = "exploit",
    backend: str | None = "gp_backend",
    execution_success: bool | None = True,
    objective_delta: float | None = 0.5,
    validation_success: bool | None = True,
    recovery_attempted: bool = False,
    recovery_success: bool | None = None,
    failure_count: int = 0,
    safety_incident_count: int = 0,
    critical_signal: bool = False,
):
    signals = [
        LoopSignal(
            name="uncertainty",
            source="optimizer",
            value=0.3,
            severity="critical" if critical_signal else "info",
        )
    ]
    outcome = LoopOutcome(
        execution_success=execution_success,
        objective_delta=objective_delta,
        validation_success=validation_success,
        recovery_attempted=recovery_attempted,
        recovery_success=recovery_success,
        failure_count=failure_count,
        safety_incident_count=safety_incident_count,
        observations={"delta_e": 6.8},
        artifacts=[{"kind": "puda_response", "id": "artifact-1"}],
    )
    return build_loop_iteration(
        iteration_id=iteration_id,
        episode_id="episode-1",
        campaign_id=campaign_id,
        round_index=round_index,
        spec=_spec(),
        signals=signals,
        decision=_decision(action=action, backend=backend),
        outcome=outcome,
    )


def test_iteration_builder_binds_spec_decision_outcome_and_reward():
    iteration = LoopIterationBuilder().build(
        iteration_id="iter-1",
        episode_id="episode-1",
        campaign_id="campaign-1",
        round_index=2,
        spec=_spec(),
        signals=[LoopSignal(name="noise", source="qc", value=0.1)],
        decision=_decision(),
        outcome=LoopOutcome(execution_success=True, objective_delta=0.4),
    )

    assert iteration.iteration_id == "iter-1"
    assert iteration.episode_id == "episode-1"
    assert iteration.campaign_id == "campaign-1"
    assert iteration.round_index == 2
    assert iteration.spec.objective_name == "delta_e"
    assert iteration.decision.action == "exploit"
    assert iteration.reward.iteration_id == "iter-1"
    assert iteration.reward.reward > 0


def test_positive_loop_outcome_scores_positive_reward():
    reward = calculate_loop_reward(
        iteration_id="iter-positive",
        outcome=LoopOutcome(
            execution_success=True,
            objective_delta=0.5,
            validation_success=True,
            recovery_attempted=True,
            recovery_success=True,
        ),
    )

    assert reward.execution_reward == 0.2
    assert reward.objective_reward == 0.15
    assert reward.validation_reward == 0.2
    assert reward.recovery_reward == 0.1
    assert reward.reward == 0.65


def test_failures_and_safety_incidents_penalize_and_clamp_reward():
    reward = LoopRewardCalculator().calculate(
        iteration_id="iter-negative",
        outcome=LoopOutcome(
            execution_success=False,
            objective_delta=-10.0,
            validation_success=False,
            recovery_attempted=True,
            recovery_success=False,
            failure_count=20,
            safety_incident_count=3,
        ),
    )

    assert reward.failure_penalty == -2.0
    assert reward.safety_penalty == -1.5
    assert reward.reward == -1.0
    assert reward.regret == 1.0
    assert reward.metadata["clamped"] is True


def test_empty_replay_summary_is_deterministic():
    summary = LoopReplayAnalyzer().analyze([], replay_id="loop-empty")

    assert summary.replay_id == "loop-empty"
    assert summary.iteration_count == 0
    assert summary.average_reward == 0.0
    assert summary.action_distribution == {}
    assert "No loop iterations" in summary.rationale


def test_replay_summary_aggregates_loop_metrics():
    summary = summarize_loop_replay(
        [
            _iteration(iteration_id="iter-1", round_index=1),
            _iteration(
                iteration_id="iter-2",
                round_index=2,
                action="recover",
                backend="built_in",
                execution_success=False,
                objective_delta=-0.3,
                validation_success=False,
                recovery_attempted=True,
                recovery_success=False,
                failure_count=2,
                safety_incident_count=1,
                critical_signal=True,
            ),
            _iteration(
                iteration_id="iter-3",
                campaign_id="campaign-2",
                round_index=3,
                action="explore",
                backend=None,
                execution_success=None,
                objective_delta=None,
                validation_success=None,
            ),
        ],
        replay_id="loop-replay",
    )

    assert summary.iteration_count == 3
    assert summary.campaign_ids == ["campaign-1", "campaign-2"]
    assert summary.round_range == {"min_round": 1, "max_round": 3}
    assert summary.action_distribution == {
        "exploit": 1,
        "recover": 1,
        "explore": 1,
    }
    assert summary.backend_distribution == {"gp_backend": 1, "built_in": 1}
    assert summary.execution_success_rate == 0.5
    assert summary.validation_success_rate == 0.5
    assert summary.recovery_success_rate == 0.0
    assert summary.failure_count == 2
    assert summary.safety_incident_count == 1
    assert summary.critical_signal_count == 1


def test_json_serialization():
    iteration = _iteration(iteration_id="iter-json")
    summary = summarize_loop_replay([iteration], replay_id="loop-json")

    dumped_iteration = iteration.model_dump(mode="json")
    dumped_summary = summary.model_dump(mode="json")
    iteration_json = iteration.model_dump_json()
    summary_json = summary.model_dump_json()

    assert dumped_iteration["iteration_id"] == "iter-json"
    assert dumped_iteration["created_at"].endswith("Z") or "+" in dumped_iteration["created_at"]
    assert dumped_summary["replay_id"] == "loop-json"
    assert '"iteration_id":"iter-json"' in iteration_json
    assert '"replay_id":"loop-json"' in summary_json


def test_created_at_must_be_timezone_aware():
    with pytest.raises(ValueError):
        _iteration(iteration_id="iter-time").model_copy(
            update={"created_at": datetime(2026, 7, 3)},
        ).model_validate(
            {
                **_iteration(iteration_id="iter-time").model_dump(),
                "created_at": datetime(2026, 7, 3),
            }
        )


def test_inputs_are_not_mutated():
    spec = _spec()
    decision = _decision()
    signal = LoopSignal(name="operator_note", source="telegram", metadata={"nested": {"v": 1}})
    outcome = LoopOutcome(observations={"delta_e": 6.8}, metadata={"nested": {"v": 1}})
    metadata = {"source": {"name": "initial"}}

    iteration = build_loop_iteration(
        iteration_id="iter-immutable",
        campaign_id="campaign-1",
        round_index=0,
        spec=spec,
        signals=[signal],
        decision=decision,
        outcome=outcome,
        metadata=metadata,
    )

    iteration.spec.constraints["new"] = "changed"
    iteration.decision.evidence[0]["weight"] = 0.1
    iteration.signals[0].metadata["nested"]["v"] = 2
    iteration.outcome.metadata["nested"]["v"] = 2
    iteration.metadata["source"]["name"] = "changed"

    assert spec.constraints == {}
    assert decision.evidence == [{"kind": "strategy_trace", "weight": 0.7}]
    assert signal.metadata == {"nested": {"v": 1}}
    assert outcome.metadata == {"nested": {"v": 1}}
    assert metadata == {"source": {"name": "initial"}}


def test_import_smoke():
    import app.services.loop_engineering  # noqa: F401
