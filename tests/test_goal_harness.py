from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.services.goal_harness import (
    GoalHarness,
    GoalHarnessState,
    ObservationEnvelope,
    ToolDescriptor,
    advance_goal,
)


def _state(**overrides):
    values = {
        "goal_id": "goal-test",
        "objective": "minimize overpotential",
        "budget_remaining": 3,
    }
    values.update(overrides)
    return GoalHarnessState(**values)


def _observation(**overrides):
    values = {
        "goal_id": "goal-test",
        "source": "puda",
        "signal_type": "measurement",
        "content": {"delta": -0.1},
        "severity": "info",
        "confidence": 0.8,
    }
    values.update(overrides)
    return ObservationEnvelope(**values)


def test_advance_proposes_tool_action_without_executing():
    state = _state()
    tool = ToolDescriptor(
        tool_name="puda_executor",
        capability="execute compiled PUDA command",
        risk="medium",
        required_permissions=("live_hardware",),
    )

    result = GoalHarness().advance(
        state,
        observations=[_observation()],
        tools=[tool],
    )

    assert result.state.status == "active"
    assert result.decision.action is not None
    assert result.decision.action.tool_name == "puda_executor"
    assert result.decision.action.intent == "execute"
    assert result.decision.action.status == "proposed"
    assert result.decision.action.required_permissions == ("live_hardware",)
    assert result.state.budget_remaining == 3
    assert (
        result.state.proposed_actions[0].action_id
        == result.decision.action.action_id
    )
    assert result.state.reflection_notes[-1].kind == "decision"


def test_budget_exhaustion_completes_goal():
    result = GoalHarness().advance(_state(budget_remaining=0))

    assert result.state.status == "completed"
    assert result.decision.should_stop is True
    assert result.decision.action is None
    assert result.decision.notes[0].kind == "completion"


def test_critical_observation_requests_human_and_blocks_autonomy():
    observation = _observation(
        signal_type="safety",
        severity="critical",
        content={"message": "unexpected pressure spike"},
    )

    result = GoalHarness().advance(_state(), observations=[observation])

    assert result.state.status == "waiting_human"
    assert result.decision.requires_human is True
    assert result.decision.action is not None
    assert result.decision.action.intent == "request_human"
    assert result.decision.action.required_permissions == ("human_review",)
    assert result.state.blockers == ["Critical observation blocked autonomous progress."]


def test_failure_observation_kills_bad_path():
    observation = _observation(
        signal_type="path_failure",
        severity="warning",
        content={
            "path_id": "route-high-temp",
            "reason": "repeated QC failure",
            "recommendation": "kill_path",
        },
    )
    tool = ToolDescriptor(tool_name="planner_agent", capability="plan next experiment")

    result = GoalHarness().advance(_state(), observations=[observation], tools=[tool])

    assert result.state.killed_paths == ["route-high-temp"]
    assert result.decision.killed_paths == ("route-high-temp",)
    path_note = result.decision.notes[0]
    assert path_note.kind == "path_kill"
    assert path_note.applies_to == "route-high-temp"
    assert result.decision.action is not None
    assert result.decision.action.params["killed_paths"] == ["route-high-temp"]


def test_missing_tool_registry_waits_for_external_registration():
    result = GoalHarness().advance(_state(), observations=[_observation()])

    assert result.state.status == "waiting_external"
    assert result.decision.action is not None
    assert result.decision.action.tool_name == "tool_registry"
    assert result.decision.action.intent == "observe"
    assert result.state.blockers == ["Waiting for additional observation before execution."]


def test_observation_goal_mismatch_is_rejected():
    with pytest.raises(ValueError, match="belongs to goal-other"):
        GoalHarness().advance(_state(), observations=[_observation(goal_id="goal-other")])


def test_inputs_are_deep_copied_not_mutated():
    state = _state()
    observation = _observation(content={"nested": {"value": 1}})
    tool = ToolDescriptor(tool_name="planner_agent", capability="plan next experiment")

    result = GoalHarness().advance(state, observations=[observation], tools=[tool])
    result.state.observations[0].content["nested"]["value"] = 9
    result.state.tool_registry[0].metadata["changed"] = True

    assert state.observations == []
    assert state.tool_registry == []
    assert observation.content["nested"]["value"] == 1
    assert tool.metadata == {}


def test_terminal_state_is_noop():
    state = _state(status="aborted")

    result = GoalHarness().advance(state, observations=[_observation()])

    assert result.state.status == "aborted"
    assert result.decision.should_stop is True
    assert result.decision.action is None


def test_timestamp_must_be_timezone_aware():
    with pytest.raises(ValueError, match="timezone-aware"):
        ObservationEnvelope(
            goal_id="goal-test",
            source="log",
            signal_type="message",
            received_at=datetime(2026, 1, 1),
        )


def test_convenience_wrapper():
    result = advance_goal(
        state=_state(),
        tools=[ToolDescriptor(tool_name="sensor", capability="observe sensor")],
    )

    assert result.decision.action is not None
    assert result.decision.action.intent == "observe"
    assert result.state.status == "waiting_external"


def test_timezone_aware_timestamp_is_accepted():
    observation = _observation(received_at=datetime(2026, 1, 1, tzinfo=UTC))

    assert observation.received_at.tzinfo is not None
