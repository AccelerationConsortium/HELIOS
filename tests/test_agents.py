"""Tests for the Agent system."""
from __future__ import annotations

import asyncio

import pytest

from app.agents.base import AgentResult, BaseAgent
from app.agents.compiler_agent import CompilerAgent, CompileInput
from app.agents.safety_agent import SafetyAgent, SafetyCheckInput
from app.agents.stop_agent import StopAgent, StopInput


def _run(coro):
    """Helper to run async code in tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# SafetyAgent
# ---------------------------------------------------------------------------


class TestSafetyAgent:
    def test_allows_safe_protocol(self):
        agent = SafetyAgent()
        input_data = SafetyCheckInput(
            compiled_graph={
                "steps": [
                    {"step_key": "s1", "primitive": "robot.home", "params": {}},
                ]
            },
            policy_snapshot={
                "max_temp_c": 95.0,
                "max_volume_ul": 1000.0,
                "allowed_primitives": ["robot.home"],
            },
        )
        result = _run(agent.run(input_data))
        assert result.success
        assert result.output is not None
        assert result.output.allowed
        assert result.output.safety_score == 1.0

    def test_blocks_disallowed_primitive(self):
        agent = SafetyAgent()
        input_data = SafetyCheckInput(
            compiled_graph={
                "steps": [
                    {"step_key": "s1", "primitive": "danger_op", "params": {}},
                ]
            },
            policy_snapshot={
                "allowed_primitives": ["robot.home"],
            },
        )
        result = _run(agent.run(input_data))
        assert result.success  # agent ran successfully
        assert result.output is not None
        assert not result.output.allowed  # but safety check failed
        assert len(result.output.violations) > 0

    def test_rejects_missing_policy(self):
        agent = SafetyAgent()
        input_data = SafetyCheckInput(
            compiled_graph={"steps": []},
            policy_snapshot={},
        )
        result = _run(agent.run(input_data))
        assert not result.success  # validation error
        assert "policy_snapshot is required" in result.errors

    def test_result_includes_trace_id(self):
        agent = SafetyAgent()
        input_data = SafetyCheckInput(
            compiled_graph={"steps": []},
            policy_snapshot={"max_temp_c": 95.0},
        )
        result = _run(agent.run(input_data, trace_id="test-trace-123"))
        assert result.trace_id == "test-trace-123"
        assert result.agent_name == "safety_agent"

    def test_duration_is_recorded(self):
        agent = SafetyAgent()
        input_data = SafetyCheckInput(
            compiled_graph={
                "steps": [
                    {"step_key": "s1", "primitive": "robot.home", "params": {}},
                ]
            },
            policy_snapshot={"allowed_primitives": ["robot.home"]},
        )
        result = _run(agent.run(input_data))
        assert result.duration_ms >= 0.0


# ---------------------------------------------------------------------------
# CompilerAgent
# ---------------------------------------------------------------------------


class TestCompilerAgent:
    def test_compiles_simple_protocol(self):
        agent = CompilerAgent()
        input_data = CompileInput(
            protocol={
                "steps": [
                    {"step_key": "s1", "primitive": "robot.home", "params": {}, "depends_on": [], "resources": []},
                    {"step_key": "s2", "primitive": "heat", "params": {"temp_c": 50}, "depends_on": ["s1"], "resources": []},
                ]
            },
        )
        result = _run(agent.run(input_data))
        assert result.success
        assert result.output is not None
        assert result.output.n_steps == 2
        assert "s1" in result.output.step_keys
        assert "s2" in result.output.step_keys
        assert len(result.output.graph_hash) == 64  # sha256 hex

    def test_rejects_missing_steps(self):
        agent = CompilerAgent()
        input_data = CompileInput(protocol={})
        result = _run(agent.run(input_data))
        assert not result.success
        assert "protocol must contain 'steps'" in result.errors

    def test_single_step(self):
        agent = CompilerAgent()
        input_data = CompileInput(
            protocol={
                "steps": [
                    {"step_key": "only", "primitive": "log", "params": {"msg": "hi"}, "depends_on": [], "resources": []},
                ]
            },
        )
        result = _run(agent.run(input_data))
        assert result.success
        assert result.output is not None
        assert result.output.n_steps == 1
        assert result.output.step_keys == ["only"]

    def test_compiler_agent_includes_deck_plan(self):
        agent = CompilerAgent()
        input_data = CompileInput(
            protocol={
                "steps": [
                    {"step_key": "s1", "primitive": "robot.aspirate", "params": {"volume_ul": 100}, "depends_on": [], "resources": []},
                    {"step_key": "s2", "primitive": "robot.dispense", "params": {"volume_ul": 100}, "depends_on": ["s1"], "resources": []},
                ]
            },
        )
        result = _run(agent.run(input_data))
        assert result.success
        assert result.output is not None
        assert isinstance(result.output.deck_plan, dict)
        assert len(result.output.deck_plan) > 0
        assert isinstance(result.output.layout_warnings, list)

    def test_compiler_agent_deck_plan_has_slots(self):
        agent = CompilerAgent()
        input_data = CompileInput(
            protocol={
                "steps": [
                    {"step_key": "s1", "primitive": "robot.aspirate", "params": {"volume_ul": 100}, "depends_on": [], "resources": []},
                    {"step_key": "s2", "primitive": "robot.dispense", "params": {"volume_ul": 100}, "depends_on": ["s1"], "resources": []},
                ]
            },
        )
        result = _run(agent.run(input_data))
        assert result.success
        assert result.output is not None
        deck_plan = result.output.deck_plan
        assert "slots" in deck_plan
        assert "pipette_right" in deck_plan


# ---------------------------------------------------------------------------
# StopAgent
# ---------------------------------------------------------------------------


class TestStopAgent:
    def test_continue_early_campaign(self):
        agent = StopAgent()
        input_data = StopInput(
            kpi_history=[100, 90, 80],
            current_round=3,
            max_rounds=20,
            direction="minimize",
        )
        result = _run(agent.run(input_data))
        assert result.success
        assert result.output is not None
        assert result.output.decision == "continue"

    def test_stop_budget(self):
        agent = StopAgent()
        input_data = StopInput(
            kpi_history=[100, 90, 80],
            current_round=20,
            max_rounds=20,
            direction="minimize",
        )
        result = _run(agent.run(input_data))
        assert result.success
        assert result.output is not None
        assert result.output.decision == "stop_budget"

    def test_stop_target_reached(self):
        agent = StopAgent()
        input_data = StopInput(
            kpi_history=[100, 50, 10],
            current_round=3,
            max_rounds=20,
            target_value=15.0,
            direction="minimize",
        )
        result = _run(agent.run(input_data))
        assert result.success
        assert result.output is not None
        assert result.output.decision == "stop_target"

    def test_best_kpi_tracked(self):
        agent = StopAgent()
        input_data = StopInput(
            kpi_history=[100, 50, 75],
            current_round=3,
            max_rounds=20,
            direction="minimize",
        )
        result = _run(agent.run(input_data))
        assert result.output is not None
        assert result.output.best_kpi == 50

    def test_stop_budget_via_run_limit(self):
        agent = StopAgent()
        input_data = StopInput(
            kpi_history=[100, 90, 80],
            current_round=3,
            max_rounds=20,
            budget_limit_runs=3,
            total_runs_so_far=3,
            direction="minimize",
        )
        result = _run(agent.run(input_data))
        assert result.success
        assert result.output is not None
        assert result.output.decision == "stop_budget"

    def test_maximize_target(self):
        agent = StopAgent()
        input_data = StopInput(
            kpi_history=[10, 50, 100],
            current_round=3,
            max_rounds=20,
            target_value=95.0,
            direction="maximize",
        )
        result = _run(agent.run(input_data))
        assert result.success
        assert result.output is not None
        assert result.output.decision == "stop_target"
        assert result.output.best_kpi == 100

    def test_rejects_invalid_max_rounds(self):
        agent = StopAgent()
        input_data = StopInput(
            kpi_history=[100],
            current_round=1,
            max_rounds=0,
        )
        result = _run(agent.run(input_data))
        assert not result.success
        assert "max_rounds must be >= 1" in result.errors
