"""Tests for the Orchestrator and Planner agents."""
from __future__ import annotations

import asyncio
import os
import tempfile
import unittest.mock as mock

# Isolate test DB before any app imports.
_tmpdir = tempfile.mkdtemp(prefix="otbot_orchestrator_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "orchestrator_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import init_db  # noqa: E402


def _run(coro):
    """Helper to run async code in tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture(autouse=True)
def _setup_db():
    """Initialize a fresh DB for each test."""
    get_settings.cache_clear()
    init_db()


class TestPlannerAgent:
    def test_basic_plan(self):
        from app.agents.planner_agent import PlannerAgent, PlannerInput

        agent = PlannerAgent()
        inp = PlannerInput(
            contract_id="tc-test",
            objective_kpi="overpotential_mv",
            direction="minimize",
            max_rounds=10,
            batch_size=5,
            strategy="lhs",
            dimensions=[
                {"param_name": "ratio", "param_type": "number", "min_value": 0.1, "max_value": 10.0},
            ],
            protocol_template={"steps": []},
        )
        result = _run(agent.run(inp))
        assert result.success
        assert len(result.output.planned_rounds) == 10
        assert result.output.total_planned_runs == 50

    def test_plan_with_history(self):
        from app.agents.planner_agent import PlannerAgent, PlannerInput

        agent = PlannerAgent()
        inp = PlannerInput(
            contract_id="tc-test",
            objective_kpi="overpotential_mv",
            direction="minimize",
            max_rounds=10,
            batch_size=5,
            strategy="lhs",
            dimensions=[
                {"param_name": "ratio", "param_type": "number", "min_value": 0.1, "max_value": 10.0},
            ],
            protocol_template={"steps": []},
            kpi_history=[100, 90, 80],
            completed_rounds=3,
        )
        result = _run(agent.run(inp))
        assert result.success
        assert len(result.output.planned_rounds) == 7  # 10 - 3

    def test_no_rounds_remaining(self):
        from app.agents.planner_agent import PlannerAgent, PlannerInput

        agent = PlannerAgent()
        inp = PlannerInput(
            contract_id="tc-test",
            objective_kpi="overpotential_mv",
            direction="minimize",
            max_rounds=5,
            batch_size=5,
            strategy="lhs",
            dimensions=[
                {"param_name": "ratio", "param_type": "number", "min_value": 0.1, "max_value": 10.0},
            ],
            protocol_template={"steps": []},
            completed_rounds=5,
        )
        result = _run(agent.run(inp))
        assert result.success
        assert len(result.output.planned_rounds) == 0

    def test_validation_error(self):
        from app.agents.planner_agent import PlannerAgent, PlannerInput

        agent = PlannerAgent()
        inp = PlannerInput(
            contract_id="tc-test",
            objective_kpi="",
            direction="minimize",
            max_rounds=10,
            batch_size=5,
            dimensions=[],
            protocol_template={"steps": []},
        )
        result = _run(agent.run(inp))
        assert not result.success


class TestOrchestratorAgent:
    def test_plan_only_mode(self):
        from app.agents.orchestrator import OrchestratorAgent, OrchestratorInput

        agent = OrchestratorAgent()
        inp = OrchestratorInput(
            contract_id="tc-test",
            objective_kpi="overpotential_mv",
            direction="minimize",
            max_rounds=5,
            batch_size=3,
            dimensions=[
                {"param_name": "ratio", "param_type": "number", "min_value": 0.1, "max_value": 10.0},
            ],
            protocol_template={"steps": [
                {"step_key": "s1", "primitive": "robot.home", "params": {}, "depends_on": [], "resources": []},
            ]},
            plan_only=True,
        )
        result = _run(agent.run(inp))
        assert result.success
        assert result.output.status == "planned"
        assert result.output.plan_summary["n_rounds"] == 5

    def test_dry_run_execution(self):
        from app.agents.orchestrator import OrchestratorAgent, OrchestratorInput

        agent = OrchestratorAgent()
        inp = OrchestratorInput(
            contract_id="tc-test",
            objective_kpi="overpotential_mv",
            direction="minimize",
            max_rounds=3,
            batch_size=2,
            dimensions=[
                {"param_name": "ratio", "param_type": "number", "min_value": 0.1, "max_value": 10.0},
            ],
            protocol_template={"steps": [
                {"step_key": "s1", "primitive": "robot.home", "params": {}, "depends_on": [], "resources": []},
            ]},
            policy_snapshot={
                "max_temp_c": 95.0,
                "max_volume_ul": 1000.0,
                "allowed_primitives": ["robot.home"],
            },
            dry_run=True,
        )

        # The DesignAgent calls generate_batch which tries to store in the DB.
        # In tests there's no campaign row for the FK, so we skip the DB store.
        with mock.patch("app.services.candidate_gen._store_batch"):
            result = _run(agent.run(inp))

        assert result.success
        assert result.output.rounds_completed > 0
        assert result.output.best_kpi is not None

    def test_orchestrator_dry_run_includes_sensing_trace(self):
        from app.agents.orchestrator import OrchestratorAgent, OrchestratorInput

        agent = OrchestratorAgent()
        inp = OrchestratorInput(
            contract_id="tc-sense",
            objective_kpi="overpotential_mv",
            direction="minimize",
            max_rounds=2,
            batch_size=2,
            dimensions=[
                {"param_name": "ratio", "param_type": "number", "min_value": 0.1, "max_value": 10.0},
            ],
            protocol_template={"steps": [
                {"step_key": "s1", "primitive": "robot.home", "params": {}, "depends_on": [], "resources": []},
            ]},
            policy_snapshot={
                "max_temp_c": 95.0,
                "max_volume_ul": 1000.0,
                "allowed_primitives": ["robot.home"],
            },
            dry_run=True,
        )

        with mock.patch("app.services.candidate_gen._store_batch"):
            result = _run(agent.run(inp))

        assert result.success
        sensing_entries = [
            entry for entry in result.output.agent_trace
            if entry.get("agent") == "sensing_agent"
        ]
        assert len(sensing_entries) > 0
        # Each sensing entry should have quality and recommendation fields
        for entry in sensing_entries:
            assert "quality" in entry
            assert "recommendation" in entry

    def test_validation_fails(self):
        from app.agents.orchestrator import OrchestratorAgent, OrchestratorInput

        agent = OrchestratorAgent()
        inp = OrchestratorInput(
            contract_id="",
            objective_kpi="",
            direction="minimize",
            max_rounds=0,
            batch_size=1,
            dimensions=[],
            protocol_template={"steps": []},
        )
        result = _run(agent.run(inp))
        assert not result.success
