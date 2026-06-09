from __future__ import annotations

import pytest


def _mission_payload(**governance_overrides):
    from app.contracts.mission import (
        AutonomyLevel,
        GovernanceEnvelope,
        MissionContract,
        MissionObjective,
    )

    governance_kwargs = {"autonomy": AutonomyLevel(level=3), "dry_run": True}
    governance_kwargs.update(governance_overrides)
    governance = GovernanceEnvelope(**governance_kwargs)
    return MissionContract(
        objective=MissionObjective(
            primary_kpi="overpotential_mv",
            direction="minimize",
            hypothesis="Find a lower-overpotential deposition condition.",
        ),
        governance=governance,
        parameter_space=[
            {
                "param_name": "deposition_time_s",
                "param_type": "number",
                "min_value": 30,
                "max_value": 120,
            }
        ],
        protocol_seed={
            "steps": [
                {
                    "step_key": "log-intent",
                    "primitive": "log",
                    "params": {"message": "mission probe"},
                }
            ]
        },
        inputs={},
    )


@pytest.fixture
def isolated_db(monkeypatch, tmp_path):
    from app.core.config import get_settings
    from app.core.db import init_db

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "data" / "orchestrator.db"))
    monkeypatch.setenv("OBJECT_STORE_DIR", str(tmp_path / "objects"))
    get_settings.cache_clear()
    init_db()


async def test_launch_mission_creates_run_for_approved_protocol(isolated_db):
    from app.services.mission_control import launch_mission
    from app.services.run_service import get_run

    result = await launch_mission(_mission_payload())

    assert result.status == "launched"
    assert len(result.proposals) == 1
    assert result.decisions[0].decision == "approved"
    assert len(result.run_ids) == 1

    run = get_run(result.run_ids[0])
    assert run is not None
    assert run["trigger_type"] == "mission_agent"
    assert run["session_key"] == result.mission_id
    assert run["trigger_payload"]["mission_id"] == result.mission_id
    assert run["trigger_payload"]["proposal"]["proposer"] == "planner_agent"


async def test_launch_mission_escalates_live_hardware_permission(isolated_db):
    from app.services.mission_control import launch_mission

    result = await launch_mission(_mission_payload(dry_run=False))

    assert result.status == "awaiting_human"
    assert result.decisions[0].decision == "needs_human"
    assert "live_hardware" in result.decisions[0].required_permissions
    assert result.run_ids == []
