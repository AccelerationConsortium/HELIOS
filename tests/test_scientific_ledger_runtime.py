from __future__ import annotations

from pathlib import Path

import pytest

from app.services.scientific_ledger_runtime import (
    finalize_scientific_decision,
    record_pending_scientific_decision,
    should_capture_decision_trace,
)
from tests.fixtures.scientific_ledger import decision_trace


@pytest.fixture
def runtime_env(monkeypatch, request, tmp_path):
    from app.core.config import get_settings
    from app.core.db import init_db

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "data" / "orchestrator.db"))
    monkeypatch.setenv("OBJECT_STORE_DIR", str(tmp_path / "objects"))
    monkeypatch.setenv("SCIENTIFIC_LEDGER_ROOT", str(tmp_path / "ledger"))
    monkeypatch.setenv("SCIENTIFIC_LEDGER_ENABLED", "true")
    monkeypatch.setenv("SCIENTIFIC_LEDGER_GIT_ENABLED", "false")
    get_settings.cache_clear()
    request.addfinalizer(get_settings.cache_clear)
    init_db()
    return tmp_path


def test_runtime_bridge_records_pending_and_closes_full_accounting(runtime_env):
    from app.services.decision_trajectory import load_trajectories

    trace = decision_trace()
    pending = record_pending_scientific_decision(trace)
    assert pending is not None
    card = Path(pending.campaign_directory) / "rounds/003/decision_003.md"
    assert "status: pending" in card.read_text()

    result = finalize_scientific_decision(
        trace,
        observed_action="propose_candidates",
        observed_backend="bo_mcp",
        candidate_count=4,
        execution_success=True,
        failure_count=1,
        objective_delta=0.2,
        recovery_attempted=True,
        recovery_success=True,
        observations=[{"yield": 0.84}],
        failures=[{"failure_type": "hardware", "root_cause": "pipette offset"}],
        recovery_events=[{"fix": "increase z offset 0.5mm", "result": "pass"}],
    )
    assert result.trajectory_id.startswith("traj-")
    assert result.accounting.reward.recovery_reward == 0.1
    assert result.ledger_result is not None
    assert "status: completed" in card.read_text()
    rows = load_trajectories("campaign-32")
    assert len(rows) == 1
    assert rows[0]["trajectory"]["outcome"]["recovery_success"] is True


def test_runtime_bridge_persists_typed_accounting_when_markdown_disabled(
    runtime_env, monkeypatch
):
    from app.core.config import get_settings
    from app.services.decision_trajectory import load_trajectories

    monkeypatch.setenv("SCIENTIFIC_LEDGER_ENABLED", "false")
    get_settings.cache_clear()
    trace = decision_trace(campaign_id="typed-only", trace_id="typed-only-trace")
    result = finalize_scientific_decision(trace, execution_success=False)
    assert result.ledger_result is None
    assert len(load_trajectories("typed-only")) == 1


def test_runtime_finalize_aligns_trace_with_observed_action(runtime_env):
    trace = decision_trace()
    result = finalize_scientific_decision(
        trace,
        observed_action="recover_failure",
        execution_success=False,
        recovery_attempted=True,
        recovery_success=False,
    )

    stored_trace = result.accounting.trace
    assert stored_trace.actual_action == "recover_failure"
    assert stored_trace.comparison["actual_action"] == "recover_failure"
    assert stored_trace.would_change_route is True
    assert result.accounting.outcome.observed_action == "recover_failure"


def test_trace_capture_gate_includes_markdown(runtime_env, monkeypatch):
    from app.core.config import get_settings

    assert should_capture_decision_trace() is True
    monkeypatch.setenv("SCIENTIFIC_LEDGER_ENABLED", "false")
    monkeypatch.setenv("CONTEXTUAL_DECISION_SHADOW_ENABLED", "false")
    get_settings.cache_clear()
    assert should_capture_decision_trace() is False


def test_trace_capture_gate_includes_live_authority(runtime_env, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setenv("SCIENTIFIC_LEDGER_ENABLED", "false")
    monkeypatch.setenv("CONTEXTUAL_DECISION_SHADOW_ENABLED", "false")
    monkeypatch.setenv("CAMPAIGN_DECISION_AUTHORITY_ENABLED", "true")
    get_settings.cache_clear()

    assert should_capture_decision_trace() is True
