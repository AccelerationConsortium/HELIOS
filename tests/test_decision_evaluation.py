"""B3+B4: retrospective audit + gated offline policy evaluation over trajectories."""
from __future__ import annotations

import pytest

from app.services.campaign_mode import CampaignMode


@pytest.fixture
def db_env(monkeypatch, request, tmp_path):
    from app.core.config import get_settings
    from app.core.db import init_db

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "data" / "orchestrator.db"))
    monkeypatch.setenv("OBJECT_STORE_DIR", str(tmp_path / "objects"))
    get_settings.cache_clear()
    request.addfinalizer(get_settings.cache_clear)
    init_db()


def _persist_one(trace_id="t", objective_delta=0.5, safety=0):
    from app.services.decision_layer import CampaignDecisionLayer
    from app.services.decision_outcome import (
        CampaignDecisionAccountingBuilder,
        CampaignDecisionOutcomeBuilder,
    )
    from app.services.decision_trace import CampaignDecisionTraceBuilder
    from app.services.decision_trajectory import persist_campaign_trajectory
    from app.services.round_context import CampaignRoundContextBuilder

    context = CampaignRoundContextBuilder().build(
        campaign_id="camp-b",
        round_index=1,
        strategy_selection_result={
            "campaign_intent": "optimize",
            "optimization_mode": "exploit",
            "candidate_generation_backend": "bo_mcp",
            "confidence": 0.75,
        },
    )
    plan = CampaignDecisionLayer().decide(context)
    trace = CampaignDecisionTraceBuilder().build(
        trace_id=trace_id, context=context, decision_plan=plan,
        actual_action="propose_candidates",
    )
    outcome = CampaignDecisionOutcomeBuilder().build(
        trace=trace, execution_success=True,
        objective_delta=objective_delta, safety_incident_count=safety,
    )
    acc = CampaignDecisionAccountingBuilder().build(trace=trace, outcome=outcome)
    return persist_campaign_trajectory(acc)


# --- B3: retrospective audit --------------------------------------------


def test_retrospective_static_is_stable(db_env):
    from app.services.decision_evaluation import retrospective_audit

    _persist_one(objective_delta=0.5)
    records = retrospective_audit("camp-b")
    assert len(records) == 1
    # same rubric as scored → no divergence
    assert records[0].verdict == "stable"
    assert records[0].delta == 0.0


def test_retrospective_underestimated_under_reweighting(db_env):
    from app.services.decision_evaluation import retrospective_audit
    from app.services.rubric import rubric_for_mode

    # objective delta scored modestly at the time; an optimization-phase rubric
    # (objective x2) values it higher in hindsight → underestimated.
    _persist_one(objective_delta=0.5)
    records = retrospective_audit(
        "camp-b", rubric_for_mode(CampaignMode.BO_OPTIMIZATION)
    )
    assert records[0].retrospective_reward > records[0].immediate_reward
    assert records[0].verdict == "underestimated"


# --- B4: offline policy evaluation (gated) ------------------------------


def test_offline_eval_gated_when_too_few(db_env):
    from app.services.decision_evaluation import offline_policy_evaluation
    from app.services.rubric import STATIC_RUBRIC

    _persist_one(trace_id="only-one")
    result = offline_policy_evaluation([STATIC_RUBRIC], "camp-b")
    assert result.ran is False
    assert result.trajectory_count == 1
    assert "insufficient" in result.reason


def test_offline_eval_runs_past_threshold(db_env):
    from app.services.decision_evaluation import offline_policy_evaluation
    from app.services.rubric import STATIC_RUBRIC, rubric_for_mode

    for i in range(3):
        _persist_one(trace_id=f"t{i}", objective_delta=0.5)
    result = offline_policy_evaluation(
        [STATIC_RUBRIC, rubric_for_mode(CampaignMode.BO_OPTIMIZATION)],
        "camp-b",
        min_trajectories=3,  # lower gate for the test
    )
    assert result.ran is True
    assert result.trajectory_count == 3
    versions = {p.rubric_version for p in result.policies}
    assert "v0.1_static" in versions
    # optimization rubric scores objective-improving decisions higher on average
    by_v = {p.rubric_version: p.mean_reward for p in result.policies}
    opt_v = next(v for v in by_v if v.startswith("v0.2"))
    assert by_v[opt_v] > by_v["v0.1_static"]
