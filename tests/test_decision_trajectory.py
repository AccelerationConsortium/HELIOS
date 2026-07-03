"""T4: append-only trajectory persistence + JSONL export."""
from __future__ import annotations

import json

import pytest


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


def _accounting(campaign_id="camp-t4", trace_id="trace-t4", **outcome_kw):
    from app.services.decision_layer import CampaignDecisionLayer
    from app.services.decision_outcome import (
        CampaignDecisionAccountingBuilder,
        CampaignDecisionOutcomeBuilder,
    )
    from app.services.decision_trace import CampaignDecisionTraceBuilder
    from app.services.round_context import CampaignRoundContextBuilder

    context = CampaignRoundContextBuilder().build(
        campaign_id=campaign_id,
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
        trace_id=trace_id,
        context=context,
        decision_plan=plan,
        actual_action="propose_candidates",
    )
    outcome = CampaignDecisionOutcomeBuilder().build(
        trace=trace,
        execution_success=True,
        objective_delta=0.5,
        **outcome_kw,
    )
    return CampaignDecisionAccountingBuilder().build(trace=trace, outcome=outcome)


def test_persist_then_load_roundtrip(db_env):
    from app.services.decision_trajectory import (
        TRAJECTORY_SCHEMA_VERSION,
        load_trajectories,
        persist_campaign_trajectory,
    )

    acc = _accounting()
    row_id = persist_campaign_trajectory(acc)
    assert row_id.startswith("traj-")

    rows = load_trajectories("camp-t4")
    assert len(rows) == 1
    row = rows[0]
    assert row["campaign_id"] == "camp-t4"
    assert row["trace_id"] == "trace-t4"
    assert row["layer"] == "campaign"
    assert row["trajectory_schema_version"] == TRAJECTORY_SCHEMA_VERSION
    assert row["reward"] == acc.reward.reward
    # per-signal verifier report is preserved
    names = {v["name"] for v in row["verifier_report"]}
    assert "execution" in names and "objective" in names
    # full replayable unit is stored
    assert row["trajectory"]["outcome"]["campaign_id"] == "camp-t4"


def test_append_only_accumulates(db_env):
    from app.services.decision_trajectory import (
        load_trajectories,
        persist_campaign_trajectory,
    )

    persist_campaign_trajectory(_accounting(trace_id="a"))
    persist_campaign_trajectory(_accounting(trace_id="b"))
    rows = load_trajectories("camp-t4")
    assert len(rows) == 2  # nothing overwritten
    assert {r["trace_id"] for r in rows} == {"a", "b"}


def test_export_jsonl_is_one_object_per_line(db_env):
    from app.services.decision_trajectory import (
        export_trajectories_jsonl,
        persist_campaign_trajectory,
    )

    persist_campaign_trajectory(_accounting(trace_id="a"))
    persist_campaign_trajectory(_accounting(trace_id="b"))
    text = export_trajectories_jsonl("camp-t4")
    lines = text.splitlines()
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)  # each line is valid JSON
        assert obj["campaign_id"] == "camp-t4"
        assert "verifier_report" in obj


def test_export_empty_when_no_rows(db_env):
    from app.services.decision_trajectory import export_trajectories_jsonl

    assert export_trajectories_jsonl("nope") == ""
