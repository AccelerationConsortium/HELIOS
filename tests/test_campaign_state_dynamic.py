from __future__ import annotations


def test_campaign_state_checkpoints_dynamic_strategy_state(monkeypatch, request, tmp_path):
    from app.core.config import get_settings
    from app.core.db import init_db
    from app.services.campaign_state import (
        checkpoint_kpi,
        create_campaign,
        load_campaign,
        load_completed_candidates,
    )

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "data" / "orchestrator.db"))
    monkeypatch.setenv("OBJECT_STORE_DIR", str(tmp_path / "objects"))
    get_settings.cache_clear()
    request.addfinalizer(get_settings.cache_clear)
    init_db()

    campaign_context = {
        "scientific_goal": "test objective",
        "current_objective_level": "performance",
        "objective_hierarchy": [
            {"level": "performance", "name": "kpi", "metric": "kpi"}
        ],
    }
    create_campaign(
        "camp-dyn",
        {"objective": "test"},
        direction="maximize",
        campaign_context=campaign_context,
    )
    checkpoint_kpi(
        "camp-dyn",
        kpi_history=[1.0],
        all_kpis=[1.0],
        all_params=[{"x": 0.2}],
        all_rounds=[1],
        best_kpi=1.0,
        total_runs=1,
        backend_failure_counts={"bomcp": 2},
        all_failed_params=[{"x": 0.9}],
        bomcp_backend_state={"dim": 6, "length": 0.8},
        latest_strategy_trace={
            "selected_intent": "optimize",
            "selected_backend": "bomcp",
        },
    )

    loaded = load_campaign("camp-dyn")
    assert loaded is not None
    assert loaded["backend_failure_counts"] == {"bomcp": 2}
    assert loaded["all_failed_params"] == [{"x": 0.9}]
    assert loaded["bomcp_backend_state"] == {"dim": 6, "length": 0.8}
    assert loaded["campaign_context"] == campaign_context
    assert loaded["latest_strategy_trace"]["selected_backend"] == "bomcp"

    restored = load_completed_candidates("camp-dyn")
    assert restored["backend_failure_counts"] == {"bomcp": 2}
    assert restored["all_failed_params"] == [{"x": 0.9}]
    assert restored["bomcp_backend_state"] == {"dim": 6, "length": 0.8}
    assert restored["campaign_context"] == campaign_context
    assert restored["latest_strategy_trace"]["selected_intent"] == "optimize"


def test_campaign_state_appends_typed_failure_events(monkeypatch, request, tmp_path):
    from app.core.config import get_settings
    from app.core.db import init_db
    from app.services.campaign_state import (
        append_failure_event,
        create_campaign,
        load_campaign,
        load_completed_candidates,
    )

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "data" / "orchestrator.db"))
    monkeypatch.setenv("OBJECT_STORE_DIR", str(tmp_path / "objects"))
    get_settings.cache_clear()
    request.addfinalizer(get_settings.cache_clear)
    init_db()

    create_campaign("camp-fail", {"objective": "test"}, direction="maximize")
    append_failure_event(
        "camp-fail",
        {
            "failure_type": "constraint",
            "reason": "safety veto",
            "backend_name": "nexus_gp_bo",
            "penalize_backend": True,
        },
    )

    loaded = load_campaign("camp-fail")
    assert loaded is not None
    assert loaded["failure_events"] == [
        {
            "failure_type": "constraint",
            "reason": "safety veto",
            "backend_name": "nexus_gp_bo",
            "penalize_backend": True,
        }
    ]

    restored = load_completed_candidates("camp-fail")
    assert restored["failure_events"][0]["failure_type"] == "constraint"


def test_campaign_state_persists_strategy_memory_and_space_revisions(
    monkeypatch, request, tmp_path
):
    from app.core.config import get_settings
    from app.core.db import init_db
    from app.services.campaign_state import (
        append_space_revision,
        create_campaign,
        load_campaign,
        load_completed_candidates,
        save_strategy_memory,
    )

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "data" / "orchestrator.db"))
    monkeypatch.setenv("OBJECT_STORE_DIR", str(tmp_path / "objects"))
    get_settings.cache_clear()
    request.addfinalizer(get_settings.cache_clear)
    init_db()

    create_campaign("camp-memory", {"objective": "test"}, direction="maximize")
    append_space_revision(
        "camp-memory",
        {
            "add_constraints": [{"type": "avoid_failed_coordinate"}],
            "reason": "constraint failures",
        },
    )
    save_strategy_memory(
        "camp-memory",
        backend_performance={"ctx|exploit|nexus_gp_bo": {"num_calls": 1}},
        strategy_bandit={"ctx|nexus_gp_bo": {"reward": 1.0, "n": 1.0}},
    )

    loaded = load_campaign("camp-memory")
    assert loaded is not None
    assert loaded["space_revisions"][0]["reason"] == "constraint failures"
    assert loaded["backend_performance"]["ctx|exploit|nexus_gp_bo"]["num_calls"] == 1
    assert loaded["strategy_bandit"]["ctx|nexus_gp_bo"]["reward"] == 1.0

    restored = load_completed_candidates("camp-memory")
    assert restored["space_revisions"][0]["add_constraints"][0]["type"] == (
        "avoid_failed_coordinate"
    )
    assert restored["strategy_bandit"]["ctx|nexus_gp_bo"]["n"] == 1.0


def test_campaign_state_logs_rewards_shadow_records_and_objective_transitions(
    monkeypatch, request, tmp_path
):
    from app.core.config import get_settings
    from app.core.db import init_db
    from app.services.campaign_state import (
        append_objective_transition,
        append_shadow_bandit_record,
        append_strategy_reward,
        create_campaign,
        load_campaign,
        load_completed_candidates,
    )

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "data" / "orchestrator.db"))
    monkeypatch.setenv("OBJECT_STORE_DIR", str(tmp_path / "objects"))
    get_settings.cache_clear()
    request.addfinalizer(get_settings.cache_clear)
    init_db()

    create_campaign("camp-policy", {"objective": "test"}, direction="maximize")
    append_strategy_reward(
        "camp-policy",
        {
            "objective_improvement": 0.2,
            "information_gain": 0.1,
            "constraint_satisfaction": 1.0,
            "data_quality_gain": 0.0,
            "novelty": 0.2,
            "failure_penalty": 0.0,
            "cost_penalty": 0.1,
            "time_penalty": 0.0,
            "composite_reward": 0.3,
            "reward_version": "strategy_reward_v1",
        },
    )
    append_shadow_bandit_record(
        "camp-policy",
        {
            "actual_action": "exploit",
            "actual_backend": "optuna_tpe",
            "suggested_action": "exploit",
            "suggested_backend": "nexus_gp_bo",
            "agrees_with_actual": False,
            "bandit_confidence": 0.7,
            "actual_reward": 0.3,
            "outcome": "completed",
        },
    )
    append_objective_transition(
        "camp-policy",
        {
            "from_level": "baseline",
            "to_level": "performance",
            "reason": "baseline coverage reached",
            "evidence": [],
            "confidence": 0.6,
            "auto_applied": False,
        },
    )

    loaded = load_campaign("camp-policy")
    assert loaded is not None
    assert loaded["strategy_rewards"][0]["reward_version"] == "strategy_reward_v1"
    assert loaded["shadow_bandit_records"][0]["suggested_backend"] == "nexus_gp_bo"
    assert loaded["objective_transitions"][0]["auto_applied"] is False

    restored = load_completed_candidates("camp-policy")
    assert restored["strategy_rewards"][0]["composite_reward"] == 0.3
    assert restored["shadow_bandit_records"][0]["agrees_with_actual"] is False
    assert restored["objective_transitions"][0]["to_level"] == "performance"
