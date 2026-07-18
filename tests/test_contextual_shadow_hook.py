from __future__ import annotations

import logging


class _Settings:
    def __init__(
        self,
        enabled: bool,
        ledger_enabled: bool = False,
        authority_enabled: bool = False,
    ) -> None:
        self.contextual_decision_shadow_enabled = enabled
        self.scientific_ledger_enabled = ledger_enabled
        self.campaign_decision_authority_enabled = authority_enabled


def test_contextual_shadow_hook_disabled_does_not_call_services(monkeypatch):
    import app.agents.orchestrator as orch

    monkeypatch.setattr(orch, "get_settings", lambda: _Settings(False))

    def _boom(**_kwargs):
        raise AssertionError("contextual builder should not be called")

    monkeypatch.setattr(orch, "build_campaign_round_context", _boom)

    trace = orch._maybe_record_contextual_shadow_decision(
        campaign_id="campaign-1",
        round_index=1,
        strategy_selection_result={"backend": "bo_mcp"},
    )

    assert trace is None


def test_contextual_shadow_hook_enabled_builds_and_logs_trace(monkeypatch, caplog):
    import app.agents.orchestrator as orch

    monkeypatch.setattr(orch, "get_settings", lambda: _Settings(True))

    with caplog.at_level(logging.INFO):
        trace = orch._maybe_record_contextual_shadow_decision(
            campaign_id="campaign-1",
            round_index=1,
            strategy_selection_result={
                "campaign_intent": "optimize",
                "optimization_mode": "exploit",
                "candidate_generation_backend": "bo_mcp",
            },
            actual_stage="candidate_generation",
        )

    assert trace is not None
    assert trace.context.campaign_id == "campaign-1"
    assert trace.decision_plan.candidate_generation_backend == "bo_mcp"
    assert trace.actual_action == "propose_candidates"
    assert "contextual_shadow_decision_trace" in caplog.text


def test_scientific_ledger_builds_trace_without_legacy_shadow_log(monkeypatch, caplog):
    import app.agents.orchestrator as orch

    monkeypatch.setattr(orch, "get_settings", lambda: _Settings(False, True))

    with caplog.at_level(logging.INFO):
        trace = orch._maybe_record_contextual_shadow_decision(
            campaign_id="campaign-ledger",
            round_index=2,
            strategy_selection_result={
                "campaign_intent": "optimize",
                "optimization_mode": "explore",
                "candidate_generation_backend": "random",
            },
        )

    assert trace is not None
    assert trace.context.campaign_id == "campaign-ledger"
    assert trace.context.round_index == 2
    assert "contextual_shadow_decision_trace" not in caplog.text


def test_live_authority_builds_trace_without_shadow_or_ledger(monkeypatch, caplog):
    import app.agents.orchestrator as orch

    monkeypatch.setattr(
        orch, "get_settings", lambda: _Settings(False, False, True)
    )

    with caplog.at_level(logging.INFO):
        trace = orch._maybe_record_contextual_shadow_decision(
            campaign_id="campaign-authority",
            round_index=3,
            strategy_selection_result={
                "campaign_intent": "optimize",
                "optimization_mode": "exploit",
                "candidate_generation_backend": "bo_mcp",
            },
        )

    assert trace is not None
    assert trace.context.campaign_id == "campaign-authority"
    assert trace.context.round_index == 3
    assert "contextual_shadow_decision_trace" not in caplog.text


def test_recovery_event_projection_deduplicates_episode_ids():
    import app.agents.orchestrator as orch

    episode = {
        "episode_id": "recovery-1",
        "phase": "exit",
        "attempts": [{"action": "retry_original", "result": "success"}],
    }
    events = orch._recovery_events_from_steps(
        [
            {"step_key": "aspirate", "recovery_episode": episode},
            {"step_key": "aspirate", "recovery_episode": episode},
            {"step_key": "measure"},
        ]
    )

    assert events == [episode]
    assert events[0] is not episode


def test_planned_strategy_decision_keeps_first_round_explainable():
    import app.agents.orchestrator as orch

    decision = orch._planned_strategy_decision("lhs")

    assert decision["campaign_intent"] == "optimize"
    assert decision["optimization_mode"] == "explore"
    assert decision["candidate_generation_backend"] == "lhs"
    assert decision["strategy_trace"]["selected_backend"] == "lhs"


def test_contextual_shadow_hook_swallows_and_logs_exceptions(monkeypatch, caplog):
    import app.agents.orchestrator as orch

    monkeypatch.setattr(orch, "get_settings", lambda: _Settings(True))

    def _boom(**_kwargs):
        raise RuntimeError("shadow failed")

    monkeypatch.setattr(orch, "build_campaign_round_context", _boom)

    with caplog.at_level(logging.WARNING):
        trace = orch._maybe_record_contextual_shadow_decision(
            campaign_id="campaign-1",
            round_index=1,
            strategy_selection_result={"backend": "bo_mcp"},
        )

    assert trace is None
    assert "Contextual shadow decision hook failed" in caplog.text


def test_contextual_shadow_hook_does_not_modify_strategy_result(monkeypatch):
    import app.agents.orchestrator as orch

    monkeypatch.setattr(orch, "get_settings", lambda: _Settings(True))
    strategy_result = {
        "candidate_generation_backend": "bo_mcp",
        "trace": {"selected_backend": "bo_mcp"},
    }

    trace = orch._maybe_record_contextual_shadow_decision(
        campaign_id="campaign-1",
        round_index=1,
        strategy_selection_result=strategy_result,
    )
    trace.context.strategy_selection_result["trace"]["selected_backend"] = "changed"

    assert strategy_result == {
        "candidate_generation_backend": "bo_mcp",
        "trace": {"selected_backend": "bo_mcp"},
    }


def test_contextual_shadow_flag_env_default_and_enabled(monkeypatch, request):
    from app.core.config import get_settings

    monkeypatch.delenv("CONTEXTUAL_DECISION_SHADOW_ENABLED", raising=False)
    get_settings.cache_clear()
    request.addfinalizer(get_settings.cache_clear)
    assert get_settings().contextual_decision_shadow_enabled is False

    monkeypatch.setenv("CONTEXTUAL_DECISION_SHADOW_ENABLED", "true")
    get_settings.cache_clear()
    assert get_settings().contextual_decision_shadow_enabled is True
