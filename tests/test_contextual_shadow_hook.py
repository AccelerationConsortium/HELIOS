from __future__ import annotations

import logging


class _Settings:
    def __init__(self, enabled: bool) -> None:
        self.contextual_decision_shadow_enabled = enabled


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
