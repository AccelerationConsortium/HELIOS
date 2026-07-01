from __future__ import annotations

import logging

from app.services.action_contracts import SafetyClass
from app.services.primitives_registry import PrimitiveSpec


class _Settings:
    def __init__(
        self,
        *,
        adaptive: bool = False,
        contextual: bool = False,
    ) -> None:
        self.adaptive_substrate_shadow_enabled = adaptive
        self.contextual_decision_shadow_enabled = contextual


class _FakeRegistry:
    def __init__(self) -> None:
        self._specs = [
            PrimitiveSpec(
                name="heat",
                description="",
                error_class="CRITICAL",
                instrument="heater",
                resource_id=None,
                skill_name="skill",
                params=(),
                safety_class=SafetyClass.REVERSIBLE,
                contract=None,
            ),
            PrimitiveSpec(
                name="log",
                description="",
                error_class="BYPASS",
                instrument=None,
                resource_id=None,
                skill_name="skill",
                params=(),
                safety_class=SafetyClass.INFORMATIONAL,
                contract=None,
            ),
        ]

    def list_primitives(self) -> list[PrimitiveSpec]:
        return list(self._specs)

    def get_primitive(self, name: str):
        return next((s for s in self._specs if s.name == name), None)

    def list_instruments(self) -> list[str]:
        return ["heater"]


def _kwargs():
    return dict(
        campaign_id="camp-1",
        round_index=0,
        objective_kpi="conductivity",
        max_rounds=10,
        failure_event_dicts=[
            {"failure_type": "hardware", "primitive": "heat", "error": "temp overshoot exceeded", "step": "s1"}
        ],
        protocol_template={"steps": [{"primitive": "heat"}]},
    )


def test_disabled_flag_records_nothing_and_skips_builder(monkeypatch):
    import app.agents.orchestrator as orch

    monkeypatch.setattr(orch, "get_settings", lambda: _Settings(adaptive=False))

    def _boom(**_kwargs):
        raise AssertionError("builder must not run when flag is off")

    monkeypatch.setattr(orch, "build_adaptive_campaign_substrate_snapshot", _boom)
    monkeypatch.setattr(orch, "get_registry", lambda: (_ for _ in ()).throw(AssertionError("no registry")))

    result = orch._maybe_record_adaptive_campaign_substrate_snapshot(**_kwargs())

    assert result is None


def test_enabled_flag_builds_and_logs_snapshot(monkeypatch, caplog):
    import app.agents.orchestrator as orch

    monkeypatch.setattr(orch, "get_settings", lambda: _Settings(adaptive=True))
    monkeypatch.setattr(orch, "get_registry", _FakeRegistry)

    with caplog.at_level(logging.INFO):
        snapshot = orch._maybe_record_adaptive_campaign_substrate_snapshot(**_kwargs())

    assert snapshot is not None
    assert snapshot.campaign_id == "camp-1"
    assert snapshot.shadow_only is True
    # Instrument failure routed the chain to calibration.
    assert snapshot.campaign_mode_decision.mode.value == "calibration"
    # available_capabilities source recorded (campaign deck subset present).
    assert snapshot.metadata["available_capabilities_source"] == "campaign_deck"
    assert snapshot.metadata["voi_ranking_advisory_only"] is True
    assert "adaptive_campaign_substrate_snapshot" in caplog.text


def test_builder_failure_is_fail_open(monkeypatch, caplog):
    import app.agents.orchestrator as orch

    monkeypatch.setattr(orch, "get_settings", lambda: _Settings(adaptive=True))
    monkeypatch.setattr(orch, "get_registry", _FakeRegistry)

    def _boom(**_kwargs):
        raise RuntimeError("substrate builder failed")

    monkeypatch.setattr(orch, "build_adaptive_campaign_substrate_snapshot", _boom)

    with caplog.at_level(logging.WARNING):
        result = orch._maybe_record_adaptive_campaign_substrate_snapshot(**_kwargs())

    assert result is None
    assert "Adaptive substrate shadow hook failed" in caplog.text


def test_hook_does_not_mutate_inputs(monkeypatch):
    import app.agents.orchestrator as orch

    monkeypatch.setattr(orch, "get_settings", lambda: _Settings(adaptive=True))
    monkeypatch.setattr(orch, "get_registry", _FakeRegistry)

    kwargs = _kwargs()
    events_before = [dict(e) for e in kwargs["failure_event_dicts"]]
    protocol_before = dict(kwargs["protocol_template"])

    orch._maybe_record_adaptive_campaign_substrate_snapshot(**kwargs)

    assert kwargs["failure_event_dicts"] == events_before
    assert kwargs["protocol_template"] == protocol_before


def test_safety_summary_routes_to_constraint_mode(monkeypatch):
    import app.agents.orchestrator as orch

    monkeypatch.setattr(orch, "get_settings", lambda: _Settings(adaptive=True))
    monkeypatch.setattr(orch, "get_registry", _FakeRegistry)

    kwargs = _kwargs()
    kwargs["failure_event_dicts"] = []  # isolate the safety signal
    snapshot = orch._maybe_record_adaptive_campaign_substrate_snapshot(
        safety_summary={"risk_level": "high"}, **kwargs
    )

    assert snapshot is not None
    assert snapshot.campaign_mode_decision.mode.value == "safety_constraint_tightening"


def test_no_safety_summary_keeps_prior_behavior(monkeypatch):
    import app.agents.orchestrator as orch

    monkeypatch.setattr(orch, "get_settings", lambda: _Settings(adaptive=True))
    monkeypatch.setattr(orch, "get_registry", _FakeRegistry)

    kwargs = _kwargs()
    kwargs["failure_event_dicts"] = []
    snapshot = orch._maybe_record_adaptive_campaign_substrate_snapshot(**kwargs)

    assert snapshot is not None
    assert snapshot.campaign_mode_decision.mode.value == "bo_optimization"


def test_both_shadow_tracks_record_in_parallel(monkeypatch, caplog):
    import app.agents.orchestrator as orch

    monkeypatch.setattr(
        orch, "get_settings", lambda: _Settings(adaptive=True, contextual=True)
    )
    monkeypatch.setattr(orch, "get_registry", _FakeRegistry)

    with caplog.at_level(logging.INFO):
        contextual = orch._maybe_record_contextual_shadow_decision(
            campaign_id="camp-1",
            round_index=0,
            strategy_selection_result={"candidate_generation_backend": "bo_mcp"},
            actual_stage="candidate_generation",
        )
        substrate = orch._maybe_record_adaptive_campaign_substrate_snapshot(**_kwargs())

    assert contextual is not None
    assert substrate is not None
    # Both tracks logged distinct, non-overwriting lines.
    assert "contextual_shadow_decision_trace" in caplog.text
    assert "adaptive_campaign_substrate_snapshot" in caplog.text
