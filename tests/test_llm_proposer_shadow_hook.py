from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from types import SimpleNamespace

from app.optimization.failure_zone_memory import FailureZone
from app.services.llm_candidate_proposer import LLMCandidateProposer
from app.services.llm_gateway import MockProvider

_NOW = datetime(2026, 7, 2, 12, 0, 0, tzinfo=UTC)

_DIMS = [
    {"param_name": "x", "param_type": "number", "min_value": 0.0, "max_value": 1.0},
    {"param_name": "c", "param_type": "categorical", "choices": ["a", "b"]},
]


class _Settings:
    def __init__(self, enabled: bool) -> None:
        self.llm_proposer_shadow_enabled = enabled


def _decision(*, plateau: bool, uncertainty: float):
    status = "plateau" if plateau else "improving"
    return SimpleNamespace(
        diagnostics=SimpleNamespace(
            convergence_status=status, model_uncertainty=uncertainty
        )
    )


def _proposer_with_points():
    provider = MockProvider(
        responses=[json.dumps({"proposals": [{"params": {"x": 0.4, "c": "a"}, "reason": "r"}]})]
    )
    return LLMCandidateProposer(provider=provider)


def _kwargs(**over):
    base = dict(
        campaign_id="camp-1",
        round_index=2,
        dimensions=_DIMS,
        protocol_template={},
        objective_kpi="conductivity",
        direction="maximize",
        strategy_decision=_decision(plateau=True, uncertainty=0.1),
        policy_snapshot={},
        proposer=_proposer_with_points(),
        now=_NOW,
    )
    base.update(over)
    return base


def _no_failure_history(monkeypatch):
    import app.agents.orchestrator as orch

    monkeypatch.setattr(orch, "recall_failure_zones", lambda *a, **k: [])


async def test_disabled_flag_returns_none_and_skips_llm(monkeypatch):
    import app.agents.orchestrator as orch

    monkeypatch.setattr(orch, "get_settings", lambda: _Settings(False))

    def _boom(**_kw):
        raise AssertionError("proposer must not be constructed/used when disabled")

    result = await orch._maybe_record_llm_proposer_shadow(**_kwargs(proposer=None))
    assert result is None


async def test_no_trigger_skips_without_calling_llm(monkeypatch):
    import app.agents.orchestrator as orch

    monkeypatch.setattr(orch, "get_settings", lambda: _Settings(True))
    provider = MockProvider(responses=[])  # would raise if called

    result = await orch._maybe_record_llm_proposer_shadow(
        **_kwargs(
            strategy_decision=_decision(plateau=False, uncertainty=0.1),
            proposer=LLMCandidateProposer(provider=provider),
        )
    )

    assert result is None
    assert provider.call_count == 0


async def test_enabled_and_triggered_records_shadow(monkeypatch, caplog):
    import app.agents.orchestrator as orch

    monkeypatch.setattr(orch, "get_settings", lambda: _Settings(True))
    _no_failure_history(monkeypatch)

    with caplog.at_level(logging.INFO):
        shadow = await orch._maybe_record_llm_proposer_shadow(**_kwargs())

    assert shadow is not None
    assert shadow.validation.accepted_points == [{"x": 0.4, "c": "a"}]
    assert "llm_proposer_shadow" in caplog.text


async def test_real_failure_zone_rejector_binds(monkeypatch):
    import app.agents.orchestrator as orch

    monkeypatch.setattr(orch, "get_settings", lambda: _Settings(True))
    # The proposed point {x:0.4,c:a} matches a recalled historical failure.
    monkeypatch.setattr(
        orch,
        "recall_failure_zones",
        lambda *a, **k: [
            FailureZone(
                params={"x": 0.4, "c": "a"},
                distance=0.0,
                error="boom",
                campaign_id="other",
                round_number=0,
                candidate_index=0,
            )
        ],
    )

    shadow = await orch._maybe_record_llm_proposer_shadow(**_kwargs())

    assert shadow.validation.accepted_points == []
    assert any(
        "failure_zone" in r
        for v in shadow.validation.validations
        for r in v.rejections
    )


async def test_real_safety_rejector_binds(monkeypatch):
    import app.agents.orchestrator as orch

    monkeypatch.setattr(orch, "get_settings", lambda: _Settings(True))
    _no_failure_history(monkeypatch)

    provider = MockProvider(
        responses=[json.dumps({"proposals": [{"params": {"temp_c": 150.0}, "reason": "hot"}]})]
    )
    shadow = await orch._maybe_record_llm_proposer_shadow(
        **_kwargs(
            dimensions=[{"param_name": "temp_c", "param_type": "number", "min_value": 0.0, "max_value": 200.0}],
            policy_snapshot={"max_temp_c": 100.0},
            proposer=LLMCandidateProposer(provider=provider),
        )
    )

    assert shadow.validation.accepted_points == []
    assert any(
        "safety" in r for v in shadow.validation.validations for r in v.rejections
    )


async def test_hook_is_fail_open(monkeypatch, caplog):
    import app.agents.orchestrator as orch

    monkeypatch.setattr(orch, "get_settings", lambda: _Settings(True))
    _no_failure_history(monkeypatch)

    def _boom(*_a, **_k):
        raise RuntimeError("validation exploded")

    monkeypatch.setattr(orch, "validate_proposal", _boom)

    with caplog.at_level(logging.WARNING):
        result = await orch._maybe_record_llm_proposer_shadow(**_kwargs())

    assert result is None
    assert "LLM proposer shadow hook failed" in caplog.text
