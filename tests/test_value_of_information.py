from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from app.services.campaign_mode import CampaignMode, CampaignModeDecision
from app.services.dynamic_action_space import ActionSpec, build_action_space_snapshot
from app.services.objective_models import ProxyGapAssessment, ProxyGapLevel
from app.services.objective_state import ObjectiveState
from app.services.value_of_information import (
    ActionValueSignals,
    ValueOfInformationScorer,
    score_value_of_information,
)

_NOW = datetime(2026, 6, 29, 12, 0, 0, tzinfo=UTC)


def _mode(mode: CampaignMode = CampaignMode.BO_OPTIMIZATION, *, rank: int = 7):
    return CampaignModeDecision(
        campaign_id="camp-1", round_index=0, mode=mode, priority_rank=rank, reason="t"
    )


def _snapshot(actions, *, available=("robot",), mode=None):
    return build_action_space_snapshot(
        mode_decision=mode or _mode(),
        actions=actions,
        available_capabilities=list(available),
        now=_NOW,
    )


def test_decision_value_combines_terms_with_default_weights():
    snapshot = _snapshot(
        [ActionSpec(name="a", kind="experiment", base_risk=0.3, cost=2.0, latency=10.0)]
    )
    signals = [
        ActionValueSignals(
            name="a",
            immediate_reward=0.5,
            expected_information_gain=0.4,
            expected_hypothesis_resolution=0.3,
            expected_proxy_gap_reduction=0.2,
            expected_future_option_value=0.1,
            reversibility=0.8,
        )
    ]

    result = score_value_of_information(
        action_space_snapshot=snapshot, value_signals=signals, now=_NOW
    )
    score = result.scores[0]

    # Single action => cost/latency normalize to 1.0; assessment risk = 0.3.
    # 1.0*0.5 + 0.8*0.2 + 0.6*0.4 + 0.6*0.3 + 0.5*0.1 - 0.4*1 - 0.7*0.3 - 0.3*1
    assert score.expected_failure_risk == 0.3
    assert score.cost_normalized == 1.0
    assert score.latency_normalized == 1.0
    assert score.decision_value == pytest.approx(0.22)


def test_ranking_orders_by_decision_value_desc():
    snapshot = _snapshot(
        [
            ActionSpec(name="low", kind="experiment", base_risk=0.6),
            ActionSpec(name="high", kind="experiment", base_risk=0.1),
        ]
    )
    signals = [
        ActionValueSignals(name="low", immediate_reward=0.1),
        ActionValueSignals(name="high", immediate_reward=0.9),
    ]

    result = score_value_of_information(
        action_space_snapshot=snapshot, value_signals=signals, now=_NOW
    )

    assert result.ranking[0] == "high"
    assert result.ranking == sorted(
        result.ranking,
        key=lambda name: -next(s.decision_value for s in result.scores if s.name == name),
    )


def test_missing_value_signals_default_to_zero():
    snapshot = _snapshot([ActionSpec(name="a", kind="experiment", base_risk=0.3)])

    result = score_value_of_information(
        action_space_snapshot=snapshot, value_signals=[], now=_NOW
    )
    score = result.scores[0]

    assert score.immediate_reward == 0.0
    assert score.expected_information_gain == 0.0
    # Pure cost/risk/latency penalty => non-positive decision value.
    assert score.decision_value <= 0.0


def test_cost_and_latency_normalized_within_batch():
    snapshot = _snapshot(
        [
            ActionSpec(name="cheap", kind="experiment", cost=1.0, latency=5.0),
            ActionSpec(name="pricey", kind="experiment", cost=4.0, latency=20.0),
        ]
    )

    result = score_value_of_information(
        action_space_snapshot=snapshot, value_signals=[], now=_NOW
    )
    by_name = {s.name: s for s in result.scores}

    assert by_name["pricey"].cost_normalized == 1.0
    assert by_name["cheap"].cost_normalized == pytest.approx(0.25)
    assert by_name["pricey"].latency_normalized == 1.0
    assert by_name["cheap"].latency_normalized == pytest.approx(0.25)


def test_higher_risk_lowers_decision_value():
    snapshot = _snapshot(
        [
            ActionSpec(name="safe", kind="experiment", base_risk=0.1),
            ActionSpec(name="risky", kind="experiment", base_risk=0.9),
        ]
    )
    signals = [
        ActionValueSignals(name="safe", immediate_reward=0.5),
        ActionValueSignals(name="risky", immediate_reward=0.5),
    ]

    result = score_value_of_information(
        action_space_snapshot=snapshot, value_signals=signals, now=_NOW
    )
    by_name = {s.name: s for s in result.scores}

    assert by_name["safe"].decision_value > by_name["risky"].decision_value


def test_proxy_gap_baseline_recorded_from_objective_state():
    snapshot = _snapshot([ActionSpec(name="a", kind="experiment")])
    objective = ObjectiveState(
        campaign_id="camp-1",
        primary_objective="x",
        proxy_gap=ProxyGapAssessment(
            score=0.7,
            level=ProxyGapLevel.HIGH,
            active_metric_names=["raw_peak_area"],
            rationale="distant proxy",
        ),
    )

    result = score_value_of_information(
        action_space_snapshot=snapshot,
        value_signals=[],
        objective_state=objective,
        now=_NOW,
    )

    assert result.metadata["proxy_gap_baseline"] == 0.7


def test_every_score_has_reason_and_evidence():
    snapshot = _snapshot([ActionSpec(name="a", kind="experiment")])

    result = score_value_of_information(
        action_space_snapshot=snapshot, value_signals=[], now=_NOW
    )

    for score in result.scores:
        assert score.reason
        assert score.evidence


def test_shadow_only_and_json_safe_and_deterministic():
    snapshot = _snapshot(
        [
            ActionSpec(name="a", kind="experiment", base_risk=0.2, cost=1.0, latency=5.0),
            ActionSpec(name="b", kind="experiment", base_risk=0.5),
        ]
    )
    signals = [ActionValueSignals(name="a", immediate_reward=0.4)]

    first = score_value_of_information(
        action_space_snapshot=snapshot, value_signals=signals, now=_NOW
    )
    second = ValueOfInformationScorer().score(
        action_space_snapshot=snapshot, value_signals=signals, now=_NOW
    )

    assert first.shadow_only is True
    assert first.created_at == _NOW
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    json.dumps(first.model_dump(mode="json"))


def test_import_smoke():
    import app.services.value_of_information  # noqa: F401
