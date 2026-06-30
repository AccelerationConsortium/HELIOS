from __future__ import annotations

import json
from datetime import UTC, datetime

from app.services.campaign_mode import (
    CampaignMode,
    CampaignModeContext,
    CampaignModeTransitionTable,
    decide_campaign_mode,
)
from app.services.failure_attribution import attribute_failure
from app.services.failure_signatures import classify_failure
from app.services.objective_models import ProxyGapAssessment, ProxyGapLevel
from app.services.objective_state import ObjectiveState, StoppingCriteria

_NOW = datetime(2026, 6, 29, 12, 0, 0, tzinfo=UTC)


def _attr(error: str, primitive: str = ""):
    return attribute_failure(
        classify_failure(step_key="s1", primitive=primitive, error_message=error),
        now=_NOW,
    )


def _objective(**kwargs) -> ObjectiveState:
    return ObjectiveState(campaign_id="camp-1", primary_objective="x", **kwargs)


def _ctx(**kwargs) -> CampaignModeContext:
    base = {"campaign_id": "camp-1", "round_index": 0}
    base.update(kwargs)
    return CampaignModeContext(**base)


def test_default_is_bo_optimization():
    decision = decide_campaign_mode(_ctx(objective_state=_objective()), now=_NOW)

    assert decision.mode == CampaignMode.BO_OPTIMIZATION
    assert decision.shadow_only is True


def test_proxy_gap_high_triggers_validation():
    objective = _objective(
        proxy_gap=ProxyGapAssessment(
            score=0.8,
            level=ProxyGapLevel.HIGH,
            active_metric_names=["raw_peak_area"],
            rationale="distant proxy",
        )
    )

    decision = decide_campaign_mode(_ctx(objective_state=objective), now=_NOW)

    assert decision.mode == CampaignMode.VALIDATION


def test_instrument_attribution_triggers_calibration():
    decision = decide_campaign_mode(
        _ctx(
            objective_state=_objective(),
            failure_attribution=_attr("temp overshoot exceeded", primitive="heat"),
        ),
        now=_NOW,
    )

    assert decision.mode == CampaignMode.CALIBRATION


def test_execution_attribution_triggers_failure_diagnosis():
    decision = decide_campaign_mode(
        _ctx(
            objective_state=_objective(),
            failure_attribution=_attr("no tips available", primitive="robot.pick_up_tip"),
        ),
        now=_NOW,
    )

    assert decision.mode == CampaignMode.FAILURE_DIAGNOSIS


def test_context_missing_triggers_literature_seeking():
    decision = decide_campaign_mode(
        _ctx(objective_state=_objective(), literature_missing=True), now=_NOW
    )

    assert decision.mode == CampaignMode.LITERATURE_CONTEXT_SEEKING


def test_proxy_gap_high_outranks_literature_missing():
    # proxy_gap HIGH means the proxy diverges from the true objective; validating
    # the proxy takes priority over seeking external context.
    objective = _objective(
        proxy_gap=ProxyGapAssessment(
            score=0.8,
            level=ProxyGapLevel.HIGH,
            active_metric_names=["raw_peak_area"],
            rationale="distant proxy",
        )
    )

    decision = decide_campaign_mode(
        _ctx(objective_state=objective, literature_missing=True), now=_NOW
    )

    assert decision.mode == CampaignMode.VALIDATION
    assert decision.priority_rank == 5


def test_low_attribution_confidence_triggers_human_observation():
    # An unmatched failure classifies as unknown with confidence 0.2 (< 0.5).
    attribution = _attr("something nobody has a rule for")
    assert attribution.confidence == 0.2

    decision = decide_campaign_mode(
        _ctx(objective_state=_objective(), failure_attribution=attribution), now=_NOW
    )

    assert decision.mode == CampaignMode.HUMAN_OBSERVATION_REQUEST


def test_stop_recommended_takes_priority_over_other_signals():
    objective = _objective(
        objective_confidence=0.95,
        stopping_criteria=StoppingCriteria(target_confidence=0.9),
        stop_recommended=True,
        stop_reason="target_confidence_reached",
    )

    # Even with an instrument failure present, stop wins by explicit priority.
    decision = decide_campaign_mode(
        _ctx(
            objective_state=objective,
            failure_attribution=_attr("temp overshoot exceeded", primitive="heat"),
        ),
        now=_NOW,
    )

    assert decision.mode == CampaignMode.STOP_RECOMMENDED
    assert decision.priority_rank == 1
    assert "target_confidence_reached" in decision.reason


def test_evidence_includes_objective_and_failure_fields():
    objective = _objective(objective_confidence=0.42)
    attribution = _attr("temp overshoot exceeded", primitive="heat")

    decision = decide_campaign_mode(
        _ctx(objective_state=objective, failure_attribution=attribution), now=_NOW
    )

    payloads = {ev.kind: ev.payload for ev in decision.evidence}
    assert "objective_state" in payloads
    assert payloads["objective_state"]["objective_confidence"] == 0.42
    assert "failure_attribution" in payloads
    assert payloads["failure_attribution"]["dominant_category"] == "instrument_failure"


def test_decision_is_deterministic_and_json_safe():
    ctx = _ctx(
        objective_state=_objective(),
        failure_attribution=_attr("temp overshoot exceeded", primitive="heat"),
    )

    first = decide_campaign_mode(ctx, now=_NOW)
    second = CampaignModeTransitionTable().decide(ctx, now=_NOW)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    json.dumps(first.model_dump(mode="json"))
    assert first.created_at == _NOW


def test_import_smoke():
    import app.services.campaign_mode  # noqa: F401
