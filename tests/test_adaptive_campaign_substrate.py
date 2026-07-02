from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime

from app.services.adaptive_campaign_substrate import (
    build_adaptive_campaign_substrate_snapshot,
)
from app.services.campaign_mode import CampaignMode
from app.services.dynamic_action_space import ActionSpec
from app.services.failure_attribution import attribute_failure
from app.services.failure_signatures import classify_failure
from app.services.objective_state import ObjectiveState
from app.services.value_of_information import ActionValueSignals

_NOW = datetime(2026, 6, 29, 12, 0, 0, tzinfo=UTC)


def _instrument_attr():
    return attribute_failure(
        classify_failure(step_key="s1", primitive="heat", error_message="temp overshoot exceeded"),
        now=_NOW,
    )


def _build(**overrides):
    base = dict(
        campaign_id="camp-1",
        round_index=0,
        objective_state=ObjectiveState(
            campaign_id="camp-1", primary_objective="x", created_at=_NOW
        ),
        failure_attribution=_instrument_attr(),
        actions=[
            ActionSpec(name="recalibrate", kind="calibration", required_capabilities=["heat"]),
            ActionSpec(name="heat_sample", kind="experiment", required_capabilities=["heat"]),
        ],
        available_capabilities=["heat"],
        value_signals=[ActionValueSignals(name="recalibrate", immediate_reward=0.5)],
        now=_NOW,
    )
    base.update(overrides)
    return build_adaptive_campaign_substrate_snapshot(**base)


def test_assembles_full_phase_1_to_5_chain():
    snapshot = _build()

    assert snapshot.objective_state is not None
    assert snapshot.failure_attribution is not None
    assert snapshot.campaign_mode_decision is not None
    assert snapshot.dynamic_action_space_snapshot is not None
    assert snapshot.value_of_information_snapshot is not None

    # The instrument failure routes the chain to calibration end-to-end.
    assert snapshot.campaign_mode_decision.mode == CampaignMode.CALIBRATION
    assert snapshot.dynamic_action_space_snapshot.mode == CampaignMode.CALIBRATION
    assert snapshot.value_of_information_snapshot.mode == CampaignMode.CALIBRATION
    # Both actions flow through to VoI scoring.
    assert {s.name for s in snapshot.value_of_information_snapshot.scores} == {
        "recalibrate",
        "heat_sample",
    }
    assert snapshot.campaign_id == "camp-1"
    assert snapshot.round_index == 0


def test_shadow_only_true_across_whole_chain():
    snapshot = _build()

    assert snapshot.shadow_only is True
    assert snapshot.campaign_mode_decision.shadow_only is True
    assert snapshot.dynamic_action_space_snapshot.shadow_only is True
    assert snapshot.value_of_information_snapshot.shadow_only is True


def test_voi_ranking_is_advisory_only():
    snapshot = _build()

    # The substrate marks the ranking advisory and exposes no selected action.
    assert snapshot.metadata["voi_ranking_advisory_only"] is True
    assert not hasattr(snapshot, "selected_action")
    # Action ordering reflects input order, not the VoI ranking (no reordering).
    assert [a.name for a in snapshot.dynamic_action_space_snapshot.assessments] == [
        "recalibrate",
        "heat_sample",
    ]
    assert isinstance(snapshot.value_of_information_snapshot.ranking, list)


def test_missing_value_signals_default_to_zero_without_crash():
    snapshot = _build(value_signals=None)

    scores = {s.name: s for s in snapshot.value_of_information_snapshot.scores}
    assert scores["recalibrate"].immediate_reward == 0.0
    assert scores["heat_sample"].immediate_reward == 0.0


def test_no_objective_or_failure_inputs_is_ok():
    snapshot = build_adaptive_campaign_substrate_snapshot(
        campaign_id="camp-1",
        round_index=2,
        objective_state=None,
        failure_attribution=None,
        actions=None,
        available_capabilities=None,
        value_signals=None,
        now=_NOW,
    )

    assert snapshot.objective_state is None
    assert snapshot.failure_attribution is None
    assert snapshot.campaign_mode_decision.mode == CampaignMode.BO_OPTIMIZATION
    assert snapshot.dynamic_action_space_snapshot.assessments == []
    assert snapshot.value_of_information_snapshot.scores == []


def test_deterministic_with_injected_now():
    first = _build()
    second = _build()

    assert first.created_at == _NOW
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_json_serializable():
    snapshot = _build()

    dumped = snapshot.model_dump(mode="json")
    json.dumps(dumped)
    # The embedded failure-attribution dataclass survives serialization.
    assert dumped["failure_attribution"]["dominant_category"] == "instrument_failure"


def test_has_provenance_evidence():
    snapshot = _build()

    assert snapshot.evidence
    kinds = {ev.kind for ev in snapshot.evidence}
    assert "substrate_chain" in kinds


def test_module_does_not_import_routing_layers():
    import app.services.adaptive_campaign_substrate as module

    source = inspect.getsource(module)
    import_lines = "\n".join(
        line
        for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    )
    assert "orchestrator" not in import_lines
    assert "strategy_selector" not in import_lines
    assert "decision_layer" not in import_lines


def test_import_smoke():
    import app.services.adaptive_campaign_substrate  # noqa: F401
