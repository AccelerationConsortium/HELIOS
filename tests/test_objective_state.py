from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.services.decision_outcome import CampaignDecisionOutcome
from app.services.objective_models import ProxyGapAssessment, ProxyGapLevel
from app.services.objective_state import (
    ObjectiveState,
    ObjectiveStateUpdater,
    StoppingCriteria,
    apply_outcome_to_objective_state,
)


def _outcome(
    *,
    round_index: int = 0,
    execution_success: bool | None = True,
    failure_count: int = 0,
    objective_delta: float | None = None,
    proxy_gap_delta: float | None = None,
    validation_success: bool | None = None,
) -> CampaignDecisionOutcome:
    return CampaignDecisionOutcome(
        trace_id="trace-1",
        campaign_id="camp-1",
        round_index=round_index,
        execution_success=execution_success,
        failure_count=failure_count,
        objective_delta=objective_delta,
        proxy_gap_delta=proxy_gap_delta,
        validation_success=validation_success,
    )


_NOW = datetime(2026, 6, 29, 12, 0, 0, tzinfo=UTC)


def test_objective_state_round_trips_through_json():
    state = ObjectiveState(
        campaign_id="camp-1",
        scientific_question="Does additive X raise device efficiency?",
        primary_objective="device_efficiency",
        proxy_objective_names=["raw_peak_area"],
        objective_confidence=0.5,
        failure_constraints=["no thermal runaway"],
        validation_requirements=["replicate best candidate 3x"],
        stopping_criteria=StoppingCriteria(target_confidence=0.9),
    )

    dumped = state.model_dump(mode="json")
    restored = ObjectiveState.model_validate(dumped)

    assert restored.primary_objective == "device_efficiency"
    assert restored.proxy_objective_names == ["raw_peak_area"]
    assert restored.stopping_criteria.target_confidence == 0.9
    assert restored.revision == 0
    assert restored.revision_history == []


def test_positive_outcome_raises_confidence_without_mutating_input():
    state = ObjectiveState(campaign_id="camp-1", primary_objective="device_efficiency")
    before = state.model_dump(mode="json")

    revised = ObjectiveStateUpdater().apply_outcome(
        state,
        _outcome(
            execution_success=True,
            objective_delta=0.4,
            validation_success=True,
        ),
        now=_NOW,
    )

    assert revised is not state
    assert state.model_dump(mode="json") == before  # input untouched
    assert revised.objective_confidence > state.objective_confidence
    assert revised.revision == 1
    assert revised.consecutive_failure_count == 0


def test_failure_outcome_lowers_confidence_and_tracks_consecutive_failures():
    updater = ObjectiveStateUpdater()
    state = ObjectiveState(
        campaign_id="camp-1",
        primary_objective="device_efficiency",
        objective_confidence=0.7,
    )

    first = updater.apply_outcome(
        state, _outcome(execution_success=False, failure_count=1), now=_NOW
    )
    second = updater.apply_outcome(
        first, _outcome(round_index=1, execution_success=False, failure_count=2), now=_NOW
    )

    assert first.objective_confidence < state.objective_confidence
    assert first.consecutive_failure_count == 1
    assert second.consecutive_failure_count == 2
    assert second.objective_confidence <= first.objective_confidence


def test_confidence_is_clamped_to_unit_interval():
    high = ObjectiveState(
        campaign_id="camp-1", primary_objective="x", objective_confidence=0.98
    )
    revised = ObjectiveStateUpdater().apply_outcome(
        high,
        _outcome(execution_success=True, objective_delta=1.0, validation_success=True),
        now=_NOW,
    )
    assert 0.0 <= revised.objective_confidence <= 1.0


def test_revision_carries_provenance_and_is_replayable():
    state = ObjectiveState(campaign_id="camp-1", primary_objective="x")

    revised = apply_outcome_to_objective_state(
        state, _outcome(objective_delta=0.2), now=_NOW
    )

    assert len(revised.revision_history) == 1
    revision = revised.revision_history[0]
    assert revision.revision == 1
    assert revision.source == "campaign_decision_outcome"
    assert revision.trace_id == "trace-1"
    assert revision.reason
    assert revision.evidence
    assert "objective_confidence" in revision.changes
    assert revision.created_at == _NOW
    # Replayable: the final confidence equals the revision's recorded "to" value.
    assert revision.changes["objective_confidence"]["to"] == pytest.approx(
        revised.objective_confidence
    )


def test_proxy_gap_snapshot_is_recorded_when_present():
    assessment = ProxyGapAssessment(
        score=0.7,
        level=ProxyGapLevel.HIGH,
        active_metric_names=["raw_peak_area"],
        rationale="distant proxy",
    )
    state = ObjectiveState(campaign_id="camp-1", primary_objective="x")

    revised = ObjectiveStateUpdater().apply_outcome(
        state,
        _outcome(proxy_gap_delta=0.3),
        proxy_gap=assessment,
        now=_NOW,
    )

    assert revised.proxy_gap is not None
    assert revised.proxy_gap.level == ProxyGapLevel.HIGH


def test_target_confidence_triggers_shadow_stop_recommendation():
    state = ObjectiveState(
        campaign_id="camp-1",
        primary_objective="x",
        objective_confidence=0.85,
        stopping_criteria=StoppingCriteria(target_confidence=0.9),
    )

    revised = ObjectiveStateUpdater().apply_outcome(
        state,
        _outcome(execution_success=True, objective_delta=0.5, validation_success=True),
        now=_NOW,
    )

    assert revised.stop_recommended is True
    assert revised.stop_reason == "target_confidence_reached"


def test_max_consecutive_failures_triggers_shadow_stop_recommendation():
    updater = ObjectiveStateUpdater()
    state = ObjectiveState(
        campaign_id="camp-1",
        primary_objective="x",
        stopping_criteria=StoppingCriteria(max_consecutive_failures=2),
    )

    first = updater.apply_outcome(
        state, _outcome(execution_success=False, failure_count=1), now=_NOW
    )
    second = updater.apply_outcome(
        first, _outcome(round_index=1, execution_success=False, failure_count=1), now=_NOW
    )

    assert first.stop_recommended is False
    assert second.stop_recommended is True
    assert second.stop_reason == "max_consecutive_failures_reached"


def test_import_smoke():
    import app.services.objective_state  # noqa: F401
