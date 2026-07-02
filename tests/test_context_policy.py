from __future__ import annotations

from app.services.context_policy import evaluate_context_gate, propose_space_revision
from app.services.strategy_models import (
    CampaignContext,
    CampaignSnapshot,
    FailureEvent,
)


def _snapshot(**kwargs):
    base = dict(
        round_number=3,
        max_rounds=10,
        n_observations=8,
        n_dimensions=2,
        has_categorical=False,
        has_log_scale=False,
        kpi_history=(0.1, 0.2, 0.3),
        all_kpis=(0.1, 0.2, 0.3),
        all_params=({"x": 0.1}, {"x": 0.2}, {"x": 0.3}),
        available_backends={"lhs": True, "built_in": True},
    )
    base.update(kwargs)
    return CampaignSnapshot(**base)


def test_context_gate_routes_measurement_failure_to_diagnose():
    snap = _snapshot(
        failure_events=(
            FailureEvent(
                failure_type="measurement",
                reason="drift in blank",
            ),
        )
    )

    decision = evaluate_context_gate(snap)

    assert decision.ready_for_optimization is False
    assert decision.requires_calibration is True
    assert decision.recommended_intent == "diagnose"


def test_context_gate_marks_mechanism_stage_for_hypothesis_update():
    snap = _snapshot(
        campaign_context=CampaignContext(
            scientific_goal="explain degradation",
            current_objective_level="mechanism",
        )
    )

    decision = evaluate_context_gate(snap)

    assert decision.ready_for_optimization is True
    assert decision.requires_hypothesis_update is True
    assert decision.recommended_intent == "validate"


def test_space_revision_adds_constraints_for_constraint_failures():
    snap = _snapshot(
        failure_events=(
            FailureEvent(
                failure_type="constraint",
                reason="safety voltage window exceeded",
                params={"voltage": 3.2},
                penalize_backend=True,
            ),
        )
    )

    revision = propose_space_revision(snap)

    assert revision is not None
    assert revision.add_constraints[0]["type"] == "avoid_failed_coordinate"
    assert revision.add_constraints[0]["params"] == {"voltage": 3.2}
    assert revision.revision_type == "constraint_update"
    assert revision.lifecycle_status == "proposed"
    assert revision.approval_required is True
    assert revision.auto_applied is False
    assert revision.affected_parameters == ("voltage",)


def test_space_revision_can_recommend_route_switch_for_generalization():
    snap = _snapshot(
        campaign_context=CampaignContext(
            current_objective_level="generalization",
            synthesis_routes=("electrodeposition", "gel"),
        )
    )

    revision = propose_space_revision(snap)

    assert revision is not None
    assert revision.switch_route == "gel"
    assert revision.revision_type == "route_switch"
    assert revision.lifecycle_status == "proposed"
    assert revision.risk_level == "high"
    assert revision.approval_required is True
    assert revision.auto_applied is False


def test_objective_level_gate_priors_cover_ladder():
    cases = {
        "feasibility": ("recover", False),
        "data_quality": ("stabilize", False),
        "baseline": ("discover", True),
        "performance": ("optimize", True),
        "mechanism": ("validate", True),
        "generalization": ("pivot", True),
    }

    for level, (intent, ready) in cases.items():
        decision = evaluate_context_gate(
            _snapshot(campaign_context=CampaignContext(current_objective_level=level))
        )
        assert decision.recommended_intent == intent
        assert decision.ready_for_optimization is ready
