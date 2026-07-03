"""Phase C / C3+C2: derived-objective overlay + gated space-change review."""
from __future__ import annotations

from app.contracts.task_contract import (
    DimensionDef,
    ExplorationSpace,
    HumanGatePolicy,
    ObjectiveSpec,
    SafetyEnvelope,
    StopCondition,
    TaskContract,
)
from app.services.space_overlay import (
    BoundaryOverlay,
    ObjectiveOverlay,
    SpaceOverlay,
    derive_contract,
    review_space_change,
)


def _contract(require_human=False):
    return TaskContract(
        contract_id="c-1",
        created_at="2026-07-03T00:00:00Z",
        created_by="test",
        objective=ObjectiveSpec(
            objective_type="single", primary_kpi="overpotential", direction="minimize"
        ),
        exploration_space=ExplorationSpace(
            dimensions=[
                DimensionDef(
                    param_name="fe_ratio", param_type="number",
                    min_value=0.0, max_value=1.0,
                ),
                DimensionDef(
                    param_name="catalyst", param_type="categorical",
                    choices=["a", "b"],
                ),
            ]
        ),
        stop_conditions=StopCondition(max_rounds=30),
        safety_envelope=SafetyEnvelope(require_human_approval=require_human),
        human_gate=HumanGatePolicy(),
        protocol_pattern_id="p-1",
    )


# --- C3: derive_contract never mutates the base -------------------------


def test_boundary_overlay_derives_without_mutating_base():
    base = _contract()
    overlay = SpaceOverlay(
        proposal_id="widen-fe",
        reason="plateau",
        confidence=0.9,
        boundary_overlays=[BoundaryOverlay(param_name="fe_ratio", new_min=-0.5, new_max=1.5)],
    )
    derived = derive_contract(base, overlay)

    d_fe = next(d for d in derived.exploration_space.dimensions if d.param_name == "fe_ratio")
    assert d_fe.min_value == -0.5 and d_fe.max_value == 1.5
    # base untouched
    b_fe = next(d for d in base.exploration_space.dimensions if d.param_name == "fe_ratio")
    assert b_fe.min_value == 0.0 and b_fe.max_value == 1.0
    # lineage recorded
    assert derived.contract_id == "c-1+ovl-widen-fe"
    assert derived.migrated_from == "c-1"


def test_objective_overlay_adds_secondary_kpi():
    base = _contract()
    overlay = SpaceOverlay(
        proposal_id="reframe",
        reason="proxy mismatch",
        confidence=0.9,
        objective_overlay=ObjectiveOverlay(add_secondary_kpis=["overpotential_robustness"]),
    )
    derived = derive_contract(base, overlay)
    assert "overpotential_robustness" in derived.objective.secondary_kpis
    assert base.objective.secondary_kpis == []  # base untouched


# --- C2: review_space_change gate ---------------------------------------


def test_unknown_dimension_rejected():
    v = review_space_change(
        SpaceOverlay(
            proposal_id="p", reason="r", confidence=0.9,
            boundary_overlays=[BoundaryOverlay(param_name="nope", new_max=2.0)],
        ),
        _contract(),
    )
    assert v.status == "rejected" and "unknown dimension" in v.reason


def test_shrinking_bound_rejected():
    v = review_space_change(
        SpaceOverlay(
            proposal_id="p", reason="r", confidence=0.9,
            boundary_overlays=[BoundaryOverlay(param_name="fe_ratio", new_max=0.5)],
        ),
        _contract(),
    )
    assert v.status == "rejected" and "shrinks" in v.reason


def test_small_expansion_high_confidence_approved():
    v = review_space_change(
        SpaceOverlay(
            proposal_id="p", reason="r", confidence=0.9,
            boundary_overlays=[BoundaryOverlay(param_name="fe_ratio", new_min=-0.2, new_max=1.2)],
        ),
        _contract(),
    )
    assert v.status == "approved"


def test_large_expansion_escalates():
    # original range 1.0; new range 5.0 → >3x → needs_human
    v = review_space_change(
        SpaceOverlay(
            proposal_id="p", reason="r", confidence=0.9,
            boundary_overlays=[BoundaryOverlay(param_name="fe_ratio", new_min=-2.0, new_max=3.0)],
        ),
        _contract(),
    )
    assert v.status == "needs_human"


def test_low_confidence_escalates():
    v = review_space_change(
        SpaceOverlay(
            proposal_id="p", reason="r", confidence=0.5,
            boundary_overlays=[BoundaryOverlay(param_name="fe_ratio", new_max=1.1)],
        ),
        _contract(),
    )
    assert v.status == "needs_human"


def test_require_human_approval_escalates():
    v = review_space_change(
        SpaceOverlay(
            proposal_id="p", reason="r", confidence=0.95,
            boundary_overlays=[BoundaryOverlay(param_name="fe_ratio", new_max=1.1)],
        ),
        _contract(require_human=True),
    )
    assert v.status == "needs_human"
