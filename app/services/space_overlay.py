"""Derived-objective / boundary overlays over a TaskContract (Phase C / C3+C2).

The headline of the self-evolving platform: break fixed-objective, fixed-boundary
BO. But mutating a versioned ``TaskContract`` mid-campaign would break every
downstream consumer and force a schema migration. So a proposed change is
expressed as a *derived overlay* — a first-class, provenance-carrying object that
layers on top of the base contract. ``derive_contract`` produces a NEW contract
with the overlay applied; the base is never mutated.

    base TaskContract + SpaceOverlay ──► derive_contract() ──► derived TaskContract
                                     └─► review_space_change() ──► verdict (gate)

C2 gate: a boundary overlay is a *deliberate* request to widen the search space.
This is categorically different from ``decision_policy._bounds_violation``, which
rejects a candidate that falls outside the CURRENT space (a hallucinated
out-of-bounds point). ``review_space_change`` evaluates the deliberate proposal:
it must reference real dimensions, must only ever WIDEN bounds (never silently
shrink), and escalates to a human when the contract requires approval, when the
advisor's confidence is low, or when the expansion is large.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.contracts.task_contract import TaskContract

__all__ = [
    "ObjectiveOverlay",
    "BoundaryOverlay",
    "SpaceOverlay",
    "SpaceChangeVerdict",
    "derive_contract",
    "review_space_change",
    "LARGE_EXPANSION_RATIO",
    "MIN_AUTO_APPROVE_CONFIDENCE",
]

# An expansion that more than triples a dimension's original range is "large"
# and always escalates to a human.
LARGE_EXPANSION_RATIO = 3.0
# Below this advisor confidence, a space change always escalates to a human.
MIN_AUTO_APPROVE_CONFIDENCE = 0.7


class ObjectiveOverlay(BaseModel):
    """A derived change to the objective (never mutates the base)."""

    new_primary_kpi: str | None = None
    add_secondary_kpis: list[str] = Field(default_factory=list)


class BoundaryOverlay(BaseModel):
    """A proposed widening of one dimension's numeric bounds."""

    param_name: str
    new_min: float | None = None
    new_max: float | None = None


class SpaceOverlay(BaseModel):
    """A first-class, auditable proposal to reframe objective / widen bounds."""

    proposal_id: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    expected_gain: str | None = None
    objective_overlay: ObjectiveOverlay | None = None
    boundary_overlays: list[BoundaryOverlay] = Field(default_factory=list)
    status: Literal["proposed", "approved", "needs_human", "rejected", "applied"] = (
        "proposed"
    )


class SpaceChangeVerdict(BaseModel):
    """Result of gating a space-change proposal."""

    status: Literal["approved", "needs_human", "rejected"]
    reason: str


def _dim_by_name(contract: TaskContract, name: str):
    for dim in contract.exploration_space.dimensions:
        if dim.param_name == name:
            return dim
    return None


def review_space_change(
    overlay: SpaceOverlay, contract: TaskContract
) -> SpaceChangeVerdict:
    """Gate a deliberate space-change proposal against the base contract.

    Distinct from ``_bounds_violation`` (which rejects out-of-bounds *candidates*
    in the current space): this evaluates a proposal to *change* the space.
    """
    escalate = False
    for bo in overlay.boundary_overlays:
        dim = _dim_by_name(contract, bo.param_name)
        if dim is None:
            return SpaceChangeVerdict(
                status="rejected",
                reason=f"unknown dimension '{bo.param_name}'",
            )
        # Must only ever widen — a proposal that shrinks the space is not this
        # channel's job and is rejected outright.
        if bo.new_min is not None and dim.min_value is not None and bo.new_min > dim.min_value:
            return SpaceChangeVerdict(
                status="rejected",
                reason=f"'{bo.param_name}' new_min {bo.new_min} shrinks lower bound {dim.min_value}",
            )
        if bo.new_max is not None and dim.max_value is not None and bo.new_max < dim.max_value:
            return SpaceChangeVerdict(
                status="rejected",
                reason=f"'{bo.param_name}' new_max {bo.new_max} shrinks upper bound {dim.max_value}",
            )
        # Large expansion escalates.
        if (
            dim.min_value is not None
            and dim.max_value is not None
            and bo.new_min is not None
            and bo.new_max is not None
        ):
            old_range = dim.max_value - dim.min_value
            new_range = bo.new_max - bo.new_min
            if old_range > 0 and new_range / old_range > LARGE_EXPANSION_RATIO:
                escalate = True

    if contract.safety_envelope.require_human_approval:
        escalate = True
    if overlay.confidence < MIN_AUTO_APPROVE_CONFIDENCE:
        escalate = True

    if escalate:
        return SpaceChangeVerdict(
            status="needs_human",
            reason="space change requires human sign-off "
            "(safety policy, low confidence, or large expansion)",
        )
    return SpaceChangeVerdict(status="approved", reason="within auto-approve policy")


def derive_contract(base: TaskContract, overlay: SpaceOverlay) -> TaskContract:
    """Return a NEW contract with *overlay* applied. Base is never mutated.

    The derived contract records its lineage in ``contract_id`` and
    ``migrated_from`` so the overlay is auditable and replayable.
    """
    objective = base.objective
    if overlay.objective_overlay is not None:
        oo = overlay.objective_overlay
        objective = objective.model_copy(
            update={
                "primary_kpi": oo.new_primary_kpi or objective.primary_kpi,
                "secondary_kpis": list(
                    dict.fromkeys([*objective.secondary_kpis, *oo.add_secondary_kpis])
                ),
            }
        )

    dimensions = [d.model_copy(deep=True) for d in base.exploration_space.dimensions]
    for bo in overlay.boundary_overlays:
        for dim in dimensions:
            if dim.param_name == bo.param_name:
                update: dict[str, float] = {}
                if bo.new_min is not None:
                    update["min_value"] = bo.new_min
                if bo.new_max is not None:
                    update["max_value"] = bo.new_max
                if update:
                    idx = dimensions.index(dim)
                    dimensions[idx] = dim.model_copy(update=update)

    exploration_space = base.exploration_space.model_copy(
        update={"dimensions": dimensions}
    )

    return base.model_copy(
        deep=True,
        update={
            "contract_id": f"{base.contract_id}+ovl-{overlay.proposal_id}",
            "migrated_from": base.contract_id,
            "objective": objective,
            "exploration_space": exploration_space,
        },
    )
