"""Dim 2 — constraint extraction beyond simplex.

Linear (in)equality constraints on inputs + outcome (response feasibility)
constraints, the feasibility checker, rejection sampling, and the bomcp spec
mapping.  See docs/plans/2026-06-26-bomcp-backend-integration-design.md (roadmap).
"""
from __future__ import annotations

import importlib.util

import pytest

from app.services.candidate_gen import (
    LinearConstraint,
    OutcomeConstraint,
    ParameterSpace,
    SearchDimension,
    is_feasible,
    sample_feasible,
)

_needs_bomcp = pytest.mark.skipif(
    importlib.util.find_spec("bo_engine") is None,
    reason="bo-engine not installed",
)


def _unit_square() -> ParameterSpace:
    return ParameterSpace(
        dimensions=(
            SearchDimension("x0", "number", min_value=0.0, max_value=1.0),
            SearchDimension("x1", "number", min_value=0.0, max_value=1.0),
        ),
        protocol_template={},
    )


# --- LinearConstraint + is_feasible -----------------------------------------


def test_linear_leq_feasibility():
    c = LinearConstraint(param_names=("x0", "x1"), coefficients=(1.0, 1.0), op="<=", bound=1.0)
    space = ParameterSpace(dimensions=_unit_square().dimensions, protocol_template={},
                           linear_constraints=(c,))
    assert is_feasible({"x0": 0.3, "x1": 0.4}, space) is True
    assert is_feasible({"x0": 0.7, "x1": 0.6}, space) is False


def test_linear_geq_feasibility():
    c = LinearConstraint(param_names=("x0", "x1"), coefficients=(1.0, 1.0), op=">=", bound=1.0)
    space = ParameterSpace(dimensions=_unit_square().dimensions, protocol_template={},
                           linear_constraints=(c,))
    assert is_feasible({"x0": 0.6, "x1": 0.6}, space) is True
    assert is_feasible({"x0": 0.2, "x1": 0.3}, space) is False


def test_weighted_linear_constraint():
    # 2*x0 + 1*x1 <= 1.5
    c = LinearConstraint(param_names=("x0", "x1"), coefficients=(2.0, 1.0), op="<=", bound=1.5)
    space = ParameterSpace(dimensions=_unit_square().dimensions, protocol_template={},
                           linear_constraints=(c,))
    assert is_feasible({"x0": 0.5, "x1": 0.4}, space) is True   # 1.0+0.4=1.4
    assert is_feasible({"x0": 0.6, "x1": 0.5}, space) is False  # 1.2+0.5=1.7


def test_no_constraints_always_feasible():
    assert is_feasible({"x0": 0.9, "x1": 0.9}, _unit_square()) is True


# --- rejection sampling ------------------------------------------------------


def test_sample_feasible_only_returns_feasible_points():
    c = LinearConstraint(param_names=("x0", "x1"), coefficients=(1.0, 1.0), op="<=", bound=1.0)
    space = ParameterSpace(dimensions=_unit_square().dimensions, protocol_template={},
                           linear_constraints=(c,))
    pts = sample_feasible(space, 20, seed=1)
    assert len(pts) == 20
    for p in pts:
        assert p["x0"] + p["x1"] <= 1.0 + 1e-9
        assert is_feasible(p, space)


# --- bomcp spec mapping ------------------------------------------------------


@_needs_bomcp
def test_bomcp_maps_sum_leq_constraint():
    from bo_engine.types import ConstraintType

    from app.optimization.bomcp_backend import to_bomcp_spec

    c = LinearConstraint(param_names=("x0", "x1"), coefficients=(1.0, 1.0), op="<=", bound=1.0)
    space = ParameterSpace(dimensions=_unit_square().dimensions, protocol_template={},
                           linear_constraints=(c,))
    spec = to_bomcp_spec(space, batch_size=2, seed=0)
    types = {sc.type for sc in spec.constraints}
    assert ConstraintType.SUM_LESS_THAN in types


@_needs_bomcp
def test_bomcp_maps_sum_geq_constraint():
    from bo_engine.types import ConstraintType

    from app.optimization.bomcp_backend import to_bomcp_spec

    c = LinearConstraint(param_names=("x0", "x1"), coefficients=(1.0, 1.0), op=">=", bound=1.0)
    space = ParameterSpace(dimensions=_unit_square().dimensions, protocol_template={},
                           linear_constraints=(c,))
    spec = to_bomcp_spec(space, batch_size=2, seed=0)
    assert any(sc.type == ConstraintType.SUM_GREATER_THAN for sc in spec.constraints)


@_needs_bomcp
def test_bomcp_maps_weighted_to_linear_with_coefficients():
    from bo_engine.types import ConstraintType

    from app.optimization.bomcp_backend import to_bomcp_spec

    c = LinearConstraint(param_names=("x0", "x1"), coefficients=(2.0, 1.0), op="<=", bound=1.5)
    space = ParameterSpace(dimensions=_unit_square().dimensions, protocol_template={},
                           linear_constraints=(c,))
    spec = to_bomcp_spec(space, batch_size=2, seed=0)
    linear = [sc for sc in spec.constraints if sc.type == ConstraintType.LINEAR]
    assert len(linear) == 1
    assert list(linear[0].coefficients) == [2.0, 1.0]


@_needs_bomcp
def test_bomcp_maps_outcome_constraint():
    from app.optimization.bomcp_backend import to_bomcp_spec

    oc = OutcomeConstraint(objective_name="yield", threshold=0.8, greater_than=True)
    space = ParameterSpace(dimensions=_unit_square().dimensions, protocol_template={},
                           outcome_constraints=(oc,))
    spec = to_bomcp_spec(space, batch_size=2, seed=0)
    assert len(spec.outcome_constraints) == 1
    assert spec.outcome_constraints[0].objective_name == "yield"
    assert spec.outcome_constraints[0].greater_than is True
