"""Dim 9 — failure-region modeling in parameter space.

HELIOS already learns failures at the *method* level (per-backend penalty/veto).
This pushes failure learning down to the *coordinate* level: a region where
experiments keep failing becomes a learned infeasible region that future
suggestions avoid -- expressed as a bomcp OutcomeConstraint on a synthetic
"feasibility" response, and enforced for any backend via candidate filtering.
"""
from __future__ import annotations

from app.optimization.failure_region import (
    FailureRegionModel,
    build_feasibility_observations,
    failure_outcome_constraint,
    filter_failure_prone,
)
from app.services.candidate_gen import OutcomeConstraint, ParameterSpace, SearchDimension
from app.services.optimization_backends import Observation


def _unit_square() -> ParameterSpace:
    return ParameterSpace(
        dimensions=(
            SearchDimension("x0", "number", min_value=0.0, max_value=1.0),
            SearchDimension("x1", "number", min_value=0.0, max_value=1.0),
        ),
        protocol_template={},
    )


def _failures_near(cx, cy, n=6):
    import random

    rng = random.Random(0)
    return [
        {"x0": cx + rng.uniform(-0.03, 0.03), "x1": cy + rng.uniform(-0.03, 0.03)}
        for _ in range(n)
    ]


# --- failure score / feasibility prediction ---------------------------------


def test_failure_score_high_near_failures_low_far_away():
    space = _unit_square()
    model = FailureRegionModel.fit(failed=_failures_near(0.8, 0.8), space=space)
    assert model.failure_score({"x0": 0.8, "x1": 0.8}) > 0.5
    assert model.failure_score({"x0": 0.1, "x1": 0.1}) < 0.1


def test_predicted_feasible_avoids_failure_cluster():
    space = _unit_square()
    model = FailureRegionModel.fit(failed=_failures_near(0.8, 0.8), space=space)
    assert model.predicted_feasible({"x0": 0.1, "x1": 0.1}) is True
    assert model.predicted_feasible({"x0": 0.8, "x1": 0.8}) is False


def test_no_failures_means_everything_feasible():
    space = _unit_square()
    model = FailureRegionModel.fit(failed=[], space=space)
    assert model.failure_score({"x0": 0.8, "x1": 0.8}) == 0.0
    assert model.predicted_feasible({"x0": 0.8, "x1": 0.8}) is True


def test_categorical_failures_match_by_category():
    space = ParameterSpace(
        dimensions=(
            SearchDimension("solvent", "categorical", choices=("water", "ethanol", "dmso")),
            SearchDimension("temp", "number", min_value=20.0, max_value=100.0),
        ),
        protocol_template={},
    )
    model = FailureRegionModel.fit(
        failed=[{"solvent": "dmso", "temp": 90.0}, {"solvent": "dmso", "temp": 95.0}],
        space=space,
    )
    # Same category + nearby temp -> failure-prone; different category -> safe.
    assert model.failure_score({"solvent": "dmso", "temp": 92.0}) > 0.5
    assert model.failure_score({"solvent": "water", "temp": 92.0}) < 0.5


# --- candidate filtering (universal enforcement) ----------------------------


def test_filter_failure_prone_removes_points_in_region():
    space = _unit_square()
    model = FailureRegionModel.fit(failed=_failures_near(0.8, 0.8), space=space)
    cands = [
        {"x0": 0.1, "x1": 0.1},   # safe
        {"x0": 0.8, "x1": 0.8},   # in failure region
        {"x0": 0.2, "x1": 0.15},  # safe
    ]
    kept = filter_failure_prone(cands, model)
    assert {"x0": 0.8, "x1": 0.8} not in kept
    assert {"x0": 0.1, "x1": 0.1} in kept
    assert len(kept) == 2


# --- bomcp expression (learned outcome constraint) --------------------------


def test_feasibility_observations_label_success_and_failure():
    succeeded = [Observation(params={"x0": 0.1, "x1": 0.1}, objective=5.0)]
    failed = [{"x0": 0.8, "x1": 0.8}]
    obs = build_feasibility_observations(succeeded, failed)
    feas = {tuple(sorted(o.parameter_values.items())): o.objective_values["feasibility"]
            for o in obs}
    assert feas[(("x0", 0.1), ("x1", 0.1))] == 1.0
    assert feas[(("x0", 0.8), ("x1", 0.8))] == 0.0


def test_failure_outcome_constraint_shape():
    oc = failure_outcome_constraint()
    assert isinstance(oc, OutcomeConstraint)
    assert oc.objective_name == "feasibility"
    assert oc.greater_than is True
    assert 0.0 < oc.threshold < 1.0


# --- P3b: failure avoidance wired into candidate generation -----------------


def test_snapshot_carries_failed_params():
    from app.services.strategy_models import CampaignSnapshot

    snap = CampaignSnapshot(
        round_number=1, max_rounds=5, n_observations=0, n_dimensions=2,
        has_categorical=False, has_log_scale=False,
        failed_params=({"x0": 0.8, "x1": 0.8},),
    )
    assert snap.failed_params == ({"x0": 0.8, "x1": 0.8},)


def test_generate_adaptive_candidates_avoids_failure_region():
    from app.services.strategy_models import CampaignSnapshot
    from app.services.strategy_selector import generate_adaptive_candidates

    space = _unit_square()
    snap = CampaignSnapshot(
        round_number=2, max_rounds=5, n_observations=5, n_dimensions=2,
        has_categorical=False, has_log_scale=False,
        user_strategy_hint="random",  # deterministic, fast backend
        failed_params=tuple(_failures_near(0.8, 0.8, n=8)),
    )
    cands, _decision = generate_adaptive_candidates(space, 10, [], snap, seed=1)
    assert len(cands) == 10
    model = FailureRegionModel.fit(failed=list(snap.failed_params), space=space)
    for c in cands:
        assert model.predicted_feasible(c), f"{c} fell in the learned failure region"


def test_generation_unchanged_without_failures():
    from app.services.strategy_models import CampaignSnapshot
    from app.services.strategy_selector import generate_adaptive_candidates

    space = _unit_square()
    snap = CampaignSnapshot(
        round_number=2, max_rounds=5, n_observations=5, n_dimensions=2,
        has_categorical=False, has_log_scale=False,
        user_strategy_hint="random",
    )
    cands, _ = generate_adaptive_candidates(space, 10, [], snap, seed=1)
    assert len(cands) == 10


def test_avoid_failure_region_filters_and_tops_up():
    from app.optimization.failure_region import avoid_failure_region

    space = _unit_square()
    failed = _failures_near(0.8, 0.8, n=8)
    cands = [
        {"x0": 0.8, "x1": 0.8},    # in region
        {"x0": 0.79, "x1": 0.81},  # in region
        {"x0": 0.1, "x1": 0.1},    # safe
        {"x0": 0.2, "x1": 0.2},    # safe
    ]
    out = avoid_failure_region(cands, space, 4, failed, seed=1)
    assert len(out) == 4  # filtered + topped up back to n
    model = FailureRegionModel.fit(failed=failed, space=space)
    for p in out:
        assert model.predicted_feasible(p)


def test_avoid_failure_region_noop_without_failures():
    from app.optimization.failure_region import avoid_failure_region

    space = _unit_square()
    cands = [{"x0": 0.8, "x1": 0.8}, {"x0": 0.1, "x1": 0.1}]
    out = avoid_failure_region(cands, space, 2, [], seed=1)
    assert out == cands


def test_design_input_carries_failed_params():
    from app.agents.design_agent import DesignInput

    di = DesignInput(
        dimensions=[{"param_name": "x", "param_type": "number", "min_value": 0, "max_value": 1}],
        protocol_template={},
        failed_params=[{"x": 0.9}],
    )
    assert di.failed_params == [{"x": 0.9}]
    # Defaults to empty so existing callers are unaffected.
    di2 = DesignInput(dimensions=di.dimensions, protocol_template={})
    assert di2.failed_params == []
