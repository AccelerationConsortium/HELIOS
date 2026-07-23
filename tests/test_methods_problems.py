"""Tests for benchmarks.methods.problems — analytic test problems.

The defining guarantee of these problems is a KNOWN ground truth: evaluating the
objective at the stated optimum_x must return the stated optimum value. Without
this, regret (and therefore the whole method comparison) has no ruler.
"""
from __future__ import annotations

import math

import pytest

from benchmarks.methods.problems import (
    OptProblem,
    ProblemTags,
    get_problem,
    get_problems,
)


def test_registry_non_empty():
    problems = get_problems()
    assert len(problems) >= 6
    assert all(isinstance(p, OptProblem) for p in problems)


@pytest.mark.parametrize(
    ("problem_id", "expected", "tol"),
    [
        ("sphere_2d", 0.0, 1e-9),
        ("branin", 0.397887, 1e-4),
        ("hartmann6", -3.32237, 5e-3),
        ("ackley_5d", 0.0, 1e-9),
        ("rosenbrock_2d", 0.0, 1e-9),
        ("rastrigin_2d", 0.0, 1e-9),
        ("early_stage_controllability", 0.0, 1e-9),
        ("early_stage_hardware_zone", 0.0, 1e-9),
        ("early_stage_objective_uncertainty", 0.0, 1e-9),
        ("early_stage_batch_effect", 0.0, 1e-9),
        ("early_stage_prior_warm_start", 0.0, 1e-9),
    ],
)
def test_objective_at_optimum_equals_known_value(problem_id, expected, tol):
    p = get_problem(problem_id)
    assert p.optimum_x is not None
    value = p.objective(p.optimum_x)
    assert math.isclose(value, expected, abs_tol=tol), (
        f"{problem_id}: f(optimum_x)={value} != {expected}"
    )
    assert math.isclose(p.optimum, expected, abs_tol=tol)


def test_early_stage_reports_are_attached_to_problem_spaces():
    for problem_id in (
        "early_stage_controllability",
        "early_stage_hardware_zone",
        "early_stage_objective_uncertainty",
        "early_stage_batch_effect",
        "early_stage_prior_warm_start",
    ):
        p = get_problem(problem_id)
        report = p.space.protocol_template.get("early_stage_report")
        assert report["contract_version"] == "early_stage_system_characterization.v1"
        assert report["risk_flags"]
        assert report["diagnostic_recommendations"]


def test_objective_uncertainty_exposes_misleading_observed_proxy():
    p = get_problem("early_stage_objective_uncertainty")
    misleading_proxy = p.evaluate(
        {"additive": "fast_yield", "temp": 86.0, "dwell_h": 1.2}
    )
    true_optimum = p.evaluate(p.optimum_x)

    assert true_optimum.raw_value == 0.0
    assert misleading_proxy.raw_value > true_optimum.raw_value
    assert misleading_proxy.optimizer_value < true_optimum.optimizer_value
    assert "candidate_kpi_stability" in misleading_proxy.observation_objectives()


def test_batch_effect_exposes_biased_observed_objective_and_correction():
    p = get_problem("early_stage_batch_effect")
    biased = p.evaluate(
        {
            "screen_protocol": "fast_screen",
            "ligand": "L1",
            "temp": 82.0,
            "hold_h": 1.2,
        }
    )
    optimum = p.evaluate(p.optimum_x)

    assert optimum.raw_value == 0.0
    assert biased.optimizer_value < optimum.optimizer_value
    assert biased.raw_value > optimum.raw_value
    assert "corrected_objective" in biased.observation_objectives()


def test_optimum_x_keys_match_space_dimensions():
    for p in get_problems():
        if p.optimum_x is None:
            continue
        dim_names = {d.param_name for d in p.space.dimensions}
        assert set(p.optimum_x) <= dim_names, p.id


def test_tags_dims_match_space():
    for p in get_problems():
        assert isinstance(p.tags, ProblemTags)
        assert p.tags.n_dims == p.space.n_dims, p.id


def test_get_problem_unknown_raises():
    with pytest.raises(KeyError):
        get_problem("does_not_exist")
