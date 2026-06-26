"""Fast smoke tests for the study runner (cheap analytic problems only)."""
from __future__ import annotations

from benchmarks.methods.problems import get_problem
from benchmarks.methods.runner import RunTrace, run_cell, run_study


def test_run_cell_well_formed():
    problem = get_problem("sphere_2d")
    trace = run_cell(problem, "random_sampling", seed=0, budget=8)
    assert isinstance(trace, RunTrace)
    assert trace.problem_id == "sphere_2d"
    assert trace.backend == "random_sampling"
    assert trace.error is None
    assert len(trace.best_so_far) == 8
    assert trace.evals == list(range(1, 9))
    assert trace.wall_s >= 0.0


def test_best_so_far_monotonic_non_increasing():
    # Minimization: best-so-far should never get worse (never increase).
    problem = get_problem("sphere_2d")
    trace = run_cell(problem, "lhs", seed=1, budget=10)
    for a, b in zip(trace.best_so_far, trace.best_so_far[1:], strict=False):
        assert b <= a + 1e-12


def test_run_study_full_grid():
    problems = [get_problem("sphere_2d"), get_problem("rastrigin_2d")]
    backends = ["lhs", "random_sampling"]
    seeds = [0, 1]
    traces = run_study(problems, backends, seeds, budget=6)
    # 2 problems x 2 backends x 2 seeds = 8 cells.
    assert len(traces) == 8
    for t in traces:
        assert t.error is None
        assert len(t.best_so_far) == 6
        # regret is non-negative (best observed >= true optimum).
        optimum = get_problem(t.problem_id).optimum
        assert min(t.best_so_far) >= optimum - 1e-9


def test_run_study_isolates_bad_backend():
    # An unknown backend name raises inside the cell; the study records the
    # error and keeps going rather than crashing.
    problems = [get_problem("sphere_2d")]
    traces = run_study(problems, ["definitely_not_a_backend"], [0], budget=4)
    assert len(traces) == 1
    assert traces[0].error is not None
