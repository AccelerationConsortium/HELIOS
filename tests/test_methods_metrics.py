"""Unit tests for benchmarks.methods.metrics with hand-computed expectations."""
from __future__ import annotations

import math

from benchmarks.methods.metrics import (
    convergence_auc,
    cost,
    evals_to_target,
    noise_robustness,
    simple_regret,
)


def test_simple_regret_scalar():
    assert simple_regret(5.0, 2.0) == 3.0


def test_simple_regret_uses_last_of_trace():
    # Best-so-far trace; regret uses the final (best) value.
    assert simple_regret([10.0, 4.0, 2.5], 2.0) == 0.5


def test_simple_regret_zero_at_optimum():
    assert simple_regret([3.0, 0.0], 0.0) == 0.0


def test_evals_to_target_first_crossing():
    # regret = [3, 1, 0.05, 0.0]; tol 0.1 first met at index 2 -> 1-based 3.
    trace = [3.0, 1.0, 0.05, 0.0]
    assert evals_to_target(trace, optimum=0.0, tol=0.1) == 3


def test_evals_to_target_immediate():
    assert evals_to_target([0.0, 0.0], optimum=0.0, tol=1e-9) == 1


def test_evals_to_target_never_reached():
    assert evals_to_target([5.0, 4.0, 3.0], optimum=0.0, tol=0.1) is None


def test_convergence_auc_trapezoid():
    # regret curve [3, 1, 0] -> trapezoid: (3+1)/2 + (1+0)/2 = 2 + 0.5 = 2.5
    assert convergence_auc([3.0, 1.0, 0.0], optimum=0.0) == 2.5


def test_convergence_auc_single_point():
    assert convergence_auc([2.0], optimum=0.0) == 2.0


def test_convergence_auc_empty():
    assert convergence_auc([], optimum=0.0) == 0.0


def test_convergence_auc_with_offset_optimum():
    # values [5, 3] with optimum 2 -> regret [3, 1] -> (3+1)/2 = 2.0
    assert convergence_auc([5.0, 3.0], optimum=2.0) == 2.0


def test_noise_robustness_pstdev():
    # population stdev of [1, 1, 4]: mean=2, var=((1+1+4)/3)... var=2, std=sqrt(2)
    vals = [1.0, 1.0, 4.0]
    assert math.isclose(noise_robustness(vals), math.sqrt(2.0), rel_tol=1e-9)


def test_noise_robustness_single_seed_zero():
    assert noise_robustness([3.0]) == 0.0


def test_cost_per_eval():
    assert cost(10.0, 5) == 2.0


def test_cost_zero_evals():
    assert cost(10.0, 0) == 0.0
