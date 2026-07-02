"""Pure metric functions over a run trace.

All metrics operate on a ``best_so_far`` list -- the best (lowest, in
minimization terms) objective value observed up to and including each
evaluation -- plus the problem's known ``optimum``.  They are the project's
"optimization objectives / criteria" (regret, sample efficiency, convergence
speed, cost, noise robustness).
"""
from __future__ import annotations

import statistics
from collections.abc import Sequence


def simple_regret(best_so_far: float, optimum: float) -> float:
    """Final-quality gap: best observed value minus the known optimum.

    In minimization terms this is ``>= 0``; 0 means the optimum was found.
    Accepts either a scalar best value or, for convenience, a trace (uses the
    last element).
    """
    best = best_so_far[-1] if isinstance(best_so_far, Sequence) else best_so_far
    return float(best) - float(optimum)


def evals_to_target(
    best_so_far: Sequence[float],
    optimum: float,
    tol: float,
) -> int | None:
    """Number of evaluations to first reach ``regret <= tol``.

    Returns the 1-based evaluation index where ``best_so_far - optimum <= tol``,
    or ``None`` if the target was never reached within the trace.
    """
    for i, best in enumerate(best_so_far):
        if (float(best) - float(optimum)) <= tol:
            return i + 1
    return None


def convergence_auc(best_so_far: Sequence[float], optimum: float = 0.0) -> float:
    """Area under the regret curve (trapezoidal). Lower = faster convergence.

    Summing regret over evaluations rewards methods that drop regret early.
    With a single evaluation the AUC is just that point's regret.
    """
    regret = [float(b) - float(optimum) for b in best_so_far]
    if not regret:
        return 0.0
    if len(regret) == 1:
        return regret[0]
    auc = 0.0
    for i in range(len(regret) - 1):
        auc += (regret[i] + regret[i + 1]) / 2.0
    return auc


def noise_robustness(final_regrets: Sequence[float]) -> float:
    """Spread of final regret across seeds. Lower = more robust.

    Population standard deviation of the per-seed final regrets; 0 for a single
    seed (no spread to measure).
    """
    vals = [float(r) for r in final_regrets]
    if len(vals) < 2:
        return 0.0
    return statistics.pstdev(vals)


def cost(wall_s: float, n_evals: int) -> float:
    """Per-evaluation wall-clock cost in seconds. 0 when no evaluations."""
    if n_evals <= 0:
        return 0.0
    return float(wall_s) / float(n_evals)
