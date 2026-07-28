"""Study runner: execute backends on problems across seeds, record traces.

For each (problem x backend x seed) cell:
1. Seed with a small random initial design.
2. Loop: ``get_backend(name).suggest(space, batch, observations, seed=...)`` ->
   evaluate ``problem.objective`` -> fold the result back as an
   ``Observation`` (sign-flipped to maximization) -> repeat until ``budget``
   evaluations are spent.
3. Track best-so-far (minimization) at each evaluation.

The loop is defensive: a backend that raises is recorded with an ``error`` and
the study continues with the other cells (one bad method never kills the run).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.services.candidate_gen import ParameterSpace, sample_random
from app.services.optimization_backends import Observation, get_backend
from benchmarks.methods.problems import OptProblem

logger = logging.getLogger(__name__)

DEFAULT_INIT = 3  # initial random design size before the optimizer takes over


@dataclass(frozen=True)
class RunTrace:
    """One (problem, backend, seed) study cell.

    ``best_so_far`` is the running minimum of the raw objective (lower = better),
    aligned with ``evals`` (1-based evaluation counts).  ``error`` is set when
    the backend failed -- the trace is still returned for bookkeeping.
    """

    problem_id: str
    backend: str
    seed: int
    best_so_far: list[float] = field(default_factory=list)
    evals: list[int] = field(default_factory=list)
    backend_history: list[str] = field(default_factory=list)
    evaluation_history: list[dict[str, Any]] = field(default_factory=list)
    wall_s: float = 0.0
    error: str | None = None


def _evaluate(problem: OptProblem, params: dict[str, Any]):
    """Evaluate the raw (minimization) objective at ``params``."""
    return problem.evaluate(params)


def run_cell(
    problem: OptProblem,
    backend_name: str,
    seed: int,
    budget: int,
    *,
    batch: int = 1,
    n_init: int = DEFAULT_INIT,
) -> RunTrace:
    """Run a single (problem, backend, seed) cell to ``budget`` evaluations."""
    space: ParameterSpace = problem.space
    best_so_far: list[float] = []
    evals: list[int] = []
    observations: list[Observation] = []
    backend_history: list[str] = []
    evaluation_history: list[dict[str, Any]] = []
    best: float | None = None
    t0 = time.perf_counter()

    def _record(params: dict[str, Any]) -> None:
        nonlocal best
        evaluation = _evaluate(problem, params)
        f = float(evaluation.raw_value)
        best = f if best is None else min(best, f)
        best_so_far.append(best)
        evals.append(len(best_so_far))
        # Backends maximize. Early-stage problems may expose an imperfect
        # observed/proxy value while true regret still uses raw_value.
        observations.append(
            Observation(
                params=dict(params),
                objective=-float(evaluation.optimizer_value),
                objectives=evaluation.observation_objectives(),
            )
        )
        evaluation_history.append(
            {
                "params": dict(params),
                "raw_value": f,
                "observed_value": float(evaluation.optimizer_value),
                "execution_success": evaluation.execution_success,
                "qc_passed": evaluation.qc_passed,
                "failure_type": evaluation.failure_type,
                "metadata": dict(evaluation.metadata),
            }
        )

    try:
        # 1. Initial random design.
        n_first = min(n_init, budget)
        for params in sample_random(space, n_first, seed=seed):
            _record(params)
        if n_first:
            backend_history.append("initial_random")

        # 2. Optimizer loop.
        backend = get_backend(backend_name)
        round_seed = seed
        while len(best_so_far) < budget:
            round_seed += 1
            take = min(batch, budget - len(best_so_far))
            candidates = backend.suggest(
                space, take, observations, seed=round_seed
            )
            backend_history.append(
                str(getattr(backend, "last_selected_backend", backend_name))
            )
            if not candidates:
                logger.warning(
                    "backend %s returned no candidates; stopping cell early",
                    backend_name,
                )
                break
            for params in candidates[:take]:
                _record(params)
    except Exception as exc:  # noqa: BLE001 -- isolate one bad backend
        logger.exception(
            "study cell failed: problem=%s backend=%s seed=%s",
            problem.id,
            backend_name,
            seed,
        )
        return RunTrace(
            problem_id=problem.id,
            backend=backend_name,
            seed=seed,
            best_so_far=best_so_far,
            evals=evals,
            backend_history=backend_history,
            evaluation_history=evaluation_history,
            wall_s=time.perf_counter() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )

    return RunTrace(
        problem_id=problem.id,
        backend=backend_name,
        seed=seed,
        best_so_far=best_so_far,
        evals=evals,
        backend_history=backend_history,
        evaluation_history=evaluation_history,
        wall_s=time.perf_counter() - t0,
    )


def run_study(
    problems: list[OptProblem],
    backends: list[str],
    seeds: list[int],
    budget: int,
    *,
    batch: int = 1,
    n_init: int = DEFAULT_INIT,
) -> list[RunTrace]:
    """Run the full cross-product of problems x backends x seeds."""
    traces: list[RunTrace] = []
    for problem in problems:
        for backend_name in backends:
            for seed in seeds:
                traces.append(
                    run_cell(
                        problem,
                        backend_name,
                        seed,
                        budget,
                        batch=batch,
                        n_init=n_init,
                    )
                )
    return traces
