"""Aggregate RunTraces into a (problem x method) score table.

Collapses the per-seed traces of a study into one :class:`MethodScore` per
(problem, backend), then exposes the table for the recommender / reporter.
Metrics come from :mod:`benchmarks.methods.metrics`.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

from benchmarks.methods.metrics import (
    convergence_auc,
    cost,
    evals_to_target,
    noise_robustness,
    simple_regret,
)
from benchmarks.methods.problems import ProblemTags, get_problem
from benchmarks.methods.runner import RunTrace

DEFAULT_TOL = 1e-2


@dataclass(frozen=True)
class MethodScore:
    """Aggregated performance of one backend on one problem across seeds."""

    problem_id: str
    backend: str
    n_seeds: int
    mean_regret: float
    mean_auc: float
    mean_evals_to_target: float | None  # None if target never reached
    target_hit_rate: float              # fraction of seeds that hit target
    regret_robustness: float            # spread of final regret (lower=better)
    mean_cost_s: float                  # mean per-eval wall-clock
    errors: int                         # seeds that raised
    tags: ProblemTags


def build_scoreboard(
    traces: list[RunTrace],
    *,
    tol: float = DEFAULT_TOL,
) -> list[MethodScore]:
    """Aggregate a study's traces into one MethodScore per (problem, backend)."""
    grouped: dict[tuple[str, str], list[RunTrace]] = defaultdict(list)
    for t in traces:
        grouped[(t.problem_id, t.backend)].append(t)

    scores: list[MethodScore] = []
    for (problem_id, backend), cell_traces in grouped.items():
        problem = get_problem(problem_id)
        optimum = problem.optimum

        final_regrets: list[float] = []
        aucs: list[float] = []
        etts: list[int] = []
        costs: list[float] = []
        errors = 0

        for t in cell_traces:
            if t.error is not None:
                errors += 1
            if not t.best_so_far:
                continue
            final_regrets.append(simple_regret(t.best_so_far, optimum))
            aucs.append(convergence_auc(t.best_so_far, optimum))
            ett = evals_to_target(t.best_so_far, optimum, tol)
            if ett is not None:
                etts.append(ett)
            costs.append(cost(t.wall_s, len(t.best_so_far)))

        n_seeds = len(cell_traces)
        scores.append(
            MethodScore(
                problem_id=problem_id,
                backend=backend,
                n_seeds=n_seeds,
                mean_regret=statistics.fmean(final_regrets)
                if final_regrets
                else float("inf"),
                mean_auc=statistics.fmean(aucs) if aucs else float("inf"),
                mean_evals_to_target=statistics.fmean(etts) if etts else None,
                target_hit_rate=len(etts) / n_seeds if n_seeds else 0.0,
                regret_robustness=noise_robustness(final_regrets),
                mean_cost_s=statistics.fmean(costs) if costs else 0.0,
                errors=errors,
                tags=problem.tags,
            )
        )

    scores.sort(key=lambda s: (s.problem_id, s.mean_regret))
    return scores
