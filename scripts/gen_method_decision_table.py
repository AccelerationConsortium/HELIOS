"""Generate the live method-advisor decision table from a benchmark study.

Runs every backend on the analytic problems in ``benchmarks/methods``, scores
them per structural bucket, and writes the ranked decision table to
``app/optimization/method_decision_table.json`` -- which the live method advisor
(:mod:`app.optimization.method_advisor`) loads to bias backend selection.

Re-run after adding problems/backends or changing metrics.  Buckets with no
benchmark problem are simply absent (the advisor falls back to its expert prior).

Usage:
    python scripts/gen_method_decision_table.py
"""
from __future__ import annotations

import json
import os

# Importing registers the classical DoE backends for the study.
import benchmarks.methods.doe_backend  # noqa: F401
from benchmarks.methods.problems import get_problems
from benchmarks.methods.recommend import recommend
from benchmarks.methods.runner import run_study
from benchmarks.methods.scoreboard import build_scoreboard

BACKENDS = [
    "lhs",
    "random_sampling",
    "built_in",
    "optuna_tpe",
    "optuna_cmaes",
    "scipy_de",
    "pymoo_nsga2",
    "bomcp",
    "full_factorial",
]
SEEDS = [0, 1]
BUDGET = 24
BATCH = 4

OUT = os.path.join("app", "optimization", "method_decision_table.json")


def main() -> None:
    problems = get_problems()
    print(f"Running study: {len(problems)} problems x {len(BACKENDS)} backends "
          f"x {len(SEEDS)} seeds, budget={BUDGET}, batch={BATCH}")
    traces = run_study(problems, BACKENDS, SEEDS, BUDGET, batch=BATCH, n_init=4)
    scores = build_scoreboard(traces)
    recs = recommend(scores)

    entries = [
        {
            "dims": r.n_dims_class,
            "modality": r.modality,
            "noise": r.noise,
            "methods": [rm.backend for rm in r.ranked],
            "_best_regret": round(r.ranked[0].mean_regret, 4) if r.ranked else None,
        }
        for r in recs
        if r.ranked
    ]
    with open(OUT, "w") as fh:
        json.dump(entries, fh, indent=2)

    print(f"Wrote {len(entries)} bucket(s) to {OUT}:")
    for e in entries:
        print(f"  ({e['dims']},{e['modality']},{e['noise']}): {e['methods']}")


if __name__ == "__main__":
    main()
