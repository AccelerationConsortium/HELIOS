"""CLI for the analytic BO/method benchmark suite.

Without a subcommand this runs the classic study (unchanged behavior).
``pathology-study`` runs the evaluation-layer harness from a study YAML::

    python -m benchmarks.methods pathology-study --study-config study.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import app.optimization  # noqa: F401 - registers optional HELIOS backends
import benchmarks.methods.doe_backend  # noqa: F401 - registers DoE baselines
import benchmarks.methods.helios_full  # noqa: F401 - registers helios_full
from app.services.optimization_backends import list_backends
from benchmarks.methods.problems import get_problem, get_problems
from benchmarks.methods.recommend import recommend
from benchmarks.methods.report import (
    family_summary,
    family_summary_to_csv,
    family_summary_to_markdown,
    recommendations_to_csv,
    recommendations_to_markdown,
    scoreboard_to_csv,
    scoreboard_to_markdown,
)
from benchmarks.methods.runner import RunTrace, run_study
from benchmarks.methods.scoreboard import MethodScore, build_scoreboard

DEFAULT_BACKENDS = (
    "helios_full",
    "gp_backend",
    "built_in",
    "scipy_de",
    "lhs",
    "random_sampling",
    "full_factorial",
    "fractional_factorial",
)


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_seeds(value: str) -> list[int]:
    if ".." in value:
        start_s, end_s = value.split("..", 1)
        start, end = int(start_s), int(end_s)
        return list(range(start, end + 1))
    return [int(part) for part in _parse_csv(value)]


def _available_default_backends() -> list[str]:
    available = list_backends()
    return [b for b in DEFAULT_BACKENDS if b == "helios_full" or available.get(b, False)]


def _trace_to_dict(trace: RunTrace) -> dict[str, Any]:
    return asdict(trace)


def _score_to_dict(score: MethodScore) -> dict[str, Any]:
    data = asdict(score)
    data["tags"] = asdict(score.tags)
    return data


def _dominance_summary(scores: list[MethodScore], reference: str) -> dict[str, Any]:
    by_problem: dict[str, dict[str, MethodScore]] = {}
    for score in scores:
        by_problem.setdefault(score.problem_id, {})[score.backend] = score

    rows: list[dict[str, Any]] = []
    for problem_id, methods in sorted(by_problem.items()):
        ref = methods.get(reference)
        if ref is None:
            continue
        for backend, score in sorted(methods.items()):
            if backend == reference:
                continue
            regret_delta = score.mean_regret - ref.mean_regret
            auc_delta = score.mean_auc - ref.mean_auc
            rows.append(
                {
                    "problem_id": problem_id,
                    "comparison": f"{reference}_vs_{backend}",
                    "reference_better_regret": regret_delta > 0,
                    "reference_better_auc": auc_delta > 0,
                    "mean_regret_delta_other_minus_reference": regret_delta,
                    "mean_auc_delta_other_minus_reference": auc_delta,
                }
            )

    baselines = sorted({r["comparison"].removeprefix(f"{reference}_vs_") for r in rows})
    aggregate: dict[str, Any] = {}
    for baseline in baselines:
        subset = [r for r in rows if r["comparison"] == f"{reference}_vs_{baseline}"]
        aggregate[baseline] = {
            "n_problems": len(subset),
            "regret_wins": sum(1 for r in subset if r["reference_better_regret"]),
            "auc_wins": sum(1 for r in subset if r["reference_better_auc"]),
            "mean_regret_delta": (
                sum(float(r["mean_regret_delta_other_minus_reference"]) for r in subset)
                / len(subset)
                if subset
                else 0.0
            ),
            "mean_auc_delta": (
                sum(float(r["mean_auc_delta_other_minus_reference"]) for r in subset)
                / len(subset)
                if subset
                else 0.0
            ),
        }
    return {"reference": reference, "by_baseline": aggregate, "per_problem": rows}


def _write_outputs(
    output_dir: Path,
    traces: list[RunTrace],
    scores: list[MethodScore],
    reference: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    recs = recommend(scores)
    summary = _dominance_summary(scores, reference)
    available = list_backends()

    (output_dir / "traces.json").write_text(
        json.dumps([_trace_to_dict(t) for t in traces], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "scoreboard.json").write_text(
        json.dumps([_score_to_dict(s) for s in scores], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "scoreboard.csv").write_text(scoreboard_to_csv(scores), encoding="utf-8")
    (output_dir / "scoreboard.md").write_text(scoreboard_to_markdown(scores), encoding="utf-8")
    (output_dir / "family_summary.csv").write_text(family_summary_to_csv(scores), encoding="utf-8")
    (output_dir / "family_summary.md").write_text(family_summary_to_markdown(scores), encoding="utf-8")
    (output_dir / "family_summary.json").write_text(
        json.dumps(family_summary(scores), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "recommendations.csv").write_text(recommendations_to_csv(recs), encoding="utf-8")
    (output_dir / "recommendations.md").write_text(recommendations_to_markdown(recs), encoding="utf-8")
    (output_dir / "dominance_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "available_backends.json").write_text(
        json.dumps(available, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _pathology_study_main(argv: list[str]) -> None:
    """Thin CLI over the study YAML: config is the interface, not flags."""
    from benchmarks.methods.ablation import (
        load_metrics_config,
        load_study_config,
        run_matrix,
        write_study_artifacts,
    )

    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.methods pathology-study",
        description="Run a pathology/ablation study from a study YAML",
    )
    parser.add_argument("--study-config", required=True, help="Study YAML path")
    parser.add_argument("--seeds", default=None, help="Override: seeds or A..B range")
    parser.add_argument("--budget", type=int, default=None, help="Override: evals per cell")
    parser.add_argument(
        "--output-dir",
        default="benchmark_results/pathology_study",
        help="Study output root (a <study_id>/ dir is created inside)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing study directory (never silent)",
    )
    args = parser.parse_args(argv)

    config = load_study_config(args.study_config)
    matrix = config.matrix
    if args.seeds is not None:
        matrix = replace(matrix, seeds=tuple(_parse_seeds(args.seeds)))
    budget = args.budget if args.budget is not None else config.budget

    metrics_config = load_metrics_config(config.metrics_config_path)
    results = run_matrix(matrix, budget, batch=config.batch, n_init=config.n_init)
    study_dir = write_study_artifacts(
        args.output_dir,
        matrix,
        results,
        metrics_config,
        reference=config.reference,
        budget=budget,
        batch=config.batch,
        n_init=config.n_init,
        description=config.description,
        problem_groups=config.problem_groups,
        comparison_families=config.comparison_families,
        force=args.force,
    )
    failed = sum(1 for r in results if r.trace.error is not None)
    print(f"cells={len(results)} failed={failed} budget={budget}")
    print(f"study={study_dir}")


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "pathology-study":
        _pathology_study_main(argv[1:])
        return

    parser = argparse.ArgumentParser(description="Run HELIOS analytic BO benchmarks")
    parser.add_argument("--problems", default="all", help="Comma-separated problem ids, or all")
    parser.add_argument(
        "--backends",
        default="default",
        help="Comma-separated backend names, or default",
    )
    parser.add_argument("--seeds", default="0..4", help="Comma-separated seeds or inclusive A..B range")
    parser.add_argument("--budget", type=int, default=25, help="Evaluations per problem/backend/seed")
    parser.add_argument("--batch", type=int, default=1, help="Candidates per optimizer round")
    parser.add_argument("--n-init", type=int, default=3, help="Initial random design size")
    parser.add_argument("--tol", type=float, default=1e-2, help="Target regret tolerance")
    parser.add_argument("--reference", default="helios_full", help="Reference backend for dominance summary")
    parser.add_argument("--output-dir", default="benchmark_results/bo_methods", help="Output artifact directory")
    args = parser.parse_args()

    problems = get_problems() if args.problems == "all" else [get_problem(p) for p in _parse_csv(args.problems)]
    backends = _available_default_backends() if args.backends == "default" else _parse_csv(args.backends)
    seeds = _parse_seeds(args.seeds)

    traces = run_study(
        problems,
        backends,
        seeds,
        args.budget,
        batch=args.batch,
        n_init=args.n_init,
    )
    scores = build_scoreboard(traces, tol=args.tol)
    out = Path(args.output_dir)
    _write_outputs(out, traces, scores, args.reference)

    print(f"problems={len(problems)} backends={len(backends)} seeds={len(seeds)} budget={args.budget}")
    print(f"outputs={out}")
    print(json.dumps(_dominance_summary(scores, args.reference)["by_baseline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
