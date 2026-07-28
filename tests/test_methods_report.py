from __future__ import annotations

from benchmarks.methods.problems import ProblemTags
from benchmarks.methods.report import benchmark_family, family_summary
from benchmarks.methods.scoreboard import MethodScore


def _score(
    backend: str,
    regret: float,
    *,
    problem_id: str,
    tags: ProblemTags,
) -> MethodScore:
    return MethodScore(
        problem_id=problem_id,
        backend=backend,
        n_seeds=3,
        mean_regret=regret,
        mean_auc=regret * 10.0,
        mean_evals_to_target=None,
        target_hit_rate=0.0,
        regret_robustness=0.0,
        mean_cost_s=0.01,
        errors=0,
        tags=tags,
    )


def test_benchmark_family_separates_early_stage_from_clean_bo():
    early = _score(
        "helios_full",
        0.1,
        problem_id="early",
        tags=ProblemTags(2, False, 0.0, False, False, "early_stage_hardware"),
    )
    clean = _score(
        "gp_backend",
        0.1,
        problem_id="clean",
        tags=ProblemTags(2, False, 0.0, True, False, "convex"),
    )

    assert benchmark_family(early) == "early_stage_imperfect_data"
    assert benchmark_family(clean) == "clean_low_dim_bo"


def test_family_summary_aggregates_by_family_and_backend():
    scores = [
        _score(
            "helios_full",
            0.1,
            problem_id="early_a",
            tags=ProblemTags(2, False, 0.0, False, False, "early_stage_hardware"),
        ),
        _score(
            "helios_full",
            0.3,
            problem_id="early_b",
            tags=ProblemTags(3, True, 0.0, False, True, "early_stage_objective_uncertainty"),
        ),
        _score(
            "gp_backend",
            1.0,
            problem_id="early_a",
            tags=ProblemTags(2, False, 0.0, False, False, "early_stage_hardware"),
        ),
    ]

    rows = family_summary(scores)
    helios = [
        row
        for row in rows
        if row["family"] == "early_stage_imperfect_data"
        and row["backend"] == "helios_full"
    ][0]

    assert helios["n_problems"] == 2
    assert helios["mean_regret"] == 0.2
