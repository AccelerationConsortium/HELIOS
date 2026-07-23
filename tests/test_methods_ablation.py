"""Ablation harness: event-grounded metrics, matrix, artifacts, CLI."""
from __future__ import annotations

import dataclasses
import json

import pytest

from benchmarks.methods.ablation import (
    DEFAULT_METRICS_CONFIG_PATH,
    ExperimentMatrix,
    PathologyBundle,
    adaptation_lag,
    constraint_adaptation_latency,
    decision_quality,
    load_metrics_config,
    load_study_config,
    paired_bootstrap,
    recovery_efficiency,
    run_matrix,
    study_id_for,
    write_study_artifacts,
)
from benchmarks.methods.pathologies import (
    NoiseDrift,
    PathologyEvent,
    SpatialFailure,
)
from benchmarks.methods.problems import get_problem

INSIDE = {"x1": 0.0, "x2": 0.0}  # normalized (0.5, 0.5)
OUTSIDE = {"x1": -5.0, "x2": -5.0}  # normalized (0.0, 0.0)
REGION = {"center": {"x1": 0.5, "x2": 0.5}, "radius": 0.2}


def _event(event_type: str, eval_index: int, region=None) -> PathologyEvent:
    return PathologyEvent(
        event_id=f"{event_type}-{eval_index:04d}",
        event_type=event_type,
        eval_index=eval_index,
        severity=1.0,
        duration=None,
        region=region,
        params={},
    )


# ---------------------------------------------------------------------------
# Metrics on synthetic traces with known answers
# ---------------------------------------------------------------------------


def test_recovery_efficiency_flatline_after_failure_is_near_zero():
    best = [10.0, 8.0, 6.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    result = recovery_efficiency(
        best, [_event("spatial_failure", 6)], optimum=0.0, window=5, epsilon=1e-6
    )
    # pre rate = (10-2)/5 = 1.6, post rate = 0 -> ratio ~ eps/1.6
    assert result.skipped == 0
    assert len(result.per_event) == 1
    assert result.mean_ratio < 1e-5


def test_recovery_efficiency_known_ratio():
    best = [10.0, 8.0, 6.0, 4.0, 2.0, 2.0, 1.6, 1.2, 0.8, 0.4, 0.0]
    result = recovery_efficiency(
        best, [_event("spatial_failure", 6)], optimum=0.0, window=5, epsilon=1e-6
    )
    # pre rate 1.6, post rate (2-0)/5 = 0.4 -> ratio 0.25
    assert result.mean_ratio == pytest.approx(0.25, rel=1e-3)


def test_recovery_efficiency_skips_events_too_close_to_boundary():
    best = [10.0, 9.0, 8.0, 7.0]
    result = recovery_efficiency(
        best, [_event("spatial_failure", 2)], optimum=0.0, window=5, epsilon=1e-6
    )
    assert result.per_event == []
    assert result.skipped == 1
    assert result.mean_ratio is None


def test_adaptation_lag_exact_recovery_point():
    raws = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0]
    result = adaptation_lag(
        raws,
        [_event("noise_drift", 4)],
        optimum=0.0,
        window=3,
        delta=0.1,
        sustain=2,
    )
    # regret_before = 1.0; threshold 1.1; post-onset windowed min recovers at
    # 0-based t=6 and holds -> lag = 6 - 3 = 3 evals.
    assert result.per_event == [3]
    assert result.mean_lag == pytest.approx(3.0)


def test_adaptation_lag_never_recovered_is_none():
    raws = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    result = adaptation_lag(
        raws,
        [_event("noise_drift", 4)],
        optimum=0.0,
        window=3,
        delta=0.1,
        sustain=2,
    )
    assert result.per_event == [None]
    assert result.mean_lag is None


def test_constraint_adaptation_latency_known_value():
    space = get_problem("sphere_2d").space
    params = (
        [INSIDE, OUTSIDE, INSIDE, OUTSIDE]  # pre-event rate 0.5
        + [INSIDE, INSIDE, INSIDE]  # keeps proposing into the bad region
        + [OUTSIDE] * 5  # adapts
    )
    result = constraint_adaptation_latency(
        params,
        [_event("spatial_failure", 5, region=REGION)],
        space,
        window=2,
        ratio_threshold=0.5,
        sustain=2,
    )
    # pre_rate 0.5, threshold 0.25; rolling rate drops below at 0-based t=8
    # sustained -> latency = 8 - 4 = 4.
    assert result.pre_rate == pytest.approx(0.5)
    assert result.latency == 4


def test_constraint_adaptation_latency_without_pre_exposure_is_none():
    space = get_problem("sphere_2d").space
    params = [OUTSIDE] * 4 + [INSIDE] + [OUTSIDE] * 5
    result = constraint_adaptation_latency(
        params,
        [_event("spatial_failure", 5, region=REGION)],
        space,
        window=2,
        ratio_threshold=0.5,
        sustain=2,
    )
    assert result.pre_rate == pytest.approx(0.0)
    assert result.latency is None


def test_decision_quality_detects_backend_switch():
    history = [
        "initial_random",
        "exploit:sample:gp_backend",
        "exploit:sample:gp_backend",
        "explore:probe:lhs",
        "explore:probe:lhs",
    ]
    result = decision_quality(
        history,
        [_event("spatial_failure", 5)],
        {"spatial_failure": ["backend_switch"]},
        window_evals=3,
        n_init=3,
        batch=1,
    )
    assert result.per_event == [True]
    assert result.hit_rate == 1.0


def test_decision_quality_miss_when_no_relevant_response():
    history = [
        "initial_random",
        "exploit:sample:gp_backend",
        "exploit:sample:gp_backend",
        "exploit:sample:gp_backend",
        "exploit:sample:gp_backend",
    ]
    result = decision_quality(
        history,
        [_event("spatial_failure", 5)],
        {"spatial_failure": ["backend_switch", "constraint_filter"]},
        window_evals=3,
        n_init=3,
        batch=1,
    )
    assert result.per_event == [False]
    assert result.hit_rate == 0.0


def test_decision_quality_detects_exploration_and_filter_classes():
    history = [
        "initial_random",
        "exploit:sample:gp_backend",
        "explore:probe:lhs",
        "exploit:sample:gp_backend:early_stage_filtered",
    ]
    exploration = decision_quality(
        history,
        [_event("noise_drift", 4)],
        {"noise_drift": ["exploration_action"]},
        window_evals=2,
        n_init=3,
        batch=1,
    )
    filtered = decision_quality(
        history,
        [_event("noise_drift", 5)],
        {"noise_drift": ["constraint_filter"]},
        window_evals=2,
        n_init=3,
        batch=1,
    )
    assert exploration.per_event == [True]
    assert filtered.per_event == [True]


def test_paired_bootstrap_constant_diff_excludes_zero():
    result = paired_bootstrap([2.0, 2.0, 2.0, 2.0], n_boot=200, seed=1)
    assert result.mean == pytest.approx(2.0)
    assert result.ci_low > 0.0


def test_paired_bootstrap_balanced_diffs_straddle_zero():
    result = paired_bootstrap([1.0, -1.0, 1.0, -1.0], n_boot=500, seed=1)
    assert result.ci_low < 0.0 < result.ci_high
    assert result.p_value > 0.05


# ---------------------------------------------------------------------------
# Matrix / config / identity
# ---------------------------------------------------------------------------


def _small_matrix() -> ExperimentMatrix:
    return ExperimentMatrix(
        problem_ids=("sphere_2d", "branin"),
        bundles=(
            PathologyBundle(id="clean", specs=()),
            PathologyBundle(
                id="failure",
                specs=(
                    SpatialFailure(
                        center={"x1": 0.5, "x2": 0.5},
                        radius=0.3,
                        p_max=0.8,
                        p_base=0.05,
                        observed_penalty=5.0,
                    ),
                ),
            ),
        ),
        config_names=("lhs", "random_sampling"),
        seeds=(0, 1),
    )


def test_matrix_expands_full_cross_product_with_stable_cell_ids():
    cells = _small_matrix().expand()
    assert len(cells) == 16
    assert cells[0].cell_id == "cell_0001"
    assert cells[-1].cell_id == "cell_0016"
    assert len({c.cell_id for c in cells}) == 16


def test_study_id_deterministic_and_sensitive():
    matrix = _small_matrix()
    a = study_id_for(matrix, metrics_config_hash="abc", git_commit="deadbeef")
    b = study_id_for(matrix, metrics_config_hash="abc", git_commit="deadbeef")
    c = study_id_for(matrix, metrics_config_hash="xyz", git_commit="deadbeef")
    d = study_id_for(matrix, metrics_config_hash="abc", git_commit="cafef00d")
    assert a == b
    assert len(a) == 12
    assert len({a, c, d}) == 3


def test_default_metrics_config_loads_and_covers_all_pathologies():
    config = load_metrics_config(DEFAULT_METRICS_CONFIG_PATH)
    assert config.schema_version == 1
    for key in (
        "epsilon",
        "recovery_window",
        "lag_window",
        "lag_delta",
        "lag_sustain",
        "constraint_window",
        "constraint_ratio_threshold",
        "constraint_sustain",
        "decision_window",
    ):
        assert key in config.thresholds
    for kind in (
        "noise_drift",
        "spatial_failure",
        "censoring",
        "proxy_gap_shift",
        "objective_shift",
    ):
        assert kind in config.response_class_mapping
    assert config.content_hash


def test_load_study_config_parses_bundles_and_seeds(tmp_path):
    study = tmp_path / "study.yaml"
    study.write_text(
        """
schema_version: 1
description: test study
problems: [sphere_2d]
pathologies:
  - id: drift
    specs:
      - type: noise_drift
        params: {start_eval: 2, rate: 0.5}
  - id: clean
    specs: []
configs: [lhs]
seeds: "0..2"
budget: 6
reference: lhs
""",
        encoding="utf-8",
    )
    config = load_study_config(study)
    assert config.matrix.problem_ids == ("sphere_2d",)
    assert [b.id for b in config.matrix.bundles] == ["drift", "clean"]
    assert config.matrix.bundles[0].specs == (NoiseDrift(start_eval=2, rate=0.5),)
    assert config.matrix.seeds == (0, 1, 2)
    assert config.budget == 6
    assert config.reference == "lhs"


# ---------------------------------------------------------------------------
# Harness + artifacts (real runner, cheap backends)
# ---------------------------------------------------------------------------


def _tiny_matrix() -> ExperimentMatrix:
    return ExperimentMatrix(
        problem_ids=("sphere_2d",),
        bundles=(
            PathologyBundle(id="clean", specs=()),
            PathologyBundle(
                id="failure",
                specs=(
                    SpatialFailure(
                        center={"x1": 0.5, "x2": 0.5},
                        radius=0.4,
                        p_max=0.9,
                        p_base=0.1,
                        observed_penalty=5.0,
                    ),
                ),
            ),
        ),
        config_names=("lhs",),
        seeds=(0, 1),
    )


def test_run_matrix_reuses_runner_and_is_deterministic():
    results_a = run_matrix(_tiny_matrix(), budget=6)
    results_b = run_matrix(_tiny_matrix(), budget=6)

    assert len(results_a) == 4
    assert all(r.trace.error is None for r in results_a)
    assert all(len(r.trace.best_so_far) == 6 for r in results_a)
    events_a = [[dataclasses.asdict(e) for e in r.events] for r in results_a]
    events_b = [[dataclasses.asdict(e) for e in r.events] for r in results_b]
    assert events_a == events_b
    # the failure bundle actually fired somewhere across its cells
    assert any(e for r in results_a if r.cell.bundle.id == "failure" for e in r.events)


def test_write_study_artifacts_round_trip(tmp_path):
    matrix = _tiny_matrix()
    results = run_matrix(matrix, budget=6)
    metrics_config = load_metrics_config(DEFAULT_METRICS_CONFIG_PATH)

    study_dir = write_study_artifacts(
        tmp_path,
        matrix,
        results,
        metrics_config,
        reference="lhs",
        budget=6,
    )

    manifest = json.loads((study_dir / "matrix.json").read_text())
    assert manifest["problem_ids"] == ["sphere_2d"]
    # full pathology params, never bare names
    failure_bundle = next(
        b for b in manifest["pathology_bundles"] if b["id"] == "failure"
    )
    assert failure_bundle["specs"][0]["params"]["radius"] == 0.4
    assert manifest["seed_list"] == [0, 1]
    assert "git_commit" in manifest
    assert "metrics_config_hash" in manifest
    assert manifest["metric_schema_version"] == 1

    metrics = json.loads((study_dir / "metrics.json").read_text())
    assert metrics["valid_cells"] == 4
    assert metrics["failed_cells"] == 0
    assert isinstance(metrics["skipped_metrics"], list)
    assert len(metrics["cells"]) == 4

    assert (study_dir / "traces" / "cell_0001.jsonl").exists()
    assert (study_dir / "events" / "cell_0001.events.jsonl").exists()
    assert (study_dir / "tables" / "main_benchmark.csv").exists()
    assert (study_dir / "tables" / "mechanism_analysis.csv").exists()
    assert (study_dir / "report.md").exists()


def test_ci_mini_matrix_end_to_end_with_helios(tmp_path):
    matrix = ExperimentMatrix(
        problem_ids=("sphere_2d", "branin"),
        bundles=(
            PathologyBundle(id="clean", specs=()),
            PathologyBundle(
                id="drift",
                specs=(NoiseDrift(start_eval=3, rate=1.0),),
            ),
        ),
        config_names=("helios_full", "lhs"),
        seeds=(0, 1),
    )
    results = run_matrix(matrix, budget=6)
    assert len(results) == 16
    assert all(r.trace.error is None for r in results)

    metrics_config = load_metrics_config(DEFAULT_METRICS_CONFIG_PATH)
    study_dir = write_study_artifacts(
        tmp_path, matrix, results, metrics_config, reference="helios_full", budget=6
    )
    metrics = json.loads((study_dir / "metrics.json").read_text())
    assert metrics["valid_cells"] == 16
    # decision quality is computed for HELIOS variants only
    for cell in metrics["cells"]:
        if not cell["config"].startswith("helios_full"):
            assert cell["metrics"].get("decision_quality") is None


# ---------------------------------------------------------------------------
# ObjectiveShift regret semantics: dynamic + epoch-local rulers
# ---------------------------------------------------------------------------


def test_dynamic_regret_auc_uses_per_eval_raw_values():
    from benchmarks.methods.ablation import dynamic_regret_auc

    # per-eval regret [2, 1, 0, 1] -> trapezoid 1.5 + 0.5 + 0.5
    assert dynamic_regret_auc([3.0, 2.0, 1.0, 2.0], optimum=1.0) == pytest.approx(2.5)


def test_epoch_metrics_split_at_objective_shift():
    from benchmarks.methods.ablation import epoch_metrics

    raws = [1.0, 0.5, 4.0, 3.0, 2.0]
    epochs = epoch_metrics(raws, [_event("objective_shift", 3)], optimum=0.0)

    assert len(epochs) == 2
    assert epochs[0]["start_eval"] == 1
    assert epochs[0]["end_eval"] == 2
    assert epochs[0]["simple_regret"] == pytest.approx(0.5)
    assert epochs[0]["regret_auc"] == pytest.approx(0.75)
    assert epochs[1]["start_eval"] == 3
    assert epochs[1]["end_eval"] == 5
    assert epochs[1]["simple_regret"] == pytest.approx(2.0)
    assert epochs[1]["regret_auc"] == pytest.approx(6.0)


def test_epoch_metrics_without_shift_is_single_epoch():
    from benchmarks.methods.ablation import epoch_metrics

    epochs = epoch_metrics([2.0, 1.0], [], optimum=0.0)
    assert len(epochs) == 1
    assert epochs[0]["simple_regret"] == pytest.approx(1.0)


def test_cell_metrics_include_epoch_semantics_for_objective_shift(tmp_path):
    from benchmarks.methods.pathologies import ObjectiveShift

    matrix = ExperimentMatrix(
        problem_ids=("sphere_2d",),
        bundles=(
            PathologyBundle(
                id="shift",
                specs=(ObjectiveShift(shift_eval=3, delta={"x1": 0.1}),),
            ),
        ),
        config_names=("lhs",),
        seeds=(0,),
    )
    results = run_matrix(matrix, budget=6)
    study_dir = write_study_artifacts(
        tmp_path,
        matrix,
        results,
        load_metrics_config(DEFAULT_METRICS_CONFIG_PATH),
        reference="lhs",
        budget=6,
    )
    cell = json.loads((study_dir / "metrics.json").read_text())["cells"][0]
    metrics = cell["metrics"]
    assert metrics["dynamic_regret_auc"] is not None
    assert metrics["final_epoch_regret"] is not None
    assert len(metrics["epochs"]) == 2  # shift at eval 4 splits the trace


# ---------------------------------------------------------------------------
# Study config: problem groups, comparison families, canonical v1 study
# ---------------------------------------------------------------------------


def test_load_study_config_supports_problem_groups_and_families(tmp_path):
    study = tmp_path / "study.yaml"
    study.write_text(
        """
schema_version: 1
problem_groups:
  clean_low_dim: [sphere_2d, branin]
  early_stage: [early_stage_controllability]
pathologies:
  - id: clean
    specs: []
configs: [helios_full, gp_backend, lhs]
seeds: "0..1"
budget: 6
reference: helios_full
comparison_families:
  common_performance:
    methods: [gp_backend, lhs]
  mechanism:
    contrasts:
      - [helios_full, gp_backend]
""",
        encoding="utf-8",
    )
    config = load_study_config(study)
    assert config.matrix.problem_ids == (
        "sphere_2d",
        "branin",
        "early_stage_controllability",
    )
    assert config.problem_groups == {
        "clean_low_dim": ("sphere_2d", "branin"),
        "early_stage": ("early_stage_controllability",),
    }
    assert "common_performance" in config.comparison_families


def test_comparisons_apply_holm_within_declared_families(tmp_path):
    matrix = ExperimentMatrix(
        problem_ids=("sphere_2d",),
        bundles=(PathologyBundle(id="clean", specs=()),),
        config_names=("lhs", "random_sampling", "full_factorial"),
        seeds=(0, 1, 2),
    )
    results = run_matrix(matrix, budget=6)
    families = {
        "family_a": {"methods": ["random_sampling"]},
        "family_b": {"methods": ["full_factorial"]},
    }
    study_dir = write_study_artifacts(
        tmp_path,
        matrix,
        results,
        load_metrics_config(DEFAULT_METRICS_CONFIG_PATH),
        reference="lhs",
        budget=6,
        comparison_families=families,
    )
    comparisons = json.loads((study_dir / "metrics.json").read_text())["comparisons"]
    by_family = {row["comparison"]: row["family"] for row in comparisons}
    assert by_family["lhs_vs_random_sampling"] == "family_a"
    assert by_family["lhs_vs_full_factorial"] == "family_b"
    # one comparison per family -> Holm within family degenerates to identity
    for row in comparisons:
        assert row["p_holm"] == pytest.approx(row["p_value"])


def test_canonical_v1_study_config_is_valid():
    from app.services.optimization_backends import list_backends

    config = load_study_config(
        "benchmarks/methods/configs/studies/pathology_study_v1.yaml"
    )
    for problem_id in config.matrix.problem_ids:
        get_problem(problem_id)  # raises on unknown ids
    registered = list_backends()
    for name in config.matrix.config_names:
        assert name in registered
    assert len(config.matrix.seeds) >= 20
    assert config.problem_groups  # tiers declared
    assert config.comparison_families  # preregistered contrasts


# ---------------------------------------------------------------------------
# Provenance + overwrite protection
# ---------------------------------------------------------------------------


def test_manifest_records_environment_provenance(tmp_path):
    matrix = ExperimentMatrix(
        problem_ids=("sphere_2d",),
        bundles=(PathologyBundle(id="clean", specs=()),),
        config_names=("lhs",),
        seeds=(0,),
    )
    results = run_matrix(matrix, budget=5)
    study_dir = write_study_artifacts(
        tmp_path,
        matrix,
        results,
        load_metrics_config(DEFAULT_METRICS_CONFIG_PATH),
        reference="lhs",
        budget=5,
    )
    manifest = json.loads((study_dir / "matrix.json").read_text())
    assert isinstance(manifest["git_dirty"], bool)
    assert manifest["git_dirty"] is False or manifest["git_diff_hash"]
    assert manifest["python_version"]
    assert manifest["dependency_lock_hash"]
    assert manifest["pathology_schema_version"] == 1
    assert manifest["metric_schema_version"] == 1


def test_write_study_artifacts_refuses_silent_overwrite(tmp_path):
    matrix = ExperimentMatrix(
        problem_ids=("sphere_2d",),
        bundles=(PathologyBundle(id="clean", specs=()),),
        config_names=("lhs",),
        seeds=(0,),
    )
    results = run_matrix(matrix, budget=5)
    metrics_config = load_metrics_config(DEFAULT_METRICS_CONFIG_PATH)

    write_study_artifacts(
        tmp_path, matrix, results, metrics_config, reference="lhs", budget=5
    )
    with pytest.raises(FileExistsError):
        write_study_artifacts(
            tmp_path, matrix, results, metrics_config, reference="lhs", budget=5
        )
    # explicit force overwrites
    write_study_artifacts(
        tmp_path, matrix, results, metrics_config, reference="lhs", budget=5, force=True
    )


def test_pathology_study_cli(tmp_path, monkeypatch):
    study = tmp_path / "study.yaml"
    study.write_text(
        """
schema_version: 1
problems: [sphere_2d]
pathologies:
  - id: clean
    specs: []
configs: [lhs]
seeds: "0..1"
budget: 5
reference: lhs
""",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmarks.methods",
            "pathology-study",
            "--study-config",
            str(study),
            "--output-dir",
            str(out_dir),
        ],
    )
    from benchmarks.methods.__main__ import main

    main()

    studies = list(out_dir.iterdir())
    assert len(studies) == 1
    assert (studies[0] / "matrix.json").exists()
    assert (studies[0] / "report.md").exists()
