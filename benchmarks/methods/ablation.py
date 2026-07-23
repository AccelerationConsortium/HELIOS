"""Ablation harness: matrix runner, event-grounded metrics, study artifacts.

Runs ``problems x pathology bundles x backend configs x seeds`` through the
**unmodified** :func:`benchmarks.methods.runner.run_cell`, grounds every
adaptation metric in the pathology event ledger, and writes a reproducible
study directory (``matrix.json`` is the single source of truth).

Two report surfaces keep the narrative honest:

- ``tables/main_benchmark.csv`` -- all methods, standard raw-value metrics
  (final regret, regret AUC, evals-to-target) under each pathology.
- ``tables/mechanism_analysis.csv`` -- HELIOS variants only: decision quality,
  recovery efficiency, adaptation lag, constraint adaptation latency.

Metric thresholds and the event-type -> response-class mapping come from a
versioned YAML config (``configs/pathology_metrics.yaml``); nothing is
hardcoded.  Design: docs/plans/2026-07-22-pathology-benchmark-evaluation-layer-design.md.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import math
import platform
import random
import statistics
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

import benchmarks.methods.helios_full  # noqa: F401 - registers helios_full + variants
from app.services.candidate_gen import ParameterSpace
from benchmarks.methods.helios_full import BACKEND_VARIANTS
from benchmarks.methods.metrics import convergence_auc, evals_to_target, simple_regret
from benchmarks.methods.pathologies import (
    PATHOLOGY_SCHEMA_VERSION,
    PathologyEvent,
    PathologySpec,
    apply_pathology,
    in_failure_region,
    spec_from_dict,
    spec_to_dict,
)
from benchmarks.methods.problems import OptProblem, get_problem, get_problems
from benchmarks.methods.runner import DEFAULT_INIT, RunTrace, run_cell
from benchmarks.methods.scoreboard import DEFAULT_TOL

DEFAULT_METRICS_CONFIG_PATH = Path(__file__).parent / "configs" / "pathology_metrics.yaml"

# Event types measured by adaptation lag (onset-style, not per-eval failures).
DRIFT_EVENT_TYPES = ("noise_drift", "proxy_gap_shift", "objective_shift", "censoring")
FAILURE_EVENT_TYPE = "spatial_failure"


# ---------------------------------------------------------------------------
# Metrics config (versioned YAML; part of the study identity)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricsConfig:
    schema_version: int
    thresholds: dict[str, float]
    response_class_mapping: dict[str, list[str]]
    content_hash: str
    raw: dict[str, Any]


def load_metrics_config(path: str | Path = DEFAULT_METRICS_CONFIG_PATH) -> MetricsConfig:
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return MetricsConfig(
        schema_version=int(data["schema_version"]),
        thresholds={k: float(v) for k, v in dict(data["thresholds"]).items()},
        response_class_mapping={
            str(k): [str(c) for c in v]
            for k, v in dict(data["response_class_mapping"]).items()
        },
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        raw=data,
    )


# ---------------------------------------------------------------------------
# Event-grounded metrics (pure functions; all thresholds injected)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryResult:
    mean_ratio: float | None
    per_event: list[float]
    skipped: int


def recovery_efficiency(
    best_so_far: list[float],
    events: list[PathologyEvent],
    optimum: float,
    *,
    window: int,
    epsilon: float,
) -> RecoveryResult:
    """Post/pre regret-improvement-rate ratio around each failure event.

    Rates use best-so-far **true** regret over ``window`` evals on each side;
    events without a full window on both sides are skipped (and counted).
    The per-event ratios aggregate as a geometric mean.
    """
    regret = [float(b) - float(optimum) for b in best_so_far]
    window = int(window)
    per_event: list[float] = []
    skipped = 0
    for event in events:
        if event.event_type != FAILURE_EVENT_TYPE:
            continue
        i = event.eval_index - 1
        if i - window < 0 or i + window >= len(regret):
            skipped += 1
            continue
        pre_rate = (regret[i - window] - regret[i]) / window
        post_rate = (regret[i] - regret[i + window]) / window
        per_event.append((post_rate + epsilon) / (pre_rate + epsilon))
    mean = (
        math.exp(statistics.fmean(math.log(r) for r in per_event))
        if per_event
        else None
    )
    return RecoveryResult(mean_ratio=mean, per_event=per_event, skipped=skipped)


@dataclass(frozen=True)
class LagResult:
    mean_lag: float | None
    per_event: list[int | None]
    skipped: int


def adaptation_lag(
    raw_values: list[float],
    events: list[PathologyEvent],
    optimum: float,
    *,
    window: int,
    delta: float,
    sustain: int,
) -> LagResult:
    """Evals from a drift/shift onset until performance recovers (change-point).

    ``regret_now(t)`` is the windowed min of post-onset raw values (the window
    never crosses the event boundary, so pre-event evals cannot fake a
    recovery).  Recovered when ``regret_now <= regret_before + delta *
    max(|regret_before|, 1)`` holds for ``sustain`` consecutive evals; the lag
    is counted to the first eval of that sustained run.  ``None`` = never
    recovered within the trace.
    """
    raws = [float(v) for v in raw_values]
    window, sustain = int(window), int(sustain)
    per_event: list[int | None] = []
    skipped = 0
    for event in events:
        if event.event_type not in DRIFT_EVENT_TYPES:
            continue
        onset = event.eval_index - 1
        pre = raws[max(0, onset - window):onset]
        if not pre:
            skipped += 1
            continue
        regret_before = min(pre) - float(optimum)
        threshold = regret_before + float(delta) * max(abs(regret_before), 1.0)
        lag: int | None = None
        run = 0
        for t in range(onset, len(raws)):
            lo = max(onset, t - window + 1)
            regret_now = min(raws[lo : t + 1]) - float(optimum)
            if regret_now <= threshold:
                run += 1
                if run >= sustain:
                    lag = (t - sustain + 1) - onset
                    break
            else:
                run = 0
        per_event.append(lag)
    recovered = [lag for lag in per_event if lag is not None]
    mean = statistics.fmean(recovered) if recovered else None
    return LagResult(mean_lag=mean, per_event=per_event, skipped=skipped)


@dataclass(frozen=True)
class LatencyResult:
    latency: int | None
    pre_rate: float | None


def constraint_adaptation_latency(
    params_list: list[dict[str, Any]],
    events: list[PathologyEvent],
    space: ParameterSpace,
    *,
    window: int,
    ratio_threshold: float,
    sustain: int,
) -> LatencyResult:
    """Evals until proposals leave the ledgered failure region.

    Baseline is the pre-event in-region proposal rate; adapted when the
    rolling in-region rate stays below ``ratio_threshold * pre_rate`` for
    ``sustain`` consecutive evals.  ``pre_rate == 0`` means there was nothing
    to adapt away from -> latency is undefined (``None``).
    """
    event = next(
        (e for e in events if e.event_type == FAILURE_EVENT_TYPE and e.region),
        None,
    )
    if event is None:
        return LatencyResult(latency=None, pre_rate=None)
    window, sustain = int(window), int(sustain)
    onset = event.eval_index - 1
    inside = [
        1.0 if in_failure_region(p, space, event.region) else 0.0
        for p in params_list
    ]
    pre = inside[:onset]
    pre_rate = statistics.fmean(pre) if pre else None
    if not pre or pre_rate == 0.0:
        return LatencyResult(latency=None, pre_rate=pre_rate)
    threshold = float(ratio_threshold) * pre_rate
    run = 0
    for t in range(onset, len(inside)):
        rate = statistics.fmean(inside[max(0, t - window + 1) : t + 1])
        if rate < threshold:
            run += 1
            if run >= sustain:
                return LatencyResult(latency=(t - sustain + 1) - onset, pre_rate=pre_rate)
        else:
            run = 0
    return LatencyResult(latency=None, pre_rate=pre_rate)


def dynamic_regret_auc(raw_values: list[float], optimum: float) -> float:
    """Trapezoidal AUC of *per-evaluation* true regret.

    Unlike best-so-far AUC this ruler stays valid across ObjectiveShift
    events: each raw value is the current landscape's value at the evaluated
    point, so no stale pre-shift incumbent can carry over.
    """
    return convergence_auc([float(v) for v in raw_values], optimum)


def epoch_metrics(
    raw_values: list[float],
    events: list[PathologyEvent],
    optimum: float,
) -> list[dict[str, Any]]:
    """Epoch-local simple regret / AUC, with epochs split at objective shifts.

    Each ``objective_shift`` event starts a new epoch at its (1-based) onset
    eval; incumbents never cross an epoch boundary, so a pre-shift best can
    never mask post-shift performance.  Without shift events the whole trace
    is one epoch (equivalent to the standard ruler).
    """
    raws = [float(v) for v in raw_values]
    if not raws:
        return []
    onsets = sorted(
        {
            e.eval_index - 1
            for e in events
            if e.event_type == "objective_shift" and 0 < e.eval_index - 1 < len(raws)
        }
    )
    boundaries = [0, *onsets, len(raws)]
    epochs: list[dict[str, Any]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        segment = raws[start:end]
        epochs.append(
            {
                "start_eval": start + 1,
                "end_eval": end,
                "simple_regret": min(segment) - float(optimum),
                "regret_auc": convergence_auc(segment, optimum),
            }
        )
    return epochs


@dataclass(frozen=True)
class QualityResult:
    hit_rate: float | None
    per_event: list[bool]


def _round_token(entry: str) -> str:
    """Terminal backend token of a backend_history entry."""
    core = entry.removesuffix(":early_stage_filtered")
    return core.split(":")[-1]


def _round_classes(history: list[str], index: int) -> set[str]:
    """Observable response classes exhibited by round ``index`` (>= 1)."""
    entry = history[index]
    classes: set[str] = set()
    if "early_stage_filtered" in entry:
        classes.add("constraint_filter")
    phase = entry.split(":")[0]
    token = _round_token(entry)
    if "explor" in phase or "lhs" in token or "random" in token:
        classes.add("exploration_action")
    if index >= 2 and _round_token(history[index - 1]) != token:
        classes.add("backend_switch")
    return classes


def decision_quality(
    backend_history: list[str],
    events: list[PathologyEvent],
    response_class_mapping: dict[str, list[str]],
    *,
    window_evals: int,
    n_init: int,
    batch: int,
) -> QualityResult:
    """Did the agent exhibit a relevant response class after each event?

    Response classes are *interpretable reaction categories, not optimal
    policies* (see configs/pathology_metrics.yaml).  Only meaningful for
    backends that expose decision provenance via ``backend_history`` --
    the harness computes it for HELIOS variants only.
    """
    window_evals, n_init, batch = int(window_evals), int(n_init), max(1, int(batch))
    per_event: list[bool] = []
    for event in events:
        relevant = set(response_class_mapping.get(event.event_type, []))
        if not relevant:
            continue
        hit = False
        for index in range(1, len(backend_history)):
            first_eval = n_init + (index - 1) * batch + 1
            last_eval = n_init + index * batch
            if last_eval < event.eval_index:
                continue
            if first_eval > event.eval_index + window_evals:
                break
            if relevant & _round_classes(backend_history, index):
                hit = True
                break
        per_event.append(hit)
    hit_rate = statistics.fmean(per_event) if per_event else None
    return QualityResult(hit_rate=hit_rate, per_event=per_event)


# ---------------------------------------------------------------------------
# Paired bootstrap + Holm correction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float
    p_value: float


def paired_bootstrap(
    diffs: list[float],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
) -> BootstrapResult:
    """Bootstrap CI + sign p-value for seed-paired metric differences."""
    values = [float(d) for d in diffs]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        statistics.fmean(rng.choices(values, k=n)) for _ in range(int(n_boot))
    )
    lo_idx = int(((1.0 - ci) / 2.0) * len(means))
    hi_idx = min(len(means) - 1, int(((1.0 + ci) / 2.0) * len(means)))
    non_positive = sum(1 for m in means if m <= 0.0) / len(means)
    non_negative = sum(1 for m in means if m >= 0.0) / len(means)
    return BootstrapResult(
        mean=statistics.fmean(values),
        ci_low=means[lo_idx],
        ci_high=means[hi_idx],
        p_value=min(1.0, 2.0 * min(non_positive, non_negative)),
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment, preserving input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathologyBundle:
    """A named, fully parameterized pathology composition ('clean' = no specs)."""

    id: str
    specs: tuple[PathologySpec, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "specs": [spec_to_dict(s) for s in self.specs]}


@dataclass(frozen=True)
class ExperimentCell:
    cell_id: str
    problem_id: str
    bundle: PathologyBundle
    config: str
    seed: int


@dataclass(frozen=True)
class ExperimentMatrix:
    """Pure data: the full study cross-product. Execution stays in runner.py."""

    problem_ids: tuple[str, ...]
    bundles: tuple[PathologyBundle, ...]
    config_names: tuple[str, ...]
    seeds: tuple[int, ...]

    def expand(self) -> list[ExperimentCell]:
        cells: list[ExperimentCell] = []
        for problem_id in self.problem_ids:
            for bundle in self.bundles:
                for config in self.config_names:
                    for seed in self.seeds:
                        cells.append(
                            ExperimentCell(
                                cell_id=f"cell_{len(cells) + 1:04d}",
                                problem_id=problem_id,
                                bundle=bundle,
                                config=config,
                                seed=seed,
                            )
                        )
        return cells

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_ids": list(self.problem_ids),
            "pathology_bundles": [b.to_dict() for b in self.bundles],
            "config_names": list(self.config_names),
            "seed_list": list(self.seeds),
        }


def _pathology_seed(bundle_id: str, cell_seed: int) -> int:
    digest = hashlib.sha256(f"{bundle_id}:{cell_seed}".encode()).hexdigest()
    return int(digest[:8], 16)


@dataclass(frozen=True)
class CellResult:
    cell: ExperimentCell
    trace: RunTrace
    events: list[PathologyEvent]


def run_matrix(
    matrix: ExperimentMatrix,
    budget: int,
    *,
    batch: int = 1,
    n_init: int = DEFAULT_INIT,
) -> list[CellResult]:
    """Expand the matrix and run every cell through the unmodified runner."""
    results: list[CellResult] = []
    for cell in matrix.expand():
        base = get_problem(cell.problem_id)
        problem: OptProblem = base
        events: list[PathologyEvent] = []
        if cell.bundle.specs:
            problem, state = apply_pathology(
                base,
                cell.bundle.specs,
                seed=_pathology_seed(cell.bundle.id, cell.seed),
                bundle_id=cell.bundle.id,
            )
        else:
            state = None
        trace = run_cell(problem, cell.config, cell.seed, budget, batch=batch, n_init=n_init)
        if state is not None:
            events = list(state.events)
        results.append(CellResult(cell=cell, trace=trace, events=events))
    return results


# ---------------------------------------------------------------------------
# Study identity + config loading
# ---------------------------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout if out.returncode == 0 else None
    except OSError:
        return None


def _git_commit() -> str:
    commit = (_git("rev-parse", "HEAD") or "").strip()
    return commit or "unknown"


def _environment_provenance() -> dict[str, Any]:
    """Reproducibility facts beyond the commit: a commit alone is not enough
    for a dirty tree, and the same code under different dependencies is a
    different experiment."""
    status = _git("status", "--porcelain")
    dirty = bool(status.strip()) if status is not None else False
    diff_hash = None
    if dirty:
        diff = _git("diff", "HEAD") or ""
        diff_hash = hashlib.sha256(diff.encode("utf-8")).hexdigest()
    repo_root = Path(__file__).resolve().parents[2]
    lock = repo_root / "uv.lock"
    dep_source = lock if lock.exists() else repo_root / "pyproject.toml"
    try:
        dep_hash = hashlib.sha256(dep_source.read_bytes()).hexdigest()
    except OSError:
        dep_hash = "unknown"
    return {
        "git_dirty": dirty,
        "git_diff_hash": diff_hash,
        "python_version": platform.python_version(),
        "dependency_lock_hash": dep_hash,
        "dependency_lock_source": dep_source.name,
        "pathology_schema_version": PATHOLOGY_SCHEMA_VERSION,
    }


def study_id_for(
    matrix: ExperimentMatrix,
    *,
    metrics_config_hash: str,
    git_commit: str,
) -> str:
    """Content-addressed study identity.

    Includes the metric definitions (config hash + schema) and the code
    version, so changing either can never silently collide with old results.
    """
    payload = json.dumps(
        {
            "matrix": matrix.to_dict(),
            "metrics_config_hash": metrics_config_hash,
            "git_commit": git_commit,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class StudyConfig:
    matrix: ExperimentMatrix
    budget: int
    batch: int
    n_init: int
    reference: str
    metrics_config_path: Path
    description: str = ""
    problem_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    comparison_families: dict[str, dict[str, Any]] = field(default_factory=dict)


def _parse_seed_field(value: Any) -> tuple[int, ...]:
    if isinstance(value, str) and ".." in value:
        start_s, end_s = value.split("..", 1)
        return tuple(range(int(start_s), int(end_s) + 1))
    if isinstance(value, list | tuple):
        return tuple(int(v) for v in value)
    return (int(value),)


def load_study_config(path: str | Path) -> StudyConfig:
    """Parse a study YAML: the CLI stays thin, the config is the interface."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    problem_groups: dict[str, tuple[str, ...]] = {
        str(name): tuple(str(p) for p in members)
        for name, members in dict(data.get("problem_groups", {})).items()
    }
    if problem_groups:
        # ordered flatten with dedup; groups are recorded for tiered reporting
        seen: dict[str, None] = {}
        for members in problem_groups.values():
            for problem_id in members:
                seen.setdefault(problem_id, None)
        problem_ids = tuple(seen)
    else:
        problems = data.get("problems", "all")
        problem_ids = (
            tuple(p.id for p in get_problems())
            if problems == "all"
            else tuple(str(p) for p in problems)
        )
    bundles = tuple(
        PathologyBundle(
            id=str(entry["id"]),
            specs=tuple(spec_from_dict(s) for s in entry.get("specs", [])),
        )
        for entry in data.get("pathologies", [{"id": "clean", "specs": []}])
    )
    matrix = ExperimentMatrix(
        problem_ids=problem_ids,
        bundles=bundles,
        config_names=tuple(str(c) for c in data["configs"]),
        seeds=_parse_seed_field(data.get("seeds", "0..4")),
    )
    return StudyConfig(
        matrix=matrix,
        budget=int(data.get("budget", 25)),
        batch=int(data.get("batch", 1)),
        n_init=int(data.get("n_init", DEFAULT_INIT)),
        reference=str(data.get("reference", "helios_full")),
        metrics_config_path=Path(
            data.get("metrics_config", DEFAULT_METRICS_CONFIG_PATH)
        ),
        description=str(data.get("description", "")),
        problem_groups=problem_groups,
        comparison_families={
            str(k): dict(v)
            for k, v in dict(data.get("comparison_families", {})).items()
        },
    )


# ---------------------------------------------------------------------------
# Per-cell metrics + aggregation
# ---------------------------------------------------------------------------


def _compute_cell_metrics(
    result: CellResult,
    metrics_config: MetricsConfig,
    *,
    n_init: int,
    batch: int,
    tol: float = DEFAULT_TOL,
) -> dict[str, Any]:
    base = get_problem(result.cell.problem_id)
    optimum = base.optimum
    trace = result.trace
    t = metrics_config.thresholds

    best = list(trace.best_so_far)
    raws = [float(h["raw_value"]) for h in trace.evaluation_history]
    params = [dict(h["params"]) for h in trace.evaluation_history]

    recovery = recovery_efficiency(
        best,
        result.events,
        optimum,
        window=int(t["recovery_window"]),
        epsilon=t["epsilon"],
    )
    lag = adaptation_lag(
        raws,
        result.events,
        optimum,
        window=int(t["lag_window"]),
        delta=t["lag_delta"],
        sustain=int(t["lag_sustain"]),
    )
    latency = constraint_adaptation_latency(
        params,
        result.events,
        base.space,
        window=int(t["constraint_window"]),
        ratio_threshold=t["constraint_ratio_threshold"],
        sustain=int(t["constraint_sustain"]),
    )
    quality = None
    if result.cell.config in BACKEND_VARIANTS:
        quality_result = decision_quality(
            list(trace.backend_history),
            result.events,
            metrics_config.response_class_mapping,
            window_evals=int(t["decision_window"]),
            n_init=n_init,
            batch=batch,
        )
        quality = dataclasses.asdict(quality_result)

    epochs = epoch_metrics(raws, result.events, optimum)
    return {
        "final_regret": simple_regret(best, optimum) if best else None,
        "regret_auc": convergence_auc(best, optimum) if best else None,
        "evals_to_target": evals_to_target(best, optimum, tol) if best else None,
        "dynamic_regret_auc": dynamic_regret_auc(raws, optimum) if raws else None,
        "epochs": epochs,
        "final_epoch_regret": epochs[-1]["simple_regret"] if epochs else None,
        "n_events": len(result.events),
        "recovery_efficiency": dataclasses.asdict(recovery),
        "adaptation_lag": dataclasses.asdict(lag),
        "constraint_adaptation": dataclasses.asdict(latency),
        "decision_quality": quality,
    }


def _mean_or_none(values: list[float | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return statistics.fmean(present) if present else None


def _group_results(
    results: list[CellResult],
) -> dict[tuple[str, str, str], list[CellResult]]:
    grouped: dict[tuple[str, str, str], list[CellResult]] = {}
    for result in results:
        key = (result.cell.problem_id, result.cell.bundle.id, result.cell.config)
        grouped.setdefault(key, []).append(result)
    return grouped


def _family_configs(family: dict[str, Any]) -> set[str]:
    configs = set(family.get("methods", []))
    for contrast in family.get("contrasts", []):
        configs.update(contrast)
    return {str(c) for c in configs}


def _assign_family(
    config: str,
    bundle_id: str,
    families: dict[str, dict[str, Any]],
) -> str:
    for name, family in families.items():
        if config not in _family_configs(family):
            continue
        pathologies = family.get("pathologies")
        if pathologies is not None and bundle_id not in {str(p) for p in pathologies}:
            continue
        return name
    return "default"


def _comparisons(
    results: list[CellResult],
    cell_metrics: dict[str, dict[str, Any]],
    reference: str,
    families: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Seed-paired bootstrap comparisons of every config against the reference.

    Holm correction is applied *within* each pre-declared comparison family
    (rows outside every family fall into "default"), not across the entire
    problem x pathology x config pool.
    """
    grouped = _group_results(results)
    rows: list[dict[str, Any]] = []
    for (problem_id, bundle_id, config), group in sorted(grouped.items()):
        if config == reference:
            continue
        reference_group = grouped.get((problem_id, bundle_id, reference))
        if not reference_group:
            continue
        ref_by_seed = {r.cell.seed: r for r in reference_group}
        diffs: list[float] = []
        for result in group:
            ref = ref_by_seed.get(result.cell.seed)
            if ref is None:
                continue
            a = cell_metrics[result.cell.cell_id].get("final_regret")
            b = cell_metrics[ref.cell.cell_id].get("final_regret")
            if a is None or b is None:
                continue
            diffs.append(float(a) - float(b))  # positive = reference better
        if not diffs:
            continue
        boot = paired_bootstrap(diffs, seed=0)
        rows.append(
            {
                "problem": problem_id,
                "pathology": bundle_id,
                "comparison": f"{reference}_vs_{config}",
                "family": _assign_family(config, bundle_id, families or {}),
                "n_pairs": len(diffs),
                "mean_regret_delta_other_minus_reference": boot.mean,
                "ci_low": boot.ci_low,
                "ci_high": boot.ci_high,
                "p_value": boot.p_value,
            }
        )
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
    for family_rows in by_family.values():
        adjusted = holm_adjust([r["p_value"] for r in family_rows])
        for row, p_holm in zip(family_rows, adjusted, strict=True):
            row["p_holm"] = p_holm
    return rows


def _main_table_rows(
    results: list[CellResult],
    cell_metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for (problem_id, bundle_id, config), group in sorted(_group_results(results).items()):
        metrics = [cell_metrics[r.cell.cell_id] for r in group]
        rows.append(
            {
                "problem": problem_id,
                "pathology": bundle_id,
                "config": config,
                "n_seeds": len(group),
                "errors": sum(1 for r in group if r.trace.error is not None),
                "mean_final_regret": _mean_or_none([m["final_regret"] for m in metrics]),
                "mean_regret_auc": _mean_or_none([m["regret_auc"] for m in metrics]),
                "mean_evals_to_target": _mean_or_none(
                    [m["evals_to_target"] for m in metrics]
                ),
                "mean_dynamic_regret_auc": _mean_or_none(
                    [m["dynamic_regret_auc"] for m in metrics]
                ),
                "mean_final_epoch_regret": _mean_or_none(
                    [m["final_epoch_regret"] for m in metrics]
                ),
                "target_hit_rate": statistics.fmean(
                    [1.0 if m["evals_to_target"] is not None else 0.0 for m in metrics]
                )
                if metrics
                else None,
            }
        )
    return rows


def _mechanism_table_rows(
    results: list[CellResult],
    cell_metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for (problem_id, bundle_id, config), group in sorted(_group_results(results).items()):
        if config not in BACKEND_VARIANTS:
            continue
        metrics = [cell_metrics[r.cell.cell_id] for r in group]
        rows.append(
            {
                "problem": problem_id,
                "pathology": bundle_id,
                "config": config,
                "n_seeds": len(group),
                "mean_decision_quality": _mean_or_none(
                    [
                        (m["decision_quality"] or {}).get("hit_rate")
                        for m in metrics
                    ]
                ),
                "mean_recovery_efficiency": _mean_or_none(
                    [m["recovery_efficiency"]["mean_ratio"] for m in metrics]
                ),
                "mean_adaptation_lag": _mean_or_none(
                    [m["adaptation_lag"]["mean_lag"] for m in metrics]
                ),
                "mean_constraint_latency": _mean_or_none(
                    [m["constraint_adaptation"]["latency"] for m in metrics]
                ),
            }
        )
    return rows


def _rows_to_csv(rows: list[dict[str, Any]], columns: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in columns})
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def write_study_artifacts(
    output_dir: str | Path,
    matrix: ExperimentMatrix,
    results: list[CellResult],
    metrics_config: MetricsConfig,
    *,
    reference: str = "helios_full",
    budget: int,
    batch: int = 1,
    n_init: int = DEFAULT_INIT,
    description: str = "",
    problem_groups: dict[str, tuple[str, ...]] | None = None,
    comparison_families: dict[str, dict[str, Any]] | None = None,
    force: bool = False,
) -> Path:
    """Write the reproducible study directory; returns its path.

    ``matrix.json`` is the single source of truth (full pathology params,
    metric config hash + schema version, git commit + dirty-tree provenance,
    run parameters, preregistered problem groups and comparison families).
    Skipped metrics are reported explicitly -- never silently dropped.
    An existing study directory is refused unless ``force=True`` -- the same
    content hash must never silently overwrite prior results.
    """
    git_commit = _git_commit()
    study_id = study_id_for(
        matrix,
        metrics_config_hash=metrics_config.content_hash,
        git_commit=git_commit,
    )
    study_dir = Path(output_dir) / study_id
    if study_dir.exists() and not force:
        raise FileExistsError(
            f"study directory already exists: {study_dir} (pass force=True / --force "
            "to overwrite explicitly)"
        )
    (study_dir / "traces").mkdir(parents=True, exist_ok=True)
    (study_dir / "events").mkdir(parents=True, exist_ok=True)
    (study_dir / "tables").mkdir(parents=True, exist_ok=True)

    skipped: list[dict[str, str]] = []
    failed_cells = 0
    cell_rows: list[dict[str, Any]] = []
    for result in results:
        cell = result.cell
        (study_dir / "traces" / f"{cell.cell_id}.jsonl").write_text(
            json.dumps(dataclasses.asdict(result.trace), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (study_dir / "events" / f"{cell.cell_id}.events.jsonl").write_text(
            "".join(
                json.dumps(dataclasses.asdict(e), sort_keys=True) + "\n"
                for e in result.events
            ),
            encoding="utf-8",
        )
        if result.trace.error is not None or not result.trace.best_so_far:
            failed_cells += 1
            reason = result.trace.error or "empty_trace"
            skipped.append({"cell": cell.cell_id, "reason": f"backend_error: {reason}"})
            metrics: dict[str, Any] | None = None
        else:
            metrics = _compute_cell_metrics(
                result, metrics_config, n_init=n_init, batch=batch
            )
        cell_rows.append(
            {
                "cell": cell.cell_id,
                "problem": cell.problem_id,
                "pathology": cell.bundle.id,
                "config": cell.config,
                "seed": cell.seed,
                "metrics": metrics,
            }
        )

    valid_results = [
        r
        for r in results
        if r.trace.error is None and r.trace.best_so_far
    ]
    metrics_by_cell = {
        row["cell"]: row["metrics"] for row in cell_rows if row["metrics"] is not None
    }
    main_rows = _main_table_rows(valid_results, metrics_by_cell)
    mechanism_rows = _mechanism_table_rows(valid_results, metrics_by_cell)
    comparison_rows = _comparisons(
        valid_results, metrics_by_cell, reference, comparison_families
    )

    manifest = {
        **matrix.to_dict(),
        "study_id": study_id,
        "description": description,
        "budget": budget,
        "batch": batch,
        "n_init": n_init,
        "reference": reference,
        "problem_groups": {k: list(v) for k, v in (problem_groups or {}).items()},
        "comparison_families": comparison_families or {},
        "metrics_config_hash": metrics_config.content_hash,
        "metric_schema_version": metrics_config.schema_version,
        "git_commit": git_commit,
        **_environment_provenance(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    (study_dir / "matrix.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (study_dir / "metrics.json").write_text(
        json.dumps(
            {
                "valid_cells": len(valid_results),
                "failed_cells": failed_cells,
                "skipped_metrics": skipped,
                "cells": cell_rows,
                "comparisons": comparison_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (study_dir / "tables" / "main_benchmark.csv").write_text(
        _rows_to_csv(
            main_rows,
            [
                "problem",
                "pathology",
                "config",
                "n_seeds",
                "errors",
                "mean_final_regret",
                "mean_regret_auc",
                "mean_evals_to_target",
                "target_hit_rate",
                "mean_dynamic_regret_auc",
                "mean_final_epoch_regret",
            ],
        ),
        encoding="utf-8",
    )
    (study_dir / "tables" / "mechanism_analysis.csv").write_text(
        _rows_to_csv(
            mechanism_rows,
            [
                "problem",
                "pathology",
                "config",
                "n_seeds",
                "mean_decision_quality",
                "mean_recovery_efficiency",
                "mean_adaptation_lag",
                "mean_constraint_latency",
            ],
        ),
        encoding="utf-8",
    )
    (study_dir / "report.md").write_text(
        _render_report(
            study_id,
            manifest,
            results,
            main_rows,
            mechanism_rows,
            comparison_rows,
            skipped,
        ),
        encoding="utf-8",
    )
    return study_dir


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_no rows_\n"
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        return "" if value is None else str(value)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _render_report(
    study_id: str,
    manifest: dict[str, Any],
    results: list[CellResult],
    main_rows: list[dict[str, Any]],
    mechanism_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    skipped: list[dict[str, str]],
) -> str:
    event_counts: dict[str, int] = {}
    for result in results:
        for event in result.events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
    parts = [
        f"# Pathology study `{study_id}`\n",
        f"- problems: {', '.join(manifest['problem_ids'])}",
        f"- pathologies: {', '.join(b['id'] for b in manifest['pathology_bundles'])}",
        f"- configs: {', '.join(manifest['config_names'])}",
        f"- seeds: {len(manifest['seed_list'])} | budget: {manifest['budget']}"
        f" | reference: {manifest['reference']}",
        f"- git commit: {manifest['git_commit']}",
        f"- cells: {len(results)} total, {len(skipped)} skipped"
        f" (see metrics.json skipped_metrics)",
        f"- ground-truth events by type: {json.dumps(event_counts, sort_keys=True)}\n",
        "## Main benchmark (all methods, true-regret ruler)\n",
        _markdown_table(
            main_rows,
            [
                "problem",
                "pathology",
                "config",
                "mean_final_regret",
                "mean_regret_auc",
                "target_hit_rate",
            ],
        ),
        "\n## Mechanism analysis (HELIOS variants only)\n",
        _markdown_table(
            mechanism_rows,
            [
                "problem",
                "pathology",
                "config",
                "mean_decision_quality",
                "mean_recovery_efficiency",
                "mean_adaptation_lag",
                "mean_constraint_latency",
            ],
        ),
        "\n## Reference comparisons (paired bootstrap, Holm-adjusted)\n",
        _markdown_table(
            comparison_rows,
            [
                "problem",
                "pathology",
                "comparison",
                "mean_regret_delta_other_minus_reference",
                "ci_low",
                "ci_high",
                "p_holm",
            ],
        ),
    ]
    return "\n".join(parts)
