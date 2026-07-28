"""Pathology layer: controlled non-stationary experimental conditions.

Wraps a clean :class:`~benchmarks.methods.problems.OptProblem` with realistic
experimental corruptions (observation drift, spatially correlated execution
failure, censoring, measurement-model shift, objective shift) without touching
the study runner or any backend: ``apply_pathology`` returns a **new**
``OptProblem`` whose evaluator is a stateful closure, so the runner still sees
an ordinary problem.

Contract (docs/plans/2026-07-22-pathology-benchmark-evaluation-layer-design.md):

- ``raw_value`` is never corrupted -- it stays the single clean regret ruler.
  Pathologies act on ``observed_value`` / success flags / metadata.  The one
  exception is :class:`ObjectiveShift`, which *translates* the landscape
  (the argmin moves) while preserving the optimum **value**.
- Every triggered condition appends a ground-truth :class:`PathologyEvent` to
  ``PathologyState.events`` -- the ruler for all adaptation metrics.
- Determinism: identical ``(specs, seed)`` produce identical event ledgers and
  observed sequences.
"""
from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass, field
from typing import Any

from app.services.candidate_gen import ParameterSpace
from benchmarks.methods.problems import OptProblem, ProblemEvaluation

# ---------------------------------------------------------------------------
# Capability each pathology tests (paper-facing naming; see design doc)
# ---------------------------------------------------------------------------

# Version of the pathology event/spec contract; recorded in study manifests.
PATHOLOGY_SCHEMA_VERSION = 1

CAPABILITY_BY_KIND = {
    "noise_drift": "observation_robustness",
    "spatial_failure": "execution_resilience",
    "censoring": "partial_observability",
    "proxy_gap_shift": "measurement_model_adaptation",
    "objective_shift": "campaign_adaptation",
}

FAILURE_TYPE = "pathology_execution_failure"


@dataclass(frozen=True)
class PathologyEvent:
    """One ground-truth ledger entry: what changed, when, where, how hard."""

    event_id: str
    event_type: str
    eval_index: int  # 1-based evaluation count when the condition applied
    severity: float
    duration: int | None  # evals; None = permanent condition
    region: dict[str, Any] | None
    params: dict[str, Any]
    details: dict[str, Any] = field(default_factory=dict)


class PathologyState:
    """Mutable per-cell state owned by the wrapped evaluator closure."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.seed = seed
        self.eval_count = 0
        self.events: list[PathologyEvent] = []
        self.history: list[dict[str, Any]] = []

    def record_event(
        self,
        event_type: str,
        *,
        severity: float,
        duration: int | None,
        region: dict[str, Any] | None,
        params: dict[str, Any],
        details: dict[str, Any] | None = None,
    ) -> PathologyEvent:
        event = PathologyEvent(
            event_id=f"{event_type}-{len(self.events) + 1:04d}",
            event_type=event_type,
            eval_index=self.eval_count,
            severity=float(severity),
            duration=duration,
            region=region,
            params=dict(params),
            details=dict(details or {}),
        )
        self.events.append(event)
        return event


# ---------------------------------------------------------------------------
# Normalized-coordinate helpers (continuous dims only)
# ---------------------------------------------------------------------------


def _normalized(params: dict[str, Any], space: ParameterSpace) -> dict[str, float]:
    coords: dict[str, float] = {}
    for dim in space.dimensions:
        if dim.min_value is None or dim.max_value is None:
            continue
        value = params.get(dim.param_name)
        if not isinstance(value, int | float):
            continue
        span = float(dim.max_value) - float(dim.min_value)
        if span <= 0:
            continue
        coords[dim.param_name] = (float(value) - float(dim.min_value)) / span
    return coords


def _region_distance(
    params: dict[str, Any],
    space: ParameterSpace,
    center: dict[str, float],
) -> float:
    coords = _normalized(params, space)
    total = 0.0
    for name, target in center.items():
        if name not in coords:
            return math.inf
        total += (coords[name] - float(target)) ** 2
    return math.sqrt(total)


def in_failure_region(
    params: dict[str, Any],
    space: ParameterSpace,
    region: dict[str, Any],
) -> bool:
    """True when ``params`` falls inside a ledgered failure region."""
    center = region.get("center", {})
    radius = float(region.get("radius", 0.0))
    return _region_distance(params, space, center) <= radius


# ---------------------------------------------------------------------------
# Pathology specs
# ---------------------------------------------------------------------------
#
# Two hook points, applied in spec order:
#   transform_params:      before the base evaluator runs (landscape stage)
#   transform_evaluation:  after the base evaluator runs (observation stage)
# Base implementations are identity; each spec overrides the stage it owns.


@dataclass(frozen=True)
class PathologySpec:
    kind = "abstract"

    def transform_params(
        self,
        state: PathologyState,
        params: dict[str, Any],
        space: ParameterSpace,
    ) -> dict[str, Any]:
        return params

    def transform_evaluation(
        self,
        state: PathologyState,
        params: dict[str, Any],
        evaluation: ProblemEvaluation,
        space: ParameterSpace,
    ) -> ProblemEvaluation:
        return evaluation


@dataclass(frozen=True)
class NoiseDrift(PathologySpec):
    """Observation robustness: observed-value bias that grows after onset."""

    start_eval: int = 10
    rate: float = 0.1
    mode: str = "linear"  # "linear" | "step"
    noise_std: float = 0.0

    kind = "noise_drift"

    def transform_evaluation(self, state, params, evaluation, space):
        elapsed = state.eval_count - self.start_eval
        if elapsed <= 0:
            return evaluation
        if elapsed == 1:
            state.record_event(
                self.kind,
                severity=self.rate,
                duration=None,
                region=None,
                params=params,
                details={"mode": self.mode, "start_eval": self.start_eval},
            )
        bias = self.rate * elapsed if self.mode == "linear" else self.rate
        noise = state.rng.gauss(0.0, self.noise_std) if self.noise_std > 0 else 0.0
        observed = float(evaluation.optimizer_value) + bias + noise
        return dataclasses.replace(
            evaluation,
            observed_value=observed,
            metadata={**evaluation.metadata, "noise_drift_bias": bias},
        )


@dataclass(frozen=True)
class SpatialFailure(PathologySpec):
    """Execution resilience: spatially correlated failure probability.

    ``p(fail | x) = p_base + (p_max - p_base) * max(0, 1 - dist(x)/radius)``
    with ``dist`` the euclidean distance to ``center`` in normalized
    continuous coordinates.  On failure the evaluation is marked
    ``execution_success=False`` and the observed value takes a penalty --
    the agent gets a corrupted signal, while ``raw_value`` (the true
    landscape) stays intact.
    """

    center: dict[str, float] = field(default_factory=dict)
    radius: float = 0.25
    p_max: float = 0.9
    p_base: float = 0.02
    observed_penalty: float = 5.0

    kind = "spatial_failure"

    def failure_probability(
        self,
        params: dict[str, Any],
        space: ParameterSpace,
    ) -> float:
        dist = _region_distance(params, space, self.center)
        if not math.isfinite(dist):
            return self.p_base
        proximity = max(0.0, 1.0 - dist / self.radius) if self.radius > 0 else 0.0
        return self.p_base + (self.p_max - self.p_base) * proximity

    @property
    def region(self) -> dict[str, Any]:
        return {"center": dict(self.center), "radius": self.radius}

    def transform_evaluation(self, state, params, evaluation, space):
        p = self.failure_probability(params, space)
        if state.rng.random() >= p:
            return evaluation
        state.record_event(
            self.kind,
            severity=p,
            duration=1,
            region=self.region,
            params=params,
            details={"failure_probability": p},
        )
        observed = float(evaluation.optimizer_value) + self.observed_penalty
        return dataclasses.replace(
            evaluation,
            observed_value=observed,
            execution_success=False,
            qc_passed=False,
            failure_type=FAILURE_TYPE,
            metadata={**evaluation.metadata, "failure_probability": p},
        )


@dataclass(frozen=True)
class Censoring(PathologySpec):
    """Partial observability: right-censoring in minimization space.

    Observed values worse than ``lod`` are reported *at* ``lod`` -- the
    optimizer cannot distinguish anything beyond the detection limit.
    """

    lod: float = 1.0

    kind = "censoring"

    def transform_evaluation(self, state, params, evaluation, space):
        observed = float(evaluation.optimizer_value)
        if observed <= self.lod:
            return evaluation
        if not any(e.event_type == self.kind for e in state.events):
            state.record_event(
                self.kind,
                severity=observed - self.lod,
                duration=None,
                region=None,
                params=params,
                details={"lod": self.lod},
            )
        return dataclasses.replace(
            evaluation,
            observed_value=self.lod,
            metadata={**evaluation.metadata, "censored": True},
        )


@dataclass(frozen=True)
class ProxyGapShift(PathologySpec):
    """Measurement-model adaptation: the observed mapping changes mid-run.

    Before ``shift_eval`` the proxy is faithful; afterwards
    ``observed = scale * observed + bias`` (a calibration change).
    """

    shift_eval: int = 15
    bias: float = 1.0
    scale: float = 1.0

    kind = "proxy_gap_shift"

    def transform_evaluation(self, state, params, evaluation, space):
        if state.eval_count <= self.shift_eval:
            return evaluation
        if state.eval_count == self.shift_eval + 1:
            state.record_event(
                self.kind,
                severity=abs(self.bias) + abs(self.scale - 1.0),
                duration=None,
                region=None,
                params=params,
                details={"bias": self.bias, "scale": self.scale},
            )
        observed = self.scale * float(evaluation.optimizer_value) + self.bias
        return dataclasses.replace(
            evaluation,
            observed_value=observed,
            metadata={**evaluation.metadata, "proxy_gap_shifted": True},
        )


@dataclass(frozen=True)
class ObjectiveShift(PathologySpec):
    """Campaign adaptation: the landscape translates mid-run.

    After ``shift_eval`` the objective is evaluated at ``x - delta * range``,
    i.e. the argmin moves by ``+delta`` (normalized units) while the optimum
    *value* is preserved, so every regret ruler stays valid.
    """

    shift_eval: int = 15
    delta: dict[str, float] = field(default_factory=dict)

    kind = "objective_shift"

    def transform_params(self, state, params, space):
        if state.eval_count <= self.shift_eval:
            return params
        if state.eval_count == self.shift_eval + 1:
            state.record_event(
                self.kind,
                severity=math.sqrt(sum(v * v for v in self.delta.values())),
                duration=None,
                region={"delta": dict(self.delta)},
                params=params,
                details={"shift_eval": self.shift_eval},
            )
        shifted = dict(params)
        for dim in space.dimensions:
            frac = self.delta.get(dim.param_name)
            if frac is None or dim.min_value is None or dim.max_value is None:
                continue
            value = shifted.get(dim.param_name)
            if not isinstance(value, int | float):
                continue
            span = float(dim.max_value) - float(dim.min_value)
            shifted[dim.param_name] = float(value) - float(frac) * span
        return shifted


SPEC_TYPES: dict[str, type[PathologySpec]] = {
    spec.kind: spec
    for spec in (NoiseDrift, SpatialFailure, Censoring, ProxyGapShift, ObjectiveShift)
}
# Also accept class names in study configs ("NoiseDrift" == "noise_drift").
SPEC_TYPES.update({cls.__name__: cls for cls in tuple(SPEC_TYPES.values())})


def spec_from_dict(data: dict[str, Any]) -> PathologySpec:
    """Build a spec from a study-config entry: {"type": ..., "params": {...}}."""
    type_name = str(data.get("type", ""))
    cls = SPEC_TYPES.get(type_name)
    if cls is None:
        raise ValueError(
            f"Unknown pathology type '{type_name}'. Known: {sorted(set(SPEC_TYPES))}"
        )
    return cls(**dict(data.get("params", {})))


def spec_to_dict(spec: PathologySpec) -> dict[str, Any]:
    """Full-parameter serialization (matrix.json must never store bare names)."""
    return {"type": spec.kind, "params": dataclasses.asdict(spec)}


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


def bundle_id_for(specs: list[PathologySpec] | tuple[PathologySpec, ...]) -> str:
    return "+".join(spec.kind for spec in specs) if specs else "clean"


def apply_pathology(
    problem: OptProblem,
    specs: list[PathologySpec] | tuple[PathologySpec, ...],
    seed: int,
    *,
    bundle_id: str | None = None,
) -> tuple[OptProblem, PathologyState]:
    """Wrap ``problem`` with pathology ``specs``; runner and backends unchanged.

    Returns a new ``OptProblem`` (id ``"<base>@<bundle>"``) plus the mutable
    :class:`PathologyState` whose ``events`` ledger grounds all adaptation
    metrics.  The base problem is untouched.
    """
    specs = tuple(specs)
    state = PathologyState(seed)
    label = bundle_id or bundle_id_for(specs)
    base_evaluate = problem.evaluate
    space = problem.space

    def evaluator(params: dict[str, Any]) -> ProblemEvaluation:
        state.eval_count += 1
        effective = dict(params)
        for spec in specs:
            effective = spec.transform_params(state, effective, space)
        evaluation = base_evaluate(effective)
        if evaluation.observed_value is None:
            evaluation = dataclasses.replace(
                evaluation, observed_value=float(evaluation.raw_value)
            )
        for spec in specs:
            evaluation = spec.transform_evaluation(state, params, evaluation, space)
        state.history.append(
            {"params": dict(params), "raw_value": float(evaluation.raw_value)}
        )
        return evaluation

    wrapped = dataclasses.replace(
        problem,
        id=f"{problem.id}@{label}",
        evaluator=evaluator,
    )
    return wrapped, state
