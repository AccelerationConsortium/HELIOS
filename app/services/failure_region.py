"""Failure-region modeling in parameter space."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.services.candidate_gen import OutcomeConstraint

if TYPE_CHECKING:  # pragma: no cover
    from app.services.candidate_gen import ParameterSpace, SearchDimension
    from app.services.optimization_backends import Observation

_FEASIBILITY = "feasibility"
_DEFAULT_BANDWIDTH = 0.15
_DEFAULT_THRESHOLD = 0.5


def _dim_distance(dim: SearchDimension, a: Any, b: Any) -> float:
    """Normalized per-dimension distance in [0, 1+]."""
    if dim.param_type in ("categorical", "boolean"):
        return 0.0 if a == b else 1.0
    lo, hi = float(dim.min_value), float(dim.max_value)
    if dim.log_scale and lo > 0 and hi > 0:
        lo, hi, a, b = math.log10(lo), math.log10(hi), math.log10(float(a)), math.log10(float(b))
    span = hi - lo
    if span <= 0:
        return 0.0
    return abs(float(a) - float(b)) / span


def _distance(space: ParameterSpace, p: dict[str, Any], q: dict[str, Any]) -> float:
    """Euclidean distance over normalized per-dimension differences."""
    total = 0.0
    for dim in space.dimensions:
        if dim.param_name in p and dim.param_name in q:
            d = _dim_distance(dim, p[dim.param_name], q[dim.param_name])
            total += d * d
    return math.sqrt(total)


@dataclass(frozen=True)
class FailureRegionModel:
    """Distance-kernel density of past failures over the parameter space."""

    space: ParameterSpace
    failed: tuple[dict[str, Any], ...]
    bandwidth: float = _DEFAULT_BANDWIDTH
    threshold: float = _DEFAULT_THRESHOLD

    @classmethod
    def fit(
        cls,
        failed: list[dict[str, Any]],
        space: ParameterSpace,
        *,
        bandwidth: float = _DEFAULT_BANDWIDTH,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> FailureRegionModel:
        return cls(
            space=space,
            failed=tuple(dict(p) for p in failed),
            bandwidth=bandwidth,
            threshold=threshold,
        )

    def failure_score(self, params: dict[str, Any]) -> float:
        """Return failure proneness in [0, 1], or 0 if no failures are known."""
        if not self.failed:
            return 0.0
        best = 0.0
        for f in self.failed:
            d = _distance(self.space, params, f)
            score = math.exp(-((d / self.bandwidth) ** 2))
            if score > best:
                best = score
        return best

    def predicted_feasible(self, params: dict[str, Any]) -> bool:
        return self.failure_score(params) < self.threshold


def filter_failure_prone(
    candidates: list[dict[str, Any]],
    model: FailureRegionModel,
) -> list[dict[str, Any]]:
    """Drop candidates predicted to fall in the learned failure region."""
    return [c for c in candidates if model.predicted_feasible(c)]


def avoid_failure_region(
    candidates: list[dict[str, Any]],
    space: ParameterSpace,
    n: int,
    failed: list[dict[str, Any]],
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Filter failure-prone candidates and top up to n with feasible points."""
    if not failed:
        return candidates

    from app.services.candidate_gen import sample_feasible

    model = FailureRegionModel.fit(failed=failed, space=space)
    kept = filter_failure_prone(candidates, model)
    if len(kept) >= n:
        return kept[:n]

    base = 0 if seed is None else seed
    tries = 0
    while len(kept) < n and tries < 50:
        for point in sample_feasible(space, n, seed=base + tries + 1):
            if model.predicted_feasible(point) and point not in kept:
                kept.append(point)
                if len(kept) == n:
                    break
        tries += 1
    return kept[:n]


def build_feasibility_observations(
    succeeded: list[Observation],
    failed: list[dict[str, Any]],
) -> list[Any]:
    """Label succeeded/failed points as binary feasibility observations."""
    from bo_engine.types import ObservationData

    obs: list[Any] = []
    for item in succeeded:
        obs.append(
            ObservationData(
                parameter_values=dict(item.params),
                objective_values={_FEASIBILITY: 1.0},
            )
        )
    for params in failed:
        obs.append(
            ObservationData(parameter_values=dict(params), objective_values={_FEASIBILITY: 0.0})
        )
    return obs


def failure_outcome_constraint(feasibility_threshold: float = 0.5) -> OutcomeConstraint:
    """The bomcp outcome constraint that keeps suggestions in the feasible region."""
    return OutcomeConstraint(
        objective_name=_FEASIBILITY,
        threshold=0.5,
        greater_than=True,
        feasibility_threshold=feasibility_threshold,
    )
