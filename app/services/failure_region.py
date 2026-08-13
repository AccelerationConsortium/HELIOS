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

# ---------------------------------------------------------------------------
# Continuous failure-objective helpers (shared across experiments)
# ---------------------------------------------------------------------------
#
# Integrations historically encoded a failure as a flat penalty (e.g. -2.0),
# which collapses every failed trial to an identical scalar.  A surrogate
# trained on those objectives cannot learn *how far* a trial was from the
# feasibility boundary, so optimization keeps re-sampling near known-success
# regions instead of steering away from failure zones (observed in the
# drug-solubilization campaign: 61% of trials had zero gradient; the GLV
# bottleneck dominated failures but got no dedicated signal).
#
# The helpers below turn a per-target *margin* (signed distance to threshold)
# into a continuous penalty, and produce a higher-is-better success objective
# for minimize-style problems.  Integrations attach per-target values to
# ``Observation.objectives`` and call :func:`continuous_failure_penalty`.


def _isfinite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def worst_margin(
    values: dict[str, Any],
    thresholds: dict[str, float],
    *,
    keys: tuple[str, ...] | None = None,
    higher_is_better: bool = True,
) -> float | None:
    """Return the worst (most negative) margin over the requested keys.

    ``margin`` is ``threshold - value`` when ``higher_is_better`` (the common
    case: a value must stay below a threshold, e.g. absorbance).  ``>= 0``
    means feasible; ``< 0`` means failed, more negative = further from
    feasibility.  Returns ``None`` when no requested key is present.
    """
    margins: list[float] = []
    for key in keys or tuple(thresholds):
        if key not in values:
            continue
        val = float(values[key])
        if not _isfinite(val):
            continue
        th = float(thresholds.get(key, 0.0))
        margins.append((th - val) if higher_is_better else (val - th))
    return min(margins) if margins else None


def continuous_failure_penalty(
    margin: float | None,
    *,
    floor: float = -2.0,
    ceiling: float = -1.0,
    width: float = 1.0,
) -> float:
    """Map a worst margin to a continuous failure penalty (higher = better).

    - ``margin >= 0``  -> ``0.0`` (feasible; success objective is the
      caller's job).
    - ``margin`` in ``[-width, 0)`` -> linear from ``ceiling`` (just below
      threshold) up to ``floor`` (at the window edge): the learning gradient.
    - ``margin < -width`` -> ``floor``.

    Defaults keep the drug campaign's old ``-2.0`` floor while adding a
    ``-2.0 -> -1.0`` ramp over the first unit of failure distance.
    """
    if margin is None or not _isfinite(margin):
        return floor
    if margin >= 0.0:
        return 0.0
    if margin <= -width:
        return floor
    t = (margin + width) / width  # 0 at -width .. 1 at 0
    return floor + (ceiling - floor) * t


def success_objective(total: float, max_total: float) -> float:
    """Minimize-style success objective expressed as higher-is-better."""
    return -total / max_total if max_total > 0 else -1.0



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
