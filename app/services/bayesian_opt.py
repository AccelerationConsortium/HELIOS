"""Bayesian Optimization MVP for experiment parameter tuning.

Pure-Python BO using a distance-weighted KNN surrogate and acquisition
functions (EI, UCB).  Integrates with the existing ParameterSpace /
Candidate structures from candidate_gen and reads past observations from
the run_kpis + batch_candidates tables.

No external dependencies -- stdlib only (math, random, etc.).

Pipeline:
1. Load past observations (params + objective values) from DB
2. Fit a KNN surrogate on observations (inverse-distance weighting)
3. Evaluate acquisition function on a large random candidate set
4. Return top-N candidates for the next batch

Offline benchmark: ``benchmark_strategies()`` compares random vs LHS vs BO
convergence on a synthetic objective.
"""
from __future__ import annotations

import logging
import math
import random
import sqlite3 as _sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from app.core.db import connection, json_dumps, parse_json, run_txn, utcnow_iso
from app.services.candidate_gen import (
    ParameterSpace,
    SearchDimension,
    _unit_to_value,
    sample_lhs,
    sample_random,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """A single observed point in parameter space.

    ``params`` stores the *normalized* [0, 1] representation of each
    dimension.  ``objective`` is the KPI value -- higher is better.
    """

    params: tuple[float, ...]  # normalised to [0,1]^d
    objective: float  # KPI value (higher = better)


# ---------------------------------------------------------------------------
# Standard normal helpers (pure Python)
# ---------------------------------------------------------------------------


def _phi(x: float) -> float:
    """Standard normal PDF: phi(x) = exp(-x^2/2) / sqrt(2*pi)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _Phi(x: float) -> float:
    """Standard normal CDF using ``math.erf``."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Surrogate model -- distance-weighted KNN
# ---------------------------------------------------------------------------


class SurrogateModel:
    """Distance-weighted K-Nearest-Neighbour surrogate for BO.

    Given a set of observations in [0,1]^d, predicts the mean and
    uncertainty (variance of k-nearest neighbours) at any query point.
    """

    def __init__(self, observations: list[Observation], k: int = 5) -> None:
        if not observations:
            raise ValueError("SurrogateModel requires at least one observation")
        self._obs = observations
        self._k = min(k, len(observations))

    # -- helpers --

    @staticmethod
    def _euclidean(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    # -- public API --

    def predict(self, x: tuple[float, ...]) -> tuple[float, float]:
        """Return ``(mean, std)`` for a query point in [0,1]^d.

        * **mean**: inverse-distance-weighted average of k-nearest objectives.
        * **std**: standard deviation of the k-nearest objectives, used as an
          uncertainty proxy.

        When a query point coincides with an observation the weight is
        clamped to a large value (``1 / eps``) to avoid division by zero.
        """
        eps = 1e-12

        # Compute distances to every observation
        dists: list[tuple[float, int]] = [
            (self._euclidean(x, obs.params), idx)
            for idx, obs in enumerate(self._obs)
        ]
        dists.sort(key=lambda t: t[0])

        # Select k-nearest
        knn = dists[: self._k]

        # Inverse-distance-weighted mean
        weights: list[float] = []
        values: list[float] = []
        for d, idx in knn:
            w = 1.0 / max(d, eps)
            weights.append(w)
            values.append(self._obs[idx].objective)

        w_sum = sum(weights)
        mean = sum(w * v for w, v in zip(weights, values)) / w_sum

        # Uncertainty: std of k-nearest objectives (unweighted)
        if self._k < 2:
            std = 0.0
        else:
            var = sum((v - mean) ** 2 for v in values) / self._k
            std = math.sqrt(max(var, 0.0))

        return mean, std


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------


def expected_improvement(
    mean: float,
    std: float,
    best_so_far: float,
    xi: float = 0.01,
) -> float:
    """Expected Improvement acquisition function.

    .. math::

        EI = (\\mu - f^* - \\xi)\\,\\Phi(z) + \\sigma\\,\\phi(z)

    where :math:`z = (\\mu - f^* - \\xi) / \\sigma`.

    Returns 0.0 when ``std`` is negligible.
    """
    if std < 1e-12:
        return 0.0
    z = (mean - best_so_far - xi) / std
    return (mean - best_so_far - xi) * _Phi(z) + std * _phi(z)


def upper_confidence_bound(
    mean: float,
    std: float,
    kappa: float = 2.0,
) -> float:
    """UCB acquisition function: ``mean + kappa * std``."""
    return mean + kappa * std


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _value_to_unit(value: Any, dim: SearchDimension) -> float:
    """Map an actual parameter value back to [0, 1].

    Inverse of ``candidate_gen._unit_to_value``.
    """
    if dim.choices is not None:
        choices = list(dim.choices)
        try:
            idx = choices.index(value)
        except ValueError:
            idx = 0
        return (idx + 0.5) / len(choices)

    if dim.param_type == "boolean":
        return 1.0 if value else 0.0

    if dim.min_value is None or dim.max_value is None:
        raise ValueError(
            f"Dimension '{dim.param_name}' requires min_value and max_value"
        )

    fval = float(value)
    if dim.log_scale:
        log_min = math.log(max(dim.min_value, 1e-12))
        log_max = math.log(max(dim.max_value, 1e-12))
        if log_max == log_min:
            return 0.5
        return (math.log(max(fval, 1e-12)) - log_min) / (log_max - log_min)

    span = dim.max_value - dim.min_value
    if span == 0:
        return 0.5
    return (fval - dim.min_value) / span


def normalize_params(
    params: dict[str, Any],
    space: ParameterSpace,
) -> tuple[float, ...]:
    """Normalise a param dict to [0,1]^d following dimension order."""
    result: list[float] = []
    for dim in space.dimensions:
        val = params.get(dim.param_name)
        if val is None:
            result.append(0.5)  # missing → midpoint
        else:
            u = _value_to_unit(val, dim)
            result.append(max(0.0, min(1.0, u)))
    return tuple(result)


def denormalize_point(
    point: tuple[float, ...] | list[float],
    space: ParameterSpace,
) -> dict[str, Any]:
    """Convert a [0,1]^d point back to an actual param dict."""
    params: dict[str, Any] = {}
    for j, dim in enumerate(space.dimensions):
        params[dim.param_name] = _unit_to_value(point[j], dim)
    return params


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def load_observations_from_db(
    space: ParameterSpace,
    campaign_id: str | None = None,
    kpi_name: str = "run_success_rate",
) -> list[Observation]:
    """Load past observations from ``run_kpis`` + ``batch_candidates``.

    Joins ``batch_candidates`` (params) with ``run_kpis`` (objective values)
    for candidates that have actually been executed
    (``selected_run_id IS NOT NULL``).
    """

    def _txn(conn: _sqlite3.Connection) -> list[Observation]:
        if campaign_id:
            sql = """
                SELECT bc.params_json, rk.kpi_value
                FROM batch_candidates bc
                JOIN batch_requests br ON br.id = bc.batch_id
                JOIN run_kpis rk ON rk.run_id = bc.selected_run_id
                WHERE bc.selected_run_id IS NOT NULL
                  AND rk.kpi_name = ?
                  AND br.campaign_id = ?
            """
            rows = conn.execute(sql, (kpi_name, campaign_id)).fetchall()
        else:
            sql = """
                SELECT bc.params_json, rk.kpi_value
                FROM batch_candidates bc
                JOIN run_kpis rk ON rk.run_id = bc.selected_run_id
                WHERE bc.selected_run_id IS NOT NULL
                  AND rk.kpi_name = ?
            """
            rows = conn.execute(sql, (kpi_name,)).fetchall()

        obs: list[Observation] = []
        for row in rows:
            raw_params = parse_json(row["params_json"], {})
            kpi_val = row["kpi_value"]
            if kpi_val is None:
                continue
            normed = normalize_params(raw_params, space)
            obs.append(Observation(params=normed, objective=float(kpi_val)))
        return obs

    return run_txn(_txn)


# ---------------------------------------------------------------------------
# BO sampler
# ---------------------------------------------------------------------------


_ACQUISITION_FNS: dict[str, str] = {
    "ei": "Expected Improvement",
    "ucb": "Upper Confidence Bound",
}

# Minimum observations before the surrogate is useful.  Below this
# threshold we fall back to LHS for better space-filling coverage.
_MIN_OBSERVATIONS = 5


def sample_bo(
    space: ParameterSpace,
    n: int,
    *,
    observations: list[Observation] | None = None,
    acquisition: str = "ei",
    n_random_candidates: int = 1000,
    seed: int | None = None,
    xi: float = 0.01,
    kappa: float = 2.0,
    k_neighbours: int = 5,
) -> list[dict[str, Any]]:
    """Bayesian-optimisation sampling -- pure Python, no external deps.

    Algorithm
    ---------
    1. If fewer than ``_MIN_OBSERVATIONS`` observations exist, fall back
       to LHS (the surrogate needs data to be meaningful).
    2. Fit a KNN surrogate on the existing observations.
    3. Generate ``n_random_candidates`` random points in [0,1]^d.
    4. Score each point with the chosen acquisition function.
    5. Return the top-*n* points (de-normalised to actual param values).

    Parameters
    ----------
    space:
        The search space definition.
    n:
        How many candidates to return.
    observations:
        Past (normalised-params, objective) pairs.  If ``None``, an
        empty list is used and the function falls back to LHS.
    acquisition:
        ``"ei"`` (Expected Improvement) or ``"ucb"`` (Upper Confidence
        Bound).
    n_random_candidates:
        Size of the random candidate pool scored by the acquisition
        function.  Larger = better coverage, slower.
    seed:
        RNG seed for reproducibility.
    xi:
        Exploration parameter for EI.
    kappa:
        Exploration parameter for UCB.
    k_neighbours:
        Number of neighbours for the KNN surrogate.
    """
    if acquisition not in _ACQUISITION_FNS:
        raise ValueError(
            f"Unknown acquisition '{acquisition}'. "
            f"Valid: {sorted(_ACQUISITION_FNS)}"
        )

    if observations is None:
        observations = []

    # --- cold-start: fall back to LHS ---
    if len(observations) < _MIN_OBSERVATIONS:
        logger.info(
            "BO cold-start (%d obs < %d): falling back to LHS",
            len(observations),
            _MIN_OBSERVATIONS,
        )
        return sample_lhs(space, n, seed=seed)

    # --- fit surrogate ---
    surrogate = SurrogateModel(observations, k=k_neighbours)
    best_so_far = max(obs.objective for obs in observations)

    # --- generate random candidate pool in [0,1]^d ---
    rng = random.Random(seed)
    n_dims = space.n_dims
    pool: list[tuple[float, ...]] = [
        tuple(rng.random() for _ in range(n_dims))
        for _ in range(n_random_candidates)
    ]

    # --- score candidates ---
    scored: list[tuple[float, tuple[float, ...]]] = []
    for point in pool:
        mean, std = surrogate.predict(point)
        if acquisition == "ei":
            acq_value = expected_improvement(mean, std, best_so_far, xi=xi)
        else:  # ucb
            acq_value = upper_confidence_bound(mean, std, kappa=kappa)
        scored.append((acq_value, point))

    # Sort descending by acquisition value (higher = more promising)
    scored.sort(key=lambda t: t[0], reverse=True)

    # --- select top-n and denormalise ---
    selected = scored[:n]
    results: list[dict[str, Any]] = []
    for _acq, point in selected:
        results.append(denormalize_point(point, space))

    logger.info(
        "BO sampled %d candidates (acquisition=%s, pool=%d, best_obj=%.4f)",
        len(results),
        acquisition,
        n_random_candidates,
        best_so_far,
    )
    return results


# ---------------------------------------------------------------------------
# Convergence benchmark
# ---------------------------------------------------------------------------


def benchmark_strategies(
    space: ParameterSpace,
    objective_fn: Callable[[dict[str, Any]], float],
    strategies: list[str] | None = None,
    n_rounds: int = 10,
    batch_size: int = 5,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Compare convergence of sampling strategies on a synthetic objective.

    Returns ``{strategy_name: [best_objective_per_round, ...]}``.

    For **BO** the observation set grows round-by-round (closed-loop).
    For **random** and **LHS** each round is independent -- the
    ``best_so_far`` is simply the running maximum across all evaluated
    candidates.

    Example synthetic objective::

        def sphere(params):
            return -sum((v - 0.5)**2 for v in params.values())

        results = benchmark_strategies(space, sphere)
    """
    if strategies is None:
        strategies = ["random", "lhs", "bo"]

    results: dict[str, list[float]] = {}

    for strat in strategies:
        rng_seed = seed
        best = float("-inf")
        history: list[float] = []
        bo_observations: list[Observation] = []

        for round_idx in range(n_rounds):
            round_seed = rng_seed + round_idx

            # Generate candidates
            if strat == "random":
                candidates = sample_random(space, batch_size, seed=round_seed)
            elif strat == "lhs":
                candidates = sample_lhs(space, batch_size, seed=round_seed)
            elif strat == "bo":
                candidates = sample_bo(
                    space,
                    batch_size,
                    observations=bo_observations if bo_observations else None,
                    acquisition="ei",
                    n_random_candidates=500,
                    seed=round_seed,
                )
            else:
                raise ValueError(f"Unknown benchmark strategy: {strat}")

            # Evaluate candidates
            for params in candidates:
                obj = objective_fn(params)
                best = max(best, obj)

                if strat == "bo":
                    normed = normalize_params(params, space)
                    bo_observations.append(
                        Observation(params=normed, objective=obj)
                    )

            history.append(best)

        results[strat] = history
        logger.info(
            "Benchmark [%s]: final best=%.6f after %d rounds",
            strat,
            best,
            n_rounds,
        )

    return results
