"""Batch candidate generation -- pure-algorithmic parameter space exploration.

Generates N candidate parameter sets for a given protocol template using
sampling strategies (LHS, grid, random, prior-guided).  ZERO LLM dependency
in the critical path -- all sampling uses Python stdlib + math module.

Pipeline:
1. Define a ParameterSpace (SearchDimensions with bounds/choices)
2. Choose a sampling strategy
3. generate_batch() produces N candidates and stores them
4. Read path: get_batch(), list_candidates(), list_batches()
"""
from __future__ import annotations

import itertools
import logging
import math
import random
import sqlite3 as _sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.core.db import connection, json_dumps, parse_json, run_txn, utcnow_iso

logger = logging.getLogger(__name__)

BATCH_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchDimension:
    """A single dimension in the parameter search space."""

    param_name: str
    param_type: str  # "number" | "integer" | "boolean" | "categorical"
    min_value: float | None = None  # for number/integer
    max_value: float | None = None  # for number/integer
    log_scale: bool = False  # sample in log space
    choices: tuple[Any, ...] | None = None  # for categorical/boolean
    step_key: str | None = None  # which protocol step this param belongs to
    primitive: str | None = None  # which primitive (for memory lookup)


@dataclass(frozen=True)
class SimplexConstraint:
    """A simplex constraint: sum of named params must equal target (default 1.0).

    Used for composition spaces where components must sum to 1 (or another value).
    After sampling, the named parameters are normalized so their sum = target.
    """

    param_names: tuple[str, ...]  # which params form the simplex
    target_sum: float = 1.0  # what they should sum to (usually 1.0)


@dataclass(frozen=True)
class ParameterSpace:
    """Complete parameter space definition for batch generation."""

    dimensions: tuple[SearchDimension, ...]
    protocol_template: dict[str, Any]  # base protocol, params to be overridden
    simplex_constraints: tuple[SimplexConstraint, ...] = ()

    @property
    def n_dims(self) -> int:
        return len(self.dimensions)


@dataclass(frozen=True)
class Candidate:
    """A single parameter candidate set."""

    index: int
    params: dict[str, Any]  # {param_name: value}
    origin: str  # strategy name
    score: float | None = None


@dataclass(frozen=True)
class BatchResult:
    """Output of a batch generation run."""

    batch_id: str
    candidates: tuple[Candidate, ...]
    strategy: str
    space: ParameterSpace


# ---------------------------------------------------------------------------
# Low-level sampling helpers
# ---------------------------------------------------------------------------


def _sample_dimension(dim: SearchDimension, rng: random.Random) -> Any:
    """Sample a single value from a SearchDimension."""
    if dim.choices is not None:
        return rng.choice(dim.choices)
    if dim.param_type == "boolean":
        return rng.choice([True, False])
    if dim.min_value is None or dim.max_value is None:
        raise ValueError(
            f"Dimension '{dim.param_name}' requires min_value and max_value"
        )
    if dim.log_scale:
        log_min = math.log(max(dim.min_value, 1e-12))
        log_max = math.log(max(dim.max_value, 1e-12))
        val = math.exp(rng.uniform(log_min, log_max))
    else:
        val = rng.uniform(dim.min_value, dim.max_value)
    if dim.param_type == "integer":
        return round(val)
    return val


def _unit_to_value(u: float, dim: SearchDimension) -> Any:
    """Map a [0, 1] unit value to the dimension's actual range."""
    if dim.choices is not None:
        idx = min(int(u * len(dim.choices)), len(dim.choices) - 1)
        return dim.choices[idx]
    if dim.param_type == "boolean":
        return u >= 0.5
    if dim.min_value is None or dim.max_value is None:
        raise ValueError(
            f"Dimension '{dim.param_name}' requires min_value and max_value"
        )
    if dim.log_scale:
        log_min = math.log(max(dim.min_value, 1e-12))
        log_max = math.log(max(dim.max_value, 1e-12))
        val = math.exp(log_min + u * (log_max - log_min))
    else:
        val = dim.min_value + u * (dim.max_value - dim.min_value)
    if dim.param_type == "integer":
        return round(val)
    return val


# ---------------------------------------------------------------------------
# LHS internal
# ---------------------------------------------------------------------------


def _lhs_unit_cube(
    n_samples: int, n_dims: int, rng: random.Random
) -> list[list[float]]:
    """Generate LHS samples in [0, 1]^n_dims.

    Standard algorithm: for each dimension, create N equal strata, randomly
    assign one sample per stratum, then sample uniformly within each stratum.
    """
    result: list[list[float]] = [[] for _ in range(n_samples)]
    for _d in range(n_dims):
        perm = list(range(n_samples))
        rng.shuffle(perm)
        for i in range(n_samples):
            low = perm[i] / n_samples
            high = (perm[i] + 1) / n_samples
            result[i].append(rng.uniform(low, high))
    return result


# ---------------------------------------------------------------------------
# Sampling strategies (all pure Python stdlib)
# ---------------------------------------------------------------------------


def sample_random(
    space: ParameterSpace, n: int, *, seed: int | None = None
) -> list[dict[str, Any]]:
    """Uniform random sampling within bounds."""
    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = []
    for _ in range(n):
        point: dict[str, Any] = {}
        for dim in space.dimensions:
            point[dim.param_name] = _sample_dimension(dim, rng)
        candidates.append(point)
    return candidates


def sample_lhs(
    space: ParameterSpace, n: int, *, seed: int | None = None
) -> list[dict[str, Any]]:
    """Latin Hypercube Sampling -- space-filling design."""
    rng = random.Random(seed)
    unit_samples = _lhs_unit_cube(n, space.n_dims, rng)
    candidates: list[dict[str, Any]] = []
    for unit_point in unit_samples:
        point: dict[str, Any] = {}
        for j, dim in enumerate(space.dimensions):
            point[dim.param_name] = _unit_to_value(unit_point[j], dim)
        candidates.append(point)
    return candidates


def sample_grid(
    space: ParameterSpace, n_per_dim: int | None = None
) -> list[dict[str, Any]]:
    """Exhaustive grid search across all dimensions.

    If *n_per_dim* is ``None``, defaults to 5 levels for continuous dimensions.
    Categorical/boolean dimensions always enumerate all choices.
    """
    dim_values: list[list[Any]] = []
    for dim in space.dimensions:
        if dim.choices is not None:
            dim_values.append(list(dim.choices))
        elif dim.param_type == "boolean":
            dim_values.append([False, True])
        else:
            levels = n_per_dim or 5
            if dim.min_value is None or dim.max_value is None:
                raise ValueError(
                    f"Dimension '{dim.param_name}' requires bounds for grid search"
                )
            if dim.log_scale:
                log_min = math.log(max(dim.min_value, 1e-12))
                log_max = math.log(max(dim.max_value, 1e-12))
                vals = [
                    math.exp(log_min + i * (log_max - log_min) / (levels - 1))
                    for i in range(levels)
                ]
            else:
                step = (dim.max_value - dim.min_value) / (levels - 1) if levels > 1 else 0
                vals = [dim.min_value + i * step for i in range(levels)]
            if dim.param_type == "integer":
                vals = sorted(set(round(v) for v in vals))
            dim_values.append(vals)

    candidates: list[dict[str, Any]] = []
    for combo in itertools.product(*dim_values):
        point: dict[str, Any] = {}
        for j, dim in enumerate(space.dimensions):
            point[dim.param_name] = combo[j]
        candidates.append(point)
    return candidates


def sample_prior_guided(
    space: ParameterSpace,
    n: int,
    *,
    k_stddev: float = 2.0,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Sample around memory priors (mean +/- k*stddev), clamped to bounds.

    Falls back to uniform random for dimensions without priors.
    """
    from app.services.memory import get_param_priors

    rng = random.Random(seed)

    # Pre-fetch priors for all dimensions
    priors: dict[str, Any] = {}
    for dim in space.dimensions:
        if dim.primitive and dim.param_type in ("number", "integer"):
            try:
                priors[dim.param_name] = get_param_priors(dim.primitive, dim.param_name)
            except Exception:
                priors[dim.param_name] = None
        else:
            priors[dim.param_name] = None

    # Pre-fetch evolved priors for tightened bounds
    evolved_bounds: dict[str, Any] = {}
    for dim in space.dimensions:
        if dim.primitive and dim.param_type in ("number", "integer"):
            try:
                from app.services.evolution import get_active_evolved_prior

                evolved_bounds[dim.param_name] = get_active_evolved_prior(
                    dim.primitive, dim.param_name
                )
            except Exception:
                evolved_bounds[dim.param_name] = None
        else:
            evolved_bounds[dim.param_name] = None

    candidates: list[dict[str, Any]] = []
    for _ in range(n):
        point: dict[str, Any] = {}
        for dim in space.dimensions:
            prior = priors.get(dim.param_name)
            if (
                prior is not None
                and prior.sample_count >= 3
                and dim.param_type in ("number", "integer")
            ):
                # Gaussian around prior mean, clamped to bounds
                stddev = prior.stddev if prior.stddev > 0 else abs(prior.mean * 0.1)
                val = rng.gauss(prior.mean, stddev * k_stddev / 2.0)

                # Use evolved bounds if available, else dimension bounds
                evolved = evolved_bounds.get(dim.param_name)
                if evolved is not None:
                    effective_min = evolved.evolved_min
                    effective_max = evolved.evolved_max
                else:
                    effective_min = dim.min_value
                    effective_max = dim.max_value

                if effective_min is not None:
                    val = max(val, effective_min)
                if effective_max is not None:
                    val = min(val, effective_max)
                if dim.param_type == "integer":
                    val = round(val)
                point[dim.param_name] = val
            else:
                point[dim.param_name] = _sample_dimension(dim, rng)
        candidates.append(point)
    return candidates


# ---------------------------------------------------------------------------
# Compositional (simplex-native) sampling
# ---------------------------------------------------------------------------


def sample_dirichlet(
    space: ParameterSpace,
    n: int,
    *,
    alpha: float | list[float] | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Sample compositions directly from a Dirichlet distribution.

    For each SimplexConstraint in the space, the constrained parameters are
    sampled from Dir(alpha) — this produces points that *naturally* lie on
    the simplex without post-normalization.

    Non-simplex dimensions are sampled uniformly (like ``sample_random``).

    Parameters
    ----------
    alpha : float | list[float] | None
        Dirichlet concentration parameter.
        - float → symmetric Dirichlet (all components share one alpha)
        - list[float] → asymmetric Dirichlet (per-component alpha)
        - None → defaults to 1.0 (uniform on the simplex)

    Returns
    -------
    list of param dicts with simplex dimensions summing to ``target_sum``.
    """
    rng = random.Random(seed)

    # Index which param names belong to which simplex constraint
    simplex_map: dict[str, SimplexConstraint] = {}
    for constraint in space.simplex_constraints:
        for name in constraint.param_names:
            simplex_map[name] = constraint

    candidates: list[dict[str, Any]] = []
    for _ in range(n):
        point: dict[str, Any] = {}

        # Sample simplex-constrained groups via Dirichlet
        done_constraints: set[int] = set()
        for dim in space.dimensions:
            if dim.param_name in simplex_map:
                constraint = simplex_map[dim.param_name]
                cid = id(constraint)
                if cid in done_constraints:
                    continue
                done_constraints.add(cid)

                k = len(constraint.param_names)
                if alpha is None:
                    alphas = [1.0] * k
                elif isinstance(alpha, (int, float)):
                    alphas = [float(alpha)] * k
                else:
                    alphas = list(alpha[:k])
                    while len(alphas) < k:
                        alphas.append(1.0)

                # Dirichlet via Gamma samples (pure stdlib)
                gammas = [_gamma_sample(a, rng) for a in alphas]
                total = sum(gammas)
                if total < 1e-15:
                    fracs = [1.0 / k] * k
                else:
                    fracs = [g / total for g in gammas]

                # Scale to target_sum
                for name, frac in zip(constraint.param_names, fracs):
                    point[name] = frac * constraint.target_sum
            else:
                # Non-simplex dimension: uniform random
                point[dim.param_name] = _sample_dimension(dim, rng)

        candidates.append(point)
    return candidates


def _gamma_sample(alpha: float, rng: random.Random) -> float:
    """Sample from Gamma(alpha, 1) using Marsaglia & Tsang's method.

    Pure stdlib implementation — no numpy required.
    """
    if alpha <= 0:
        return 0.0

    if alpha < 1.0:
        # Boost: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
        g = _gamma_sample(alpha + 1.0, rng)
        u = rng.random()
        return g * (u ** (1.0 / alpha)) if g > 0 else 0.0

    # Marsaglia & Tsang for alpha >= 1
    d = alpha - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)

    while True:
        while True:
            x = rng.gauss(0, 1)
            v = 1.0 + c * x
            if v > 0:
                break
        v = v * v * v
        u = rng.random()

        if u < 1.0 - 0.0331 * (x * x) * (x * x):
            return d * v
        if math.log(max(u, 1e-300)) < 0.5 * x * x + d * (1.0 - v + math.log(max(v, 1e-300))):
            return d * v


# ---------------------------------------------------------------------------
# Simplex constraint enforcement (post-normalization fallback)
# ---------------------------------------------------------------------------


def _apply_simplex_constraints(
    candidates: list[dict[str, Any]],
    constraints: tuple[SimplexConstraint, ...],
) -> list[dict[str, Any]]:
    """Normalize simplex-constrained parameter groups.

    For each candidate, each simplex constraint normalizes the named params
    so they sum to ``target_sum``.  If all values are zero, assigns equal
    fractions.  Negative values are clamped to zero before normalization.

    This is a post-sampling step — works with any sampling strategy.
    """
    result: list[dict[str, Any]] = []
    for point in candidates:
        point = dict(point)  # shallow copy
        for constraint in constraints:
            names = constraint.param_names
            target = constraint.target_sum
            # Collect current values; treat missing as 0
            vals = [max(0.0, float(point.get(n, 0.0))) for n in names]
            total = sum(vals)
            if total < 1e-15:
                # All zero → equal fractions
                equal = target / max(len(names), 1)
                for n in names:
                    point[n] = equal
            else:
                # Scale proportionally
                scale = target / total
                for n, v in zip(names, vals):
                    point[n] = v * scale
        result.append(point)
    return result


# ---------------------------------------------------------------------------
# Scoring (advisory, never blocks)
# ---------------------------------------------------------------------------


def _score_candidate(params: dict[str, Any], space: ParameterSpace) -> float | None:
    """Score a candidate based on proximity to memory priors.

    Returns mean z-score distance across dimensions with available priors.
    Lower = closer to known-good values.  ``None`` if no priors available.

    When evolved priors exist, also penalizes values outside evolved bounds.
    """
    from app.services.memory import get_param_priors

    scores: list[float] = []
    for dim in space.dimensions:
        if dim.primitive and dim.param_type in ("number", "integer"):
            try:
                prior = get_param_priors(dim.primitive, dim.param_name)
            except Exception:
                continue
            if prior is not None and prior.sample_count >= 3 and prior.stddev > 0:
                val = params.get(dim.param_name)
                if isinstance(val, (int, float)):
                    z = abs(float(val) - prior.mean) / prior.stddev

                    # Penalize values outside evolved bounds
                    try:
                        from app.services.evolution import get_active_evolved_prior

                        evolved = get_active_evolved_prior(dim.primitive, dim.param_name)
                        if evolved is not None:
                            fval = float(val)
                            if fval < evolved.evolved_min or fval > evolved.evolved_max:
                                z += 1.0  # penalty for out-of-evolved-bounds
                    except Exception:
                        pass

                    scores.append(z)
    return sum(scores) / len(scores) if scores else None


# ---------------------------------------------------------------------------
# Strategy dispatch
# ---------------------------------------------------------------------------


_STRATEGIES = {
    "lhs": "Latin Hypercube Sampling",
    "grid": "Grid Search",
    "random": "Random Sampling",
    "prior_guided": "Prior-Guided Sampling",
    "bayesian": "Bayesian Optimization (KNN surrogate + EI/UCB)",
    "adaptive": "Adaptive Strategy Selection (auto-selects best method per round)",
    "dirichlet": "Dirichlet Simplex Sampling (native compositional sampling)",
}


def generate_batch(
    space: ParameterSpace,
    strategy: str = "lhs",
    n_candidates: int = 10,
    *,
    seed: int | None = None,
    created_by: str = "system",
    campaign_id: str | None = None,
    acquisition: str = "ei",
    kpi_name: str = "run_success_rate",
    store: bool = True,
) -> BatchResult:
    """Generate a batch of candidate parameter sets and store them.

    This is the main entry point.  Pure algorithmic, no LLM calls.

    Parameters
    ----------
    acquisition:
        Acquisition function for Bayesian strategy: ``"ei"`` or ``"ucb"``.
    kpi_name:
        KPI to optimize when using Bayesian strategy.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Valid: {sorted(_STRATEGIES)}"
        )
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")

    # Generate raw parameter dicts
    if strategy == "lhs":
        raw_params = sample_lhs(space, n_candidates, seed=seed)
    elif strategy == "grid":
        raw_params = sample_grid(space)[:n_candidates]
    elif strategy == "random":
        raw_params = sample_random(space, n_candidates, seed=seed)
    elif strategy == "prior_guided":
        raw_params = sample_prior_guided(space, n_candidates, seed=seed)
    elif strategy == "bayesian":
        from app.services.bayesian_opt import load_observations_from_db, sample_bo

        observations = load_observations_from_db(
            space, campaign_id=campaign_id, kpi_name=kpi_name,
        )
        raw_params = sample_bo(
            space, n_candidates,
            observations=observations,
            acquisition=acquisition,
            seed=seed,
        )
    elif strategy == "dirichlet":
        raw_params = sample_dirichlet(space, n_candidates, seed=seed)
    elif strategy == "adaptive":
        # Delegate to the adaptive strategy selector
        from app.services.optimization_backends import Observation as OptObs
        from app.services.strategy_selector import (
            CampaignSnapshot,
            generate_adaptive_candidates,
        )
        from app.services.bayesian_opt import load_observations_from_db

        bo_obs = load_observations_from_db(
            space, campaign_id=campaign_id, kpi_name=kpi_name,
        )
        opt_obs = [
            OptObs(params={}, objective=obs.objective)
            for obs in bo_obs
        ]
        # Build snapshot from available context
        snapshot = CampaignSnapshot(
            round_number=max(1, len(bo_obs) // max(n_candidates, 1) + 1),
            max_rounds=20,  # sensible default; orchestrator passes real value
            n_observations=len(bo_obs),
            n_dimensions=space.n_dims,
            has_categorical=any(d.choices is not None for d in space.dimensions),
            has_log_scale=any(d.log_scale for d in space.dimensions),
            kpi_history=tuple(obs.objective for obs in bo_obs),
            direction="maximize",
        )
        raw_params, _decision = generate_adaptive_candidates(
            space, n_candidates, opt_obs, snapshot, seed=seed,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Apply simplex constraints (normalize composition groups)
    if space.simplex_constraints:
        raw_params = _apply_simplex_constraints(raw_params, space.simplex_constraints)

    # Score candidates (advisory, never fails)
    candidates: list[Candidate] = []
    for i, params in enumerate(raw_params):
        try:
            score = _score_candidate(params, space)
        except Exception:
            score = None
        candidates.append(Candidate(index=i, params=params, origin=strategy, score=score))

    # Sort by score if scores available (lower = better)
    scored = [c for c in candidates if c.score is not None]
    if scored:
        scored.sort(key=lambda c: c.score)  # type: ignore[arg-type]
        unscored = [c for c in candidates if c.score is None]
        candidates = scored + unscored
        # Re-index after sort
        candidates = [
            Candidate(index=i, params=c.params, origin=c.origin, score=c.score)
            for i, c in enumerate(candidates)
        ]

    batch_id = str(uuid.uuid4())
    result = BatchResult(
        batch_id=batch_id,
        candidates=tuple(candidates),
        strategy=strategy,
        space=space,
    )

    if store:
        _store_batch(result, created_by=created_by, campaign_id=campaign_id)
    return result


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def _serialize_space(space: ParameterSpace) -> dict[str, Any]:
    """Serialize a ParameterSpace for JSON storage."""
    return {
        "schema_version": BATCH_SCHEMA_VERSION,
        "dimensions": [
            {
                "param_name": d.param_name,
                "param_type": d.param_type,
                "min_value": d.min_value,
                "max_value": d.max_value,
                "log_scale": d.log_scale,
                "choices": list(d.choices) if d.choices else None,
                "step_key": d.step_key,
                "primitive": d.primitive,
            }
            for d in space.dimensions
        ],
    }


def _store_batch(
    result: BatchResult, *, created_by: str, campaign_id: str | None
) -> None:
    """Persist batch request and candidates to DB."""
    now = utcnow_iso()

    def _txn(conn: _sqlite3.Connection) -> None:
        # Orchestrator campaigns live in campaign_state, not campaigns.
        # Only reference campaign_id when it exists in the campaigns table
        # to avoid FK constraint violations.
        stored_campaign_id: str | None = None
        if campaign_id is not None:
            row = conn.execute(
                "SELECT 1 FROM campaigns WHERE id = ? LIMIT 1", (campaign_id,)
            ).fetchone()
            if row is not None:
                stored_campaign_id = campaign_id

        conn.execute(
            "INSERT INTO batch_requests "
            "(id, campaign_id, protocol_template_json, space_json, strategy, "
            "n_candidates, status, created_by, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'generated', ?, ?)",
            (
                result.batch_id,
                stored_campaign_id,
                json_dumps(result.space.protocol_template),
                json_dumps(_serialize_space(result.space)),
                result.strategy,
                len(result.candidates),
                created_by,
                now,
            ),
        )
        for c in result.candidates:
            conn.execute(
                "INSERT INTO batch_candidates "
                "(id, batch_id, candidate_index, params_json, origin, score, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    result.batch_id,
                    c.index,
                    json_dumps(c.params),
                    c.origin,
                    c.score,
                    now,
                ),
            )

    run_txn(_txn)


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------


def get_batch(batch_id: str) -> dict[str, Any] | None:
    """Return a batch request with all its candidates."""

    def _txn(conn: _sqlite3.Connection) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM batch_requests WHERE id = ?",
            (batch_id,),
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["protocol_template"] = parse_json(
            item.pop("protocol_template_json"), {}
        )
        item["space"] = parse_json(item.pop("space_json"), {})

        cand_rows = conn.execute(
            "SELECT * FROM batch_candidates WHERE batch_id = ? "
            "ORDER BY candidate_index ASC",
            (batch_id,),
        ).fetchall()
        item["candidates"] = [
            {
                "id": r["id"],
                "candidate_index": r["candidate_index"],
                "params": parse_json(r["params_json"], {}),
                "origin": r["origin"],
                "score": r["score"],
                "selected_run_id": r["selected_run_id"],
            }
            for r in cand_rows
        ]
        return item

    return run_txn(_txn)


def list_candidates(batch_id: str) -> list[dict[str, Any]]:
    """Return all candidates for a batch, ordered by index."""

    def _txn(conn: _sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT * FROM batch_candidates WHERE batch_id = ? "
            "ORDER BY candidate_index ASC",
            (batch_id,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "batch_id": r["batch_id"],
                "candidate_index": r["candidate_index"],
                "params": parse_json(r["params_json"], {}),
                "origin": r["origin"],
                "score": r["score"],
                "selected_run_id": r["selected_run_id"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    return run_txn(_txn)


def list_batches(
    campaign_id: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """Return batch requests, optionally filtered by campaign."""

    def _txn(conn: _sqlite3.Connection) -> list[dict[str, Any]]:
        if campaign_id:
            rows = conn.execute(
                "SELECT * FROM batch_requests WHERE campaign_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (campaign_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM batch_requests "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            item["protocol_template"] = parse_json(
                item.pop("protocol_template_json"), {}
            )
            item["space"] = parse_json(item.pop("space_json"), {})
            result.append(item)
        return result

    return run_txn(_txn)
