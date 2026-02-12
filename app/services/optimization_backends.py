"""Pluggable optimization backends for experiment parameter tuning.

Provides a unified interface over multiple optimization engines:
- **built_in**: Pure-Python KNN-surrogate BO (no external deps, always available)
- **optuna_tpe**: Optuna's Tree-structured Parzen Estimator (optional)
- **optuna_cmaes**: Optuna's CMA-ES sampler (optional)
- **scipy_de**: SciPy Differential Evolution (optional)
- **pymoo_nsga2**: pymoo NSGA-II evolutionary (optional, multi-objective)

Each backend conforms to a ``BackendProtocol`` so the strategy selector
can swap them transparently.  Backends that require optional dependencies
gracefully degrade to the built-in fallback.

No backend modifies the database -- they return raw parameter dicts that
the caller (candidate_gen / design_agent) persists.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from app.services.candidate_gen import (
    ParameterSpace,
    SearchDimension,
    _unit_to_value,
    sample_lhs,
    sample_random,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Observation data (shared across backends)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """A past evaluation: param dict → objective value(s)."""
    params: dict[str, Any]
    objective: float  # primary KPI (higher = better after direction flip)
    objectives: dict[str, float] | None = None  # for multi-objective


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class BackendProtocol(Protocol):
    """All optimization backends implement this interface."""

    name: str

    def suggest(
        self,
        space: ParameterSpace,
        n: int,
        observations: list[Observation],
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return *n* candidate parameter dicts."""
        ...

    @staticmethod
    def is_available() -> bool:
        """Return True if the backend's dependencies are installed."""
        ...


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type] = {}


def register_backend(cls: type) -> type:
    """Decorator to register a backend class."""
    _BACKENDS[cls.name] = cls
    return cls


def get_backend(name: str) -> BackendProtocol:
    """Instantiate a backend by name, with fallback to built_in."""
    cls = _BACKENDS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {sorted(_BACKENDS)}"
        )
    instance = cls()
    if not instance.is_available():
        logger.warning(
            "Backend '%s' not available (missing deps), falling back to 'built_in'",
            name,
        )
        return _BACKENDS["built_in"]()
    return instance


def list_backends() -> dict[str, bool]:
    """Return {name: is_available} for all registered backends."""
    result = {}
    for name, cls in _BACKENDS.items():
        try:
            result[name] = cls.is_available()
        except Exception:
            result[name] = False
    return result


# ---------------------------------------------------------------------------
# Helper: ParameterSpace → normalized [0,1] <-> actual value
# ---------------------------------------------------------------------------

def _normalize_params(
    params: dict[str, Any], space: ParameterSpace,
) -> list[float]:
    """Convert actual params to [0,1]^d (dimension order)."""
    from app.services.bayesian_opt import _value_to_unit

    result = []
    for dim in space.dimensions:
        val = params.get(dim.param_name)
        if val is None:
            result.append(0.5)
        else:
            u = _value_to_unit(val, dim)
            result.append(max(0.0, min(1.0, u)))
    return result


def _denormalize_point(
    point: list[float], space: ParameterSpace,
) -> dict[str, Any]:
    """Convert [0,1]^d back to actual param dict."""
    params: dict[str, Any] = {}
    for j, dim in enumerate(space.dimensions):
        params[dim.param_name] = _unit_to_value(point[j], dim)
    return params


# ===================================================================
# Backend: Built-in KNN Surrogate BO (always available)
# ===================================================================

@register_backend
class BuiltInBO:
    """Pure-Python KNN-surrogate BO -- the original implementation."""

    name = "built_in"

    def suggest(
        self,
        space: ParameterSpace,
        n: int,
        observations: list[Observation],
        *,
        seed: int | None = None,
        acquisition: str = "ei",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        from app.services.bayesian_opt import (
            Observation as BOObs,
            normalize_params,
            sample_bo,
        )

        bo_obs = [
            BOObs(
                params=tuple(normalize_params(obs.params, space)),
                objective=obs.objective,
            )
            for obs in observations
        ]
        return sample_bo(
            space, n,
            observations=bo_obs,
            acquisition=acquisition,
            seed=seed,
        )

    @staticmethod
    def is_available() -> bool:
        return True  # always available -- pure Python


# ===================================================================
# Backend: Optuna TPE
# ===================================================================

@register_backend
class OptunaTPE:
    """Optuna Tree-structured Parzen Estimator.

    Requires: ``pip install optuna``
    """

    name = "optuna_tpe"

    def suggest(
        self,
        space: ParameterSpace,
        n: int,
        observations: list[Observation],
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Replay past observations via enqueue + tell
        for obs in observations:
            trial = optuna.trial.create_trial(
                params=_optuna_params_dict(obs.params, space),
                distributions=_optuna_distributions(space),
                values=[obs.objective],
            )
            study.add_trial(trial)

        # Ask for n new trials
        results: list[dict[str, Any]] = []
        for _ in range(n):
            trial = study.ask(_optuna_distributions(space))
            params = _optuna_trial_to_params(trial, space)
            results.append(params)
            # Tell the study a dummy value so it can inform future asks
            # (we don't have the real value yet -- use 0.0 as placeholder)
            study.tell(trial, 0.0, state=optuna.trial.TrialState.PRUNED)

        return results

    @staticmethod
    def is_available() -> bool:
        try:
            import optuna  # noqa: F401
            return True
        except ImportError:
            return False


def _optuna_distributions(space: ParameterSpace) -> dict:
    """Build Optuna distributions from ParameterSpace."""
    import optuna

    dists = {}
    for dim in space.dimensions:
        if dim.choices is not None:
            dists[dim.param_name] = optuna.distributions.CategoricalDistribution(
                list(dim.choices)
            )
        elif dim.param_type == "boolean":
            dists[dim.param_name] = optuna.distributions.CategoricalDistribution(
                [True, False]
            )
        elif dim.param_type == "integer":
            dists[dim.param_name] = optuna.distributions.IntDistribution(
                int(dim.min_value or 0),
                int(dim.max_value or 100),
                log=dim.log_scale,
            )
        else:
            dists[dim.param_name] = optuna.distributions.FloatDistribution(
                dim.min_value or 0.0,
                dim.max_value or 1.0,
                log=dim.log_scale,
            )
    return dists


def _optuna_params_dict(params: dict[str, Any], space: ParameterSpace) -> dict:
    """Ensure params match Optuna distribution types."""
    result = {}
    for dim in space.dimensions:
        val = params.get(dim.param_name)
        if val is None:
            continue
        if dim.param_type == "integer":
            result[dim.param_name] = int(val)
        elif dim.param_type in ("number",):
            result[dim.param_name] = float(val)
        else:
            result[dim.param_name] = val
    return result


def _optuna_trial_to_params(trial: Any, space: ParameterSpace) -> dict[str, Any]:
    """Extract params from an Optuna trial."""
    params = {}
    for dim in space.dimensions:
        params[dim.param_name] = trial.params[dim.param_name]
    return params


# ===================================================================
# Backend: Optuna CMA-ES
# ===================================================================

@register_backend
class OptunaCMAES:
    """Optuna CMA-ES sampler — best for continuous low-dimensional spaces.

    Requires: ``pip install optuna``
    """

    name = "optuna_cmaes"

    def suggest(
        self,
        space: ParameterSpace,
        n: int,
        observations: list[Observation],
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.CmaEsSampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Replay past observations
        for obs in observations:
            trial = optuna.trial.create_trial(
                params=_optuna_params_dict(obs.params, space),
                distributions=_optuna_distributions(space),
                values=[obs.objective],
            )
            study.add_trial(trial)

        results: list[dict[str, Any]] = []
        for _ in range(n):
            trial = study.ask(_optuna_distributions(space))
            params = _optuna_trial_to_params(trial, space)
            results.append(params)
            study.tell(trial, 0.0, state=optuna.trial.TrialState.PRUNED)

        return results

    @staticmethod
    def is_available() -> bool:
        try:
            import optuna  # noqa: F401
            return True
        except ImportError:
            return False


# ===================================================================
# Backend: SciPy Differential Evolution
# ===================================================================

@register_backend
class ScipyDE:
    """SciPy Differential Evolution — robust global optimizer.

    Good for multi-modal landscapes and mixed discrete/continuous spaces.
    Requires: ``pip install scipy``
    """

    name = "scipy_de"

    def suggest(
        self,
        space: ParameterSpace,
        n: int,
        observations: list[Observation],
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        from scipy.optimize import differential_evolution
        from scipy.stats.qmc import LatinHypercube

        # Build bounds for continuous dimensions
        bounds = []
        for dim in space.dimensions:
            if dim.choices is not None:
                bounds.append((0.0, len(dim.choices) - 0.001))
            elif dim.param_type == "boolean":
                bounds.append((0.0, 1.0))
            else:
                bounds.append((
                    dim.min_value if dim.min_value is not None else 0.0,
                    dim.max_value if dim.max_value is not None else 1.0,
                ))

        # Build surrogate from observations for the objective
        if observations:
            from app.services.bayesian_opt import SurrogateModel, Observation as BOObs

            bo_obs = [
                BOObs(
                    params=tuple(_normalize_params(obs.params, space)),
                    objective=obs.objective,
                )
                for obs in observations
            ]
            surrogate = SurrogateModel(bo_obs, k=min(5, len(bo_obs)))

            def neg_surrogate(x: list[float]) -> float:
                # DE minimizes, we want to maximize
                mean, _ = surrogate.predict(tuple(x))
                return -mean

            objective_fn = neg_surrogate
            x_bounds = [(0.0, 1.0)] * space.n_dims
        else:
            # No observations — use LHS fallback
            return sample_lhs(space, n, seed=seed)

        # Run DE to find multiple promising points
        results: list[dict[str, Any]] = []
        for i in range(n):
            iter_seed = (seed or 42) + i
            result = differential_evolution(
                objective_fn,
                bounds=x_bounds,
                seed=iter_seed,
                maxiter=50,
                popsize=10,
                tol=0.01,
            )
            params = _denormalize_point(list(result.x), space)
            results.append(params)

        return results

    @staticmethod
    def is_available() -> bool:
        try:
            from scipy.optimize import differential_evolution  # noqa: F401
            return True
        except ImportError:
            return False


# ===================================================================
# Backend: pymoo NSGA-II (multi-objective evolutionary)
# ===================================================================

@register_backend
class PymooNSGA2:
    """pymoo NSGA-II evolutionary algorithm.

    Best for multi-objective optimization and discrete/combinatorial spaces.
    Requires: ``pip install pymoo``
    """

    name = "pymoo_nsga2"

    def suggest(
        self,
        space: ParameterSpace,
        n: int,
        observations: list[Observation],
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        import numpy as np
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.soo.nonconvex.ga import GA
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.lhs import LHS as PymooLHS
        from pymoo.optimize import minimize as pymoo_minimize
        from pymoo.termination import get_termination

        n_dims = space.n_dims

        # Build surrogate from observations
        if not observations:
            return sample_lhs(space, n, seed=seed)

        from app.services.bayesian_opt import SurrogateModel, Observation as BOObs

        bo_obs = [
            BOObs(
                params=tuple(_normalize_params(obs.params, space)),
                objective=obs.objective,
            )
            for obs in observations
        ]
        surrogate = SurrogateModel(bo_obs, k=min(5, len(bo_obs)))

        class SurrogateProblem(ElementwiseProblem):
            def __init__(self):
                super().__init__(
                    n_var=n_dims,
                    n_obj=1,
                    xl=np.zeros(n_dims),
                    xu=np.ones(n_dims),
                )

            def _evaluate(self, x, out, *args, **kw):
                mean, _ = surrogate.predict(tuple(x))
                out["F"] = [-mean]  # pymoo minimizes; negate for maximization

        problem = SurrogateProblem()
        algorithm = GA(
            pop_size=max(n * 4, 20),
            sampling=PymooLHS(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
        )

        termination = get_termination("n_gen", 30)

        result = pymoo_minimize(
            problem,
            algorithm,
            termination,
            seed=seed or 42,
            verbose=False,
        )

        # Extract top-n solutions
        if result.X is not None:
            if result.X.ndim == 1:
                points = [result.X.tolist()]
            else:
                # Sort by fitness and take top n
                indices = result.F[:, 0].argsort()[:n]
                points = [result.X[i].tolist() for i in indices]
        else:
            return sample_lhs(space, n, seed=seed)

        # Pad if needed
        while len(points) < n:
            rng = random.Random((seed or 42) + len(points))
            points.append([rng.random() for _ in range(n_dims)])

        return [_denormalize_point(pt, space) for pt in points[:n]]

    @staticmethod
    def is_available() -> bool:
        try:
            import pymoo  # noqa: F401
            import numpy  # noqa: F401
            return True
        except ImportError:
            return False


# ===================================================================
# Backend: LHS (pure-Python, always available)
# ===================================================================

@register_backend
class LHSBackend:
    """Latin Hypercube Sampling — space-filling exploration."""

    name = "lhs"

    def suggest(
        self,
        space: ParameterSpace,
        n: int,
        observations: list[Observation],
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return sample_lhs(space, n, seed=seed)

    @staticmethod
    def is_available() -> bool:
        return True


@register_backend
class RandomBackend:
    """Uniform random sampling."""

    name = "random_sampling"

    def suggest(
        self,
        space: ParameterSpace,
        n: int,
        observations: list[Observation],
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return sample_random(space, n, seed=seed)

    @staticmethod
    def is_available() -> bool:
        return True
