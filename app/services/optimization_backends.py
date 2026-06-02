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
from typing import Any, Callable, Protocol, runtime_checkable

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


# ===================================================================
# Unified Optimizer Layer
# ===================================================================
#
# The backends above are *stateless* one-shot samplers: every ``suggest()``
# call must be handed the full observation history.  This works, but it forces
# the campaign loop to re-load observations from the DB on every round
# (latency + missed mid-campaign learning) and keeps the Bayesian and RL code
# paths completely siloed.
#
# ``UnifiedOptimizer`` is a *stateful*, async-friendly facade over the
# stateless backends.  It owns an in-memory ``ObservationBuffer`` so that a
# completed run can be folded into the model immediately via
# ``incremental_observe()`` -- without waiting for the next ``suggest()`` call
# and without a DB round-trip.  Both BO and RL implementations share the same
# interface so ``strategy_router`` can route by phase through one object.

import asyncio
import time as _time
from abc import ABC, abstractmethod
from threading import Lock as _Lock

# Action -> backend mapping is owned by the RL module; import lazily inside
# RLOptimizer to avoid a hard import cycle (rl_strategy_selector imports from
# strategy_selector which imports from this module).


@dataclass
class ObservationBuffer:
    """In-memory, thread-safe ring of recent observations for one campaign.

    Holds *de-normalised* param dicts (what the rest of HELIOS speaks) plus the
    objective.  ``to_observations()`` converts to the stateless
    :class:`Observation` understood by the backends.  The buffer is the live
    feedback channel: ``add()`` is called the instant a run completes, so the
    surrogate / replay buffer sees the result before the next ``suggest()``.
    """

    capacity: int = 512
    _records: list[Observation] = field(default_factory=list)
    _lock: _Lock = field(default_factory=_Lock, repr=False)

    def add(self, params: dict[str, Any], objective: float) -> None:
        rec = Observation(params=dict(params), objective=float(objective))
        with self._lock:
            self._records.append(rec)
            if len(self._records) > self.capacity:
                # Drop oldest; recent observations dominate the surrogate.
                del self._records[: len(self._records) - self.capacity]

    def get_recent(self, n: int) -> list[Observation]:
        with self._lock:
            if n <= 0:
                return []
            return list(self._records[-n:])

    def to_observations(self) -> list[Observation]:
        with self._lock:
            return list(self._records)

    def merge(self, observations: list[Observation]) -> None:
        """Seed the buffer (e.g. with DB history) without duplicating records."""
        with self._lock:
            seen = {
                (tuple(sorted(r.params.items())), r.objective)
                for r in self._records
            }
            for obs in observations:
                key = (tuple(sorted(obs.params.items())), obs.objective)
                if key not in seen:
                    self._records.append(obs)
                    seen.add(key)
            if len(self._records) > self.capacity:
                del self._records[: len(self._records) - self.capacity]

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


@dataclass(frozen=True)
class SuggestResult:
    """Outcome of a unified ``suggest()`` call."""

    candidates: list[dict[str, Any]]
    backend_name: str
    n_observations: int
    metadata: dict[str, Any] = field(default_factory=dict)


class UnifiedOptimizer(ABC):
    """Stateful optimizer facade with an incremental feedback channel.

    Subclasses wrap a stateless backend (BO) or an agent (RL) and maintain a
    per-campaign :class:`ObservationBuffer`.  All public methods are async so
    the campaign orchestrator can await them without blocking the event loop;
    CPU-bound model work is dispatched to a worker thread.
    """

    name: str = "unified"

    def __init__(
        self,
        space: ParameterSpace,
        *,
        buffer: ObservationBuffer | None = None,
        retrain_threshold: int = 1,
    ) -> None:
        self.space = space
        self.buffer = buffer if buffer is not None else ObservationBuffer()
        self._retrain_threshold = max(1, retrain_threshold)
        self._pending_since_train = 0
        self._log = logging.getLogger(f"{__name__}.{type(self).__name__}")

    # -- feedback channel (the critical addition) --

    async def incremental_observe(
        self,
        params: dict[str, Any],
        objective: float,
    ) -> None:
        """Fold one completed run into the model *immediately*.

        Called from the campaign loop's ``on_run_complete`` hook right after a
        KPI is computed.  Updates the in-memory buffer and -- once enough new
        observations have accrued -- refreshes the underlying model so the next
        ``suggest()`` reflects the latest data with zero DB latency.
        """
        self.buffer.add(params, objective)
        self._pending_since_train += 1
        self._log.info(
            "incremental_observe",
            extra={
                "optimizer": self.name,
                "objective": float(objective),
                "buffer_size": len(self.buffer),
                "pending": self._pending_since_train,
            },
        )
        if self._pending_since_train >= self._retrain_threshold:
            await self._on_retrain(self.buffer.to_observations())
            self._pending_since_train = 0

    def seed(self, observations: list[Observation]) -> None:
        """Warm-start the buffer from DB history (cold-start recovery)."""
        self.buffer.merge(observations)

    # -- suggestion (shared async wrapper around the abstract sync core) --

    async def suggest(
        self,
        n: int,
        *,
        observations: list[Observation] | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> SuggestResult:
        """Return *n* candidates using buffer observations plus any extras.

        ``observations`` are merged with the live buffer so a cold optimizer
        can still be primed with DB history on the very first call.
        """
        obs = self.buffer.to_observations()
        if observations:
            extra_keys = {
                (tuple(sorted(o.params.items())), o.objective) for o in obs
            }
            for o in observations:
                key = (tuple(sorted(o.params.items())), o.objective)
                if key not in extra_keys:
                    obs.append(o)
                    extra_keys.add(key)

        candidates = await asyncio.to_thread(
            self._suggest_sync, n, obs, seed, kwargs
        )
        return SuggestResult(
            candidates=candidates,
            backend_name=self.name,
            n_observations=len(obs),
            metadata={"requested": n, "returned": len(candidates)},
        )

    # -- abstract hooks for subclasses --

    @abstractmethod
    def _suggest_sync(
        self,
        n: int,
        observations: list[Observation],
        seed: int | None,
        kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Synchronous, CPU-bound candidate generation (runs in a thread)."""

    async def _on_retrain(self, observations: list[Observation]) -> None:
        """Hook invoked when the retrain threshold is crossed. No-op default."""
        return None


class BayesianOptimizer(UnifiedOptimizer):
    """UnifiedOptimizer backed by a stateless BO backend (default ``built_in``).

    The surrogate is rebuilt from buffer observations on each ``suggest()`` /
    retrain, so ``incremental_observe()`` makes new KPIs available without the
    DB polling that ``load_observations_from_db`` previously required.
    """

    def __init__(
        self,
        space: ParameterSpace,
        *,
        backend_name: str = "built_in",
        acquisition: str = "ei",
        buffer: ObservationBuffer | None = None,
        retrain_threshold: int = 1,
    ) -> None:
        super().__init__(space, buffer=buffer, retrain_threshold=retrain_threshold)
        self.name = f"bo:{backend_name}"
        self._backend: BackendProtocol = get_backend(backend_name)
        self._acquisition = acquisition

    def _suggest_sync(
        self,
        n: int,
        observations: list[Observation],
        seed: int | None,
        kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        call_kwargs = dict(kwargs)
        call_kwargs.setdefault("acquisition", self._acquisition)
        t0 = _time.perf_counter()
        candidates = self._backend.suggest(
            self.space, n, observations, seed=seed, **call_kwargs
        )
        self._log.info(
            "bo_suggest",
            extra={
                "optimizer": self.name,
                "n_obs": len(observations),
                "n_returned": len(candidates),
                "latency_ms": round((_time.perf_counter() - t0) * 1000, 2),
            },
        )
        return candidates

    async def _on_retrain(self, observations: list[Observation]) -> None:
        # KNN surrogate is rebuilt lazily inside suggest(); nothing to persist.
        self._log.debug(
            "bo_buffer_refreshed",
            extra={"optimizer": self.name, "n_obs": len(observations)},
        )


class RLOptimizer(UnifiedOptimizer):
    """UnifiedOptimizer backed by the RL strategy selector + experience replay.

    ``suggest()`` asks the RL agent which *backend* to use for the current
    phase, then delegates candidate generation to that backend.
    ``incremental_observe()`` pushes the latest run into the agent's experience
    replay buffer and applies a Q-update, so the policy keeps learning online.
    """

    def __init__(
        self,
        space: ParameterSpace,
        *,
        buffer: ObservationBuffer | None = None,
        retrain_threshold: int = 1,
        explore: bool = True,
        selector: Any | None = None,
        snapshot_provider: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(space, buffer=buffer, retrain_threshold=retrain_threshold)
        self.name = "rl"
        self._explore = explore
        self._snapshot_provider = snapshot_provider
        self._prev_state: Any | None = None
        self._prev_action: int | None = None
        self._prev_best: float | None = None

        if selector is not None:
            self._selector = selector
        else:
            from app.services.rl_strategy_selector import get_rl_selector

            self._selector = get_rl_selector()

    def _current_diagnostics(self) -> tuple[Any, Any]:
        from app.services.rl_strategy_selector import compute_diagnostics

        if self._snapshot_provider is None:
            raise RuntimeError(
                "RLOptimizer requires a snapshot_provider to build campaign state"
            )
        snapshot = self._snapshot_provider()
        diagnostics = compute_diagnostics(snapshot)
        return snapshot, diagnostics

    def _suggest_sync(
        self,
        n: int,
        observations: list[Observation],
        seed: int | None,
        kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        from app.services.rl_strategy_selector import RLState

        snapshot, diagnostics = self._current_diagnostics()
        action, backend_name = self._selector.select_action(
            snapshot, diagnostics, explore=self._explore
        )
        # Remember the (state, action) so the next incremental_observe() can
        # close the loop with a reward.
        self._prev_state = RLState.from_snapshot(snapshot, diagnostics)
        self._prev_action = action

        backend = get_backend(backend_name)
        candidates = backend.suggest(self.space, n, observations, seed=seed, **kwargs)
        self._log.info(
            "rl_suggest",
            extra={
                "optimizer": self.name,
                "action": action,
                "backend": backend_name,
                "n_obs": len(observations),
                "n_returned": len(candidates),
            },
        )
        return candidates

    async def _on_retrain(self, observations: list[Observation]) -> None:
        """Convert the newest observation into an RL transition and learn."""
        if self._prev_state is None or self._prev_action is None:
            return  # no suggest() preceded this observation

        from app.services.rl_strategy_selector import RLState

        best = max((o.objective for o in observations), default=None)
        if best is None:
            return
        prev_best = self._prev_best if self._prev_best is not None else best
        cfg = self._selector.config
        reward = (best - prev_best) * cfg.kpi_improvement_scale + cfg.round_cost
        self._prev_best = best

        next_state: Any | None = None
        done = False
        try:
            snapshot, diagnostics = self._current_diagnostics()
            next_state = RLState.from_snapshot(snapshot, diagnostics)
            done = snapshot.round_number >= snapshot.max_rounds
        except RuntimeError:
            pass  # no snapshot provider; terminal transition

        await asyncio.to_thread(
            self._selector.learn_from_experience,
            self._prev_state,
            self._prev_action,
            reward,
            next_state,
            done,
        )
        self._log.info(
            "rl_online_update",
            extra={
                "optimizer": self.name,
                "action": self._prev_action,
                "reward": round(reward, 4),
                "done": done,
            },
        )


# ---------------------------------------------------------------------------
# Facade factory
# ---------------------------------------------------------------------------

@register_backend
class GPBackend:
    """Research-grade GP backend (ARD-Matern52, multiple acquisition functions).

    Replaces BuiltInBO as the default when sufficient observations exist (≥ 5×n_dims).
    Falls back to BuiltInBO automatically when gp_surrogate is unavailable.
    """

    name = "gp_backend"

    def suggest(
        self,
        space: ParameterSpace,
        n: int,
        observations: list[Observation],
        *,
        seed: int | None = None,
        acquisition: str = "ei",
        optimize_hyperparams: bool = True,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        try:
            from app.services.gp_surrogate import sample_bo_gp
            return sample_bo_gp(
                space, n, observations,
                acquisition=acquisition,
                seed=seed,
                optimize_hyperparams=optimize_hyperparams and len(observations) >= 10,
            )
        except ImportError:
            logger.warning("gp_surrogate not available, falling back to built_in")
            return _BACKENDS["built_in"]().suggest(space, n, observations, seed=seed)

    @staticmethod
    def is_available() -> bool:
        try:
            from app.services.gp_surrogate import GPSurrogate  # noqa: F401
            return True
        except ImportError:
            return False


def build_optimizer(
    space: ParameterSpace,
    *,
    mode: str = "bayesian",
    buffer: ObservationBuffer | None = None,
    retrain_threshold: int = 1,
    **kwargs: Any,
) -> UnifiedOptimizer:
    """Construct a :class:`UnifiedOptimizer` for a campaign.

    ``mode`` is ``"bayesian"`` or ``"rl"``.  The returned object is stateful and
    should be cached per-campaign so its ObservationBuffer / RL state persists
    across rounds.
    """
    mode_norm = mode.strip().lower()
    if mode_norm in ("bayesian", "bo", "built_in"):
        return BayesianOptimizer(
            space, buffer=buffer, retrain_threshold=retrain_threshold, **kwargs
        )
    if mode_norm in ("rl", "reinforcement"):
        return RLOptimizer(
            space, buffer=buffer, retrain_threshold=retrain_threshold, **kwargs
        )
    raise ValueError(f"Unknown optimizer mode '{mode}'. Use 'bayesian' or 'rl'.")
