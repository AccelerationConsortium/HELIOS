"""Bridge the ``bo-engine`` (bo-mcp) BoTorch optimizer into HELIOS's backend registry.

``bo-engine`` is a deep, single-paradigm Bayesian-optimization engine (real
Gaussian-process surrogates, qLogNEI/qLogNEHVI acquisition, TuRBO, SAASBO, ...).
HELIOS's built-in BO is a pure-Python KNN heuristic; this bridge lets the
strategy selector borrow real GP-BO on exploit rounds while HELIOS keeps
authority over phase selection, execution, recovery, and provenance.

Adaptation (see docs/plans/2026-06-26-bomcp-backend-integration-design.md):

* ``to_bomcp_spec``         -- HELIOS ``ParameterSpace`` -> bo-engine ``OptimizationSpec``
* ``to_bomcp_observations`` -- HELIOS ``Observation``    -> bo-engine ``ObservationData``
* ``from_bomcp_suggestion`` -- bo-engine ``SuggestionResult`` -> HELIOS param dict

Two transforms carry real risk and are unit-tested:

1. **Direction** -- HELIOS's ``Observation.objective`` is already maximization
   form (higher = better), so the objective maps to ``minimize=False`` and the
   value is fed *un-negated*.  Re-flipping on ``direction`` would invert
   minimize problems (the bug this bridge is written to avoid).
2. **log_scale inputs** -- bo-engine has no per-parameter log sampling, so a
   ``log_scale`` dimension is fit in ``log10`` space (bounds and observed values
   are ``log10``'d going in; suggested values are ``10**x``'d coming out).

The module imports cleanly even without bo-engine installed; ``bo_engine`` is
imported lazily inside ``suggest`` so ``is_available()`` (not an ``ImportError``)
governs whether the backend is used.  Statelessness is deliberate for v1: the
full observation history is passed every round and the GP is refit, so no
``turbo_state`` round-trip is needed (TuRBO continuity / pending points are a
documented follow-up).
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from app.services.optimization_backends import Observation, register_backend

if TYPE_CHECKING:  # pragma: no cover
    from app.services.candidate_gen import ParameterSpace

logger = logging.getLogger(__name__)

_DEFAULT_SEED = 42
_OBJECTIVE_NAME = "objective"
# At/above this dimensionality, enable TuRBO (trust-region BO) for an
# all-continuous space.  TuRBO is single-objective + continuous only, which the
# bomcp bridge always is (single maximization objective; categorical dims
# disable it).  Below this, the standard qLogNEI path is used (no state).
_TURBO_MIN_DIMS = 5


def _bomcp_available() -> bool:
    """Return True if the bo-engine library can be imported."""
    try:
        import bo_engine  # noqa: F401
    except Exception:  # ImportError, or env/dep issues -- never used if unavailable
        return False
    return True


# ---------------------------------------------------------------------------
# Per-dimension transforms (kept tiny + pure for unit testing)
# ---------------------------------------------------------------------------


def _is_log(dim: Any) -> bool:
    return bool(getattr(dim, "log_scale", False))


def _encode_value(dim: Any, value: Any) -> Any:
    """HELIOS value -> bo-engine value (going *into* the engine)."""
    if dim.param_type in ("categorical", "boolean"):
        return str(value)
    if _is_log(dim):
        return math.log10(float(value))
    return float(value)


def _decode_value(dim: Any, value: Any) -> Any:
    """bo-engine value -> HELIOS value (coming *out* of the engine)."""
    if dim.param_type in ("categorical", "boolean"):
        # Map the category string back to the original choice object.
        for choice in dim.choices or ():
            if str(choice) == str(value):
                return choice
        return value
    if _is_log(dim):
        return 10.0 ** float(value)
    if dim.param_type == "integer":
        return int(round(float(value)))
    return float(value)


def _encoded_bounds(dim: Any) -> tuple[float, float]:
    lo, hi = float(dim.min_value), float(dim.max_value)
    if _is_log(dim):
        return (math.log10(lo), math.log10(hi))
    return (lo, hi)


# ---------------------------------------------------------------------------
# Conversions (HELIOS <-> bo-engine data models)
# ---------------------------------------------------------------------------


def to_bomcp_spec(space: ParameterSpace, batch_size: int, seed: int | None) -> Any:
    """Convert a HELIOS ``ParameterSpace`` to a bo-engine ``OptimizationSpec``."""
    from bo_engine.types import (
        ConstraintSpec,
        ConstraintType,
        ObjectiveSpec,
        OptimizationSpec,
        OutcomeConstraintSpec,
        ParameterSpec,
        ParameterType,
        TurboConfig,
    )

    params: list[Any] = []
    for dim in space.dimensions:
        if dim.param_type in ("categorical", "boolean"):
            params.append(
                ParameterSpec(
                    name=dim.param_name,
                    type=ParameterType.CATEGORICAL,
                    categories=[str(c) for c in (dim.choices or ())],
                )
            )
        else:  # number / integer (integer rounded on the way out)
            params.append(
                ParameterSpec(
                    name=dim.param_name,
                    type=ParameterType.CONTINUOUS,
                    bounds=_encoded_bounds(dim),
                )
            )

    # HELIOS objective is already higher = better -> maximize (minimize=False).
    objectives = [ObjectiveSpec(name=_OBJECTIVE_NAME, minimize=False)]

    def _log_in(names: Any) -> bool:
        # A linear form over log-scaled params has no linear analogue in log
        # space; skip such constraints (rare) rather than emit an incorrect one.
        return any(_is_log(d) for d in space.dimensions if d.param_name in set(names))

    constraints: list[Any] = []

    # Simplex maps cleanly to SUM_EQUALS.
    for sc in space.simplex_constraints:
        if _log_in(sc.param_names):
            logger.debug("Skipping simplex constraint over log-scaled params for bomcp")
            continue
        constraints.append(
            ConstraintSpec(
                type=ConstraintType.SUM_EQUALS,
                parameters=list(sc.param_names),
                value=float(sc.target_sum),
            )
        )

    # Linear (in)equalities: unit coefficients collapse to the SUM_* forms;
    # weighted coefficients use the general LINEAR form.
    _op_sum = {"<=": ConstraintType.SUM_LESS_THAN,
               ">=": ConstraintType.SUM_GREATER_THAN,
               "==": ConstraintType.SUM_EQUALS}
    for lc in space.linear_constraints:
        if _log_in(lc.param_names):
            logger.debug("Skipping linear constraint over log-scaled params for bomcp")
            continue
        unit = all(abs(c - 1.0) < 1e-12 for c in lc.coefficients)
        if unit and lc.op in _op_sum:
            constraints.append(
                ConstraintSpec(
                    type=_op_sum[lc.op],
                    parameters=list(lc.param_names),
                    value=float(lc.bound),
                )
            )
        else:
            constraints.append(
                ConstraintSpec(
                    type=ConstraintType.LINEAR,
                    parameters=list(lc.param_names),
                    value=float(lc.bound),
                    coefficients=[float(c) for c in lc.coefficients],
                )
            )

    # Outcome constraints: learned feasibility on a measured response.
    outcome_constraints = [
        OutcomeConstraintSpec(
            objective_name=oc.objective_name,
            threshold=float(oc.threshold),
            greater_than=bool(oc.greater_than),
            feasibility_threshold=float(oc.feasibility_threshold),
        )
        for oc in space.outcome_constraints
    ]

    # High-dimensional, all-continuous spaces use TuRBO (trust-region BO), whose
    # trust region is threaded across rounds as backend_state (see suggest()).
    has_categorical = any(d.param_type in ("categorical", "boolean") for d in space.dimensions)
    turbo_config = (
        TurboConfig()
        if len(space.dimensions) >= _TURBO_MIN_DIMS and not has_categorical
        else None
    )

    return OptimizationSpec(
        parameters=params,
        objectives=objectives,
        constraints=constraints,
        outcome_constraints=outcome_constraints,
        batch_size=batch_size,
        random_seed=_DEFAULT_SEED if seed is None else int(seed),
        turbo_config=turbo_config,
    )


def to_bomcp_observations(observations: list[Observation], space: ParameterSpace) -> list[Any]:
    """Convert HELIOS observations to bo-engine ``ObservationData``.

    The objective is fed **un-negated** (HELIOS is already maximization form);
    log-scale parameters are ``log10``'d to match :func:`to_bomcp_spec` bounds.
    """
    from bo_engine.types import ObservationData

    dims_by_name = {d.param_name: d for d in space.dimensions}
    out: list[Any] = []
    for obs in observations:
        encoded = {
            name: _encode_value(dims_by_name[name], val)
            for name, val in obs.params.items()
            if name in dims_by_name
        }
        out.append(
            ObservationData(
                parameter_values=encoded,
                objective_values={_OBJECTIVE_NAME: float(obs.objective)},
            )
        )
    return out


def from_bomcp_suggestion(suggestion: Any, space: ParameterSpace) -> dict[str, Any]:
    """Convert a bo-engine ``SuggestionResult`` to a HELIOS parameter dict."""
    dims_by_name = {d.param_name: d for d in space.dimensions}
    params = suggestion.parameter_values
    return {
        name: _decode_value(dims_by_name[name], val)
        for name, val in params.items()
        if name in dims_by_name
    }


def _state_to_dict(turbo_state: Any) -> dict[str, Any] | None:
    """Serialize a bo-engine ``TurboState`` to a JSON-safe dict (None passes through)."""
    if turbo_state is None:
        return None
    import dataclasses

    return dataclasses.asdict(turbo_state)


def _state_from_dict(state: dict[str, Any] | None) -> Any:
    """Reconstruct a ``TurboState`` from a persisted dict (None passes through)."""
    if not state:
        return None
    from bo_engine.turbo import TurboState

    return TurboState(**state)


def _suggestion_info(suggestion: Any) -> dict[str, Any]:
    """Extract per-point explanation/provenance from a ``SuggestionResult``."""
    return {
        "generation_method": getattr(suggestion, "generation_method", None),
        "acquisition_function": getattr(suggestion, "acquisition_function", None),
        "acquisition_value": getattr(suggestion, "acquisition_value", None),
        "model_type": getattr(suggestion, "model_type", None),
        "model_uncertainty": getattr(suggestion, "model_uncertainty", None),
        "confidence_level": getattr(suggestion, "confidence_level", None),
        "explanation": getattr(suggestion, "explanation", None),
        "predicted_objectives": getattr(suggestion, "predicted_objectives", None),
    }


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


@register_backend
class BoMcpBackend:
    """Real GP Bayesian optimization via bo-engine, exposed as a HELIOS backend."""

    name = "bomcp"

    def __init__(self) -> None:
        # Populated after each suggest() so callers can surface per-point
        # explanations into StrategyDecision / provenance (dimensions 5 & 7).
        self.last_method_info: list[dict[str, Any]] = []
        # JSON-safe TuRBO trust-region state for high-dim runs; None on the
        # standard qLogNEI path.  The caller persists it and passes it back via
        # the ``backend_state`` kwarg next round to keep the trust region warm.
        self.last_backend_state: dict[str, Any] | None = None

    def suggest(
        self,
        space: ParameterSpace,
        n: int,
        observations: list[Observation],
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        # No history yet: fall back to HELIOS's own space-filling sampler
        # (bomcp is normally invoked on exploit rounds, so this is a backstop).
        if not observations:
            from app.services.candidate_gen import sample_random

            self.last_method_info = []
            self.last_backend_state = None
            return sample_random(space, n, seed=seed)

        from bo_engine import generate_next_batch

        spec = to_bomcp_spec(space, n, seed)
        obs = to_bomcp_observations(observations, space)
        iteration = int(kwargs.get("round_number", len(observations)))
        turbo_state = _state_from_dict(kwargs.get("backend_state"))

        suggestions, new_state = generate_next_batch(
            spec, obs, batch_size=n, iteration=iteration, turbo_state=turbo_state
        )
        self.last_method_info = [_suggestion_info(s) for s in suggestions]
        self.last_backend_state = _state_to_dict(new_state)
        return [from_bomcp_suggestion(s, space) for s in suggestions]

    @staticmethod
    def is_available() -> bool:
        return _bomcp_available()
