"""Bridge Nexus algorithm plugins into HELIOS's optimization-backend registry.

Nexus (``optimization_copilot``) exposes a portfolio of pure-Python optimizers
via the ``AlgorithmPlugin`` interface (a *fit-then-suggest* lifecycle that
minimizes its objective).  HELIOS's ``BackendProtocol`` is stateless
(``suggest(space, n, observations, *, seed)``) and treats the objective as
*higher = better* (already direction-flipped upstream).

This module adapts the two:

* ``to_nexus_specs``        -- HELIOS ``ParameterSpace`` -> Nexus ``ParameterSpec``
* ``to_nexus_observations`` -- HELIOS ``Observation``    -> Nexus ``Observation``
                               (with the objective **negated**, because Nexus
                               minimizes and HELIOS maximizes)
* ``NexusBackendBridge``    -- base ``BackendProtocol`` adapter
* concrete ``nexus_*`` backends, registered in the shared registry

The module imports cleanly even without Nexus installed; the Nexus plugin
classes are imported lazily inside ``suggest`` so that ``is_available()`` (not
an ``ImportError``) governs whether a backend is used.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.services.optimization_backends import Observation, register_backend

if TYPE_CHECKING:  # pragma: no cover
    from app.services.candidate_gen import ParameterSpace

logger = logging.getLogger(__name__)

_DEFAULT_SEED = 42
_DEFAULT_OBJECTIVE = "objective"


def _nexus_available() -> bool:
    """Return True if the Nexus optimization core can be imported."""
    try:
        import optimization_copilot.backends.builtin  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Conversions (HELIOS <-> Nexus data models)
# ---------------------------------------------------------------------------


def to_nexus_specs(space: ParameterSpace) -> list[Any]:
    """Convert a HELIOS ``ParameterSpace`` to a list of Nexus ``ParameterSpec``."""
    from optimization_copilot.core.models import ParameterSpec, VariableType

    specs: list[Any] = []
    for dim in space.dimensions:
        if dim.param_type in ("categorical", "boolean"):
            specs.append(
                ParameterSpec(
                    name=dim.param_name,
                    type=VariableType.CATEGORICAL,
                    categories=list(dim.choices or ()),
                )
            )
        elif dim.param_type == "integer":
            specs.append(
                ParameterSpec(
                    name=dim.param_name,
                    type=VariableType.DISCRETE,
                    lower=dim.min_value,
                    upper=dim.max_value,
                )
            )
        else:  # "number" (and any unknown numeric type) -> continuous
            specs.append(
                ParameterSpec(
                    name=dim.param_name,
                    type=VariableType.CONTINUOUS,
                    lower=dim.min_value,
                    upper=dim.max_value,
                )
            )
    return specs


def to_nexus_observations(
    observations: list[Observation],
    objective_name: str = _DEFAULT_OBJECTIVE,
    *,
    negate: bool = True,
) -> list[Any]:
    """Convert HELIOS observations to Nexus observations.

    The HELIOS ``objective`` is *higher = better*.  Nexus's built-in plugins
    **minimize**, so for the candidate-generation path the value is negated
    (``negate=True``): minimizing ``-objective`` maximizes ``objective``.

    For the *advisory* path (profiler / diagnostics) the snapshot carries the
    real objective with an explicit ``maximize`` direction, so pass
    ``negate=False`` to keep diagnostic trends in natural units.
    """
    from optimization_copilot.core.models import Observation as NexusObservation

    sign = -1.0 if negate else 1.0
    nexus_obs: list[Any] = []
    for i, obs in enumerate(observations):
        nexus_obs.append(
            NexusObservation(
                iteration=i,
                parameters=dict(obs.params),
                kpi_values={objective_name: sign * float(obs.objective)},
            )
        )
    return nexus_obs


# ---------------------------------------------------------------------------
# Backend bridge
# ---------------------------------------------------------------------------


class NexusBackendBridge:
    """Adapt a Nexus ``AlgorithmPlugin`` to HELIOS's ``BackendProtocol``."""

    name: str = "nexus_base"
    plugin_name: str = ""  # name of the class in ``optimization_copilot.backends.builtin``

    def _make_plugin(self) -> Any:
        from optimization_copilot.backends import builtin

        return getattr(builtin, self.plugin_name)()

    def suggest(
        self,
        space: ParameterSpace,
        n: int,
        observations: list[Observation],
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        plugin = self._make_plugin()
        plugin.fit(to_nexus_observations(observations), to_nexus_specs(space))
        return plugin.suggest(
            n_suggestions=n,
            seed=_DEFAULT_SEED if seed is None else seed,
        )

    @staticmethod
    def is_available() -> bool:
        return _nexus_available()


# ---------------------------------------------------------------------------
# Concrete portfolio -- registered in the shared backend registry
# ---------------------------------------------------------------------------


@register_backend
class NexusGaussianProcessBackend(NexusBackendBridge):
    name = "nexus_gp_bo"
    plugin_name = "GaussianProcessBO"


@register_backend
class NexusTPEBackend(NexusBackendBridge):
    name = "nexus_tpe"
    plugin_name = "TPESampler"


@register_backend
class NexusCMAESBackend(NexusBackendBridge):
    name = "nexus_cmaes"
    plugin_name = "CMAESSampler"


@register_backend
class NexusNSGA2Backend(NexusBackendBridge):
    name = "nexus_nsga2"
    plugin_name = "NSGA2Sampler"


@register_backend
class NexusTuRBOBackend(NexusBackendBridge):
    name = "nexus_turbo"
    plugin_name = "TuRBOSampler"


@register_backend
class NexusDifferentialEvolutionBackend(NexusBackendBridge):
    name = "nexus_de"
    plugin_name = "DifferentialEvolution"


@register_backend
class NexusRandomForestBackend(NexusBackendBridge):
    name = "nexus_rf_bo"
    plugin_name = "RandomForestBO"


@register_backend
class NexusLatinHypercubeBackend(NexusBackendBridge):
    name = "nexus_lhs"
    plugin_name = "LatinHypercubeSampler"


@register_backend
class NexusSobolBackend(NexusBackendBridge):
    name = "nexus_sobol"
    plugin_name = "SobolSampler"


@register_backend
class NexusRandomBackend(NexusBackendBridge):
    name = "nexus_random"
    plugin_name = "RandomSampler"
