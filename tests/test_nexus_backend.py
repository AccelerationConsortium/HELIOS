"""Tests for the Nexus -> HELIOS optimization backend bridge.

The bridge adapts a Nexus ``AlgorithmPlugin`` (fit-then-suggest lifecycle,
minimization-oriented) to HELIOS's stateless ``BackendProtocol`` (higher
objective = better).  The most important behaviour is that the HELIOS
objective is *negated* when handed to Nexus, because Nexus's built-in
plugins minimize.
"""
from __future__ import annotations

import pytest

from app.services.candidate_gen import ParameterSpace, SearchDimension
from app.services.optimization_backends import Observation, list_backends

# Skip the whole module cleanly if Nexus is not installed in this env.
pytest.importorskip("optimization_copilot")

from optimization_copilot.core.models import VariableType  # noqa: E402

from app.optimization import nexus_backend as nb  # noqa: E402


def _space() -> ParameterSpace:
    return ParameterSpace(
        dimensions=(
            SearchDimension(param_name="x", param_type="number", min_value=0.0, max_value=10.0),
            SearchDimension(param_name="k", param_type="integer", min_value=1, max_value=5),
            SearchDimension(param_name="cat", param_type="categorical", choices=("a", "b", "c")),
        ),
        protocol_template={},
    )


def test_to_nexus_specs_maps_number_to_continuous():
    specs = {s.name: s for s in nb.to_nexus_specs(_space())}
    assert specs["x"].type == VariableType.CONTINUOUS
    assert specs["x"].lower == 0.0
    assert specs["x"].upper == 10.0


def test_to_nexus_specs_maps_integer_to_discrete():
    specs = {s.name: s for s in nb.to_nexus_specs(_space())}
    assert specs["k"].type == VariableType.DISCRETE
    assert specs["k"].lower == 1
    assert specs["k"].upper == 5


def test_to_nexus_specs_maps_categorical_with_choices():
    specs = {s.name: s for s in nb.to_nexus_specs(_space())}
    assert specs["cat"].type == VariableType.CATEGORICAL
    assert specs["cat"].categories == ["a", "b", "c"]


def test_to_nexus_observations_negates_objective():
    """Nexus minimizes; HELIOS objective is higher=better -> negate."""
    obs = [Observation(params={"x": 1.0}, objective=5.0)]
    nexus_obs = nb.to_nexus_observations(obs)
    assert nexus_obs[0].kpi_values["objective"] == -5.0
    assert nexus_obs[0].parameters == {"x": 1.0}
    assert nexus_obs[0].iteration == 0


def test_bridge_suggest_returns_valid_in_bounds_candidates():
    backend = nb.NexusGaussianProcessBackend()
    cands = backend.suggest(_space(), n=3, observations=[], seed=42)
    assert len(cands) == 3
    for c in cands:
        assert set(c.keys()) == {"x", "k", "cat"}
        assert 0.0 <= c["x"] <= 10.0
        assert 1 <= c["k"] <= 5
        assert c["cat"] in ("a", "b", "c")


def test_bridge_suggest_is_deterministic_with_seed():
    backend = nb.NexusGaussianProcessBackend()
    a = backend.suggest(_space(), n=3, observations=[], seed=7)
    b = backend.suggest(_space(), n=3, observations=[], seed=7)
    assert a == b


def test_is_available_true_when_installed():
    assert nb.NexusGaussianProcessBackend.is_available() is True


def test_nexus_backends_registered_in_shared_registry():
    backends = list_backends()
    assert "nexus_gp_bo" in backends
    assert backends["nexus_gp_bo"] is True
    # The portfolio should expose more than one algorithm.
    assert "nexus_tpe" in backends
    assert "nexus_cmaes" in backends
