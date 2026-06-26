"""PoC validation for the bo-mcp (bo-engine) optimization backend bridge.

See docs/plans/2026-06-26-bomcp-backend-integration-design.md §4.

These tests are pure-memory: no hardware, no database.  The end-to-end campaign
test runs the *real* bo-engine GP (skipped automatically if bo-engine is not
installed), proving HELIOS can borrow real Bayesian optimization on a backend
that beats the built-in KNN heuristic.
"""
from __future__ import annotations

import math

import pytest

# Importing the package registers the ``bomcp`` backend in the shared registry.
import app.optimization  # noqa: F401
from app.optimization import bomcp_backend as bb
from app.services.candidate_gen import ParameterSpace, SearchDimension, SimplexConstraint
from app.services.optimization_backends import Observation, get_backend, list_backends

_HAVE_BOMCP = bb._bomcp_available()
_needs_bomcp = pytest.mark.skipif(not _HAVE_BOMCP, reason="bo-engine not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _branin_space() -> ParameterSpace:
    return ParameterSpace(
        dimensions=(
            SearchDimension("x0", "number", min_value=-5.0, max_value=10.0),
            SearchDimension("x1", "number", min_value=0.0, max_value=15.0),
        ),
        protocol_template={},
    )


def _neg_branin(params: dict) -> float:
    """Branin negated so that higher = better (HELIOS maximization form).

    True branin minimum is ~0.3979, so the negated maximum is ~-0.3979.
    """
    x0, x1 = float(params["x0"]), float(params["x1"])
    t1 = x1 - 5.1 / (4 * math.pi**2) * x0**2 + 5 / math.pi * x0 - 6
    branin = t1**2 + 10 * (1 - 1 / (8 * math.pi)) * math.cos(x0) + 10
    return -branin


def _seed_observations(space, objfn, n, seed):
    import random

    rng = random.Random(seed)
    obs = []
    for _ in range(n):
        p = {
            d.param_name: rng.uniform(d.min_value, d.max_value)
            for d in space.dimensions
        }
        obs.append(Observation(params=p, objective=objfn(p)))
    return obs


def _run_campaign(backend_name, space, objfn, *, rounds, batch, seed, init_obs):
    obs = list(init_obs)
    backend = get_backend(backend_name)
    assert backend.name == backend_name, f"expected {backend_name}, got {backend.name}"
    for r in range(rounds):
        for cand in backend.suggest(space, batch, obs, seed=seed + r):
            obs.append(Observation(params=cand, objective=objfn(cand)))
    return max(o.objective for o in obs)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_bomcp_registered():
    assert "bomcp" in list_backends()


@_needs_bomcp
def test_bomcp_available_and_selected():
    assert list_backends()["bomcp"] is True
    assert get_backend("bomcp").name == "bomcp"


# ---------------------------------------------------------------------------
# A. End-to-end toy campaign: bomcp (real GP) beats built_in (KNN) on Branin
# ---------------------------------------------------------------------------


@_needs_bomcp
def test_bomcp_beats_builtin_on_branin():
    space = _branin_space()
    init = _seed_observations(space, _neg_branin, n=8, seed=7)

    bomcp_best = _run_campaign(
        "bomcp", space, _neg_branin, rounds=5, batch=3, seed=100, init_obs=init
    )
    builtin_best = _run_campaign(
        "built_in", space, _neg_branin, rounds=5, batch=3, seed=100, init_obs=init
    )

    optimum = -0.397887
    # bomcp should get meaningfully close to the optimum...
    assert bomcp_best > optimum - 5.0, f"bomcp_best={bomcp_best:.3f} too far from {optimum}"
    # ...and be at least as good as the KNN heuristic from the same start.
    assert bomcp_best >= builtin_best - 1e-6, (
        f"bomcp_best={bomcp_best:.3f} < builtin_best={builtin_best:.3f}"
    )


# ---------------------------------------------------------------------------
# B. Graceful degradation
# ---------------------------------------------------------------------------


def test_degrades_to_builtin_when_unavailable(monkeypatch):
    monkeypatch.setattr(bb, "_bomcp_available", lambda: False)
    # get_backend swaps an unavailable backend for built_in.
    backend = get_backend("bomcp")
    assert backend.name == "built_in"


def test_empty_history_uses_space_filling_fallback():
    # bomcp is normally called on exploit rounds (history present); with no
    # observations it must still return valid candidates, not crash.
    space = _branin_space()
    cands = bb.BoMcpBackend().suggest(space, 4, [], seed=1)
    assert len(cands) == 4
    for c in cands:
        assert -5.0 <= c["x0"] <= 10.0
        assert 0.0 <= c["x1"] <= 15.0


# ---------------------------------------------------------------------------
# C. Double-flip guard (the bug this bridge is written to avoid)
# ---------------------------------------------------------------------------


@_needs_bomcp
def test_minimize_problem_moves_toward_true_optimum():
    """A minimize problem (true cost minimized at x=2) flipped to HELIOS form.

    HELIOS stores objective = -cost (higher = better).  If the bridge re-flipped
    on direction, bomcp would chase the *worst* region instead of the best.
    """
    space = ParameterSpace(
        dimensions=(SearchDimension("x", "number", min_value=-10.0, max_value=10.0),),
        protocol_template={},
    )

    def neg_cost(params):  # cost = (x-2)^2 ; HELIOS maximizes -cost
        return -((float(params["x"]) - 2.0) ** 2)

    # Dense history so the GP has a clear gradient toward x=2.
    import random

    rng = random.Random(3)
    obs = [
        Observation(params={"x": (xv := rng.uniform(-10, 10))}, objective=neg_cost({"x": xv}))
        for _ in range(12)
    ]
    cands = get_backend("bomcp").suggest(space, 4, obs, seed=5)
    mean_x = sum(c["x"] for c in cands) / len(cands)
    # Suggestions should sit nearer the true optimum (2.0) than the far edges.
    assert abs(mean_x - 2.0) < 6.0, f"mean suggested x={mean_x:.2f} not heading to optimum"


# ---------------------------------------------------------------------------
# D. log_scale round-trip (pure transform unit tests)
# ---------------------------------------------------------------------------


def test_log_scale_encode_decode_roundtrip():
    dim = SearchDimension("k", "number", min_value=1e-3, max_value=1e2, log_scale=True)
    # encode: original value -> log10 space
    assert bb._encode_value(dim, 1.0) == pytest.approx(0.0)
    assert bb._encode_value(dim, 100.0) == pytest.approx(2.0)
    # bounds get log10'd for the spec
    lo, hi = bb._encoded_bounds(dim)
    assert (lo, hi) == pytest.approx((-3.0, 2.0))
    # decode: log10 space -> original, within bounds
    assert bb._decode_value(dim, 0.0) == pytest.approx(1.0)
    assert bb._decode_value(dim, 2.0) == pytest.approx(100.0)
    decoded = bb._decode_value(dim, 1.5)
    assert 1e-3 <= decoded <= 1e2


def test_integer_decodes_to_rounded_int():
    dim = SearchDimension("n", "integer", min_value=1, max_value=10)
    out = bb._decode_value(dim, 4.6)
    assert out == 5 and isinstance(out, int)


def test_categorical_decodes_back_to_original_choice():
    dim = SearchDimension("solvent", "categorical", choices=("water", "ethanol", "dmso"))
    assert bb._encode_value(dim, "ethanol") == "ethanol"
    assert bb._decode_value(dim, "dmso") == "dmso"


def test_boolean_choice_roundtrip():
    dim = SearchDimension("heated", "boolean", choices=(True, False))
    assert bb._encode_value(dim, True) == "True"
    # decode maps the category string back to the real boolean choice object.
    assert bb._decode_value(dim, "True") is True
    assert bb._decode_value(dim, "False") is False


@_needs_bomcp
def test_simplex_maps_to_sum_equals_constraint():
    from bo_engine.types import ConstraintType

    space = ParameterSpace(
        dimensions=(
            SearchDimension("a", "number", min_value=0.0, max_value=1.0),
            SearchDimension("b", "number", min_value=0.0, max_value=1.0),
        ),
        protocol_template={},
        simplex_constraints=(SimplexConstraint(param_names=("a", "b"), target_sum=1.0),),
    )
    spec = bb.to_bomcp_spec(space, batch_size=2, seed=0)
    assert len(spec.constraints) == 1
    c = spec.constraints[0]
    assert c.type == ConstraintType.SUM_EQUALS
    assert set(c.parameters) == {"a", "b"}
    assert c.value == pytest.approx(1.0)
