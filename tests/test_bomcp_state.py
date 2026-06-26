"""(c) bomcp stateful enhancement -- TuRBO trust-region state across rounds.

For high-dimensional problems the bomcp backend enables TuRBO and emits an
opaque, JSON-serializable ``backend_state`` (the trust-region) that the caller
persists and passes back next round, so the trust region shrinks/grows
continuously instead of resetting every round.  Low-dimensional problems use the
standard qLogNEI path and emit no state.
"""
from __future__ import annotations

import json
import math
import random

import pytest

from app.optimization import bomcp_backend as bb
from app.services.candidate_gen import ParameterSpace, SearchDimension
from app.services.optimization_backends import Observation

_needs_bomcp = pytest.mark.skipif(not bb._bomcp_available(), reason="bo-engine not installed")


def _space(n_dims: int) -> ParameterSpace:
    return ParameterSpace(
        dimensions=tuple(
            SearchDimension(f"x{i}", "number", min_value=-5.0, max_value=5.0)
            for i in range(n_dims)
        ),
        protocol_template={},
    )


def _neg_sphere_obs(n_dims, n, seed):
    rng = random.Random(seed)
    obs = []
    for _ in range(n):
        p = {f"x{i}": rng.uniform(-5, 5) for i in range(n_dims)}
        obs.append(Observation(params=p, objective=-sum(v * v for v in p.values())))
    return obs


@_needs_bomcp
def test_high_dim_emits_serializable_turbo_state():
    space = _space(6)
    backend = bb.BoMcpBackend()
    cands = backend.suggest(space, 4, _neg_sphere_obs(6, 12, seed=1), seed=0)
    assert len(cands) == 4
    assert backend.last_backend_state is not None
    # Opaque but JSON-serializable so the orchestrator can persist it.
    json.dumps(backend.last_backend_state)
    assert backend.last_backend_state["dim"] == 6


@_needs_bomcp
def test_low_dim_emits_no_state():
    space = _space(2)
    backend = bb.BoMcpBackend()
    backend.suggest(space, 3, _neg_sphere_obs(2, 8, seed=1), seed=0)
    assert backend.last_backend_state is None  # standard qLogNEI path, no TuRBO


@_needs_bomcp
def test_turbo_state_round_trips():
    space = _space(6)
    backend = bb.BoMcpBackend()
    obs = _neg_sphere_obs(6, 12, seed=2)
    backend.suggest(space, 4, obs, seed=0)
    state1 = backend.last_backend_state
    assert state1 is not None

    # Feed the prior state back; accepted, produces a valid batch + new state.
    more = obs + _neg_sphere_obs(6, 4, seed=3)
    cands2 = backend.suggest(space, 4, more, seed=1, backend_state=state1)
    assert len(cands2) == 4
    assert backend.last_backend_state is not None
    json.dumps(backend.last_backend_state)
