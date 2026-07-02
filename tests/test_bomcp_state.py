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


# --- live thread: generate_adaptive_candidates surfaces state on the decision --


def test_generate_adaptive_candidates_exposes_backend_state_on_decision():
    from app.services.strategy_models import CampaignSnapshot
    from app.services.strategy_selector import generate_adaptive_candidates

    space = _space(2)
    snap = CampaignSnapshot(
        round_number=2, max_rounds=5, n_observations=5, n_dimensions=2,
        has_categorical=False, has_log_scale=False,
        user_strategy_hint="random",  # deterministic backend with no state
    )
    cands, decision = generate_adaptive_candidates(space, 3, [], snap, seed=1)
    assert len(cands) == 3
    assert hasattr(decision, "backend_state")
    assert decision.backend_state is None  # random backend emits no state

    # Accepts an incoming backend_state without breaking the 2-tuple contract.
    cands2, _ = generate_adaptive_candidates(space, 3, [], snap, seed=1, backend_state=None)
    assert len(cands2) == 3


def test_generate_batch_threads_backend_state():
    from app.services.candidate_gen import generate_batch

    space = _space(2)
    # Adaptive path, no DB campaign -> empty obs -> still produces a batch and
    # carries a backend_state field (None for this low-dim/no-data case).
    result = generate_batch(space, strategy="adaptive", n_candidates=3, store=False)
    assert hasattr(result, "backend_state")
    assert len(result.candidates) == 3

    # Accepts an incoming backend_state without error.
    result2 = generate_batch(
        space, strategy="lhs", n_candidates=3, store=False, backend_state=None
    )
    assert result2.backend_state is None


def test_design_contract_threads_backend_state():
    from app.agents.design_agent import DesignInput, DesignOutput

    di = DesignInput(
        dimensions=[{"param_name": "x", "param_type": "number", "min_value": 0, "max_value": 1}],
        protocol_template={},
        backend_state={"length": 0.8},
    )
    assert di.backend_state == {"length": 0.8}
    # Default None keeps existing callers unaffected.
    assert DesignInput(dimensions=di.dimensions, protocol_template={}).backend_state is None
    assert "backend_state" in DesignOutput.model_fields
