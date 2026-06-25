"""Tests for OptimizationDecisionPolicy.arbitrate -- two-stage verdict (Δ1).

Stage 1 (hard gate): bounds / dedup / safety -- absolute, strategy-free.
Stage 2 (soft score): delegate to the authority's utilities to rank survivors.

The legacy ``evaluate`` (hard-gate-only over a CandidateSuggestion) is unchanged
and covered by ``test_decision_policy.py``.
"""
from __future__ import annotations

from app.optimization.decision_policy import OptimizationDecisionPolicy
from app.optimization.schemas import (
    CandidatePool,
    DecisionResult,
    OptimizationRequest,
    PooledCandidate,
)
from app.services.candidate_gen import ParameterSpace, SearchDimension
from app.services.optimization_backends import Observation
from app.services.strategy_models import ActionCandidate, StrategyDecision, WeightsUsed


def _space() -> ParameterSpace:
    return ParameterSpace(
        dimensions=(SearchDimension("x", "number", 0.0, 10.0),),
        protocol_template={},
    )


def _request(observations=()) -> OptimizationRequest:
    return OptimizationRequest(
        campaign_id="c", space=_space(), observations=tuple(observations), n=2
    )


def _decision() -> StrategyDecision:
    return StrategyDecision(
        backend_name="nexus_gp_bo",
        phase="exploitation",
        reason="test",
        confidence=0.8,
        actions_considered=(
            ActionCandidate("exploit", "nexus_gp_bo", 0.70, 0.30, 0.20, 0.55, "x"),
            ActionCandidate("explore", "lhs", 0.40, 0.80, 0.10, 0.45, "x"),
        ),
        weights_used=WeightsUsed(0.45, 0.35, 0.20, "test"),
    )


def _pool(candidates) -> CandidatePool:
    return CandidatePool(candidates=tuple(candidates))


def test_arbitrate_hard_gate_then_soft_rank():
    pool = _pool([
        PooledCandidate({"x": 5.0}, "local", "explore"),    # valid, base 0.45
        PooledCandidate({"x": 2.0}, "nexus", "exploit"),    # valid, base 0.55
        PooledCandidate({"x": 99.0}, "sobol", "explore"),   # out of bounds
    ])
    res = OptimizationDecisionPolicy().arbitrate(pool, _request(), _decision())

    assert isinstance(res, DecisionResult)
    assert res.accepted is True
    # exploit (0.55) ranks above explore (0.45)
    assert res.final_candidates[0] == {"x": 2.0}
    assert res.final_candidates[1] == {"x": 5.0}
    # OOB candidate hard-gated out
    assert {"x": 99.0} in res.rejected
    assert any("bounds" in r for r in res.rejection_reasons)
    # only survivors are scored
    assert len(res.scored_pool) == 2
    assert res.requires_human_review is False


def test_arbitrate_respects_n():
    pool = _pool([
        PooledCandidate({"x": 2.0}, "nexus", "exploit"),
        PooledCandidate({"x": 5.0}, "local", "explore"),
    ])
    req = OptimizationRequest(campaign_id="c", space=_space(), n=1)
    res = OptimizationDecisionPolicy().arbitrate(pool, req, _decision())
    assert len(res.final_candidates) == 1
    assert res.final_candidates[0] == {"x": 2.0}  # top-scored
    # but the full portfolio is still recorded for audit
    assert len(res.scored_pool) == 2


def test_arbitrate_empty_after_gate_requires_human_review():
    pool = _pool([PooledCandidate({"x": 99.0}, "nexus", "exploit")])
    res = OptimizationDecisionPolicy().arbitrate(pool, _request(), _decision())
    assert res.accepted is False
    assert res.final_candidates == ()
    assert res.requires_human_review is True


def test_arbitrate_dedups_against_history():
    history = [Observation(params={"x": 2.0}, objective=1.0)]
    pool = _pool([
        PooledCandidate({"x": 2.0}, "nexus", "exploit"),   # duplicate of history
        PooledCandidate({"x": 5.0}, "local", "explore"),
    ])
    res = OptimizationDecisionPolicy().arbitrate(pool, _request(observations=history), _decision())
    assert {"x": 2.0} in res.rejected
    assert any("duplicate" in r.lower() for r in res.rejection_reasons)
    assert res.final_candidates == ({"x": 5.0},)


def test_arbitrate_safety_hook_rejects_in_pool():
    policy = OptimizationDecisionPolicy(safety_check=lambda c, req: c["x"] < 8.0)
    pool = _pool([
        PooledCandidate({"x": 9.0}, "nexus", "exploit"),   # unsafe
        PooledCandidate({"x": 3.0}, "local", "explore"),
    ])
    res = policy.arbitrate(pool, _request(), _decision())
    assert {"x": 9.0} in res.rejected
    assert any("safety" in r.lower() for r in res.rejection_reasons)
    assert res.final_candidates == ({"x": 3.0},)
