"""Tests for the delegating candidate scorer (Δ1).

The scorer is the bridge that makes "one source of utility truth" literal: it
ranks *concrete* candidates by projecting them onto the *abstract* action
archetypes that the HELIOS authority (``strategy_selector``) already scored,
reusing the authority's ``weights_used`` vector verbatim.  It never defines its
own weights and never recomputes a phase.

Key invariants under test:
  * No per-candidate diagnostics  -> rank == archetype-utility order, delta == 0.
  * Per-candidate diagnostics      -> score shifts by exactly the same weights.
  * Near-duplicates                -> redundancy penalty demotes the later one.
  * Unknown archetype              -> graceful (no crash, zero base).
"""
from __future__ import annotations

import math

from app.optimization.candidate_scorer import score_pool
from app.optimization.schemas import PooledCandidate, ScoredCandidate
from app.services.candidate_gen import ParameterSpace, SearchDimension
from app.services.strategy_models import ActionCandidate, StrategyDecision, WeightsUsed


def _space() -> ParameterSpace:
    return ParameterSpace(
        dimensions=(SearchDimension("x", "number", 0.0, 10.0),),
        protocol_template={},
    )


def _decision() -> StrategyDecision:
    """Authority output: exploit (utility 0.55) beats explore (0.45)."""
    return StrategyDecision(
        backend_name="nexus_gp_bo",
        phase="exploitation",
        reason="test",
        confidence=0.8,
        actions_considered=(
            ActionCandidate("exploit", "nexus_gp_bo", 0.70, 0.30, 0.20, 0.55, "x"),
            ActionCandidate("explore", "lhs", 0.40, 0.80, 0.10, 0.45, "x"),
            ActionCandidate("refine", "scipy_de", 0.30, 0.20, 0.30, 0.20, "x"),
            ActionCandidate("stabilize", "built_in", 0.10, 0.40, 0.10, 0.30, "x"),
        ),
        weights_used=WeightsUsed(0.45, 0.35, 0.20, "test weights"),
    )


def test_empty_per_candidate_ranks_by_archetype_utility():
    survivors = [
        PooledCandidate(params={"x": 5.0}, source="nexus", source_action="explore"),
        PooledCandidate(params={"x": 2.0}, source="nexus", source_action="exploit"),
    ]
    scored = score_pool(survivors, _decision(), _space())

    assert all(isinstance(s, ScoredCandidate) for s in scored)
    # exploit archetype (0.55) outranks explore (0.45) despite input order
    assert scored[0].candidate.source_action == "exploit"
    assert scored[1].candidate.source_action == "explore"
    # with no per-candidate diagnostics, utility == archetype base, delta == 0
    assert scored[0].base_utility == 0.55
    assert scored[0].delta == 0.0
    assert scored[0].redundancy == 0.0
    assert math.isclose(scored[0].utility, 0.55)


def test_per_candidate_diagnostics_shift_score_with_same_weights():
    # exploit archetype has expected_improvement 0.70; this candidate beats it.
    survivors = [
        PooledCandidate(
            params={"x": 2.0},
            source="nexus",
            source_action="exploit",
            expected_improvement=0.90,
        ),
    ]
    scored = score_pool(survivors, _decision(), _space())

    # delta = w_improvement * (0.90 - 0.70) = 0.45 * 0.20 = 0.09
    assert math.isclose(scored[0].delta, 0.09, abs_tol=1e-9)
    assert math.isclose(scored[0].utility, 0.55 + 0.09, abs_tol=1e-9)


def test_redundancy_penalty_demotes_near_duplicate():
    survivors = [
        PooledCandidate(params={"x": 5.00}, source="nexus", source_action="exploit"),
        PooledCandidate(params={"x": 5.01}, source="local", source_action="exploit"),
    ]
    scored = score_pool(survivors, _decision(), _space())

    # both share the same archetype; the second is a near-duplicate of the first
    by_x = {s.candidate.params["x"]: s for s in scored}
    assert by_x[5.00].redundancy == 0.0
    assert by_x[5.01].redundancy > 0.0
    # the penalised one ends up ranked last
    assert scored[-1].candidate.params["x"] == 5.01


def test_unknown_archetype_degrades_gracefully():
    survivors = [
        PooledCandidate(params={"x": 1.0}, source="nexus", source_action="expand"),
    ]
    scored = score_pool(survivors, _decision(), _space())

    assert len(scored) == 1
    assert scored[0].base_utility == 0.0
    assert scored[0].delta == 0.0


def test_missing_weights_used_falls_back_without_crashing():
    decision = StrategyDecision(
        backend_name="built_in",
        phase="exploitation",
        reason="no adaptive weights",
        confidence=0.5,
        actions_considered=(
            ActionCandidate("exploit", "built_in", 0.7, 0.3, 0.2, 0.55, "x"),
        ),
        weights_used=None,  # adaptive weights disabled
    )
    survivors = [
        PooledCandidate(
            params={"x": 2.0}, source="nexus", source_action="exploit",
            expected_improvement=0.9,
        ),
    ]
    scored = score_pool(survivors, decision, _space())
    # falls back to PhaseConfig default weights; still produces a sane score
    assert scored[0].base_utility == 0.55
    assert scored[0].delta > 0.0
