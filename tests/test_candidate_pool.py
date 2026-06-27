"""Tests for CandidatePoolBuilder (Δ1).

The builder assembles the concrete portfolio the boundary layer arbitrates over.
The binding rule (guiding principle): **HELIOS authority decides the archetype;
the backend only generates.**  So a candidate produced by ``nexus_gp_bo`` during
an exploration phase is tagged ``source_action="explore"`` (authority), with the
backend recorded separately as ``generator_backend`` for audit.

source_action precedence under test:
  1. explicit per-candidate archetype (if a provider supplies one) wins;
  2. else Nexus/local inherit the authority-selected action;
  3. local baselines carry their own explicit archetype (replicate→stabilize, etc.).
"""
from __future__ import annotations

from app.optimization.arbitration_config import ArbitrationConfig
from app.optimization.candidate_pool import CandidatePoolBuilder
from app.optimization.schemas import CandidatePool, CandidateSuggestion, OptimizationRequest
from app.services.candidate_gen import ParameterSpace, SearchDimension
from app.services.strategy_models import (
    ActionCandidate,
    StabilizeSpec,
    StrategyDecision,
)


def _space() -> ParameterSpace:
    return ParameterSpace(
        dimensions=(SearchDimension("x", "number", 0.0, 10.0),),
        protocol_template={},
    )


def _request(seed: int = 7) -> OptimizationRequest:
    return OptimizationRequest(campaign_id="c", space=_space(), n=2, seed=seed)


def _decision(
    best: str = "explore",
    phase: str = "exploration",
    stabilize_spec: StabilizeSpec | None = None,
) -> StrategyDecision:
    """Authority decision whose top-ranked action is ``best``."""
    actions = {
        "explore": ActionCandidate("explore", "lhs", 0.4, 0.8, 0.1, 0.60, "x"),
        "exploit": ActionCandidate("exploit", "nexus_gp_bo", 0.7, 0.3, 0.2, 0.55, "x"),
        "refine": ActionCandidate("refine", "scipy_de", 0.3, 0.2, 0.3, 0.20, "x"),
        "stabilize": ActionCandidate("stabilize", "built_in", 0.1, 0.4, 0.1, 0.30, "x"),
    }
    ordered = [actions[best]] + [a for k, a in actions.items() if k != best]
    return StrategyDecision(
        backend_name=actions[best].backend_name,
        phase=phase,
        reason="test",
        confidence=0.8,
        actions_considered=tuple(ordered),
        stabilize_spec=stabilize_spec,
    )


def _nexus_suggestion(per_candidate=()) -> CandidateSuggestion:
    return CandidateSuggestion(
        candidates=({"x": 1.0}, {"x": 2.0}),
        algorithm="nexus_gp_bo",
        source="nexus",
        per_candidate=tuple(per_candidate),
    )


def test_nexus_candidates_inherit_authority_archetype_not_backend():
    pool = CandidatePoolBuilder().build(
        _request(),
        _decision(best="explore", phase="exploration"),
        nexus_suggestion=_nexus_suggestion(),
    )
    assert isinstance(pool, CandidatePool)
    nexus = [c for c in pool.candidates if c.source == "nexus"]
    assert len(nexus) == 2
    for c in nexus:
        # authority said explore; backend was gp_bo -> archetype must be explore
        assert c.source_action == "explore"
        assert c.generator_backend == "nexus_gp_bo"


def test_explicit_per_candidate_archetype_overrides_authority():
    sug = _nexus_suggestion(per_candidate=[{"source_action": "refine"}, {}])
    pool = CandidatePoolBuilder().build(
        _request(), _decision(best="explore"), nexus_suggestion=sug
    )
    nexus = [c for c in pool.candidates if c.source == "nexus"]
    by_x = {c.params["x"]: c for c in nexus}
    assert by_x[1.0].source_action == "refine"  # explicit wins
    assert by_x[2.0].source_action == "explore"  # falls back to authority


def test_replicate_best_uses_stabilize_spec():
    spec = StabilizeSpec(
        strategy="best",
        points_to_replicate=({"x": 5.0},),
        n_replicates=2,
        reason="replicate top point",
    )
    pool = CandidatePoolBuilder().build(
        _request(),
        _decision(best="stabilize", phase="stabilize", stabilize_spec=spec),
    )
    rep = [c for c in pool.candidates if c.source == "replicate"]
    assert len(rep) == 1
    assert rep[0].params == {"x": 5.0}
    assert rep[0].source_action == "stabilize"


def test_sobol_added_only_in_exploration_phase():
    explore_pool = CandidatePoolBuilder().build(
        _request(), _decision(best="explore", phase="exploration")
    )
    assert any(c.source == "sobol" for c in explore_pool.candidates)
    assert all(c.source_action == "explore" for c in explore_pool.candidates if c.source == "sobol")

    exploit_pool = CandidatePoolBuilder().build(
        _request(), _decision(best="exploit", phase="exploitation")
    )
    assert not any(c.source == "sobol" for c in exploit_pool.candidates)


def test_sources_dropped_recorded_when_nexus_absent():
    pool = CandidatePoolBuilder().build(
        _request(), _decision(best="exploit", phase="exploitation"), nexus_suggestion=None
    )
    assert "nexus" in pool.sources_dropped
    assert "nexus" not in pool.sources_used


def test_local_baseline_inherits_authority_archetype():
    local = CandidateSuggestion(
        candidates=({"x": 3.0},), algorithm="built_in", source="local_fallback"
    )
    pool = CandidatePoolBuilder(
        ArbitrationConfig(include_sobol_in_exploration=False)
    ).build(
        _request(), _decision(best="refine", phase="refinement"), local_suggestion=local
    )
    loc = [c for c in pool.candidates if c.source == "local"]
    assert len(loc) == 1
    assert loc[0].source_action == "refine"
    assert loc[0].generator_backend == "built_in"
