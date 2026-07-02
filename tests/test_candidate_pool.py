"""Δ3/Δ7: multi-source candidate pool with archetype-projected utility.

The pool gathers candidates from several tagged sources, hard-gates them
(bounds / dedup / safety), scores survivors with a utility whose component
weights are *projected from the problem archetype* (derived from the Nexus
fingerprint), selects the top-N, and records full scored-pool provenance.
"""
from __future__ import annotations

from app.optimization.candidate_pool import (
    COMPONENTS,
    CandidatePoolBuilder,
    PooledCandidate,
    PoolResult,
    archetype_weights,
    build_candidate_pool,
    derive_archetype,
)
from app.optimization.arbitration_config import ArbitrationConfig
from app.optimization.schemas import (
    CandidatePool,
    CandidateSuggestion,
    OptimizationRequest,
)
from app.services.candidate_gen import ParameterSpace, SearchDimension
from app.services.optimization_backends import Observation
from app.services.strategy_models import (
    ActionCandidate,
    StabilizeSpec,
    StrategyDecision,
)


def _request(observations=(), n=2, seed=1) -> OptimizationRequest:
    space = ParameterSpace(
        dimensions=(SearchDimension("x", "number", 0.0, 10.0),),
        protocol_template={},
    )
    return OptimizationRequest(
        campaign_id="c", space=space, observations=tuple(observations), n=n, seed=seed
    )


# --- archetype derivation + weights -----------------------------------------

def test_derive_archetype_multi_objective():
    assert derive_archetype({"objective_form": "multi_objective"}) == "multi_objective"


def test_derive_archetype_high_noise():
    assert derive_archetype({"noise_regime": "high"}) == "high_noise"


def test_derive_archetype_tiny_data():
    assert derive_archetype({"data_scale": "tiny"}) == "tiny_data"


def test_derive_archetype_high_dim():
    assert derive_archetype({"effective_dimensionality": 15}) == "high_dim"


def test_derive_archetype_standard_default():
    assert derive_archetype({}) == "standard"


def test_archetype_weights_are_normalized_over_components():
    for archetype in ("standard", "high_noise", "tiny_data", "multi_objective", "high_dim"):
        w = archetype_weights(archetype)
        assert set(w) == set(COMPONENTS)
        assert abs(sum(w.values()) - 1.0) < 1e-9


# --- pool assembly + provenance ---------------------------------------------

def test_pool_selects_n_and_records_full_provenance():
    obs = [Observation(params={"x": 5.0}, objective=1.0)]
    res = build_candidate_pool(_request(obs), fingerprint={}, backend_name="built_in", k=3, select_n=2)
    assert isinstance(res, PoolResult)
    assert len(res.selected) == 2
    assert res.candidates
    assert all(isinstance(c, PooledCandidate) for c in res.candidates)
    assert sum(1 for c in res.candidates if c.selected) == 2
    # every candidate carries source + scores + gate status + archetype weights
    for c in res.candidates:
        assert c.source
        assert set(c.raw_scores) == set(COMPONENTS)
        assert c.gate_status
        assert set(c.archetype_weights) == set(COMPONENTS)


def test_replicate_best_reproposes_best_point_and_is_dedup_exempt():
    obs = [
        Observation(params={"x": 7.0}, objective=9.0),
        Observation(params={"x": 1.0}, objective=2.0),
    ]
    res = build_candidate_pool(
        _request(obs), fingerprint={}, sources=("replicate_best",), select_n=1
    )
    repl = [c for c in res.candidates if c.source == "replicate_best"]
    assert repl and repl[0].params["x"] == 7.0  # the best point
    assert repl[0].raw_scores["replication"] == 1.0
    assert repl[0].gate_status == "passed"  # duplicate of history, but exempt


# --- archetype-projected utility changes selection --------------------------

def test_standard_archetype_favours_acquisition_source():
    obs = [Observation(params={"x": 7.0}, objective=9.0)]
    res = build_candidate_pool(
        _request(obs), fingerprint={},  # standard
        sources=("local_baseline", "replicate_best"), k=2, select_n=1,
    )
    assert res.archetype == "standard"
    assert any(c.source == "local_baseline" and c.selected for c in res.candidates)


def test_high_noise_archetype_favours_robust_replication():
    obs = [Observation(params={"x": 7.0}, objective=9.0)]
    res = build_candidate_pool(
        _request(obs), fingerprint={"noise_regime": "high"},
        sources=("local_baseline", "replicate_best"), k=2, select_n=1,
    )
    assert res.archetype == "high_noise"
    assert any(c.source == "replicate_best" and c.selected for c in res.candidates)


# --- hard gate ---------------------------------------------------------------

def test_hard_gate_safety_check_rejects_and_excludes_from_selection():
    obs = [Observation(params={"x": 5.0}, objective=1.0)]
    res = build_candidate_pool(
        _request(obs, n=2), fingerprint={}, sources=("local_baseline",), k=4, select_n=2,
        safety_check=lambda params, req: params["x"] < 3.0,
    )
    assert any(c.gate_status.startswith("rejected") for c in res.candidates)
    assert all(c.params["x"] < 3.0 for c in res.candidates if c.selected)


# --- determinism + source distribution --------------------------------------

def test_pool_is_deterministic_under_seed():
    obs = [Observation(params={"x": 5.0}, objective=1.0)]
    a = build_candidate_pool(_request(obs), fingerprint={}, k=3, select_n=2)
    b = build_candidate_pool(_request(obs), fingerprint={}, k=3, select_n=2)
    assert a.selected == b.selected
    assert a.source_distribution == b.source_distribution


def test_source_distribution_counts_selected_by_source():
    obs = [Observation(params={"x": 5.0}, objective=1.0)]
    res = build_candidate_pool(_request(obs), fingerprint={}, k=4, select_n=3)
    assert sum(res.source_distribution.values()) == 3


# --- authority-bound arbitration pool ---------------------------------------

def _decision(
    best: str = "explore",
    phase: str = "exploration",
    stabilize_spec: StabilizeSpec | None = None,
) -> StrategyDecision:
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


def test_arbitration_nexus_candidates_inherit_authority_archetype_not_backend():
    pool = CandidatePoolBuilder().build(
        _request(),
        _decision(best="explore", phase="exploration"),
        nexus_suggestion=_nexus_suggestion(),
    )
    assert isinstance(pool, CandidatePool)
    nexus = [c for c in pool.candidates if c.source == "nexus"]
    assert len(nexus) == 2
    for c in nexus:
        assert c.source_action == "explore"
        assert c.generator_backend == "nexus_gp_bo"


def test_arbitration_explicit_per_candidate_archetype_overrides_authority():
    sug = _nexus_suggestion(per_candidate=[{"source_action": "refine"}, {}])
    pool = CandidatePoolBuilder().build(
        _request(), _decision(best="explore"), nexus_suggestion=sug
    )
    nexus = [c for c in pool.candidates if c.source == "nexus"]
    by_x = {c.params["x"]: c for c in nexus}
    assert by_x[1.0].source_action == "refine"
    assert by_x[2.0].source_action == "explore"


def test_arbitration_replicate_best_uses_stabilize_spec():
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


def test_arbitration_sobol_added_only_in_exploration_phase():
    explore_pool = CandidatePoolBuilder().build(
        _request(), _decision(best="explore", phase="exploration")
    )
    assert any(c.source == "sobol" for c in explore_pool.candidates)
    assert all(
        c.source_action == "explore"
        for c in explore_pool.candidates
        if c.source == "sobol"
    )

    exploit_pool = CandidatePoolBuilder().build(
        _request(), _decision(best="exploit", phase="exploitation")
    )
    assert not any(c.source == "sobol" for c in exploit_pool.candidates)


def test_arbitration_sources_dropped_recorded_when_nexus_absent():
    pool = CandidatePoolBuilder().build(
        _request(), _decision(best="exploit", phase="exploitation"), nexus_suggestion=None
    )
    assert "nexus" in pool.sources_dropped
    assert "nexus" not in pool.sources_used


def test_arbitration_local_baseline_inherits_authority_archetype():
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
