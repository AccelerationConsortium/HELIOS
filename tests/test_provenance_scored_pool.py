"""Δ1: provenance records the full scored portfolio behind a verdict.

Every round must answer "why this candidate, why not the alternatives" from the
stored record alone -- so the scored pool (with decomposed utility + which were
selected) is serialised alongside the existing proposed/accepted/rejected fields.
"""
from __future__ import annotations

from app.optimization.provenance import ProvenanceLogger
from app.optimization.schemas import (
    CandidateSuggestion,
    DecisionResult,
    OptimizationRequest,
    PooledCandidate,
    ScoredCandidate,
)
from app.services.candidate_gen import ParameterSpace, SearchDimension


def _request() -> OptimizationRequest:
    space = ParameterSpace(
        dimensions=(SearchDimension("x", "number", 0.0, 10.0),),
        protocol_template={},
    )
    return OptimizationRequest(campaign_id="c", space=space, round_index=3)


def _scored(params, action, util, selected_base) -> ScoredCandidate:
    return ScoredCandidate(
        candidate=PooledCandidate(
            params=params, source="nexus", source_action=action,
            generator_backend="nexus_gp_bo",
        ),
        utility=util,
        base_utility=selected_base,
        delta=0.0,
        redundancy=0.0,
    )


def test_provenance_records_scored_pool_with_selection_flags():
    suggestion = CandidateSuggestion(
        candidates=({"x": 2.0},), algorithm="nexus_gp_bo", source="nexus"
    )
    decision = DecisionResult(
        accepted=True,
        final_candidates=({"x": 2.0},),
        scored_pool=(
            _scored({"x": 2.0}, "exploit", 0.55, 0.55),
            _scored({"x": 5.0}, "explore", 0.45, 0.45),
        ),
        decision_trace=("arbitrated",),
    )
    rec = ProvenanceLogger().build(_request(), suggestion, decision)

    pool = rec["candidate_pool"]
    assert len(pool) == 2
    entry = pool[0]
    assert {
        "params", "source", "source_action", "generator_backend",
        "base_utility", "delta", "utility", "selected",
    } <= set(entry)
    # the chosen candidate is flagged selected; the runner-up is not
    by_x = {e["params"]["x"]: e for e in pool}
    assert by_x[2.0]["selected"] is True
    assert by_x[5.0]["selected"] is False


def test_provenance_scored_pool_empty_when_legacy_evaluate_used():
    # legacy evaluate() produces no scored_pool -> record carries an empty list,
    # never errors (back-compat).
    suggestion = CandidateSuggestion(
        candidates=({"x": 2.0},), algorithm="nexus_gp_bo", source="nexus"
    )
    decision = DecisionResult(accepted=True, final_candidates=({"x": 2.0},))
    rec = ProvenanceLogger().build(_request(), suggestion, decision)
    assert rec["candidate_pool"] == []
