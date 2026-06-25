"""Tests for OptimizationDecisionPolicy.

The policy is HELIOS's authority gate over a provider's suggestion.  It
sequences campaign-level checks (search-space bounds, deduplication, an
optional safety hook) into a single auditable verdict.  It does NOT generate
candidates and does NOT reimplement the safety agent -- it orchestrates.
"""
from __future__ import annotations

from app.optimization.decision_policy import OptimizationDecisionPolicy
from app.optimization.schemas import CandidateSuggestion, DecisionResult, OptimizationRequest
from app.services.candidate_gen import ParameterSpace, SearchDimension
from app.services.optimization_backends import Observation


def _request(observations=()) -> OptimizationRequest:
    space = ParameterSpace(
        dimensions=(
            SearchDimension("x", "number", 0.0, 10.0),
            SearchDimension("cat", "categorical", choices=("a", "b")),
        ),
        protocol_template={},
    )
    return OptimizationRequest(campaign_id="c", space=space, observations=tuple(observations))


def _suggestion(candidates) -> CandidateSuggestion:
    return CandidateSuggestion(candidates=tuple(candidates), algorithm="nexus_gp_bo", source="nexus")


def test_accepts_in_bounds_unique_candidate():
    res = OptimizationDecisionPolicy().evaluate(
        _suggestion([{"x": 5.0, "cat": "a"}]), _request()
    )
    assert isinstance(res, DecisionResult)
    assert res.accepted is True
    assert res.final_candidates == ({"x": 5.0, "cat": "a"},)
    assert res.decision_trace  # non-empty audit trail


def test_rejects_out_of_bounds_candidate():
    res = OptimizationDecisionPolicy().evaluate(
        _suggestion([{"x": 99.0, "cat": "a"}]), _request()
    )
    assert res.accepted is False
    assert res.final_candidates == ()
    assert res.rejected == ({"x": 99.0, "cat": "a"},)
    assert any("bounds" in r for r in res.rejection_reasons)


def test_rejects_invalid_category():
    res = OptimizationDecisionPolicy().evaluate(
        _suggestion([{"x": 1.0, "cat": "z"}]), _request()
    )
    assert res.accepted is False
    assert any("categor" in r.lower() for r in res.rejection_reasons)


def test_rejects_duplicate_of_existing_observation():
    history = [Observation(params={"x": 5.0, "cat": "a"}, objective=1.0)]
    res = OptimizationDecisionPolicy().evaluate(
        _suggestion([{"x": 5.0, "cat": "a"}]), _request(observations=history)
    )
    assert res.accepted is False
    assert any("duplicate" in r.lower() for r in res.rejection_reasons)


def test_rejects_duplicate_within_batch():
    res = OptimizationDecisionPolicy().evaluate(
        _suggestion([{"x": 3.0, "cat": "a"}, {"x": 3.0, "cat": "a"}]), _request()
    )
    assert len(res.final_candidates) == 1
    assert any("duplicate" in r.lower() for r in res.rejection_reasons)


def test_safety_hook_can_reject():
    policy = OptimizationDecisionPolicy(safety_check=lambda c, req: c["x"] < 8.0)
    res = policy.evaluate(_suggestion([{"x": 9.0, "cat": "a"}]), _request())
    assert res.accepted is False
    assert any("safety" in r.lower() for r in res.rejection_reasons)


def test_requires_human_review_when_nothing_accepted():
    res = OptimizationDecisionPolicy().evaluate(
        _suggestion([{"x": 99.0, "cat": "a"}]), _request()
    )
    assert res.requires_human_review is True
