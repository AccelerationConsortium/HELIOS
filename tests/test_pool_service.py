"""Phase B-2: CandidatePoolService composes candidate sources for arbitration.

The service is the unified candidate-and-strategy layer the authority's
``arbitrate_next`` calls: it gathers candidates from pluggable sources (Nexus,
local baseline, nexus archetype pool, bomcp proposals), assembles them into a
``CandidatePool`` via the arbitration builder, and applies the failure-zone
penalty.  The hard constraint gate and advisor hints live downstream in
``policy.arbitrate`` / the authority ``StrategyDecision``.
"""
from __future__ import annotations

import pytest

from app.optimization.pool_service import CandidatePoolService
from app.optimization.schemas import CandidateSuggestion, OptimizationRequest
from app.services.candidate_gen import ParameterSpace, SearchDimension
from app.services.strategy_models import StrategyDecision


def _space() -> ParameterSpace:
    return ParameterSpace(
        dimensions=(
            SearchDimension("x0", "number", min_value=0.0, max_value=1.0),
            SearchDimension("x1", "number", min_value=0.0, max_value=1.0),
        ),
        protocol_template={},
    )


def _decision() -> StrategyDecision:
    return StrategyDecision(
        backend_name="built_in", phase="exploitation", reason="t", confidence=0.8,
    )


class _FakeSource:
    def __init__(self, name, params):
        self.name = name
        self._params = params

    def propose(self, request, decision):
        return CandidateSuggestion(
            candidates=tuple(self._params), algorithm=f"{self.name}_algo", source=self.name,
        )


def test_service_composes_multiple_sources_into_pool():
    req = OptimizationRequest(campaign_id="c", space=_space(), n=2)
    service = CandidatePoolService(sources=[
        _FakeSource("nexus", [{"x0": 0.1, "x1": 0.1}]),
        _FakeSource("archetype", [{"x0": 0.2, "x1": 0.2}]),
        _FakeSource("bomcp", [{"x0": 0.3, "x1": 0.3}]),
    ])
    pool = service.build_pool(req, _decision())
    sources = {c.source for c in pool.candidates}
    assert {"nexus", "archetype", "bomcp"} <= sources
    assert len(pool.candidates) == 3


def test_service_applies_failure_zone_penalty():
    # Failed experiments cluster near (0.8, 0.8); a candidate there is dropped.
    req = OptimizationRequest(
        campaign_id="c", space=_space(), n=2,
        context={"failed_params": [{"x0": 0.8, "x1": 0.8}, {"x0": 0.81, "x1": 0.79}]},
    )
    service = CandidatePoolService(sources=[
        _FakeSource("nexus", [{"x0": 0.1, "x1": 0.1}, {"x0": 0.8, "x1": 0.8}]),
    ])
    pool = service.build_pool(req, _decision())
    kept = [c.params for c in pool.candidates]
    assert {"x0": 0.1, "x1": 0.1} in kept
    assert {"x0": 0.8, "x1": 0.8} not in kept  # in the learned failure zone


def test_service_never_returns_empty_even_if_all_failure_prone():
    req = OptimizationRequest(
        campaign_id="c", space=_space(), n=1,
        context={"failed_params": [{"x0": 0.5, "x1": 0.5}]},
    )
    service = CandidatePoolService(sources=[
        _FakeSource("nexus", [{"x0": 0.5, "x1": 0.5}]),  # the only candidate is failure-prone
    ])
    pool = service.build_pool(req, _decision())
    assert len(pool.candidates) >= 1  # never strand the round empty


def test_builtin_sources_factory_present():
    # The default production source set is discoverable + names are stable.
    from app.optimization.pool_service import default_sources

    names = {s.name for s in default_sources()}
    assert {"nexus", "local", "archetype", "bomcp"} <= names


def test_arbitrate_next_consumes_the_pool_service():
    # The authority entry (arbitrate_next) builds its pool via the service when
    # one is supplied -- realizing arbitrate_next -> CandidatePoolService -> sources.
    from app.optimization.service import arbitrate_next

    req = OptimizationRequest(campaign_id="c", space=_space(), n=2)
    service = CandidatePoolService(sources=[
        _FakeSource("nexus", [{"x0": 0.15, "x1": 0.15}, {"x0": 0.25, "x1": 0.25}]),
    ])
    outcome = arbitrate_next(req, _decision(), pool_service=service)
    assert outcome.decision is not None
    # The scored pool that the authority arbitrated came from the service's source.
    assert any(s.candidate.source == "nexus" for s in outcome.decision.scored_pool)
