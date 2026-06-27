"""Δ1 capstone: arbitrate_next composes the full deep path.

providers (top-k + local baseline) -> CandidatePoolBuilder -> decision policy
(hard gate + delegated soft score) -> provenance.

The service takes the authority ``StrategyDecision`` as input -- it never
computes weights or phases itself (single source of truth).  A Nexus outage
degrades to the local baseline; the round is never empty.
"""
from __future__ import annotations

from app.optimization.schemas import CandidateSuggestion, OptimizationRequest
from app.optimization.service import OptimizationOutcome, arbitrate_next
from app.services.candidate_gen import ParameterSpace, SearchDimension
from app.services.strategy_models import ActionCandidate, StrategyDecision, WeightsUsed


def _space() -> ParameterSpace:
    return ParameterSpace(
        dimensions=(SearchDimension("x", "number", 0.0, 10.0),),
        protocol_template={},
    )


def _request() -> OptimizationRequest:
    return OptimizationRequest(campaign_id="c", space=_space(), n=2, seed=11)


def _decision() -> StrategyDecision:
    return StrategyDecision(
        backend_name="nexus_gp_bo",
        phase="exploitation",
        reason="test",
        confidence=0.8,
        actions_considered=(
            ActionCandidate("exploit", "nexus_gp_bo", 0.7, 0.3, 0.2, 0.55, "x"),
            ActionCandidate("explore", "lhs", 0.4, 0.8, 0.1, 0.45, "x"),
        ),
        weights_used=WeightsUsed(0.45, 0.35, 0.20, "test"),
    )


class _FakeNexus:
    available = True

    def is_available(self) -> bool:
        return self.available

    def suggest_top_k(self, request, k=3):
        return CandidateSuggestion(
            candidates=tuple({"x": float(i)} for i in range(1, k + 1)),
            algorithm="nexus_gp_bo",
            source="nexus",
        )

    def suggest(self, request):  # pragma: no cover - not used here
        return self.suggest_top_k(request, request.n)


class _FakeLocal:
    def is_available(self) -> bool:
        return True

    def suggest(self, request):
        return CandidateSuggestion(
            candidates=({"x": 8.0},), algorithm="built_in", source="local_fallback"
        )


def test_arbitrate_next_runs_deep_path_with_scored_pool():
    out = arbitrate_next(
        _request(), _decision(), provider=_FakeNexus(), fallback=_FakeLocal()
    )
    assert isinstance(out, OptimizationOutcome)
    assert out.decision.accepted is True
    # pool contains nexus top-k + local baseline, all scored
    assert len(out.decision.scored_pool) >= 4
    sources = {s.candidate.source for s in out.decision.scored_pool}
    assert "nexus" in sources and "local" in sources
    # provenance captured the scored portfolio
    assert out.provenance["candidate_pool"]
    assert any(e["selected"] for e in out.provenance["candidate_pool"])


def test_arbitrate_next_degrades_when_nexus_unavailable():
    nexus = _FakeNexus()
    nexus.available = False
    out = arbitrate_next(_request(), _decision(), provider=nexus, fallback=_FakeLocal())
    assert out.decision.accepted is True
    sources = {s.candidate.source for s in out.decision.scored_pool}
    assert "nexus" not in sources
    assert "local" in sources  # never empty
    assert out.provenance["optimizer_source"] == "local_fallback"


def test_arbitrate_next_does_not_recompute_authority():
    # The decision we pass in must be used verbatim: scoring uses its weights and
    # archetype utilities, not a recomputed strategy.
    out = arbitrate_next(
        _request(), _decision(), provider=_FakeNexus(), fallback=_FakeLocal()
    )
    exploit = [s for s in out.decision.scored_pool if s.candidate.source_action == "exploit"]
    assert exploit and exploit[0].base_utility == 0.55  # straight from our decision
