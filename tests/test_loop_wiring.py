"""Loop-wiring regression tests (feat/arbitrate-next-loop-wiring).

Wires arbitrate_next into the campaign loop behind a feature flag via a thin,
gated seam.  The seam is the unit that embodies "the loop's use of deep
arbitration"; the four required regressions live here:

  1. flag off  -> seam returns None, so the loop uses its existing path unchanged
  2. Nexus outage -> arbitration still produces a batch (degrades to local)
  3. pool failure -> seam returns None (no downgrade, no campaign break)
  4. flag on + Δ1&Δ2 active -> provenance carries BOTH the Δ2 backend-selection
     trace and the Δ1 scored candidate-pool trace

Guardrails unchanged: strategy_selector stays the authority; the boundary layer
defines no weights and recomputes no phase.
"""
from __future__ import annotations

from app.optimization.loop_integration import (
    arbitrate_round_if_enabled,
    build_optimization_request,
)
from app.optimization.provenance import ProvenanceLogger
from app.optimization.schemas import (
    CandidateSuggestion,
    DecisionResult,
    OptimizationRequest,
    PooledCandidate,
    ScoredCandidate,
)
from app.optimization.service import OptimizationOutcome
from app.services.candidate_gen import ParameterSpace, SearchDimension
from app.services.strategy_models import ActionCandidate, StrategyDecision, WeightsUsed


def _space() -> ParameterSpace:
    return ParameterSpace(
        dimensions=(SearchDimension("x", "number", 0.0, 10.0),),
        protocol_template={},
    )


def _request() -> OptimizationRequest:
    return OptimizationRequest(campaign_id="c", space=_space(), n=2, seed=11)


def _decision(recommended_backends=("nexus_gp_bo", "nexus_tpe")) -> StrategyDecision:
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
        recommended_backends=recommended_backends,
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

    def suggest(self, request):  # pragma: no cover
        return self.suggest_top_k(request, request.n)


class _FakeLocal:
    def is_available(self) -> bool:
        return True

    def suggest(self, request):
        return CandidateSuggestion(
            candidates=({"x": 8.0},), algorithm="built_in", source="local_fallback"
        )


class _BrokenPoolBuilder:
    def build(self, *a, **k):
        raise RuntimeError("candidate pool construction blew up")


# --- 1. flag off -> existing behavior unchanged ----------------------------


def test_flag_off_returns_none_so_loop_uses_legacy_path():
    out = arbitrate_round_if_enabled(
        _request(), _decision(), enabled=False,
        provider=_FakeNexus(), fallback=_FakeLocal(),
    )
    assert out is None


# --- 2. Nexus outage degrades gracefully -----------------------------------


def test_nexus_outage_degrades_to_local_never_empty():
    nexus = _FakeNexus()
    nexus.available = False
    out = arbitrate_round_if_enabled(
        _request(), _decision(), enabled=True, provider=nexus, fallback=_FakeLocal()
    )
    assert isinstance(out, OptimizationOutcome)
    assert out.decision.accepted is True
    sources = {s.candidate.source for s in out.decision.scored_pool}
    assert "nexus" not in sources and "local" in sources


# --- 3. pool failure -> no downgrade / no campaign break -------------------


def test_pool_failure_returns_none_without_raising():
    # A pool-construction failure must not break the campaign: the seam swallows
    # it and returns None so the caller falls back to its legacy generation path.
    out = arbitrate_round_if_enabled(
        _request(), _decision(), enabled=True,
        provider=_FakeNexus(), fallback=_FakeLocal(),
        pool_builder=_BrokenPoolBuilder(),
    )
    assert out is None


# --- 4. both-traces provenance when Δ1 + Δ2 active -------------------------


def test_both_traces_provenance_present_when_active():
    out = arbitrate_round_if_enabled(
        _request(), _decision(recommended_backends=("nexus_gp_bo", "nexus_tpe")),
        enabled=True, provider=_FakeNexus(), fallback=_FakeLocal(),
    )
    assert out is not None
    prov = out.provenance
    # Δ2 backend-selection trace
    bsel = prov["backend_selection"]
    assert bsel["chosen_backend"] == "nexus_gp_bo"
    assert bsel["phase"] == "exploitation"
    assert list(bsel["recommended_backends"]) == ["nexus_gp_bo", "nexus_tpe"]
    # Δ1 scored candidate-pool trace
    assert prov["candidate_pool"]
    assert any(e["selected"] for e in prov["candidate_pool"])


# --- provenance unit: backend_selection only when a decision is supplied ----


def test_provenance_backend_selection_present_with_decision():
    suggestion = CandidateSuggestion(
        candidates=({"x": 2.0},), algorithm="nexus_gp_bo", source="nexus"
    )
    result = DecisionResult(
        accepted=True,
        final_candidates=({"x": 2.0},),
        scored_pool=(
            ScoredCandidate(
                candidate=PooledCandidate({"x": 2.0}, "nexus", "exploit", "nexus_gp_bo"),
                utility=0.55, base_utility=0.55, delta=0.0, redundancy=0.0,
            ),
        ),
    )
    rec = ProvenanceLogger().build(
        _request(), suggestion, result, strategy_decision=_decision()
    )
    assert rec["backend_selection"]["recommended_backends"] == ["nexus_gp_bo", "nexus_tpe"]


def test_provenance_backend_selection_empty_without_decision():
    # legacy callers (suggest_next) pass no StrategyDecision -> empty trace, no error
    suggestion = CandidateSuggestion(
        candidates=({"x": 2.0},), algorithm="built_in", source="local_fallback"
    )
    result = DecisionResult(accepted=True, final_candidates=({"x": 2.0},))
    rec = ProvenanceLogger().build(_request(), suggestion, result)
    assert rec["backend_selection"] == {}


# --- feature flag (Settings) ------------------------------------------------


def test_settings_candidate_arbitration_flag_defaults_off(monkeypatch):
    from app.core.config import Settings

    monkeypatch.delenv("ENABLE_CANDIDATE_ARBITRATION", raising=False)
    assert Settings().enable_candidate_arbitration is False
    monkeypatch.setenv("ENABLE_CANDIDATE_ARBITRATION", "true")
    assert Settings().enable_candidate_arbitration is True


# --- authority surfaces recommended_backends onto the decision (Δ2 trace) ---


def test_select_strategy_surfaces_recommended_backends(monkeypatch):
    import app.services.optimization_intelligence as oi
    from app.services.optimization_intelligence import OptimizationIntelligence
    from app.services.strategy_models import CampaignSnapshot, PhaseConfig
    from app.services.strategy_selector import select_strategy

    class _FakeAdvisor:
        def advise(self, snapshot, **kwargs):
            return OptimizationIntelligence(
                recommended_backends=("nexus_gp_bo", "nexus_tpe"),
                sources=("nexus_fingerprint",),
            )

    monkeypatch.setattr(oi, "OptimizationIntelligenceAdvisor", _FakeAdvisor)

    snapshot = CampaignSnapshot(
        round_number=3,
        max_rounds=20,
        n_observations=12,
        n_dimensions=2,
        has_categorical=False,
        has_log_scale=False,
        kpi_history=(0.1, 0.2, 0.3, 0.4),
        direction="maximize",
    )
    decision = select_strategy(snapshot, PhaseConfig(enable_optimization_intelligence=True))
    assert decision.recommended_backends == ("nexus_gp_bo", "nexus_tpe")


# --- request builder (campaign context -> OptimizationRequest) --------------


def test_build_optimization_request_from_campaign_context():
    req = build_optimization_request(
        campaign_id="camp-9",
        dimensions=[{"param_name": "x", "param_type": "number",
                     "min_value": 0.0, "max_value": 10.0}],
        protocol_template={},
        all_params=[{"x": 1.0}, {"x": 2.0}],
        all_kpis=[0.5, 0.8],
        objective_name="CE",
        direction="maximize",
        n=3,
        seed=7,
        round_index=4,
    )
    assert req.campaign_id == "camp-9"
    assert req.n == 3 and req.seed == 7 and req.round_index == 4
    assert req.objective_name == "CE" and req.direction == "maximize"
    # space reconstructed from dimension dicts
    assert len(req.space.dimensions) == 1
    dim = req.space.dimensions[0]
    assert dim.param_name == "x" and dim.min_value == 0.0 and dim.max_value == 10.0
    # observations zip params with kpis
    assert len(req.observations) == 2
    assert req.observations[0].params == {"x": 1.0}
    assert req.observations[1].objective == 0.8


def test_build_optimization_request_tolerates_missing_history():
    req = build_optimization_request(
        campaign_id="c",
        dimensions=[{"param_name": "x", "param_type": "number",
                     "min_value": 0.0, "max_value": 1.0}],
        protocol_template={},
        all_params=[],
        all_kpis=[],
    )
    assert req.observations == ()
