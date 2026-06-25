from __future__ import annotations

from app.services.optimization_intelligence import OptimizationIntelligence
from app.services.strategy_models import CampaignSnapshot, EvidenceItem, PhaseConfig
from app.services.strategy_selector import select_strategy


def _snapshot() -> CampaignSnapshot:
    params = (
        {"temperature": 40.0, "ph": 6.5},
        {"temperature": 45.0, "ph": 6.8},
        {"temperature": 50.0, "ph": 7.0},
        {"temperature": 55.0, "ph": 7.2},
        {"temperature": 60.0, "ph": 7.5},
        {"temperature": 65.0, "ph": 7.8},
    )
    kpis = (0.10, 0.14, 0.18, 0.23, 0.25, 0.31)
    return CampaignSnapshot(
        round_number=3,
        max_rounds=8,
        n_observations=len(kpis),
        n_dimensions=2,
        has_categorical=False,
        has_log_scale=False,
        kpi_history=kpis,
        available_backends={
            "lhs": True,
            "built_in": True,
            "optuna_tpe": False,
            "optuna_cmaes": False,
            "scipy_de": False,
            "pymoo_nsga2": False,
            "random_sampling": True,
        },
        last_batch_kpis=kpis[-3:],
        last_batch_params=params[-3:],
        best_kpi_so_far=max(kpis),
        all_params=params,
        all_kpis=kpis,
    )


def test_optimization_intelligence_failure_degrades_to_local_strategy(monkeypatch):
    class FailingAdvisor:
        def advise(self, snapshot):  # noqa: ANN001, ANN201
            raise RuntimeError("nexus unavailable")

    monkeypatch.setattr(
        "app.services.optimization_intelligence.OptimizationIntelligenceAdvisor",
        lambda: FailingAdvisor(),
    )

    decision = select_strategy(
        _snapshot(),
        PhaseConfig(enable_optimization_intelligence=True),
    )

    assert decision.backend_name
    assert all(not e.signal_name.startswith("nexus_") for e in decision.evidence)


def test_optimization_intelligence_adds_weight_adjustments_and_evidence(monkeypatch):
    injected = EvidenceItem(
        signal_name="nexus_causal_temperature_to_kpi",
        signal_value=0.82,
        target_action="exploit",
        contribution=0.123,
        description="Nexus causal edge temperature->kpi strength=0.82 supports exploit.",
    )

    class FakeAdvisor:
        def advise(self, snapshot):  # noqa: ANN001, ANN201
            return OptimizationIntelligence(
                evidence=(injected,),
                weight_adjustments={"w_improvement": 0.2, "w_info_gain": -0.05},
                recommended_phase="exploitation",
                sources=("nexus_meta_learning", "nexus_diagnostics"),
            )

    monkeypatch.setattr(
        "app.services.optimization_intelligence.OptimizationIntelligenceAdvisor",
        lambda: FakeAdvisor(),
    )

    decision = select_strategy(
        _snapshot(),
        PhaseConfig(enable_optimization_intelligence=True),
    )

    evidence_names = {e.signal_name for e in decision.evidence}
    assert "nexus_causal_temperature_to_kpi" in evidence_names
    assert decision.weights_used is not None
    assert "optimization intelligence meta-learning adj" in decision.weights_used.reason
