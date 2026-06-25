"""Δ2e: end-to-end soft-bias behaviour through select_strategy.

Covers the acceptance criteria at the integration level:
  1. nexus_* backends are selectable through PhaseConfig,
  2. missing Nexus still degrades to built_in,
  3. fingerprint recommendation can change backend ranking,
  4. recommendation cannot select an unavailable or phase-incompatible backend,
  5. backend choice is deterministic under seed,
  6. selection reasoning is captured in provenance.

The recommendation is injected through the existing OptimizationIntelligence
seam (monkeypatched advisor), exactly how Nexus's fingerprint would arrive.
"""
from __future__ import annotations

from app.services.optimization_intelligence import OptimizationIntelligence
from app.services.strategy_models import CampaignSnapshot, PhaseConfig
from app.services.strategy_selector import select_strategy

# An "exploration"-yielding snapshot whose backend pool is
# ('lhs', 'nexus_lhs', 'nexus_sobol') -- all three available below.
_ALL_AVAIL = {
    "lhs": True, "built_in": True, "random_sampling": True,
    "optuna_tpe": False, "optuna_cmaes": False, "scipy_de": False, "pymoo_nsga2": False,
    "nexus_lhs": True, "nexus_sobol": True, "nexus_tpe": True, "nexus_gp_bo": True,
}
_NO_NEXUS = {
    "lhs": False, "built_in": True, "random_sampling": False,
    "optuna_tpe": False, "optuna_cmaes": False, "scipy_de": False, "pymoo_nsga2": False,
}


def _explore_snapshot(available: dict[str, bool]) -> CampaignSnapshot:
    params = tuple({"x": float(i), "y": float(i) * 0.5} for i in range(12))
    kpis = tuple(0.1 + 0.02 * i for i in range(12))
    return CampaignSnapshot(
        round_number=5, max_rounds=10, n_observations=12, n_dimensions=2,
        has_categorical=False, has_log_scale=False, kpi_history=kpis, direction="maximize",
        available_backends=available, last_batch_kpis=kpis[-3:], last_batch_params=params[-3:],
        best_kpi_so_far=max(kpis), all_params=params, all_kpis=kpis,
    )


def _plateau_snapshot(available: dict[str, bool]) -> CampaignSnapshot:
    params = tuple({"x": float(i % 3)} for i in range(30))
    kpis = tuple([0.1, 0.3, 0.6, 0.85, 0.9, 0.92] + [0.93] * 24)
    return CampaignSnapshot(
        round_number=9, max_rounds=10, n_observations=30, n_dimensions=2,
        has_categorical=False, has_log_scale=False, kpi_history=kpis, direction="maximize",
        available_backends=available, last_batch_kpis=kpis[-3:], last_batch_params=params[-3:],
        best_kpi_so_far=max(kpis), all_params=params, all_kpis=kpis,
    )


def _patch_recommendation(monkeypatch, recommended: tuple[str, ...]) -> None:
    class FakeAdvisor:
        def advise(self, snapshot):  # noqa: ANN001, ANN201
            return OptimizationIntelligence(recommended_backends=recommended)

    monkeypatch.setattr(
        "app.services.optimization_intelligence.OptimizationIntelligenceAdvisor",
        lambda: FakeAdvisor(),
    )


# Conservative-but-flip-capable weight for the 3-item explore pool.
def _cfg(**kw) -> PhaseConfig:
    return PhaseConfig(enable_optimization_intelligence=True, backend_fingerprint_weight=0.4, **kw)


# --- Criterion 6: provenance is always captured -----------------------------

def test_backend_selection_provenance_is_populated():
    decision = select_strategy(_explore_snapshot(_ALL_AVAIL), PhaseConfig())
    bs = decision.backend_selection
    assert bs is not None
    assert bs.phase
    assert bs.candidate_backends
    assert bs.score_components
    assert bs.selected_backend == decision.backend_name
    assert bs.reason


# --- Criterion 1 + 3: nexus selectable; recommendation changes ranking -------

def test_fingerprint_recommendation_changes_selected_backend(monkeypatch):
    baseline = select_strategy(_explore_snapshot(_ALL_AVAIL), _cfg())
    assert baseline.backend_name == "lhs"  # phase default with no recommendation

    _patch_recommendation(monkeypatch, ("nexus_lhs",))
    biased = select_strategy(_explore_snapshot(_ALL_AVAIL), _cfg())
    assert biased.backend_name == "nexus_lhs"  # recommendation promoted it
    assert biased.backend_selection.fingerprint_recommendation == ("nexus_lhs",)


# --- Criterion 4: recommendation cannot pull in unavailable backend ----------

def test_recommendation_cannot_select_unavailable_backend(monkeypatch):
    available = {**_ALL_AVAIL, "nexus_sobol": False}
    _patch_recommendation(monkeypatch, ("nexus_sobol",))
    decision = select_strategy(_explore_snapshot(available), _cfg())
    assert decision.backend_name != "nexus_sobol"
    assert decision.backend_name == "lhs"


# --- Criterion 4: recommendation cannot pull in phase-incompatible backend ----

def test_recommendation_cannot_select_out_of_pool_backend(monkeypatch):
    # nexus_nsga2 is not in the exploration pool.
    _patch_recommendation(monkeypatch, ("nexus_nsga2",))
    decision = select_strategy(_explore_snapshot(_ALL_AVAIL), _cfg())
    assert decision.backend_name == "lhs"


# --- Criterion 2: missing Nexus still degrades to built_in -------------------

def test_missing_nexus_degrades_to_built_in():
    decision = select_strategy(_plateau_snapshot(_NO_NEXUS), PhaseConfig())
    assert decision.backend_name == "built_in"
    assert decision.backend_selection.fallback is True


def test_select_strategy_runs_without_nexus_available():
    # Whole pipeline must not raise when no Nexus backend is available.
    decision = select_strategy(_explore_snapshot(_NO_NEXUS), PhaseConfig())
    assert decision.backend_name  # a valid, usable backend


# --- Criterion 5: deterministic under seed -----------------------------------

def test_backend_choice_is_deterministic(monkeypatch):
    _patch_recommendation(monkeypatch, ("nexus_lhs",))
    a = select_strategy(_explore_snapshot(_ALL_AVAIL), _cfg())
    b = select_strategy(_explore_snapshot(_ALL_AVAIL), _cfg())
    assert a.backend_name == b.backend_name
    assert a.backend_selection == b.backend_selection


# --- Phase policy still dominant: phase is never overridden by fingerprint ----

def test_fingerprint_does_not_change_phase(monkeypatch):
    baseline = select_strategy(_explore_snapshot(_ALL_AVAIL), _cfg())
    _patch_recommendation(monkeypatch, ("nexus_lhs",))
    biased = select_strategy(_explore_snapshot(_ALL_AVAIL), _cfg())
    assert baseline.phase == biased.phase == "exploration"
