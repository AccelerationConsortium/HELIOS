"""P3a — feed the benchmark decision table into the live selector.

The method advisor derives a coarse problem profile (dims / modality / noise /
categorical) from the live CampaignSnapshot + diagnostics and returns a ranked
list of backends for that problem class, drawn from a decision table.  This is
the production-side mirror of the offline benchmarks/methods/recommend table;
dependency direction is app <- (data), never app -> benchmarks.
"""
from __future__ import annotations

from app.optimization.method_advisor import problem_profile, recommend_backends
from app.services.strategy_models import CampaignSnapshot, DiagnosticSignals


def _snap(n_dims, *, categorical=False):
    return CampaignSnapshot(
        round_number=2,
        max_rounds=5,
        n_observations=20,
        n_dimensions=n_dims,
        has_categorical=categorical,
        has_log_scale=False,
    )


def _diag(*, smoothness=0.9, noise=0.0):
    return DiagnosticSignals(
        space_coverage=0.5,
        model_uncertainty=0.2,
        noise_ratio=noise,
        replicate_need_score=None,
        batch_kpi_cv=None,
        improvement_velocity=0.1,
        ei_decay_proxy=None,
        kpi_var_ratio=None,
        convergence_status="improving",
        convergence_confidence=0.5,
        local_smoothness=smoothness,
        batch_param_spread=None,
    )


# --- profile derivation ------------------------------------------------------


def test_profile_low_dim_smooth_unimodal():
    assert problem_profile(_snap(2), _diag(smoothness=0.9, noise=0.0)) == ("low", "unimodal", "low")


def test_profile_high_dim_multimodal_noisy():
    prof = problem_profile(_snap(12), _diag(smoothness=0.1, noise=0.4))
    assert prof == ("high", "multimodal", "high")


# --- recommendations ---------------------------------------------------------


def test_low_dim_smooth_prefers_bo_family():
    recs = recommend_backends(_snap(2), _diag(smoothness=0.9, noise=0.0))
    assert "bomcp" in recs
    # BO leads evolutionary methods on smooth low-dim surfaces.
    assert recs.index("bomcp") < recs.index("pymoo_nsga2")


def test_multimodal_prefers_global_search():
    recs = recommend_backends(_snap(3), _diag(smoothness=0.05, noise=0.0))
    # A global/evolutionary method should out-rank plain GP-BO on rugged surfaces.
    assert recs.index("optuna_cmaes") < recs.index("bomcp")


def test_high_dim_prefers_scalable_methods():
    recs = recommend_backends(_snap(15), _diag(smoothness=0.5, noise=0.0))
    assert recs[0] in {"pymoo_nsga2", "optuna_tpe", "bomcp"}


def test_categorical_promotes_tpe():
    recs = recommend_backends(_snap(4, categorical=True), _diag())
    assert recs[0] == "optuna_tpe"


def test_recommendations_are_known_backends():
    known = {
        "bomcp", "built_in", "optuna_tpe", "optuna_cmaes", "scipy_de",
        "pymoo_nsga2", "lhs", "random_sampling", "gp_backend",
    }
    recs = recommend_backends(_snap(5), _diag())
    assert recs, "advisor must return at least one backend"
    assert set(recs) <= known


# --- live integration: advice reaches rank_backends -------------------------


def test_selector_merges_method_advice_into_recommendation():
    from app.services.strategy_selector import select_strategy

    snap = CampaignSnapshot(
        round_number=3,
        max_rounds=6,
        n_observations=18,
        n_dimensions=2,
        has_categorical=False,
        has_log_scale=False,
        kpi_history=(0.1, 0.3, 0.5, 0.55, 0.6),
        all_kpis=(0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6),
        all_params=tuple({"x0": 0.1 * i, "x1": 0.2 * i} for i in range(7)),
        best_kpi_so_far=0.6,
    )
    decision = select_strategy(snap)
    rec = decision.backend_selection.fingerprint_recommendation
    # The method advisor's picks for a low-dim smooth problem should be present.
    assert any(b in rec for b in ("bomcp", "built_in", "optuna_tpe"))
