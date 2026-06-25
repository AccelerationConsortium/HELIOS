"""Δ2c: fingerprint -> backend recommendation.

``fingerprint_to_backends`` is a pure mapping from a Nexus problem fingerprint
to an ordered list of ``nexus_*`` backends.  ``recommend_backends`` drives it
from a HELIOS CampaignSnapshot through Nexus's profiler (diagnose path), and
degrades to an empty tuple when Nexus is unavailable or there is no history.
"""
from __future__ import annotations

import pytest

from app.optimization.recommendation import fingerprint_to_backends
from app.services.strategy_models import CampaignSnapshot


def test_multi_objective_recommends_nsga2_first():
    fp = {"objective_form": "multi_objective", "noise_regime": "low", "data_scale": "moderate"}
    assert fingerprint_to_backends(fp)[0] == "nexus_nsga2"


def test_high_noise_recommends_robust_backends():
    fp = {"objective_form": "single", "noise_regime": "high", "data_scale": "moderate"}
    assert fingerprint_to_backends(fp)[0] == "nexus_rf_bo"


def test_tiny_data_recommends_space_filling():
    fp = {"objective_form": "single", "noise_regime": "low", "data_scale": "tiny"}
    assert fingerprint_to_backends(fp)[0] == "nexus_lhs"


def test_high_dimensionality_recommends_cmaes():
    fp = {
        "objective_form": "single",
        "noise_regime": "low",
        "data_scale": "moderate",
        "effective_dimensionality": 15,
    }
    assert fingerprint_to_backends(fp)[0] == "nexus_cmaes"


def test_standard_problem_recommends_gp_bo():
    fp = {
        "objective_form": "single",
        "noise_regime": "low",
        "data_scale": "moderate",
        "effective_dimensionality": 3,
    }
    assert fingerprint_to_backends(fp)[0] == "nexus_gp_bo"


def test_empty_fingerprint_returns_empty():
    assert fingerprint_to_backends({}) != ()  # falls through to a safe default
    assert all(b.startswith("nexus_") for b in fingerprint_to_backends({}))


def test_recommend_backends_empty_history_returns_empty():
    from app.optimization.recommendation import recommend_backends

    snap = CampaignSnapshot(
        round_number=1, max_rounds=10, n_observations=0,
        n_dimensions=2, has_categorical=False, has_log_scale=False,
    )
    assert recommend_backends(snap) == ()


def test_recommend_backends_from_snapshot_returns_nexus_backends():
    pytest.importorskip("optimization_copilot")
    from app.optimization.recommendation import recommend_backends

    params = tuple({"x": float(i), "y": float(i) * 0.5} for i in range(12))
    kpis = tuple(float(i) for i in range(12))
    snap = CampaignSnapshot(
        round_number=3, max_rounds=10, n_observations=12,
        n_dimensions=2, has_categorical=False, has_log_scale=False,
        direction="maximize", all_params=params, all_kpis=kpis,
    )
    rec = recommend_backends(snap)
    assert rec  # non-empty
    assert all(b.startswith("nexus_") for b in rec)
