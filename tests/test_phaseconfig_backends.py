"""Δ2a: nexus_* backends are selectable through PhaseConfig preference tuples.

Ordering rule: existing optional backends first (no behaviour change when they
are installed), then the Nexus equivalents (enrichment when optuna/scipy/pymoo
are absent), then ``built_in`` as the guaranteed fallback.  Selection stays
deterministic (first available in preference order).
"""
from __future__ import annotations

from app.services.strategy_actions import _pick_first_available
from app.services.strategy_models import PhaseConfig


def test_nexus_selected_for_exploitation_when_optuna_absent():
    cfg = PhaseConfig()
    available = {"optuna_tpe": False, "nexus_tpe": True, "nexus_gp_bo": True, "built_in": True}
    assert _pick_first_available(cfg.exploitation_backends, available) == "nexus_tpe"


def test_optuna_still_preferred_when_available():
    cfg = PhaseConfig()
    available = {"optuna_tpe": True, "nexus_tpe": True, "built_in": True}
    assert _pick_first_available(cfg.exploitation_backends, available) == "optuna_tpe"


def test_built_in_fallback_when_no_optional_backends():
    cfg = PhaseConfig()
    available = {"optuna_tpe": False, "built_in": True}
    assert _pick_first_available(cfg.exploitation_backends, available) == "built_in"


def test_nexus_in_refinement_and_high_dim_tuples():
    cfg = PhaseConfig()
    assert any(b.startswith("nexus_") for b in cfg.refinement_backends)
    assert any(b.startswith("nexus_") for b in cfg.high_dim_backends)
    # built_in remains the final fallback everywhere
    assert cfg.exploitation_backends[-1] == "built_in"
    assert cfg.refinement_backends[-1] == "built_in"
    assert cfg.high_dim_backends[-1] == "built_in"


def test_nexus_high_dim_selected_when_pymoo_absent():
    cfg = PhaseConfig()
    available = {"pymoo_nsga2": False, "nexus_nsga2": True, "optuna_tpe": False, "built_in": True}
    assert _pick_first_available(cfg.high_dim_backends, available) == "nexus_nsga2"
