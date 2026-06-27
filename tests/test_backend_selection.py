"""Δ2b: rank_backends -- conservative fingerprint-biased backend ranking.

Rules:
  * phase policy is dominant (preference order = base score),
  * fingerprint recommendation is a secondary, additive boost,
  * availability and phase-incompatibility are hard vetoes (only pool members
    that are available can ever be selected),
  * recent failure history penalizes and (past a threshold) vetoes,
  * selection is deterministic, with ties broken by preference order,
  * with no recommendation and no failures it reduces to "first available".
"""
from __future__ import annotations

from app.services.backend_selection import BackendSelection, rank_backends
from app.services.strategy_actions import _pick_first_available

POOL = ("optuna_tpe", "nexus_tpe", "nexus_gp_bo", "built_in")
ALL_AVAIL = {b: True for b in POOL}


def test_no_recommendation_matches_first_available():
    avail = {"optuna_tpe": False, "nexus_tpe": True, "nexus_gp_bo": True, "built_in": True}
    sel = rank_backends("exploitation", POOL, avail)
    assert sel.selected_backend == _pick_first_available(POOL, avail) == "nexus_tpe"
    assert sel.fallback is False


def test_returns_backend_selection_with_provenance():
    sel = rank_backends("exploitation", POOL, ALL_AVAIL, recommended=("nexus_gp_bo",))
    assert isinstance(sel, BackendSelection)
    assert sel.phase == "exploitation"
    assert sel.fingerprint_recommendation == ("nexus_gp_bo",)
    assert sel.score_components  # per-candidate breakdown present
    assert sel.reason


def test_fingerprint_promotes_lower_ranked_available_backend():
    # nexus_tpe is 2nd preference; recommendation should let it overtake optuna_tpe.
    sel = rank_backends("exploitation", POOL, ALL_AVAIL, recommended=("nexus_tpe",))
    assert sel.selected_backend == "nexus_tpe"


def test_fingerprint_cannot_select_unavailable_backend():
    avail = {"optuna_tpe": True, "nexus_tpe": False, "nexus_gp_bo": False, "built_in": True}
    sel = rank_backends("exploitation", POOL, avail, recommended=("nexus_tpe",))
    assert sel.selected_backend == "optuna_tpe"  # recommended one is unavailable -> ignored


def test_fingerprint_cannot_select_phase_incompatible_backend():
    # nexus_nsga2 is not in the exploitation pool -> recommendation has no effect.
    sel = rank_backends("exploitation", POOL, ALL_AVAIL, recommended=("nexus_nsga2",))
    assert sel.selected_backend == "optuna_tpe"


def test_failure_history_vetoes_backend():
    sel = rank_backends(
        "exploitation", POOL, ALL_AVAIL, failure_counts={"optuna_tpe": 3}, failure_veto_threshold=3
    )
    assert sel.selected_backend == "nexus_tpe"  # optuna vetoed by failures


def test_failure_penalty_changes_ranking_without_veto():
    sel = rank_backends(
        "exploitation", POOL, ALL_AVAIL, failure_counts={"optuna_tpe": 2}, failure_veto_threshold=3
    )
    assert sel.selected_backend == "nexus_tpe"  # penalised below 2nd preference


def test_deterministic_under_same_inputs():
    a = rank_backends("exploitation", POOL, ALL_AVAIL, recommended=("nexus_tpe",))
    b = rank_backends("exploitation", POOL, ALL_AVAIL, recommended=("nexus_tpe",))
    assert a == b


def test_degrades_to_built_in_with_fallback_flag():
    avail = {"optuna_tpe": False, "nexus_tpe": False, "nexus_gp_bo": False, "built_in": True}
    sel = rank_backends("exploitation", POOL, avail)
    assert sel.selected_backend == "built_in"
    assert sel.fallback is True


def test_built_in_selection_is_marked_as_fallback():
    # fallback status == "running on the universal fallback optimizer".
    sel = rank_backends("stabilize", ("built_in",), {"built_in": True})
    assert sel.selected_backend == "built_in"
    assert sel.fallback is True


def test_phase_policy_dominates_low_ranked_recommendation():
    # nexus_gp_bo is 3rd preference; a single conservative boost must not let it
    # overtake the top phase preference.
    sel = rank_backends("exploitation", POOL, ALL_AVAIL, recommended=("nexus_gp_bo",))
    assert sel.selected_backend == "optuna_tpe"
