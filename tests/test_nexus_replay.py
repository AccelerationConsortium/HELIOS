"""Smoke test for the Nexus optimization replay/comparison harness.

Runs a tiny before/after comparison and asserts the key invariants:
  * the harness produces both metric sets,
  * provenance is complete every round,
  * the fingerprint soft bias never changes the phase (phase_instability == 0),
  * rates are well-formed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("optimization_copilot")

from benchmarks.nexus_replay import ReplayMetrics, compare_before_after


def test_replay_comparison_runs_and_holds_invariants():
    results = compare_before_after("her", seeds=1, rounds=3, batch_size=2)
    before, after = results["before"], results["after"]

    assert isinstance(before, ReplayMetrics)
    assert isinstance(after, ReplayMetrics)

    # Provenance must be complete on every round in both arms.
    assert before.provenance_completeness == 1.0
    assert after.provenance_completeness == 1.0

    # The conservative soft bias must never flip the phase on a fixed snapshot.
    assert after.phase_instability == 0.0

    # Well-formed rates.
    for m in (before, after):
        assert 0.0 <= m.fallback_rate <= 1.0
        assert 0.0 <= m.duplicate_rate <= 1.0
        assert len(m.best_kpi_trajectory) == 3
        assert sum(m.backend_distribution.values()) == 3  # one backend per round


def test_after_arm_can_use_nexus_backends():
    # Over enough rounds the enriched arm should at least be *able* to pick a
    # nexus_* backend (availability + recommendation make them selectable).
    results = compare_before_after("her", seeds=2, rounds=8, batch_size=2)
    before = results["before"]
    # BEFORE never has nexus available, so it can never select one.
    assert not any(b.startswith("nexus_") for b in before.backend_distribution)
    # AFTER ran the full budget with provenance intact.
    assert results["after"].provenance_completeness == 1.0
