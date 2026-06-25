"""Tests for per-backend failure-history accumulation.

A round's execution outcome (did it produce QC failures?) is attributed to the
backend that suggested it.  Counts accumulate toward the veto threshold on
failure and *heal* on success, so the signal reflects *recent* reliability
rather than a permanent blacklist.
"""
from __future__ import annotations

from app.optimization.failure_history import update_backend_failures


def test_failure_increments_count():
    assert update_backend_failures({}, "nexus_tpe", round_had_failure=True) == {"nexus_tpe": 1}


def test_success_heals_count():
    assert update_backend_failures({"nexus_tpe": 2}, "nexus_tpe", round_had_failure=False) == {"nexus_tpe": 1}


def test_success_removes_zeroed_entry():
    assert update_backend_failures({"nexus_tpe": 1}, "nexus_tpe", round_had_failure=False) == {}


def test_success_on_clean_backend_is_noop():
    assert update_backend_failures({}, "nexus_tpe", round_had_failure=False) == {}


def test_none_backend_is_noop():
    assert update_backend_failures({"a": 1}, None, round_had_failure=True) == {"a": 1}


def test_does_not_mutate_input():
    counts = {"a": 1}
    update_backend_failures(counts, "a", round_had_failure=True)
    assert counts == {"a": 1}


def test_accumulates_across_rounds_to_veto_threshold():
    counts: dict[str, int] = {}
    for _ in range(3):
        counts = update_backend_failures(counts, "b", round_had_failure=True)
    assert counts["b"] == 3
