"""Tests for the shared verifiable-reward core (Phase A / RLVR wedge).

These lock the RewardVerification contract and prove the canonical component
scores match the values the two legacy calculators (decision_outcome,
loop_engineering) already produce, so the delegation refactor is a bit-for-bit
no-op (regression IRON RULE).
"""
from __future__ import annotations

import pytest

from app.services.verifiable_reward import (
    OUTCOME_SIGNALS,
    PROCESS_SIGNALS,
    RewardVerification,
    run_verifier,
    split_reward,
    verify_context,
    verify_execution,
    verify_failure,
    verify_objective,
    verify_proxy_gap,
    verify_recovery,
    verify_safety,
    verify_validation,
)

# --- RewardVerification contract -----------------------------------------


def test_verification_shape():
    v = verify_execution(True)
    assert isinstance(v, RewardVerification)
    assert v.name == "execution"
    assert v.passed is True
    assert v.score == 0.2
    assert v.verifier_type == "state_transition"
    assert "execution_success" in v.evidence


# --- execution: True / False / None (passed tri-state) -------------------


@pytest.mark.parametrize(
    "value,passed,score",
    [(True, True, 0.2), (False, False, -0.3), (None, None, 0.0)],
)
def test_execution(value, passed, score):
    v = verify_execution(value)
    assert v.passed is passed
    assert v.score == score


# --- objective: sign-based, coefficient 0.3, clamp positive at 1.0 -------


@pytest.mark.parametrize(
    "delta,passed,score",
    [
        (None, None, 0.0),
        (0.5, True, round(0.5 * 0.3, 10)),
        (2.0, True, round(1.0 * 0.3, 10)),  # positive clamped to 1.0 before scaling
        (-0.4, False, round(-0.4 * 0.3, 10)),
    ],
)
def test_objective(delta, passed, score):
    v = verify_objective(delta)
    assert v.passed is passed
    assert v.score == score
    assert v.verifier_type == "outcome_metric"


# --- proxy_gap: reduction (negative delta) is good -----------------------


@pytest.mark.parametrize(
    "delta,passed,score",
    [
        (None, None, 0.0),
        (-0.5, True, round(0.5 * 0.3, 10)),
        (0.5, False, round(-0.5 * 0.3, 10)),
    ],
)
def test_proxy_gap(delta, passed, score):
    v = verify_proxy_gap(delta)
    assert v.passed is passed
    assert v.score == score


# --- validation / failure / safety / recovery / context ------------------


@pytest.mark.parametrize(
    "value,passed,score",
    [(True, True, 0.2), (False, False, -0.2), (None, None, 0.0)],
)
def test_validation(value, passed, score):
    v = verify_validation(value)
    assert v.passed is passed
    assert v.score == score


@pytest.mark.parametrize(
    "count,passed,score",
    [(0, True, 0.0), (2, False, round(-0.1 * 2, 10))],
)
def test_failure(count, passed, score):
    v = verify_failure(count)
    assert v.passed is passed
    assert v.score == score


@pytest.mark.parametrize(
    "count,passed,score",
    [(0, True, 0.0), (1, False, round(-0.5 * 1, 10))],
)
def test_safety(count, passed, score):
    v = verify_safety(count)
    assert v.passed is passed
    assert v.score == score
    assert v.verifier_type == "safety_rule"


@pytest.mark.parametrize(
    "attempted,success,passed,score",
    [
        (False, None, None, 0.0),
        (True, True, True, 0.1),
        (True, False, False, -0.1),
        (True, None, None, 0.0),
    ],
)
def test_recovery(attempted, success, passed, score):
    v = verify_recovery(attempted=attempted, success=success)
    assert v.passed is passed
    assert v.score == score


@pytest.mark.parametrize(
    "fulfilled,passed,score",
    [(True, True, 0.1), (False, None, 0.0), (None, None, 0.0)],
)
def test_context(fulfilled, passed, score):
    v = verify_context(fulfilled)
    assert v.passed is passed
    assert v.score == score


# --- verifier error handling: never swallow, passed=None + evidence ------


def test_run_verifier_catches_and_marks_none():
    def boom():
        raise ValueError("kaboom")

    v = run_verifier("explode", boom, verifier_type="outcome_metric")
    assert v.passed is None
    assert v.score == 0.0
    assert "error" in v.evidence
    assert "kaboom" in v.evidence["error"]


def test_run_verifier_passes_through_success():
    v = run_verifier("ok", lambda: verify_execution(True), verifier_type="state_transition")
    assert v.passed is True
    assert v.score == 0.2


# --- process / outcome split sums to the total ---------------------------


def test_split_reward_partitions_signals():
    verifications = [
        verify_execution(True),      # outcome
        verify_objective(0.5),       # outcome
        verify_proxy_gap(-0.5),      # process
        verify_context(True),        # process
    ]
    process, outcome = split_reward(verifications)
    expected_process = verify_proxy_gap(-0.5).score + verify_context(True).score
    expected_outcome = verify_execution(True).score + verify_objective(0.5).score
    assert process == round(expected_process, 10)
    assert outcome == round(expected_outcome, 10)


def test_signal_partition_is_total():
    # every declared signal is classified exactly once (no orphans, no dupes)
    assert PROCESS_SIGNALS.isdisjoint(OUTCOME_SIGNALS)
    all_signals = PROCESS_SIGNALS | OUTCOME_SIGNALS
    assert {"execution", "objective", "validation", "failure", "safety",
            "proxy_gap", "context", "recovery"} <= all_signals
