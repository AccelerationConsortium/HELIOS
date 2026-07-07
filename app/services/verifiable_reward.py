"""Shared, verifiable reward core (Phase A / RLVR wedge).

Single source of truth for the deterministic reward *components* that both
``decision_outcome.CampaignDecisionRewardCalculator`` and
``loop_engineering.LoopRewardCalculator`` compute. Historically each file
carried its own verbatim copies of these helpers; this module de-duplicates
them and, on top of the raw scalar, exposes each signal as a
``RewardVerification`` — an auditable ``(passed, score, evidence)`` record that
makes the reward RLVR-ready (per-signal verifiable) instead of a single opaque
number.

Design invariants (regression IRON RULE): the scalar returned by each
``*_score`` function is bit-for-bit identical to the value the legacy helpers
produced, so the two calculators can delegate here without changing any
existing reward number.

    signal ──► *_score() ──► float          (what the legacy calculators need)
           └─► verify_*() ──► RewardVerification(passed, score, evidence)

``passed`` is tri-state:
    True  — the "good" condition held (execution succeeded, objective improved)
    False — the "bad" condition held (execution failed, safety incident)
    None  — the signal was not observed / not applicable (delta is None)

A verifier that raises is never swallowed: ``run_verifier`` converts it into a
``passed=None`` record carrying the error in ``evidence`` (CLAUDE.md rule 8).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "RewardVerification",
    "VerifierType",
    "PROCESS_SIGNALS",
    "OUTCOME_SIGNALS",
    "RUBRIC_VERSION_DEFAULT",
    "round_component",
    "clamp",
    "execution_score",
    "objective_score",
    "proxy_gap_score",
    "validation_score",
    "failure_score",
    "safety_score",
    "recovery_score",
    "context_score",
    "verify_execution",
    "verify_objective",
    "verify_proxy_gap",
    "verify_validation",
    "verify_failure",
    "verify_safety",
    "verify_recovery",
    "verify_context",
    "run_verifier",
    "split_reward",
]

VerifierType = Literal[
    "unit_test",
    "state_transition",
    "safety_rule",
    "outcome_metric",
    "retrospective_audit",
]

# Legacy default so historical reward records (which predate rubric
# versioning) load with a stable, comparable version tag.
RUBRIC_VERSION_DEFAULT = "v0.1_static"


class RewardVerification(BaseModel):
    """One auditable signal within a decision's reward.

    ``score`` is the signal's contribution to the (pre-clamp) raw reward.
    ``passed`` records the qualitative verdict (see module docstring).
    ``evidence`` carries the raw inputs so a reviewer can reproduce the verdict.
    """

    name: str
    passed: bool | None
    score: float
    verifier_type: VerifierType
    evidence: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rounding (canonical — matches both legacy files exactly)
# ---------------------------------------------------------------------------


def round_component(value: float) -> float:
    return round(value, 10)


def clamp(value: float) -> float:
    return round_component(max(-1.0, min(1.0, value)))


# ---------------------------------------------------------------------------
# Scalar component scores (bit-for-bit identical to the legacy helpers)
# ---------------------------------------------------------------------------


def execution_score(execution_success: bool | None) -> float:
    if execution_success is True:
        return 0.2
    if execution_success is False:
        return -0.3
    return 0.0


def objective_score(
    objective_delta: float | None, *, positive_clamp: bool = True
) -> float:
    """Objective-improvement reward, coefficient 0.3.

    ``positive_clamp=True`` (decision layer) caps a positive delta at 1.0 before
    scaling; ``positive_clamp=False`` (loop layer) scales the raw delta. The two
    layers historically differed only in this clamp — expressing both here keeps
    a single source of truth without changing either layer's numbers.
    """
    if objective_delta is None:
        return 0.0
    if positive_clamp and objective_delta > 0:
        return round_component(min(objective_delta, 1.0) * 0.3)
    return round_component(objective_delta * 0.3)


def proxy_gap_score(proxy_gap_delta: float | None) -> float:
    if proxy_gap_delta is None:
        return 0.0
    if proxy_gap_delta < 0:
        return round_component(abs(proxy_gap_delta) * 0.3)
    return round_component(-proxy_gap_delta * 0.3)


def validation_score(validation_success: bool | None) -> float:
    if validation_success is True:
        return 0.2
    if validation_success is False:
        return -0.2
    return 0.0


def failure_score(failure_count: int) -> float:
    return round_component(-0.1 * failure_count)


def safety_score(safety_incident_count: int) -> float:
    return round_component(-0.5 * safety_incident_count)


def recovery_score(*, attempted: bool, success: bool | None) -> float:
    if not attempted:
        return 0.0
    if success is True:
        return 0.1
    if success is False:
        return -0.1
    return 0.0


def context_score(context_request_fulfilled: bool | None) -> float:
    return 0.1 if context_request_fulfilled is True else 0.0


# ---------------------------------------------------------------------------
# Verifiers: scalar + qualitative verdict + reproducible evidence
# ---------------------------------------------------------------------------


def verify_execution(execution_success: bool | None) -> RewardVerification:
    return RewardVerification(
        name="execution",
        passed=execution_success,
        score=execution_score(execution_success),
        verifier_type="state_transition",
        evidence={"execution_success": execution_success},
    )


def verify_objective(
    objective_delta: float | None, *, positive_clamp: bool = True
) -> RewardVerification:
    passed = None if objective_delta is None else objective_delta > 0
    return RewardVerification(
        name="objective",
        passed=passed,
        score=objective_score(objective_delta, positive_clamp=positive_clamp),
        verifier_type="outcome_metric",
        evidence={"objective_delta": objective_delta, "positive_clamp": positive_clamp},
    )


def verify_proxy_gap(proxy_gap_delta: float | None) -> RewardVerification:
    # Reducing the proxy gap (negative delta) is the desirable direction.
    passed = None if proxy_gap_delta is None else proxy_gap_delta < 0
    return RewardVerification(
        name="proxy_gap",
        passed=passed,
        score=proxy_gap_score(proxy_gap_delta),
        verifier_type="outcome_metric",
        evidence={"proxy_gap_delta": proxy_gap_delta},
    )


def verify_validation(validation_success: bool | None) -> RewardVerification:
    return RewardVerification(
        name="validation",
        passed=validation_success,
        score=validation_score(validation_success),
        verifier_type="outcome_metric",
        evidence={"validation_success": validation_success},
    )


def verify_failure(failure_count: int) -> RewardVerification:
    return RewardVerification(
        name="failure",
        passed=failure_count == 0,
        score=failure_score(failure_count),
        verifier_type="outcome_metric",
        evidence={"failure_count": failure_count},
    )


def verify_safety(safety_incident_count: int) -> RewardVerification:
    return RewardVerification(
        name="safety",
        passed=safety_incident_count == 0,
        score=safety_score(safety_incident_count),
        verifier_type="safety_rule",
        evidence={"safety_incident_count": safety_incident_count},
    )


def verify_recovery(*, attempted: bool, success: bool | None) -> RewardVerification:
    if not attempted:
        passed: bool | None = None
    else:
        passed = success
    return RewardVerification(
        name="recovery",
        passed=passed,
        score=recovery_score(attempted=attempted, success=success),
        verifier_type="state_transition",
        evidence={"attempted": attempted, "success": success},
    )


def verify_context(context_request_fulfilled: bool | None) -> RewardVerification:
    passed = True if context_request_fulfilled is True else None
    return RewardVerification(
        name="context",
        passed=passed,
        score=context_score(context_request_fulfilled),
        verifier_type="state_transition",
        evidence={"context_request_fulfilled": context_request_fulfilled},
    )


def run_verifier(
    name: str,
    fn: Callable[[], RewardVerification],
    *,
    verifier_type: VerifierType,
) -> RewardVerification:
    """Run a verifier, converting any exception into an auditable failure.

    Never swallows the error: it lands in ``evidence['error']`` with
    ``passed=None`` and ``score=0.0`` so a broken verifier degrades the record
    rather than the whole reward (CLAUDE.md rule 8: fail loudly, not silently).
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — deliberately broad; recorded, not hidden
        return RewardVerification(
            name=name,
            passed=None,
            score=0.0,
            verifier_type=verifier_type,
            evidence={"error": f"{type(exc).__name__}: {exc}"},
        )


# ---------------------------------------------------------------------------
# process / outcome split (design A2)
# ---------------------------------------------------------------------------
#   outcome  — observed results of having acted
#   process  — whether the move was reasonable given the state
# Net: process_reward + outcome_reward == sum(scores) (pre-clamp raw reward).

OUTCOME_SIGNALS: frozenset[str] = frozenset(
    {"execution", "objective", "validation", "failure", "safety", "recovery"}
)
PROCESS_SIGNALS: frozenset[str] = frozenset({"proxy_gap", "context"})


def split_reward(
    verifications: list[RewardVerification],
) -> tuple[float, float]:
    """Return ``(process_reward, outcome_reward)`` from a verification list."""
    process = sum(v.score for v in verifications if v.name in PROCESS_SIGNALS)
    outcome = sum(v.score for v in verifications if v.name in OUTCOME_SIGNALS)
    return round_component(process), round_component(outcome)
