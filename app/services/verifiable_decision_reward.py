"""Verifiable reward wrapper for contextual shadow campaign decisions (RLVR-0).

This module wraps the seven deterministic components already produced by
``CampaignDecisionReward`` (see :mod:`app.services.decision_outcome`) into an
auditable, RLVR-style verification report. Each component becomes a
``RewardVerification`` carrying a verifier type, an explicit pass/neutral
status, a numeric score, and the raw evidence it was derived from. The report
also splits process-level from outcome-level credit.

Boundaries (intentional):

* This layer is pure DTO and deterministic scoring. It does not read or write
  the database, call live services, mutate campaign runtime, touch candidate
  selection, or influence learned-policy promotion.
* It reuses the existing reward components verbatim; it does not recompute or
  change any reward logic in ``decision_outcome``.
* Reports are ``shadow_only=True`` and fully serialize/round-trip via pydantic.
* Missing signals never imply a strong pass: a component whose underlying
  observation is ``None`` is reported as ``status="neutral_due_to_missing_signal"``
  with ``passed=None`` and ``evidence["signal_present"] = False``.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.services.decision_outcome import (
    CampaignDecisionAccounting,
    CampaignDecisionOutcome,
    CampaignDecisionReward,
)

__all__ = [
    "RUBRIC_VERSION",
    "RewardVerification",
    "VerifiableDecisionRewardReport",
    "VerifiableDecisionRewardVerifier",
    "verify_campaign_decision_reward",
]

#: Version tag for the fixed component -> verifier rubric encoded in this module.
RUBRIC_VERSION = "vdr_v1"

VerifierType = Literal[
    "unit_test",
    "state_transition",
    "safety_rule",
    "outcome_metric",
    "retrospective_audit",
    "failure_rule",
    "proxy_metric",
    "validation_rule",
    "context_rule",
]

VerificationStatus = Literal["verified", "neutral_due_to_missing_signal"]

#: Components whose credit reflects the quality of the scientific decision
#: process (safe/clean execution, avoiding failures, seeking context, and the
#: proxy/validation decision-quality signals).
_PROCESS_COMPONENTS = ("execution", "safety", "failure", "context", "proxy_gap", "validation")

#: The single direct outcome metric: did the primary objective improve.
_OUTCOME_COMPONENTS = ("objective",)


class RewardVerification(BaseModel):
    """One component of a decision reward, verified with provenance."""

    name: str
    verifier_type: VerifierType
    status: VerificationStatus
    passed: bool | None
    score: float
    evidence: dict[str, Any] = Field(default_factory=dict)


class VerifiableDecisionRewardReport(BaseModel):
    """RLVR-style verification report for one shadow decision's reward."""

    decision_id: str
    campaign_id: str
    round_index: int = Field(ge=0)
    rubric_version: str = RUBRIC_VERSION
    total_reward: float
    raw_component_sum: float
    clamped: bool
    process_reward: float
    outcome_reward: float
    verifications: list[RewardVerification] = Field(default_factory=list)
    shadow_only: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class VerifiableDecisionRewardVerifier:
    """Verify a decision's reward components into an auditable report."""

    def verify(
        self,
        accounting: CampaignDecisionAccounting,
        *,
        now: datetime | None = None,
    ) -> VerifiableDecisionRewardReport:
        timestamp = now or datetime.now(UTC)
        reward = accounting.reward
        outcome = accounting.outcome

        verifications = [
            _verify_execution(reward, outcome),
            _verify_safety(reward, outcome),
            _verify_failure(reward, outcome),
            _verify_context(reward, outcome),
            _verify_proxy_gap(reward, outcome),
            _verify_validation(reward, outcome),
            _verify_objective(reward, outcome),
        ]
        by_name = {verification.name: verification.score for verification in verifications}

        process_reward = _round(
            sum(by_name[name] for name in _PROCESS_COMPONENTS)
        )
        outcome_reward = _round(
            sum(by_name[name] for name in _OUTCOME_COMPONENTS)
        )
        # raw_component_sum is the unclamped sum of every verification score;
        # by construction process_reward + outcome_reward == raw_component_sum.
        raw_component_sum = _round(process_reward + outcome_reward)
        total_reward = _round(reward.reward)
        clamped = raw_component_sum != total_reward

        return VerifiableDecisionRewardReport(
            decision_id=accounting.trace.trace_id,
            campaign_id=accounting.trace.campaign_id,
            round_index=accounting.trace.round_index,
            rubric_version=RUBRIC_VERSION,
            total_reward=total_reward,
            raw_component_sum=raw_component_sum,
            clamped=clamped,
            process_reward=process_reward,
            outcome_reward=outcome_reward,
            verifications=verifications,
            shadow_only=True,
            created_at=timestamp,
            metadata={
                "source_trace_id": reward.trace_id,
                "source_raw_reward": reward.metadata.get("raw_reward"),
            },
        )


def verify_campaign_decision_reward(
    accounting: CampaignDecisionAccounting,
    *,
    now: datetime | None = None,
) -> VerifiableDecisionRewardReport:
    """Verify a decision reward with the default verifier."""
    return VerifiableDecisionRewardVerifier().verify(accounting, now=now)


def _verify_execution(
    reward: CampaignDecisionReward,
    outcome: CampaignDecisionOutcome,
) -> RewardVerification:
    # execution credit lives in the source reward metadata, not on a field.
    score = _round(float(reward.metadata.get("execution_reward", 0.0)))
    present = outcome.execution_success is not None
    return _build(
        name="execution",
        verifier_type="state_transition",
        score=score,
        present=present,
        passed=(outcome.execution_success is True) if present else None,
        evidence={"execution_success": outcome.execution_success},
    )


def _verify_safety(
    reward: CampaignDecisionReward,
    outcome: CampaignDecisionOutcome,
) -> RewardVerification:
    # safety incident count always present (defaults to 0).
    return _build(
        name="safety",
        verifier_type="safety_rule",
        score=_round(reward.safety_penalty),
        present=True,
        passed=reward.safety_penalty == 0.0,
        evidence={"safety_incident_count": outcome.safety_incident_count},
    )


def _verify_failure(
    reward: CampaignDecisionReward,
    outcome: CampaignDecisionOutcome,
) -> RewardVerification:
    # failure count always present (defaults to 0).
    return _build(
        name="failure",
        verifier_type="failure_rule",
        score=_round(reward.failure_penalty),
        present=True,
        passed=reward.failure_penalty == 0.0,
        evidence={"failure_count": outcome.failure_count},
    )


def _verify_context(
    reward: CampaignDecisionReward,
    outcome: CampaignDecisionOutcome,
) -> RewardVerification:
    present = outcome.context_request_fulfilled is not None
    return _build(
        name="context",
        verifier_type="context_rule",
        score=_round(reward.context_reward),
        present=present,
        passed=(outcome.context_request_fulfilled is True) if present else None,
        evidence={"context_request_fulfilled": outcome.context_request_fulfilled},
    )


def _verify_proxy_gap(
    reward: CampaignDecisionReward,
    outcome: CampaignDecisionOutcome,
) -> RewardVerification:
    present = outcome.proxy_gap_delta is not None
    return _build(
        name="proxy_gap",
        verifier_type="proxy_metric",
        score=_round(reward.proxy_gap_reward),
        present=present,
        # proxy_gap_reward is non-negative only when the gap did not widen.
        passed=(reward.proxy_gap_reward >= 0.0) if present else None,
        evidence={"proxy_gap_delta": outcome.proxy_gap_delta},
    )


def _verify_validation(
    reward: CampaignDecisionReward,
    outcome: CampaignDecisionOutcome,
) -> RewardVerification:
    present = outcome.validation_success is not None
    return _build(
        name="validation",
        verifier_type="validation_rule",
        score=_round(reward.validation_reward),
        present=present,
        passed=(outcome.validation_success is True) if present else None,
        evidence={"validation_success": outcome.validation_success},
    )


def _verify_objective(
    reward: CampaignDecisionReward,
    outcome: CampaignDecisionOutcome,
) -> RewardVerification:
    present = outcome.objective_delta is not None
    return _build(
        name="objective",
        verifier_type="outcome_metric",
        score=_round(reward.objective_reward),
        present=present,
        # objective_reward is non-negative only when the objective did not regress.
        passed=(reward.objective_reward >= 0.0) if present else None,
        evidence={"objective_delta": outcome.objective_delta},
    )


def _build(
    *,
    name: str,
    verifier_type: VerifierType,
    score: float,
    present: bool,
    passed: bool | None,
    evidence: dict[str, Any],
) -> RewardVerification:
    status: VerificationStatus = (
        "verified" if present else "neutral_due_to_missing_signal"
    )
    resolved_evidence = {"signal_present": present, **evidence}
    return RewardVerification(
        name=name,
        verifier_type=verifier_type,
        status=status,
        passed=passed if present else None,
        score=score,
        evidence=resolved_evidence,
    )


def _round(value: float) -> float:
    return round(value, 10)
