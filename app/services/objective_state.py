"""Objective state and revision accounting for contextual campaign decisions.

This module upgrades the objective layer from a static metric hierarchy plus a
computed reward into an evolving, versioned ``ObjectiveState``. The updater is
pure and shadow-only: it consumes an observed ``CampaignDecisionOutcome`` and
returns a NEW revised state with full provenance. It performs no database
writes, external calls, or live routing changes, and it never mutates its
inputs.

The state separates the scientific goal (``primary_objective`` /
``scientific_question``) from optimization proxies (``proxy_objective_names`` /
``proxy_gap``) and tracks ``objective_confidence``, ``failure_constraints``,
``validation_requirements``, ``stopping_criteria`` and an append-only
``revision_history`` so every change is inspectable and replayable.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.services.decision_outcome import CampaignDecisionOutcome
from app.services.objective_models import ProxyGapAssessment
from app.services.scientific_evidence import ClaimAssessment, PromotionDecision

__all__ = [
    "ObjectiveRevision",
    "ObjectiveState",
    "ObjectiveStateUpdater",
    "ObjectiveConfidenceMethod",
    "StoppingCriteria",
    "apply_evidence_to_objective_state",
    "apply_outcome_to_objective_state",
]


class ObjectiveConfidenceMethod(StrEnum):
    """How ``objective_confidence`` was most recently updated."""

    HEURISTIC_OUTCOME_DELTA = "heuristic_outcome_delta"
    SCIENTIFIC_EVIDENCE_POSTERIOR = "scientific_evidence_posterior"


class StoppingCriteria(BaseModel):
    """Deterministic, evaluable stopping conditions for a campaign objective."""

    max_rounds: int | None = Field(default=None, ge=0)
    target_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    max_consecutive_failures: int | None = Field(default=None, ge=0)


class ObjectiveRevision(BaseModel):
    """Provenance record for one objective-state revision."""

    revision: int = Field(ge=1)
    reason: str
    source: str
    trace_id: str | None = None
    changes: dict[str, Any] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class ObjectiveState(BaseModel):
    """Evolving campaign objective state, separate from live strategy selection."""

    campaign_id: str
    primary_objective: str
    scientific_question: str | None = None
    proxy_objective_names: list[str] = Field(default_factory=list)
    objective_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    objective_confidence_method: ObjectiveConfidenceMethod = (
        ObjectiveConfidenceMethod.HEURISTIC_OUTCOME_DELTA
    )
    evidence_claim_id: str | None = None
    evidence_assessment: ClaimAssessment | None = None
    promotion_decision: PromotionDecision | None = None
    proxy_gap: ProxyGapAssessment | None = None
    failure_constraints: list[str] = Field(default_factory=list)
    validation_requirements: list[str] = Field(default_factory=list)
    stopping_criteria: StoppingCriteria | None = None
    revision: int = Field(default=0, ge=0)
    rounds_observed: int = Field(default=0, ge=0)
    consecutive_failure_count: int = Field(default=0, ge=0)
    stop_recommended: bool = False
    stop_reason: str | None = None
    revision_history: list[ObjectiveRevision] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class ObjectiveStateUpdater:
    """Turn an observed decision outcome into a revised objective state.

    Deterministic and shadow-only: returns a new ``ObjectiveState`` and never
    mutates the input state or outcome.
    """

    def apply_outcome(
        self,
        state: ObjectiveState,
        outcome: CampaignDecisionOutcome,
        *,
        proxy_gap: ProxyGapAssessment | None = None,
        now: datetime | None = None,
    ) -> ObjectiveState:
        timestamp = now or datetime.now(UTC)

        if state.objective_confidence_method == ObjectiveConfidenceMethod.SCIENTIFIC_EVIDENCE_POSTERIOR:
            delta = 0.0
            evidence = [
                "heuristic confidence update skipped because the objective is bound to a scientific evidence posterior"
            ]
        else:
            delta, evidence = _confidence_delta(outcome)
        old_confidence = state.objective_confidence
        new_confidence = _clamp_unit(_round(old_confidence + delta))

        had_failure = outcome.execution_success is False or outcome.failure_count > 0
        new_consecutive = (
            state.consecutive_failure_count + 1 if had_failure else 0
        )
        new_rounds = state.rounds_observed + 1
        new_proxy_gap = proxy_gap if proxy_gap is not None else state.proxy_gap

        stop_recommended, stop_reason = _evaluate_stopping(
            criteria=state.stopping_criteria,
            confidence=new_confidence,
            consecutive_failures=new_consecutive,
            rounds_observed=new_rounds,
        )

        changes: dict[str, Any] = {
            "objective_confidence": {"from": old_confidence, "to": new_confidence},
            "consecutive_failure_count": {
                "from": state.consecutive_failure_count,
                "to": new_consecutive,
            },
            "rounds_observed": {"from": state.rounds_observed, "to": new_rounds},
        }
        if proxy_gap is not None:
            changes["proxy_gap"] = {
                "from": state.proxy_gap.level.value if state.proxy_gap else None,
                "to": proxy_gap.level.value,
            }
        if (stop_recommended, stop_reason) != (state.stop_recommended, state.stop_reason):
            changes["stop_recommended"] = {
                "from": state.stop_recommended,
                "to": stop_recommended,
            }
            changes["stop_reason"] = {"from": state.stop_reason, "to": stop_reason}

        revision = ObjectiveRevision(
            revision=state.revision + 1,
            reason=_revision_reason(delta, stop_reason),
            source="campaign_decision_outcome",
            trace_id=outcome.trace_id,
            changes=changes,
            evidence=evidence,
            created_at=timestamp,
        )

        return state.model_copy(
            update={
                "objective_confidence": new_confidence,
                "proxy_gap": new_proxy_gap,
                "consecutive_failure_count": new_consecutive,
                "rounds_observed": new_rounds,
                "stop_recommended": stop_recommended,
                "stop_reason": stop_reason,
                "revision": state.revision + 1,
                "revision_history": [*state.revision_history, revision],
                "updated_at": timestamp,
            }
        )

    def apply_evidence_assessment(
        self,
        state: ObjectiveState,
        assessment: ClaimAssessment,
        *,
        promotion_decision: PromotionDecision | None = None,
        trace_id: str | None = None,
        now: datetime | None = None,
    ) -> ObjectiveState:
        """Bind an auditable claim posterior to objective state.

        This replaces the objective confidence with the assessed posterior but
        does not apply a promoted objective, constraint, or search-space
        change.  A promotion decision is stored solely as shadow evidence.
        """

        if state.evidence_claim_id is not None and state.evidence_claim_id != assessment.claim_id:
            raise ValueError(
                "objective state is already bound to another scientific claim: "
                f"{state.evidence_claim_id!r}"
            )
        if state.evidence_assessment is not None and (
            state.evidence_assessment.prior_probability != assessment.prior_probability
            or state.evidence_assessment.prior_version != assessment.prior_version
        ):
            raise ValueError(
                "scientific claim prior changed after binding; create a new versioned claim "
                "instead of silently rewriting prior odds"
            )
        if promotion_decision is not None and promotion_decision.claim_id != assessment.claim_id:
            raise ValueError("promotion decision and assessment must target the same claim")
        timestamp = now or datetime.now(UTC)
        old_confidence = state.objective_confidence
        old_method = state.objective_confidence_method
        new_confidence = assessment.posterior_probability
        stop_recommended, stop_reason = _evaluate_stopping(
            criteria=state.stopping_criteria,
            confidence=new_confidence,
            consecutive_failures=state.consecutive_failure_count,
            rounds_observed=state.rounds_observed,
        )
        changes: dict[str, Any] = {
            "objective_confidence": {"from": old_confidence, "to": new_confidence},
            "objective_confidence_method": {
                "from": old_method.value,
                "to": ObjectiveConfidenceMethod.SCIENTIFIC_EVIDENCE_POSTERIOR.value,
            },
            "evidence_claim_id": {
                "from": state.evidence_claim_id,
                "to": assessment.claim_id,
            },
            "evidence_status": {
                "from": state.evidence_assessment.status.value if state.evidence_assessment else None,
                "to": assessment.status.value,
            },
        }
        if promotion_decision is not None:
            changes["promotion_allowed"] = {
                "from": (
                    state.promotion_decision.promotion_allowed
                    if state.promotion_decision is not None
                    else None
                ),
                "to": promotion_decision.promotion_allowed,
            }
        if (stop_recommended, stop_reason) != (state.stop_recommended, state.stop_reason):
            changes["stop_recommended"] = {
                "from": state.stop_recommended,
                "to": stop_recommended,
            }
            changes["stop_reason"] = {"from": state.stop_reason, "to": stop_reason}

        revision = ObjectiveRevision(
            revision=state.revision + 1,
            reason=(
                "Objective confidence replaced by an auditable scientific evidence posterior; "
                "no live objective change was auto-applied."
            ),
            source="scientific_evidence_posterior",
            trace_id=trace_id,
            changes=changes,
            evidence=[
                f"claim_id={assessment.claim_id}",
                f"status={assessment.status.value}",
                f"posterior_probability={assessment.posterior_probability:.6g}",
                f"scored_evidence_count={assessment.scored_evidence_count}",
                f"prospective_evidence_count={assessment.prospective_evidence_count}",
                f"independent_block_count={assessment.independent_block_count}",
                (
                    f"promotion_allowed={promotion_decision.promotion_allowed}"
                    if promotion_decision is not None
                    else "promotion_not_evaluated"
                ),
            ],
            created_at=timestamp,
            metadata={"shadow_only": True, "auto_applied": False},
        )
        return state.model_copy(
            update={
                "objective_confidence": new_confidence,
                "objective_confidence_method": (
                    ObjectiveConfidenceMethod.SCIENTIFIC_EVIDENCE_POSTERIOR
                ),
                "evidence_claim_id": assessment.claim_id,
                "evidence_assessment": assessment.model_copy(deep=True),
                "promotion_decision": (
                    promotion_decision.model_copy(deep=True)
                    if promotion_decision is not None
                    else None
                ),
                "stop_recommended": stop_recommended,
                "stop_reason": stop_reason,
                "revision": state.revision + 1,
                "revision_history": [*state.revision_history, revision],
                "updated_at": timestamp,
            }
        )


def apply_outcome_to_objective_state(
    state: ObjectiveState,
    outcome: CampaignDecisionOutcome,
    *,
    proxy_gap: ProxyGapAssessment | None = None,
    now: datetime | None = None,
) -> ObjectiveState:
    """Apply an outcome with the default ObjectiveStateUpdater."""
    return ObjectiveStateUpdater().apply_outcome(
        state, outcome, proxy_gap=proxy_gap, now=now
    )


def apply_evidence_to_objective_state(
    state: ObjectiveState,
    assessment: ClaimAssessment,
    *,
    promotion_decision: PromotionDecision | None = None,
    trace_id: str | None = None,
    now: datetime | None = None,
) -> ObjectiveState:
    """Bind a scientific evidence posterior with the default updater."""

    return ObjectiveStateUpdater().apply_evidence_assessment(
        state,
        assessment,
        promotion_decision=promotion_decision,
        trace_id=trace_id,
        now=now,
    )


def _confidence_delta(outcome: CampaignDecisionOutcome) -> tuple[float, list[str]]:
    delta = 0.0
    evidence: list[str] = []

    if outcome.execution_success is True:
        delta += 0.1
        evidence.append("execution_success=True (+0.1)")
    elif outcome.execution_success is False:
        delta -= 0.1
        evidence.append("execution_success=False (-0.1)")

    if outcome.objective_delta is not None:
        contribution = _round(_clamp_signed(outcome.objective_delta) * 0.2)
        delta += contribution
        evidence.append(f"objective_delta={outcome.objective_delta:.3g} ({contribution:+.3g})")

    if outcome.proxy_gap_delta is not None:
        # Positive proxy_gap_delta means the gap widened (worse for confidence).
        contribution = _round(-_clamp_signed(outcome.proxy_gap_delta) * 0.2)
        delta += contribution
        evidence.append(f"proxy_gap_delta={outcome.proxy_gap_delta:.3g} ({contribution:+.3g})")

    if outcome.validation_success is True:
        delta += 0.1
        evidence.append("validation_success=True (+0.1)")
    elif outcome.validation_success is False:
        delta -= 0.1
        evidence.append("validation_success=False (-0.1)")

    if outcome.failure_count > 0:
        contribution = _round(-0.05 * outcome.failure_count)
        delta += contribution
        evidence.append(f"failure_count={outcome.failure_count} ({contribution:+.3g})")

    if not evidence:
        evidence.append("no confidence-relevant outcome components")

    return _round(delta), evidence


def _evaluate_stopping(
    *,
    criteria: StoppingCriteria | None,
    confidence: float,
    consecutive_failures: int,
    rounds_observed: int,
) -> tuple[bool, str | None]:
    if criteria is None:
        return False, None
    if (
        criteria.max_consecutive_failures is not None
        and consecutive_failures >= criteria.max_consecutive_failures
    ):
        return True, "max_consecutive_failures_reached"
    if (
        criteria.target_confidence is not None
        and confidence >= criteria.target_confidence
    ):
        return True, "target_confidence_reached"
    if criteria.max_rounds is not None and rounds_observed >= criteria.max_rounds:
        return True, "max_rounds_reached"
    return False, None


def _revision_reason(delta: float, stop_reason: str | None) -> str:
    direction = "raised" if delta > 0 else "lowered" if delta < 0 else "held"
    base = f"Objective confidence {direction} by {delta:+.3g} from observed outcome."
    if stop_reason is not None:
        return f"{base} Stopping criterion: {stop_reason}."
    return base


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _round(value: float) -> float:
    return round(value, 10)
