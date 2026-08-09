"""Typed scientific claims, evidence accounting, and promotion gates.

This module is the evidence boundary between descriptive campaign signals and
scientific claims.  It deliberately does not manufacture confidence from
generic success/failure events.  Posterior claim probabilities move only when
an evidence item supplies an auditable likelihood ratio (stored as a log Bayes
factor), and dependent evidence cannot be double counted because every item
must have a unique ``independence_key``.

All APIs are pure and shadow-only.  A positive promotion decision means that
predeclared evidence requirements were met; it never mutates a live objective,
search space, constraint, or hardware workflow.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "ClaimAssessment",
    "ClaimStatus",
    "EvidenceAssessmentPolicy",
    "EvidenceDesign",
    "EvidenceItem",
    "EvidenceSet",
    "PromotionCriteria",
    "PromotionDecision",
    "ScientificClaim",
    "ValidationCheck",
    "assess_claim_evidence",
    "evaluate_claim_promotion",
]


class ClaimStatus(StrEnum):
    """Evidence state for one scientific claim."""

    PROPOSED = "proposed"
    INCONCLUSIVE = "inconclusive"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    BLOCKED = "blocked"


class EvidenceDesign(StrEnum):
    """Study design that produced an evidence item."""

    RETROSPECTIVE = "retrospective"
    PROSPECTIVE_OBSERVATIONAL = "prospective_observational"
    PROSPECTIVE_INTERVENTIONAL = "prospective_interventional"
    INDEPENDENT_REPLICATION = "independent_replication"
    EXTERNAL_VALIDATION = "external_validation"


_PROSPECTIVE_DESIGNS = {
    EvidenceDesign.PROSPECTIVE_OBSERVATIONAL,
    EvidenceDesign.PROSPECTIVE_INTERVENTIONAL,
    EvidenceDesign.INDEPENDENT_REPLICATION,
    EvidenceDesign.EXTERNAL_VALIDATION,
}

_INTERVENTIONAL_DESIGNS = {
    EvidenceDesign.PROSPECTIVE_INTERVENTIONAL,
    EvidenceDesign.INDEPENDENT_REPLICATION,
}


class ScientificClaim(BaseModel):
    """A falsifiable claim whose evidence is tracked independently of prose."""

    claim_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    prior_probability: float = Field(default=0.5, gt=0.0, lt=1.0)
    prior_version: str = Field(default="v1", min_length=1)
    prior_rationale: str | None = None
    competing_claim_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    falsifying_observations: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    blocked_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("competing_claim_ids")
    @classmethod
    def _claim_must_not_compete_with_itself(cls, values: list[str], info: Any) -> list[str]:
        claim_id = info.data.get("claim_id")
        if claim_id in values:
            raise ValueError("a scientific claim cannot compete with itself")
        if len(values) != len(set(values)):
            raise ValueError("competing_claim_ids must be unique")
        return values


class EvidenceItem(BaseModel):
    """One independently interpretable unit of evidence for a claim.

    ``log_bayes_factor`` is positive when the observed data are more likely
    under the claim than its declared alternative, negative when they favor
    the alternative, and zero when they do not discriminate.  ``None`` keeps
    descriptive evidence in the ledger without pretending that it updates a
    posterior.
    """

    evidence_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    independence_key: str = Field(min_length=1)
    design: EvidenceDesign
    source: str = Field(min_length=1)
    log_bayes_factor: float | None = None
    analysis_method: str | None = None
    dataset_hash: str | None = None
    protocol_version: str | None = None
    registered_before_observation: bool = False
    replicate_count: int = Field(default=1, ge=1)
    block_ids: list[str] = Field(default_factory=list)
    effect_estimate: float | None = None
    standard_error: float | None = Field(default=None, gt=0.0)
    falsifier_triggered: bool = False
    safety_incident_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @field_validator("block_ids")
    @classmethod
    def _block_ids_must_be_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("block_ids must be unique")
        return values

    @model_validator(mode="after")
    def _scored_evidence_requires_an_auditable_method(self) -> EvidenceItem:
        if self.log_bayes_factor is not None:
            if not math.isfinite(self.log_bayes_factor):
                raise ValueError("log_bayes_factor must be finite")
            if not self.analysis_method:
                raise ValueError("scored evidence requires analysis_method")
        return self


class EvidenceSet(BaseModel):
    """Evidence for one claim with independence and identity invariants."""

    claim_id: str = Field(min_length=1)
    items: list[EvidenceItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _items_must_be_aligned_and_independent(self) -> EvidenceSet:
        evidence_ids = [item.evidence_id for item in self.items]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence_id values must be unique")
        independence_keys = [item.independence_key for item in self.items]
        if len(independence_keys) != len(set(independence_keys)):
            raise ValueError(
                "independence_key values must be unique; aggregate dependent observations "
                "into one evidence item before posterior updating"
            )
        mismatched = [item.evidence_id for item in self.items if item.claim_id != self.claim_id]
        if mismatched:
            raise ValueError(f"evidence items target another claim: {mismatched}")
        return self


class EvidenceAssessmentPolicy(BaseModel):
    """Predeclared thresholds for interpreting a claim posterior."""

    support_probability: float = Field(default=0.95, gt=0.5, lt=1.0)
    refute_probability: float = Field(default=0.05, gt=0.0, lt=0.5)
    min_scored_evidence: int = Field(default=1, ge=1)
    min_prospective_evidence: int = Field(default=1, ge=0)
    min_independent_blocks: int = Field(default=1, ge=0)
    require_interventional_evidence: bool = False


class ClaimAssessment(BaseModel):
    """Posterior and design-quality summary for a scientific claim."""

    claim_id: str
    status: ClaimStatus
    prior_probability: float = Field(gt=0.0, lt=1.0)
    prior_version: str
    prior_rationale_recorded: bool
    posterior_probability: float = Field(ge=0.0, le=1.0)
    cumulative_log_bayes_factor: float
    scored_evidence_count: int = Field(ge=0)
    unscored_evidence_count: int = Field(ge=0)
    prospective_evidence_count: int = Field(ge=0)
    interventional_evidence_count: int = Field(ge=0)
    preregistered_evidence_count: int = Field(ge=0)
    independent_block_count: int = Field(ge=0)
    safety_incident_count: int = Field(ge=0)
    falsifier_triggered: bool = False
    evidence_ids: list[str] = Field(default_factory=list)
    unmet_requirements: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    method: str = "posterior_odds_from_independent_log_bayes_factors"
    shadow_only: bool = True


class ValidationCheck(BaseModel):
    """One predeclared non-posterior condition for promotion."""

    name: str = Field(min_length=1)
    passed: bool
    required: bool = True
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str | None = None


class PromotionCriteria(BaseModel):
    """Evidence and governance requirements for considering a claim promotable."""

    min_posterior_probability: float = Field(default=0.95, gt=0.5, lt=1.0)
    min_scored_evidence: int = Field(default=2, ge=1)
    min_prospective_evidence: int = Field(default=2, ge=0)
    min_interventional_evidence: int = Field(default=0, ge=0)
    min_independent_blocks: int = Field(default=2, ge=0)
    min_preregistered_evidence: int = Field(default=1, ge=0)
    max_safety_incidents: int = Field(default=0, ge=0)
    require_supported_status: bool = True
    require_prior_rationale: bool = True
    require_human_approval: bool = True


class PromotionDecision(BaseModel):
    """Shadow-only decision about whether evidence justifies human promotion review."""

    claim_id: str
    evidence_criteria_satisfied: bool
    human_approval_required: bool
    human_approved: bool
    promotion_allowed: bool
    reasons: list[str] = Field(default_factory=list)
    checks: list[ValidationCheck] = Field(default_factory=list)
    auto_applied: bool = False
    shadow_only: bool = True

    @model_validator(mode="after")
    def _live_mutation_is_forbidden(self) -> PromotionDecision:
        if self.auto_applied:
            raise ValueError("scientific evidence promotion cannot be auto-applied")
        return self


def assess_claim_evidence(
    claim: ScientificClaim,
    evidence: EvidenceSet,
    *,
    policy: EvidenceAssessmentPolicy | None = None,
) -> ClaimAssessment:
    """Update claim odds from independent, auditable likelihood ratios."""

    if evidence.claim_id != claim.claim_id:
        raise ValueError("claim and evidence set must have the same claim_id")
    effective_policy = policy or EvidenceAssessmentPolicy()
    scored_values = [
        item.log_bayes_factor
        for item in evidence.items
        if item.log_bayes_factor is not None
    ]
    scored = [item for item in evidence.items if item.log_bayes_factor is not None]
    unscored = [item for item in evidence.items if item.log_bayes_factor is None]
    cumulative_log_bf = sum(scored_values, 0.0)
    log_prior_odds = math.log(claim.prior_probability) - math.log1p(-claim.prior_probability)
    posterior = _logistic(log_prior_odds + cumulative_log_bf)
    prospective = [item for item in scored if item.design in _PROSPECTIVE_DESIGNS]
    interventional = [item for item in scored if item.design in _INTERVENTIONAL_DESIGNS]
    preregistered = [item for item in prospective if item.registered_before_observation]
    block_ids = {block_id for item in scored for block_id in item.block_ids}
    falsifier_triggered = any(item.falsifier_triggered for item in evidence.items)
    safety_incidents = sum(item.safety_incident_count for item in evidence.items)

    unmet = _assessment_requirements(
        effective_policy,
        scored_count=len(scored),
        prospective_count=len(prospective),
        interventional_count=len(interventional),
        block_count=len(block_ids),
    )
    warnings: list[str] = []
    if unscored:
        warnings.append(
            f"{len(unscored)} evidence item(s) were recorded descriptively and did not update the posterior"
        )
    if any(item.dataset_hash is None for item in scored):
        warnings.append("one or more scored evidence items lack a dataset_hash")
    if any(item.design in _PROSPECTIVE_DESIGNS and not item.registered_before_observation for item in scored):
        warnings.append("one or more prospective evidence items were not preregistered")
    if not claim.prior_rationale:
        warnings.append("claim prior has no recorded rationale")

    if claim.blocked_reason:
        status = ClaimStatus.BLOCKED
        unmet = [*unmet, f"claim blocked: {claim.blocked_reason}"]
    elif falsifier_triggered or posterior <= effective_policy.refute_probability:
        status = ClaimStatus.REFUTED
    elif posterior >= effective_policy.support_probability and not unmet:
        status = ClaimStatus.SUPPORTED
    elif not evidence.items:
        status = ClaimStatus.PROPOSED
    else:
        status = ClaimStatus.INCONCLUSIVE

    return ClaimAssessment(
        claim_id=claim.claim_id,
        status=status,
        prior_probability=claim.prior_probability,
        prior_version=claim.prior_version,
        prior_rationale_recorded=bool(claim.prior_rationale),
        posterior_probability=round(posterior, 12),
        cumulative_log_bayes_factor=round(cumulative_log_bf, 12),
        scored_evidence_count=len(scored),
        unscored_evidence_count=len(unscored),
        prospective_evidence_count=len(prospective),
        interventional_evidence_count=len(interventional),
        preregistered_evidence_count=len(preregistered),
        independent_block_count=len(block_ids),
        safety_incident_count=safety_incidents,
        falsifier_triggered=falsifier_triggered,
        evidence_ids=[item.evidence_id for item in evidence.items],
        unmet_requirements=unmet,
        warnings=warnings,
    )


def evaluate_claim_promotion(
    assessment: ClaimAssessment,
    *,
    criteria: PromotionCriteria | None = None,
    validation_checks: list[ValidationCheck] | None = None,
    human_approved: bool = False,
) -> PromotionDecision:
    """Evaluate a promotion gate without applying any live change."""

    effective = criteria or PromotionCriteria()
    checks = list(validation_checks or [])
    reasons: list[str] = []

    if effective.require_supported_status and assessment.status != ClaimStatus.SUPPORTED:
        reasons.append(f"claim status is {assessment.status.value}, not supported")
    if effective.require_prior_rationale and not assessment.prior_rationale_recorded:
        reasons.append("claim prior has no recorded rationale")
    if assessment.posterior_probability < effective.min_posterior_probability:
        reasons.append(
            "posterior probability "
            f"{assessment.posterior_probability:.4f} < {effective.min_posterior_probability:.4f}"
        )
    _append_count_failure(
        reasons,
        "scored evidence",
        assessment.scored_evidence_count,
        effective.min_scored_evidence,
    )
    _append_count_failure(
        reasons,
        "prospective evidence",
        assessment.prospective_evidence_count,
        effective.min_prospective_evidence,
    )
    _append_count_failure(
        reasons,
        "interventional evidence",
        assessment.interventional_evidence_count,
        effective.min_interventional_evidence,
    )
    _append_count_failure(
        reasons,
        "independent blocks",
        assessment.independent_block_count,
        effective.min_independent_blocks,
    )
    _append_count_failure(
        reasons,
        "preregistered evidence",
        assessment.preregistered_evidence_count,
        effective.min_preregistered_evidence,
    )
    if assessment.safety_incident_count > effective.max_safety_incidents:
        reasons.append(
            f"safety incidents {assessment.safety_incident_count} > {effective.max_safety_incidents}"
        )
    if assessment.falsifier_triggered:
        reasons.append("a predeclared falsifier was triggered")
    for check in checks:
        if check.required and not check.passed:
            reasons.append(f"required validation check failed: {check.name}")

    evidence_criteria_satisfied = not reasons
    promotion_allowed = evidence_criteria_satisfied and (
        human_approved or not effective.require_human_approval
    )
    if evidence_criteria_satisfied and effective.require_human_approval and not human_approved:
        reasons.append("explicit human approval is required")

    return PromotionDecision(
        claim_id=assessment.claim_id,
        evidence_criteria_satisfied=evidence_criteria_satisfied,
        human_approval_required=effective.require_human_approval,
        human_approved=human_approved,
        promotion_allowed=promotion_allowed,
        reasons=reasons,
        checks=checks,
    )


def _assessment_requirements(
    policy: EvidenceAssessmentPolicy,
    *,
    scored_count: int,
    prospective_count: int,
    interventional_count: int,
    block_count: int,
) -> list[str]:
    unmet: list[str] = []
    _append_count_failure(unmet, "scored evidence", scored_count, policy.min_scored_evidence)
    _append_count_failure(
        unmet,
        "prospective evidence",
        prospective_count,
        policy.min_prospective_evidence,
    )
    _append_count_failure(unmet, "independent blocks", block_count, policy.min_independent_blocks)
    if policy.require_interventional_evidence and interventional_count < 1:
        unmet.append("interventional evidence 0 < 1")
    return unmet


def _append_count_failure(reasons: list[str], name: str, observed: int, required: int) -> None:
    if observed < required:
        reasons.append(f"{name} {observed} < {required}")


def _logistic(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)
