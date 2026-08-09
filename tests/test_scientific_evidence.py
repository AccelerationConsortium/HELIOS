from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from app.services.scientific_evidence import (
    ClaimStatus,
    EvidenceAssessmentPolicy,
    EvidenceDesign,
    EvidenceItem,
    EvidenceSet,
    PromotionCriteria,
    PromotionDecision,
    ScientificClaim,
    ValidationCheck,
    assess_claim_evidence,
    evaluate_claim_promotion,
)


def _claim(**updates):
    values = {
        "claim_id": "scalarization-cliff",
        "statement": "Hard scalarization destroys useful feasibility signal.",
        "scope": "multi-drug solubilization campaign",
        "prior_probability": 0.5,
        "prior_rationale": "Balanced prior before prospective validation.",
        "falsifying_observations": [
            "An independently validated constrained model does not improve predictive calibration."
        ],
    }
    values.update(updates)
    return ScientificClaim(**values)


def _evidence(
    evidence_id: str,
    *,
    log_bayes_factor: float | None,
    block_id: str,
    design: EvidenceDesign = EvidenceDesign.PROSPECTIVE_INTERVENTIONAL,
    falsifier_triggered: bool = False,
):
    return EvidenceItem(
        evidence_id=evidence_id,
        claim_id="scalarization-cliff",
        independence_key=f"independent-{evidence_id}",
        design=design,
        source="predeclared analysis",
        log_bayes_factor=log_bayes_factor,
        analysis_method="held-out likelihood ratio" if log_bayes_factor is not None else None,
        dataset_hash=f"sha256:{evidence_id}",
        registered_before_observation=True,
        replicate_count=3,
        block_ids=[block_id],
        falsifier_triggered=falsifier_triggered,
    )


def test_independent_log_bayes_factors_update_posterior_odds():
    evidence = EvidenceSet(
        claim_id="scalarization-cliff",
        items=[
            _evidence("plate-a", log_bayes_factor=math.log(9.0), block_id="plate-a"),
            _evidence("plate-b", log_bayes_factor=math.log(3.0), block_id="plate-b"),
        ],
    )

    assessment = assess_claim_evidence(
        _claim(),
        evidence,
        policy=EvidenceAssessmentPolicy(
            support_probability=0.95,
            min_scored_evidence=2,
            min_prospective_evidence=2,
            min_independent_blocks=2,
            require_interventional_evidence=True,
        ),
    )

    assert assessment.posterior_probability == pytest.approx(27 / 28)
    assert assessment.status == ClaimStatus.SUPPORTED
    assert assessment.interventional_evidence_count == 2
    assert assessment.preregistered_evidence_count == 2
    assert assessment.independent_block_count == 2
    assert assessment.unmet_requirements == []


def test_descriptive_evidence_is_recorded_but_does_not_move_posterior():
    assessment = assess_claim_evidence(
        _claim(prior_probability=0.4),
        EvidenceSet(
            claim_id="scalarization-cliff",
            items=[_evidence("audit", log_bayes_factor=None, block_id="audit")],
        ),
    )

    assert assessment.posterior_probability == pytest.approx(0.4)
    assert assessment.scored_evidence_count == 0
    assert assessment.unscored_evidence_count == 1
    assert assessment.status == ClaimStatus.INCONCLUSIVE
    assert assessment.warnings


def test_dependent_evidence_cannot_be_double_counted():
    first = _evidence("first", log_bayes_factor=1.0, block_id="plate-a")
    second = _evidence("second", log_bayes_factor=1.0, block_id="plate-a").model_copy(
        update={"independence_key": first.independence_key}
    )

    with pytest.raises(ValidationError, match="independence_key values must be unique"):
        EvidenceSet(claim_id="scalarization-cliff", items=[first, second])


def test_scored_evidence_requires_analysis_method():
    with pytest.raises(ValidationError, match="requires analysis_method"):
        EvidenceItem(
            evidence_id="untraceable",
            claim_id="scalarization-cliff",
            independence_key="plate-a",
            design=EvidenceDesign.RETROSPECTIVE,
            source="unknown",
            log_bayes_factor=2.0,
        )


def test_predeclared_falsifier_refutes_even_when_other_evidence_supports():
    assessment = assess_claim_evidence(
        _claim(),
        EvidenceSet(
            claim_id="scalarization-cliff",
            items=[
                _evidence("support", log_bayes_factor=8.0, block_id="plate-a"),
                _evidence(
                    "falsifier",
                    log_bayes_factor=0.0,
                    block_id="plate-b",
                    falsifier_triggered=True,
                ),
            ],
        ),
    )

    assert assessment.posterior_probability > 0.99
    assert assessment.falsifier_triggered is True
    assert assessment.status == ClaimStatus.REFUTED


def test_promotion_requires_evidence_checks_and_explicit_human_approval():
    assessment = assess_claim_evidence(
        _claim(),
        EvidenceSet(
            claim_id="scalarization-cliff",
            items=[
                _evidence("plate-a", log_bayes_factor=math.log(9.0), block_id="plate-a"),
                _evidence("plate-b", log_bayes_factor=math.log(3.0), block_id="plate-b"),
            ],
        ),
        policy=EvidenceAssessmentPolicy(
            min_scored_evidence=2,
            min_prospective_evidence=2,
            min_independent_blocks=2,
            require_interventional_evidence=True,
        ),
    )
    criteria = PromotionCriteria(
        min_posterior_probability=0.95,
        min_scored_evidence=2,
        min_prospective_evidence=2,
        min_interventional_evidence=1,
        min_independent_blocks=2,
        min_preregistered_evidence=2,
    )
    calibration = ValidationCheck(
        name="held-out feasibility calibration",
        passed=True,
        evidence_ids=["plate-a", "plate-b"],
    )

    waiting = evaluate_claim_promotion(
        assessment,
        criteria=criteria,
        validation_checks=[calibration],
    )
    approved = evaluate_claim_promotion(
        assessment,
        criteria=criteria,
        validation_checks=[calibration],
        human_approved=True,
    )

    assert waiting.evidence_criteria_satisfied is True
    assert waiting.promotion_allowed is False
    assert "explicit human approval is required" in waiting.reasons
    assert approved.promotion_allowed is True
    assert approved.auto_applied is False
    assert approved.shadow_only is True


def test_promotion_model_forbids_auto_application():
    with pytest.raises(ValidationError, match="cannot be auto-applied"):
        PromotionDecision(
            claim_id="scalarization-cliff",
            evidence_criteria_satisfied=True,
            human_approval_required=False,
            human_approved=False,
            promotion_allowed=True,
            auto_applied=True,
        )


def test_blocked_claim_cannot_be_supported():
    assessment = assess_claim_evidence(
        _claim(blocked_reason="toxicity metadata is a random placeholder"),
        EvidenceSet(
            claim_id="scalarization-cliff",
            items=[_evidence("plate-a", log_bayes_factor=10.0, block_id="plate-a")],
        ),
    )

    assert assessment.status == ClaimStatus.BLOCKED
    assert any("claim blocked" in reason for reason in assessment.unmet_requirements)


def test_missing_prior_rationale_blocks_default_promotion():
    assessment = assess_claim_evidence(
        _claim(prior_rationale=None),
        EvidenceSet(
            claim_id="scalarization-cliff",
            items=[
                _evidence("plate-a", log_bayes_factor=8.0, block_id="plate-a"),
                _evidence("plate-b", log_bayes_factor=2.0, block_id="plate-b"),
            ],
        ),
        policy=EvidenceAssessmentPolicy(
            min_scored_evidence=2,
            min_prospective_evidence=2,
            min_independent_blocks=2,
        ),
    )

    decision = evaluate_claim_promotion(assessment, human_approved=True)

    assert decision.promotion_allowed is False
    assert "claim prior has no recorded rationale" in decision.reasons
