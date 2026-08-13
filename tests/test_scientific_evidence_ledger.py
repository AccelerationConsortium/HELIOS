from __future__ import annotations

import math
from pathlib import Path

from app.services.hypothesis_experiment_planner import (
    DiscriminationExperiment,
    ExperimentPrediction,
    HypothesisPriorScenario,
    rank_discrimination_experiments,
)
from app.services.scientific_evidence import (
    EvidenceAssessmentPolicy,
    EvidenceDesign,
    EvidenceItem,
    EvidenceSet,
    PromotionCriteria,
    ScientificClaim,
    assess_claim_evidence,
    evaluate_claim_promotion,
)
from app.services.scientific_ledger import ScientificLedger, safe_path_component


def _evidence_bundle():
    claim = ScientificClaim(
        claim_id="objective/validity",
        statement="Robust feasibility is a better proxy than hard scalar desirability.",
        scope="multi-drug solubilization",
        prior_rationale="Balanced prior registered before the comparison.",
        falsifying_observations=["No calibration improvement on an independent plate."],
    )
    evidence = EvidenceSet(
        claim_id=claim.claim_id,
        items=[
            EvidenceItem(
                evidence_id=f"plate-{suffix}",
                claim_id=claim.claim_id,
                independence_key=f"plate-{suffix}",
                design=EvidenceDesign.PROSPECTIVE_INTERVENTIONAL,
                source="held-out comparison",
                log_bayes_factor=log_bf,
                analysis_method="predictive likelihood ratio",
                dataset_hash=f"sha256:{suffix}",
                registered_before_observation=True,
                replicate_count=3,
                block_ids=[f"plate-{suffix}"],
            )
            for suffix, log_bf in (("a", math.log(9)), ("b", math.log(3)))
        ],
    )
    assessment = assess_claim_evidence(
        claim,
        evidence,
        policy=EvidenceAssessmentPolicy(
            min_scored_evidence=2,
            min_prospective_evidence=2,
            min_independent_blocks=2,
            require_interventional_evidence=True,
        ),
    )
    promotion = evaluate_claim_promotion(
        assessment,
        criteria=PromotionCriteria(
            min_scored_evidence=2,
            min_prospective_evidence=2,
            min_interventional_evidence=1,
            min_independent_blocks=2,
            min_preregistered_evidence=2,
        ),
    )
    return claim, evidence, assessment, promotion


def test_ledger_records_claim_posterior_and_promotion_gate(tmp_path):
    ledger = ScientificLedger(tmp_path / "ledger")
    claim, evidence, assessment, promotion = _evidence_bundle()

    result = ledger.record_claim_evidence(
        campaign_id="campaign-1",
        claim=claim,
        evidence=evidence,
        assessment=assessment,
        promotion_decision=promotion,
    )

    campaign_dir = Path(result.campaign_directory)
    claim_path = campaign_dir / result.changed_paths[0]
    content = claim_path.read_text()
    assert "artifact_type: scientific_claim_evidence" in content
    assert "Posterior probability" in content
    assert "explicit human approval is required" in content
    assert "Auto-applied: no" in content
    assert (campaign_dir / "evidence/index.md").exists()
    assert "objective/validity" in (campaign_dir / "evidence/index.md").read_text()


def test_claim_evidence_write_is_idempotent(tmp_path):
    ledger = ScientificLedger(tmp_path / "ledger")
    claim, evidence, assessment, promotion = _evidence_bundle()
    kwargs = {
        "campaign_id": "campaign-1",
        "claim": claim,
        "evidence": evidence,
        "assessment": assessment,
        "promotion_decision": promotion,
    }

    first = ledger.record_claim_evidence(**kwargs)
    second = ledger.record_claim_evidence(**kwargs)

    assert first.changed_paths
    assert second.changed_paths == ()
    assert set(second.unchanged_paths) == {
        f"evidence/claims/{safe_path_component(claim.claim_id)}.md",
        "evidence/index.md",
    }


def test_ledger_records_shadow_experiment_plan(tmp_path):
    plan = rank_discrimination_experiments(
        [
            DiscriminationExperiment(
                experiment_id="anchor-replicate",
                description="Repeat the feasible anchor across an independent plate.",
                safety_approved=True,
                predictions=[
                    ExperimentPrediction(
                        hypothesis_id="scalar-cliff",
                        outcome_probabilities={"reproduces": 0.9, "fails": 0.1},
                    ),
                    ExperimentPrediction(
                        hypothesis_id="assay-drift",
                        outcome_probabilities={"reproduces": 0.2, "fails": 0.8},
                    ),
                ],
            )
        ],
        [
            HypothesisPriorScenario(
                scenario_id="balanced",
                probabilities={"scalar-cliff": 0.5, "assay-drift": 0.5},
            )
        ],
        plan_id="discovery-round-1",
    )
    ledger = ScientificLedger(tmp_path / "ledger")

    result = ledger.record_experiment_plan(campaign_id="campaign-1", plan=plan)

    campaign_dir = Path(result.campaign_directory)
    content = (campaign_dir / "evidence/plans/discovery-round-1.md").read_text()
    assert "hypothesis_discrimination_plan" in content
    assert "anchor-replicate" in content
    assert "advisory and cannot execute" in content
    assert "discovery-round-1" in (campaign_dir / "evidence/index.md").read_text()
