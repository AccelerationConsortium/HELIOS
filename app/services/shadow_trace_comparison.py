"""Shadow trace comparison / calibration prep (Phase 5.7).

A pure, read-only analyzer that compares the two parallel shadow tracks on a
round:

* the legacy contextual decision plan (``CampaignDecisionAction``), and
* the new adaptive campaign substrate snapshot (``CampaignMode`` + action
  labels + VoI ranking).

It answers three questions for calibration:

1. Do the tracks agree? (via an explicit equivalence-class mapping)
2. Where do they diverge?
3. Does the substrate produce anything absurd, and do the input mappers
   (kind / capability / failure attribution) need correction?

This module changes nothing at runtime: it only reads the two artifacts and
emits typed findings. All outputs are deterministic and JSON-safe.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.services.adaptive_campaign_substrate import (
    AdaptiveCampaignSubstrateSnapshot,
)
from app.services.campaign_mode import CampaignMode
from app.services.decision_models import CampaignDecisionAction, CampaignDecisionPlan
from app.services.dynamic_action_space import ActionShadowLabel
from app.services.failure_attribution import FailureAttributionCategory

__all__ = [
    "ShadowComparisonSummary",
    "ShadowEquivalenceClass",
    "ShadowFinding",
    "ShadowTrackComparison",
    "compare_shadow_tracks",
    "parse_substrate_log_line",
    "summarize_comparisons",
]

_LOW_CONFIDENCE_THRESHOLD = 0.5


class ShadowEquivalenceClass(StrEnum):
    """Shared vocabulary that both shadow tracks map into for comparison."""

    STOP = "stop"
    OPTIMIZATION = "optimization"
    OBJECTIVE_INTEGRITY = "objective_integrity"
    FAILURE_HANDLING = "failure_handling"
    HUMAN_OBSERVATION = "human_observation"
    CONTEXT_SEEKING = "context_seeking"
    CONSTRAINT = "constraint"
    UNKNOWN = "unknown"


_ACTION_TO_CLASS: dict[CampaignDecisionAction, ShadowEquivalenceClass] = {
    CampaignDecisionAction.STOP_CAMPAIGN: ShadowEquivalenceClass.STOP,
    CampaignDecisionAction.PROPOSE_CANDIDATES: ShadowEquivalenceClass.OPTIMIZATION,
    CampaignDecisionAction.REVISE_OBJECTIVE: ShadowEquivalenceClass.OBJECTIVE_INTEGRITY,
    CampaignDecisionAction.RUN_VALIDATION: ShadowEquivalenceClass.OBJECTIVE_INTEGRITY,
    CampaignDecisionAction.RECOVER_FAILURE: ShadowEquivalenceClass.FAILURE_HANDLING,
    CampaignDecisionAction.REQUEST_HUMAN_OBSERVATION: ShadowEquivalenceClass.HUMAN_OBSERVATION,
    CampaignDecisionAction.QUERY_LITERATURE: ShadowEquivalenceClass.CONTEXT_SEEKING,
    CampaignDecisionAction.TIGHTEN_CONSTRAINTS: ShadowEquivalenceClass.CONSTRAINT,
}

_MODE_TO_CLASS: dict[CampaignMode, ShadowEquivalenceClass] = {
    CampaignMode.STOP_RECOMMENDED: ShadowEquivalenceClass.STOP,
    CampaignMode.BO_OPTIMIZATION: ShadowEquivalenceClass.OPTIMIZATION,
    CampaignMode.VALIDATION: ShadowEquivalenceClass.OBJECTIVE_INTEGRITY,
    CampaignMode.CALIBRATION: ShadowEquivalenceClass.FAILURE_HANDLING,
    CampaignMode.FAILURE_DIAGNOSIS: ShadowEquivalenceClass.FAILURE_HANDLING,
    CampaignMode.HUMAN_OBSERVATION_REQUEST: ShadowEquivalenceClass.HUMAN_OBSERVATION,
    CampaignMode.LITERATURE_CONTEXT_SEEKING: ShadowEquivalenceClass.CONTEXT_SEEKING,
    CampaignMode.SAFETY_CONSTRAINT_TIGHTENING: ShadowEquivalenceClass.CONSTRAINT,
}


class ShadowFinding(BaseModel):
    """One typed finding from the comparison (divergence / sanity / calibration)."""

    category: str  # "divergence" | "sanity" | "calibration"
    code: str
    severity: str  # "info" | "warning" | "concern"
    summary: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class ShadowTrackComparison(BaseModel):
    """Comparison of the two shadow tracks for one round."""

    campaign_id: str
    round_index: int = Field(ge=0)
    decision_action: str
    decision_class: ShadowEquivalenceClass
    substrate_mode: CampaignMode
    substrate_class: ShadowEquivalenceClass
    agree: bool
    divergences: list[ShadowFinding] = Field(default_factory=list)
    sanity_findings: list[ShadowFinding] = Field(default_factory=list)
    calibration_flags: list[ShadowFinding] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class ShadowComparisonSummary(BaseModel):
    """Aggregate over many round comparisons."""

    total: int = Field(ge=0)
    agreement_count: int = Field(ge=0)
    agreement_rate: float = Field(ge=0.0, le=1.0)
    divergence_histogram: dict[str, int] = Field(default_factory=dict)
    sanity_histogram: dict[str, int] = Field(default_factory=dict)
    calibration_histogram: dict[str, int] = Field(default_factory=dict)


def compare_shadow_tracks(
    decision_plan: CampaignDecisionPlan,
    substrate_snapshot: AdaptiveCampaignSubstrateSnapshot,
    *,
    now: datetime | None = None,
) -> ShadowTrackComparison:
    """Compare the legacy decision plan with the substrate snapshot for a round."""
    timestamp = now or datetime.now(UTC)
    action = decision_plan.action_type
    mode = substrate_snapshot.campaign_mode_decision.mode
    decision_class = _ACTION_TO_CLASS.get(action, ShadowEquivalenceClass.UNKNOWN)
    substrate_class = _MODE_TO_CLASS.get(mode, ShadowEquivalenceClass.UNKNOWN)
    agree = decision_class == substrate_class

    divergences = _divergences(decision_class, substrate_class, action, mode)
    sanity_findings = _sanity_checks(substrate_snapshot)
    calibration_flags = _calibration_checks(substrate_snapshot)

    return ShadowTrackComparison(
        campaign_id=substrate_snapshot.campaign_id,
        round_index=substrate_snapshot.round_index,
        decision_action=action.value,
        decision_class=decision_class,
        substrate_mode=mode,
        substrate_class=substrate_class,
        agree=agree,
        divergences=divergences,
        sanity_findings=sanity_findings,
        calibration_flags=calibration_flags,
        created_at=timestamp,
    )


def summarize_comparisons(
    comparisons: list[ShadowTrackComparison],
) -> ShadowComparisonSummary:
    """Aggregate agreement rate and finding histograms over many rounds."""
    total = len(comparisons)
    agreement_count = sum(1 for c in comparisons if c.agree)
    divergence: Counter[str] = Counter()
    sanity: Counter[str] = Counter()
    calibration: Counter[str] = Counter()
    for comparison in comparisons:
        divergence.update(f.code for f in comparison.divergences)
        sanity.update(f.code for f in comparison.sanity_findings)
        calibration.update(f.code for f in comparison.calibration_flags)
    return ShadowComparisonSummary(
        total=total,
        agreement_count=agreement_count,
        agreement_rate=(agreement_count / total) if total else 0.0,
        divergence_histogram=dict(divergence),
        sanity_histogram=dict(sanity),
        calibration_histogram=dict(calibration),
    )


def parse_substrate_log_line(line: str) -> AdaptiveCampaignSubstrateSnapshot | None:
    """Reconstruct a substrate snapshot from its shadow log line, or None."""
    return _parse_log_line(line, "adaptive_campaign_substrate_snapshot", AdaptiveCampaignSubstrateSnapshot)


def _parse_log_line(line: str, key: str, model: type[Any]) -> Any | None:
    marker = f"{key} "
    index = line.find(marker)
    if index < 0:
        return None
    payload = line[index + len(marker):].strip()
    try:
        return model.model_validate(json.loads(payload))
    except Exception:
        return None


def _divergences(
    decision_class: ShadowEquivalenceClass,
    substrate_class: ShadowEquivalenceClass,
    action: CampaignDecisionAction,
    mode: CampaignMode,
) -> list[ShadowFinding]:
    if decision_class == substrate_class:
        return []
    if decision_class == ShadowEquivalenceClass.CONSTRAINT:
        return [
            ShadowFinding(
                category="divergence",
                code="substrate_missing_constraint_mode",
                severity="info",
                summary=(
                    "Legacy track proposed constraint tightening; the substrate has "
                    "no constraint/safety mode yet."
                ),
                evidence={"decision_action": action.value, "substrate_mode": mode.value},
            )
        ]
    return [
        ShadowFinding(
            category="divergence",
            code="class_mismatch",
            severity="warning",
            summary=(
                f"Tracks disagree: decision '{action.value}' maps to "
                f"{decision_class.value} but substrate mode '{mode.value}' maps to "
                f"{substrate_class.value}."
            ),
            evidence={
                "decision_action": action.value,
                "decision_class": decision_class.value,
                "substrate_mode": mode.value,
                "substrate_class": substrate_class.value,
            },
        )
    ]


def _sanity_checks(
    snapshot: AdaptiveCampaignSubstrateSnapshot,
) -> list[ShadowFinding]:
    findings: list[ShadowFinding] = []
    das = snapshot.dynamic_action_space_snapshot
    voi = snapshot.value_of_information_snapshot
    mode = snapshot.campaign_mode_decision.mode
    attribution = snapshot.failure_attribution
    label_by_name = {a.name: a.label for a in das.assessments}

    if voi.ranking and label_by_name.get(voi.ranking[0]) == ActionShadowLabel.PROPOSED_DISABLED:
        findings.append(
            ShadowFinding(
                category="sanity",
                code="voi_recommends_disabled",
                severity="concern",
                summary=(
                    f"VoI ranks a proposed-disabled action first: '{voi.ranking[0]}'."
                ),
                evidence={"top_ranked": voi.ranking[0]},
            )
        )

    if (
        mode == CampaignMode.BO_OPTIMIZATION
        and attribution is not None
        and attribution.confidence >= _LOW_CONFIDENCE_THRESHOLD
    ):
        findings.append(
            ShadowFinding(
                category="sanity",
                code="failure_ignored_by_mode",
                severity="concern",
                summary=(
                    "Mode is optimization while a confident failure attribution is "
                    "present."
                ),
                evidence={
                    "dominant_category": attribution.dominant_category.value,
                    "confidence": attribution.confidence,
                },
            )
        )

    if (
        das.assessments
        and len(das.proposed_disabled_actions) == len(das.assessments)
        and mode != CampaignMode.STOP_RECOMMENDED
    ):
        findings.append(
            ShadowFinding(
                category="sanity",
                code="all_disabled_non_stop",
                severity="warning",
                summary="All actions are proposed-disabled but the mode is not stop.",
                evidence={"mode": mode.value, "action_count": len(das.assessments)},
            )
        )

    if mode == CampaignMode.CALIBRATION and (
        attribution is None
        or attribution.dominant_category != FailureAttributionCategory.INSTRUMENT
    ):
        findings.append(
            ShadowFinding(
                category="sanity",
                code="calibration_without_instrument_failure",
                severity="warning",
                summary="Calibration mode without a dominant instrument failure.",
                evidence={
                    "dominant_category": (
                        attribution.dominant_category.value if attribution else None
                    )
                },
            )
        )

    if mode == CampaignMode.FAILURE_DIAGNOSIS and attribution is None:
        findings.append(
            ShadowFinding(
                category="sanity",
                code="diagnosis_without_failure",
                severity="warning",
                summary="Failure-diagnosis mode without any failure attribution.",
                evidence={"mode": mode.value},
            )
        )

    return findings


def _calibration_checks(
    snapshot: AdaptiveCampaignSubstrateSnapshot,
) -> list[ShadowFinding]:
    findings: list[ShadowFinding] = []
    das = snapshot.dynamic_action_space_snapshot
    available = set(das.available_capabilities)

    for assessment in das.assessments:
        overlap = set(assessment.missing_capabilities) & available
        if overlap:
            findings.append(
                ShadowFinding(
                    category="calibration",
                    code="capability_mapping_inconsistency",
                    severity="concern",
                    summary=(
                        f"Action '{assessment.name}' reports missing capabilities that "
                        f"are actually available: {sorted(overlap)}."
                    ),
                    evidence={"action": assessment.name, "overlap": sorted(overlap)},
                )
            )
        if _action_kind(assessment) == "experiment" and not assessment.required_capabilities:
            findings.append(
                ShadowFinding(
                    category="calibration",
                    code="experiment_without_capability",
                    severity="info",
                    summary=(
                        f"Action '{assessment.name}' is kind 'experiment' but requires "
                        "no capability; the kind mapper may have mislabeled it."
                    ),
                    evidence={"action": assessment.name},
                )
            )

    attribution = snapshot.failure_attribution
    if (
        attribution is not None
        and attribution.dominant_category == FailureAttributionCategory.EXTERNAL_CONTEXT_MISSING
        and attribution.source_failure_type not in (None, "unknown")
    ):
        findings.append(
            ShadowFinding(
                category="calibration",
                code="attribution_known_type_as_external",
                severity="warning",
                summary=(
                    "Failure attributed to external-context-missing despite a known "
                    f"failure type '{attribution.source_failure_type}'."
                ),
                evidence={"source_failure_type": attribution.source_failure_type},
            )
        )

    return findings


def _action_kind(assessment: Any) -> str | None:
    for evidence in assessment.evidence:
        kind = evidence.payload.get("action_kind")
        if kind is not None:
            return str(kind)
    return None
