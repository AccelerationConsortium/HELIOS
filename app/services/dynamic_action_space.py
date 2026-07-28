"""Dynamic action space — deterministic, shadow-only action assessment.

Given a per-round ``CampaignModeDecision`` (Phase 3) and a list of read-only
action descriptors mapped from the primitives registry / action contracts, this
module produces a *shadow snapshot* that labels each action as ``preferred``,
``risky``, ``proposed_disabled`` or ``neutral`` and records its risk, cost,
latency, required capabilities, missing capabilities, and context dependencies.

Boundaries (intentional):

* Strictly read-only. It never mutates the primitives registry, enables or
  disables any action, registers composite actions, or touches the orchestrator,
  strategy selector, candidate selection, or decision-layer routing.
* All labels are *proposals* (``shadow_only=True``). ``proposed_disabled`` means
  "a downstream policy could consider disabling this", not "this is disabled".
* The snapshot is deterministic and every assessment carries a reason and
  evidence; the whole snapshot is JSON-safe and replayable via pydantic.

``ActionSpec`` is a decoupled input descriptor. Callers build it from
``PrimitiveSpec`` / ``ActionContract`` (e.g. ``safety_class=spec.safety_class.name``)
so this module never depends on live registry state.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.services.campaign_mode import CampaignMode, CampaignModeDecision
from app.services.decision_models import CampaignDecisionEvidence
from app.services.failure_attribution import FailureAttributionDistribution
from app.services.objective_state import ObjectiveState

__all__ = [
    "ActionAssessment",
    "ActionShadowLabel",
    "ActionSpec",
    "DynamicActionSpaceAssessor",
    "DynamicActionSpaceSnapshot",
    "build_action_space_snapshot",
]

#: base_risk at or above this is flagged ``risky`` in optimization mode.
_HIGH_RISK_THRESHOLD = 0.7
#: Risk bump applied to actions that touch an implicated capability.
_IMPLICATED_RISK_BUMP = 0.2
#: Risk floor by safety class name (read-only mapping; higher class = riskier).
_SAFETY_RISK_FLOOR: dict[str, float] = {
    "informational": 0.1,
    "reversible": 0.3,
    "careful": 0.5,
    "hazardous": 0.8,
}


class ActionShadowLabel(StrEnum):
    """Shadow-only recommendation label for an action."""

    PREFERRED = "preferred"
    RISKY = "risky"
    PROPOSED_DISABLED = "proposed_disabled"
    NEUTRAL = "neutral"


class ActionSpec(BaseModel):
    """Read-only descriptor of one candidate action for shadow assessment."""

    name: str
    kind: str = "experiment"
    safety_class: str | None = None
    base_risk: float = Field(default=0.3, ge=0.0, le=1.0)
    cost: float | None = None
    latency: float | None = None
    required_capabilities: list[str] = Field(default_factory=list)
    context_dependencies: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionAssessment(BaseModel):
    """Shadow assessment of a single action under a campaign mode."""

    name: str
    label: ActionShadowLabel
    risk: float = Field(ge=0.0, le=1.0)
    cost: float | None = None
    latency: float | None = None
    required_capabilities: list[str] = Field(default_factory=list)
    missing_capabilities: list[str] = Field(default_factory=list)
    context_dependencies: list[str] = Field(default_factory=list)
    reason: str
    evidence: list[CampaignDecisionEvidence] = Field(default_factory=list)


class DynamicActionSpaceSnapshot(BaseModel):
    """A deterministic, shadow-only snapshot of the action space for one round."""

    campaign_id: str
    round_index: int = Field(ge=0)
    mode: CampaignMode
    assessments: list[ActionAssessment] = Field(default_factory=list)
    preferred_actions: list[str] = Field(default_factory=list)
    risky_actions: list[str] = Field(default_factory=list)
    proposed_disabled_actions: list[str] = Field(default_factory=list)
    available_capabilities: list[str] = Field(default_factory=list)
    shadow_only: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class DynamicActionSpaceAssessor:
    """Build deterministic shadow action-space snapshots from a mode decision."""

    def snapshot(
        self,
        *,
        mode_decision: CampaignModeDecision,
        actions: list[ActionSpec],
        available_capabilities: list[str],
        failure_attribution: FailureAttributionDistribution | None = None,
        objective_state: ObjectiveState | None = None,  # noqa: ARG002 (reserved, read-only)
        now: datetime | None = None,
    ) -> DynamicActionSpaceSnapshot:
        timestamp = now or datetime.now(UTC)
        available = set(available_capabilities)
        implicated = _implicated_capabilities(failure_attribution)

        assessments = [
            _assess_action(
                action=action,
                mode=mode_decision.mode,
                available=available,
                implicated=implicated,
            )
            for action in actions
        ]

        return DynamicActionSpaceSnapshot(
            campaign_id=mode_decision.campaign_id,
            round_index=mode_decision.round_index,
            mode=mode_decision.mode,
            assessments=assessments,
            preferred_actions=[
                a.name for a in assessments if a.label == ActionShadowLabel.PREFERRED
            ],
            risky_actions=[
                a.name for a in assessments if a.label == ActionShadowLabel.RISKY
            ],
            proposed_disabled_actions=[
                a.name
                for a in assessments
                if a.label == ActionShadowLabel.PROPOSED_DISABLED
            ],
            available_capabilities=sorted(available),
            shadow_only=True,
            created_at=timestamp,
        )


def build_action_space_snapshot(
    *,
    mode_decision: CampaignModeDecision,
    actions: list[ActionSpec],
    available_capabilities: list[str],
    failure_attribution: FailureAttributionDistribution | None = None,
    objective_state: ObjectiveState | None = None,
    now: datetime | None = None,
) -> DynamicActionSpaceSnapshot:
    """Build an action-space snapshot with the default assessor."""
    return DynamicActionSpaceAssessor().snapshot(
        mode_decision=mode_decision,
        actions=actions,
        available_capabilities=available_capabilities,
        failure_attribution=failure_attribution,
        objective_state=objective_state,
        now=now,
    )


def _assess_action(
    *,
    action: ActionSpec,
    mode: CampaignMode,
    available: set[str],
    implicated: set[str],
) -> ActionAssessment:
    missing = [cap for cap in action.required_capabilities if cap not in available]
    touches_implicated = bool(
        implicated & (set(action.required_capabilities) | {action.name})
    )
    risk = _action_risk(action, mode, touches_implicated)

    if missing:
        label = ActionShadowLabel.PROPOSED_DISABLED
        reason = (
            f"Action '{action.name}' is missing required capabilities: "
            f"{', '.join(missing)}."
        )
    else:
        label, reason = _label_for_mode(action, mode, touches_implicated)

    evidence = [
        CampaignDecisionEvidence(
            source="dynamic_action_space",
            kind="action_assessment",
            summary=reason,
            payload={
                "mode": mode.value,
                "action_kind": action.kind,
                "label": label.value,
                "risk": risk,
                "missing_capabilities": missing,
                "touches_implicated_capability": touches_implicated,
            },
        )
    ]

    return ActionAssessment(
        name=action.name,
        label=label,
        risk=risk,
        cost=action.cost,
        latency=action.latency,
        required_capabilities=list(action.required_capabilities),
        missing_capabilities=missing,
        context_dependencies=list(action.context_dependencies),
        reason=reason,
        evidence=evidence,
    )


def _label_for_mode(
    action: ActionSpec,
    mode: CampaignMode,
    touches_implicated: bool,
) -> tuple[ActionShadowLabel, str]:
    kind = action.kind

    if mode == CampaignMode.STOP_RECOMMENDED:
        # Report and cleanup/maintenance actions may still run while stopping;
        # experiment / preparation / workflow actions should not start a new run.
        if kind in {"report", "cleanup"}:
            return ActionShadowLabel.NEUTRAL, (
                f"Stop recommended; '{kind}' action '{action.name}' may still run."
            )
        return ActionShadowLabel.PROPOSED_DISABLED, (
            f"Stop recommended; action '{action.name}' should not start a new run."
        )

    if mode == CampaignMode.CALIBRATION:
        if kind == "calibration":
            return ActionShadowLabel.PREFERRED, (
                f"Calibration mode prefers calibration action '{action.name}'."
            )
        if touches_implicated:
            return ActionShadowLabel.RISKY, (
                f"Action '{action.name}' uses an implicated instrument during "
                "calibration."
            )
        return ActionShadowLabel.NEUTRAL, (
            f"Action '{action.name}' is unrelated to the calibration target."
        )

    if mode == CampaignMode.FAILURE_DIAGNOSIS:
        if kind == "diagnostic":
            return ActionShadowLabel.PREFERRED, (
                f"Failure-diagnosis mode prefers diagnostic action '{action.name}'."
            )
        if touches_implicated:
            return ActionShadowLabel.RISKY, (
                f"Action '{action.name}' reuses the failing capability before "
                "diagnosis."
            )
        return ActionShadowLabel.NEUTRAL, (
            f"Action '{action.name}' is unrelated to the failing capability."
        )

    if mode == CampaignMode.LITERATURE_CONTEXT_SEEKING:
        if kind == "literature":
            return ActionShadowLabel.PREFERRED, (
                f"Context-seeking mode prefers literature action '{action.name}'."
            )
        return ActionShadowLabel.NEUTRAL, (
            f"Action '{action.name}' does not gather external context."
        )

    if mode == CampaignMode.VALIDATION:
        if kind == "validation":
            return ActionShadowLabel.PREFERRED, (
                f"Validation mode prefers validation action '{action.name}'."
            )
        return ActionShadowLabel.NEUTRAL, (
            f"Action '{action.name}' does not validate the active objective."
        )

    if mode == CampaignMode.SAFETY_CONSTRAINT_TIGHTENING:
        if kind in {"diagnostic", "calibration"}:
            return ActionShadowLabel.PREFERRED, (
                f"Constraint-tightening mode prefers assessment action '{action.name}'."
            )
        if kind in {"experiment", "preparation", "workflow"}:
            return ActionShadowLabel.RISKY, (
                f"Constraint-tightening mode holds '{kind}' action '{action.name}' "
                "until constraints are tightened."
            )
        return ActionShadowLabel.NEUTRAL, (
            f"Action '{action.name}' may continue during constraint tightening."
        )

    if mode == CampaignMode.HUMAN_OBSERVATION_REQUEST:
        if kind in {"experiment", "optimization"}:
            return ActionShadowLabel.RISKY, (
                f"Human observation pending; action '{action.name}' is risky to run."
            )
        return ActionShadowLabel.NEUTRAL, (
            f"Action '{action.name}' may proceed while awaiting human observation."
        )

    if mode == CampaignMode.OBJECTIVE_DISCOVERY:
        if kind in {"objective_discovery", "kpi_discovery", "diagnostic"}:
            return ActionShadowLabel.PREFERRED, (
                f"Objective-discovery mode prefers action '{action.name}'."
            )
        if kind in {"optimization", "experiment"}:
            return ActionShadowLabel.RISKY, (
                f"Objective-discovery mode holds '{kind}' action '{action.name}' "
                "until the campaign objective is clarified."
            )
        return ActionShadowLabel.NEUTRAL, (
            f"Action '{action.name}' does not clarify the campaign objective."
        )

    if mode == CampaignMode.CONTROLLABILITY_MAPPING:
        if kind in {"controllability_mapping", "control_probe", "calibration", "diagnostic"}:
            return ActionShadowLabel.PREFERRED, (
                f"Controllability-mapping mode prefers action '{action.name}'."
            )
        if kind == "optimization":
            return ActionShadowLabel.RISKY, (
                f"Controllability-mapping mode gates optimization action '{action.name}'."
            )
        return ActionShadowLabel.NEUTRAL, (
            f"Action '{action.name}' does not map target-vs-actual control."
        )

    if mode == CampaignMode.HARDWARE_FEASIBILITY_DISCOVERY:
        if kind in {"hardware_feasibility", "feasibility_mapping", "diagnostic"}:
            return ActionShadowLabel.PREFERRED, (
                f"Hardware-feasibility mode prefers action '{action.name}'."
            )
        if kind in {"optimization", "experiment"} and touches_implicated:
            return ActionShadowLabel.RISKY, (
                f"Action '{action.name}' touches implicated hardware before "
                "feasibility is mapped."
            )
        return ActionShadowLabel.NEUTRAL, (
            f"Action '{action.name}' does not map hardware feasibility."
        )

    if mode == CampaignMode.DATA_QUALITY_DIAGNOSTIC:
        if kind in {"data_quality_diagnostic", "replicate", "sensor_qc", "diagnostic"}:
            return ActionShadowLabel.PREFERRED, (
                f"Data-quality diagnostic mode prefers action '{action.name}'."
            )
        if kind == "optimization":
            return ActionShadowLabel.RISKY, (
                f"Data-quality diagnostic mode gates optimization action '{action.name}'."
            )
        return ActionShadowLabel.NEUTRAL, (
            f"Action '{action.name}' does not diagnose data quality."
        )

    if mode == CampaignMode.EARLY_STAGE_SYSTEM_CHARACTERIZATION:
        if kind in {"diagnostic", "characterization", "replicate", "control_probe"}:
            return ActionShadowLabel.PREFERRED, (
                f"Early-stage characterization mode prefers action '{action.name}'."
            )
        if kind == "optimization":
            return ActionShadowLabel.RISKY, (
                f"Early-stage characterization mode delays optimization action "
                f"'{action.name}'."
            )
        return ActionShadowLabel.NEUTRAL, (
            f"Action '{action.name}' is neutral for early-stage characterization."
        )

    # Default: BO_OPTIMIZATION.
    if kind in {"experiment", "optimization"}:
        floor = _SAFETY_RISK_FLOOR.get((action.safety_class or "").lower(), 0.0)
        if max(action.base_risk, floor) >= _HIGH_RISK_THRESHOLD:
            return ActionShadowLabel.RISKY, (
                f"Optimization mode flags high-risk action '{action.name}'."
            )
        return ActionShadowLabel.PREFERRED, (
            f"Optimization mode prefers low-risk action '{action.name}'."
        )
    return ActionShadowLabel.NEUTRAL, (
        f"Action '{action.name}' is not an optimization action."
    )


def _action_risk(
    action: ActionSpec,
    mode: CampaignMode,
    touches_implicated: bool,
) -> float:
    floor = _SAFETY_RISK_FLOOR.get((action.safety_class or "").lower(), 0.0)
    risk = max(action.base_risk, floor)
    if touches_implicated and mode in {
        CampaignMode.CALIBRATION,
        CampaignMode.FAILURE_DIAGNOSIS,
    }:
        risk += _IMPLICATED_RISK_BUMP
    return _round(_clamp_unit(risk))


def _implicated_capabilities(
    failure_attribution: FailureAttributionDistribution | None,
) -> set[str]:
    if failure_attribution is None:
        return set()
    implicated: set[str] = set()
    if failure_attribution.primitive:
        implicated.add(failure_attribution.primitive)
    return implicated


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _round(value: float) -> float:
    return round(value, 10)
