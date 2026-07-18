"""Bounded live authority for campaign-level decision plans.

The contextual decision layer originally produced shadow-only envelopes.  This
module is the narrow promotion boundary: when explicitly enabled, it converts a
``CampaignDecisionPlan`` into a deterministic verdict that the orchestrator can
consume before candidate generation.

The module is intentionally pure.  It does not mutate campaign state, write to
the database, call agents, or execute tools.  The orchestrator owns all effects
and records every consumed verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.decision_models import (
    CampaignContextRequest,
    CampaignDecisionAction,
    CampaignDecisionPlan,
)


@dataclass(frozen=True)
class CampaignAuthorityStateUpdate:
    """A state update the orchestrator should persist if the verdict is consumed."""

    update_type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CampaignAuthorityVerdict:
    """Live-routing verdict derived from a campaign decision plan."""

    enabled: bool
    consumed: bool
    proceed_to_candidates: bool
    terminal: bool
    action_type: CampaignDecisionAction
    stop_reason: str | None = None
    round_status: str = "continue"
    reason: str = ""
    state_updates: tuple[CampaignAuthorityStateUpdate, ...] = ()
    event_payload: dict[str, Any] = field(default_factory=dict)


_NON_CANDIDATE_ACTIONS = {
    CampaignDecisionAction.RECOVER_FAILURE,
    CampaignDecisionAction.RUN_VALIDATION,
    CampaignDecisionAction.QUERY_LITERATURE,
    CampaignDecisionAction.REQUEST_HUMAN_OBSERVATION,
    CampaignDecisionAction.REVISE_OBJECTIVE,
    CampaignDecisionAction.TIGHTEN_CONSTRAINTS,
}


def evaluate_campaign_decision_authority(
    plan: CampaignDecisionPlan,
    *,
    enabled: bool,
) -> CampaignAuthorityVerdict:
    """Return the live authority verdict for a decision-layer plan.

    With ``enabled=False`` this is a no-op and candidate generation continues.
    With ``enabled=True``:

    * ``PROPOSE_CANDIDATES`` continues normally.
    * ``STOP_CAMPAIGN`` terminates the campaign before more candidates.
    * validation/recovery/context/objective/constraint decisions block candidate
      generation for the current round and return explicit state updates.
    """
    if not enabled:
        return CampaignAuthorityVerdict(
            enabled=False,
            consumed=False,
            proceed_to_candidates=True,
            terminal=False,
            action_type=plan.action_type,
            reason="campaign decision authority disabled",
            event_payload=_event_payload(plan, consumed=False, enabled=False),
        )

    if plan.action_type == CampaignDecisionAction.PROPOSE_CANDIDATES:
        return CampaignAuthorityVerdict(
            enabled=True,
            consumed=False,
            proceed_to_candidates=True,
            terminal=False,
            action_type=plan.action_type,
            reason="decision layer selected candidate proposal path",
            event_payload=_event_payload(plan, consumed=False, enabled=True),
        )

    updates = _state_updates_for_plan(plan)
    if plan.action_type == CampaignDecisionAction.STOP_CAMPAIGN:
        return CampaignAuthorityVerdict(
            enabled=True,
            consumed=True,
            proceed_to_candidates=False,
            terminal=True,
            action_type=plan.action_type,
            stop_reason="campaign_decision_authority_stop",
            round_status="completed",
            reason=plan.rationale,
            state_updates=updates,
            event_payload=_event_payload(plan, consumed=True, enabled=True),
        )

    if plan.action_type in _NON_CANDIDATE_ACTIONS:
        return CampaignAuthorityVerdict(
            enabled=True,
            consumed=True,
            proceed_to_candidates=False,
            terminal=False,
            action_type=plan.action_type,
            round_status="deferred",
            reason=plan.rationale,
            state_updates=updates,
            event_payload=_event_payload(plan, consumed=True, enabled=True),
        )

    return CampaignAuthorityVerdict(
        enabled=True,
        consumed=False,
        proceed_to_candidates=True,
        terminal=False,
        action_type=plan.action_type,
        reason=f"unknown authority action {plan.action_type}; continuing safely",
        event_payload=_event_payload(plan, consumed=False, enabled=True),
    )


def _state_updates_for_plan(
    plan: CampaignDecisionPlan,
) -> tuple[CampaignAuthorityStateUpdate, ...]:
    updates: list[CampaignAuthorityStateUpdate] = []

    if plan.objective_patch is not None:
        updates.append(
            CampaignAuthorityStateUpdate(
                update_type="objective_transition",
                payload={
                    "reason": plan.objective_patch.reason,
                    "proposed_changes": dict(plan.objective_patch.proposed_changes),
                    "source_action": plan.action_type.value,
                    "confidence": plan.confidence,
                    "auto_applied": False,
                },
            )
        )

    if plan.constraint_patch is not None:
        updates.append(
            CampaignAuthorityStateUpdate(
                update_type="space_revision",
                payload={
                    "revision_type": "constraint_update",
                    "reason": plan.constraint_patch.reason,
                    "proposed_changes": dict(plan.constraint_patch.proposed_changes),
                    "source_action": plan.action_type.value,
                    "confidence": plan.confidence,
                    "approval_required": True,
                    "auto_applied": False,
                },
            )
        )

    requests = tuple(plan.context_requests)
    if not requests:
        synthetic = _synthetic_context_request(plan)
        requests = (synthetic,) if synthetic is not None else ()

    for request in requests:
        updates.append(
            CampaignAuthorityStateUpdate(
                update_type="context_request",
                payload={
                    "request_type": request.request_type,
                    "reason": request.reason,
                    "priority": request.priority,
                    "target": request.target,
                    "payload": dict(request.payload),
                    "source_action": plan.action_type.value,
                    "confidence": plan.confidence,
                },
            )
        )

    if plan.action_type == CampaignDecisionAction.RECOVER_FAILURE:
        updates.append(
            CampaignAuthorityStateUpdate(
                update_type="recovery_request",
                payload={
                    "reason": plan.rationale,
                    "route_target": plan.route_target or "recovery",
                    "source_action": plan.action_type.value,
                    "confidence": plan.confidence,
                },
            )
        )

    if plan.action_type == CampaignDecisionAction.RUN_VALIDATION:
        updates.append(
            CampaignAuthorityStateUpdate(
                update_type="validation_request",
                payload={
                    "reason": plan.rationale,
                    "route_target": plan.route_target or "validation",
                    "source_action": plan.action_type.value,
                    "confidence": plan.confidence,
                },
            )
        )

    return tuple(updates)


def _synthetic_context_request(
    plan: CampaignDecisionPlan,
) -> CampaignContextRequest | None:
    if plan.action_type == CampaignDecisionAction.QUERY_LITERATURE:
        return CampaignContextRequest(
            request_type="literature_context",
            reason=plan.rationale,
            priority="high",
            target=plan.route_target or "literature",
        )
    if plan.action_type == CampaignDecisionAction.REQUEST_HUMAN_OBSERVATION:
        return CampaignContextRequest(
            request_type="human_observation",
            reason=plan.rationale,
            priority="high",
            target=plan.route_target or "human_observation",
        )
    return None


def _event_payload(
    plan: CampaignDecisionPlan,
    *,
    consumed: bool,
    enabled: bool,
) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "consumed": consumed,
        "action_type": plan.action_type.value,
        "rationale": plan.rationale,
        "confidence": plan.confidence,
        "shadow_only_plan": plan.shadow_only,
        "route_target": plan.route_target,
        "fallback_action": (
            plan.fallback_action.value if plan.fallback_action is not None else None
        ),
        "context_requests": [
            request.model_dump(mode="json") for request in plan.context_requests
        ],
        "metadata": dict(plan.metadata),
    }
