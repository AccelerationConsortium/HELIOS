"""Context-aware reasoning gate and parameter-space revision policy."""
from __future__ import annotations

from app.services.strategy_models import (
    CampaignIntent,
    CampaignSnapshot,
    ContextGateDecision,
    FailureType,
    ObjectiveLevel,
    SpaceRevision,
)


def evaluate_context_gate(snapshot: CampaignSnapshot) -> ContextGateDecision:
    """Decide whether the next loop is ready for numerical optimization."""
    failure_types = {
        getattr(event.failure_type, "value", event.failure_type)
        for event in snapshot.failure_events
    }
    context = snapshot.campaign_context
    level = (
        getattr(context.current_objective_level, "value", context.current_objective_level)
        if context is not None
        else ObjectiveLevel.PERFORMANCE.value
    )

    if FailureType.HARDWARE.value in failure_types:
        return ContextGateDecision(
            ready_for_optimization=False,
            requires_human_review=True,
            recommended_intent=CampaignIntent.RECOVER,
            reason="hardware failure observed; recover execution path before optimizer changes",
        )
    if FailureType.MEASUREMENT.value in failure_types:
        return ContextGateDecision(
            ready_for_optimization=False,
            requires_calibration=True,
            recommended_intent=CampaignIntent.DIAGNOSE,
            reason="measurement failure observed; calibrate or diagnose before optimization",
        )
    if FailureType.CONSTRAINT.value in failure_types:
        return ContextGateDecision(
            ready_for_optimization=False,
            requires_space_revision=True,
            recommended_intent=CampaignIntent.RECOVER,
            reason="constraint failure observed; revise feasible region before optimization",
        )
    if FailureType.BACKEND.value in failure_types:
        return ContextGateDecision(
            ready_for_optimization=False,
            requires_human_review=True,
            recommended_intent=CampaignIntent.RECOVER,
            reason="backend failure observed; review optimizer/backend before continuing",
        )
    if FailureType.SCIENTIFIC_NEGATIVE.value in failure_types:
        return ContextGateDecision(
            ready_for_optimization=True,
            recommended_intent=CampaignIntent.VALIDATE,
            reason="scientific negative result is evidence; continue with hypothesis-aware validation",
        )
    if level == ObjectiveLevel.FEASIBILITY.value:
        return ContextGateDecision(
            ready_for_optimization=False,
            requires_human_review=True,
            recommended_intent=CampaignIntent.RECOVER,
            reason="objective level is feasibility; favor recover/diagnose/stabilize before optimization",
        )
    if level == ObjectiveLevel.DATA_QUALITY.value:
        return ContextGateDecision(
            ready_for_optimization=False,
            requires_calibration=True,
            recommended_intent=CampaignIntent.STABILIZE,
            reason="objective level is data_quality; favor replicate/calibration",
        )
    if level == ObjectiveLevel.BASELINE.value:
        return ContextGateDecision(
            ready_for_optimization=True,
            recommended_intent=CampaignIntent.DISCOVER,
            reason="objective level is baseline; favor broad exploration",
        )
    if level == ObjectiveLevel.PERFORMANCE.value:
        return ContextGateDecision(
            ready_for_optimization=True,
            recommended_intent=CampaignIntent.OPTIMIZE,
            reason="objective level is performance; favor BO/TuRBO/TPE-style optimization",
        )
    if level == ObjectiveLevel.MECHANISM.value:
        return ContextGateDecision(
            ready_for_optimization=True,
            requires_hypothesis_update=True,
            recommended_intent=CampaignIntent.VALIDATE,
            reason="mechanism objective active; update/validate hypotheses while optimizing",
        )
    if level == ObjectiveLevel.GENERALIZATION.value:
        return ContextGateDecision(
            ready_for_optimization=True,
            requires_route_switch=True,
            recommended_intent=CampaignIntent.PIVOT,
            reason="generalization objective active; favor transfer/pivot",
        )
    return ContextGateDecision()


def propose_space_revision(snapshot: CampaignSnapshot) -> SpaceRevision | None:
    """Propose a parameter-space or route revision from context/failure signals."""
    context = snapshot.campaign_context
    constraint_events = [
        event for event in snapshot.failure_events
        if getattr(event.failure_type, "value", event.failure_type)
        == FailureType.CONSTRAINT.value
    ]
    if constraint_events:
        constraints = tuple(
            {
                "type": "avoid_failed_coordinate",
                "params": dict(event.params),
                "reason": event.reason,
            }
            for event in constraint_events
        )
        return SpaceRevision(
            revision_type="constraint_update",
            add_constraints=constraints,
            reason="constraint failures indicate infeasible parameter region",
            risk_level="medium",
            approval_required=True,
            affected_parameters=tuple(
                sorted({
                    str(name)
                    for event in constraint_events
                    for name in event.params
                })
            ),
        )

    if snapshot.failed_params:
        constraints = tuple(
            {"type": "avoid_failed_coordinate", "params": dict(params)}
            for params in snapshot.failed_params
        )
        return SpaceRevision(
            revision_type="constraint_update",
            add_constraints=constraints,
            reason="failed experiment coordinates should be avoided in future search",
            risk_level="medium",
            approval_required=True,
            affected_parameters=tuple(
                sorted({str(name) for params in snapshot.failed_params for name in params})
            ),
        )

    if context is not None and context.synthesis_routes:
        gate = evaluate_context_gate(snapshot)
        if gate.requires_route_switch and len(context.synthesis_routes) > 1:
            return SpaceRevision(
                revision_type="route_switch",
                switch_route=context.synthesis_routes[1],
                reason="context gate recommends route transfer/generalization",
                risk_level="high",
                approval_required=True,
                affected_routes=tuple(context.synthesis_routes),
            )

    return None
