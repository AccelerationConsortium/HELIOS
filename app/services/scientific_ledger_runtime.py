"""Thin runtime bridge from campaign decisions to the Scientific Ledger."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.services.decision_outcome import (
    CampaignDecisionAccounting,
    CampaignDecisionAccountingBuilder,
    CampaignDecisionOutcomeBuilder,
)
from app.services.decision_trace import CampaignDecisionTrace
from app.services.scientific_ledger import LedgerWriteResult, get_scientific_ledger


@dataclass(frozen=True)
class RuntimeDecisionAccountingResult:
    accounting: CampaignDecisionAccounting
    trajectory_id: str
    ledger_result: LedgerWriteResult | None = None


def record_pending_scientific_decision(
    trace: CampaignDecisionTrace,
    *,
    campaign_metadata: Mapping[str, Any] | None = None,
    policy_snapshot: Mapping[str, Any] | None = None,
) -> LedgerWriteResult | None:
    """Record the pre-execution card when Markdown projection is enabled."""
    if not _ledger_enabled():
        return None
    return get_scientific_ledger().record_pending(
        trace,
        campaign_metadata=campaign_metadata,
        policy_snapshot=policy_snapshot,
    )


def finalize_scientific_decision(
    trace: CampaignDecisionTrace,
    *,
    observed_action: str | None = None,
    observed_backend: str | None = None,
    candidate_count: int | None = None,
    execution_success: bool | None = None,
    failure_count: int = 0,
    safety_incident_count: int = 0,
    objective_delta: float | None = None,
    proxy_gap_delta: float | None = None,
    validation_success: bool | None = None,
    recovery_attempted: bool = False,
    recovery_success: bool | None = None,
    context_request_fulfilled: bool | None = None,
    human_override: bool | None = None,
    metadata: Mapping[str, Any] | None = None,
    campaign_metadata: Mapping[str, Any] | None = None,
    policy_snapshot: Mapping[str, Any] | None = None,
    observations: Sequence[Any] | None = None,
    failures: Sequence[Any] | None = None,
    recovery_events: Sequence[Any] | None = None,
) -> RuntimeDecisionAccountingResult:
    """Close Trace -> Outcome -> Reward, persist it, then project to Markdown."""
    from app.services.decision_trajectory import persist_campaign_trajectory

    trace = _trace_with_observed_action(trace, observed_action)
    outcome = CampaignDecisionOutcomeBuilder().build(
        trace=trace,
        observed_action=observed_action,
        observed_backend=observed_backend,
        candidate_count=candidate_count,
        execution_success=execution_success,
        failure_count=failure_count,
        safety_incident_count=safety_incident_count,
        objective_delta=objective_delta,
        proxy_gap_delta=proxy_gap_delta,
        validation_success=validation_success,
        recovery_attempted=recovery_attempted,
        recovery_success=recovery_success,
        context_request_fulfilled=context_request_fulfilled,
        human_override=human_override,
        metadata=dict(metadata or {}),
    )
    accounting = CampaignDecisionAccountingBuilder().build(
        trace=trace,
        outcome=outcome,
        metadata=dict(metadata or {}),
    )
    trajectory_id = persist_campaign_trajectory(accounting)
    ledger_result = None
    if _ledger_enabled():
        ledger_result = get_scientific_ledger().record_completed(
            accounting,
            campaign_metadata=campaign_metadata,
            policy_snapshot=policy_snapshot,
            observations=observations,
            failures=failures,
            recovery_events=recovery_events,
        )
    return RuntimeDecisionAccountingResult(
        accounting=accounting,
        trajectory_id=trajectory_id,
        ledger_result=ledger_result,
    )


def should_capture_decision_trace() -> bool:
    """A trace is needed for shadow logging, ledger projection, or live authority."""
    from app.core.config import get_settings

    settings = get_settings()
    return bool(
        getattr(settings, "contextual_decision_shadow_enabled", False)
        or getattr(settings, "scientific_ledger_enabled", False)
        or getattr(settings, "campaign_decision_authority_enabled", False)
    )


def _ledger_enabled() -> bool:
    from app.core.config import get_settings

    return bool(getattr(get_settings(), "scientific_ledger_enabled", False))


def _trace_with_observed_action(
    trace: CampaignDecisionTrace,
    observed_action: str | None,
) -> CampaignDecisionTrace:
    """Return a trace whose runtime comparison matches the consumed action."""
    if observed_action is None or trace.actual_action == observed_action:
        return trace
    shadow_action = trace.shadow_action.value
    would_change_route = observed_action != shadow_action
    comparison = {
        **dict(trace.comparison),
        "actual_action": observed_action,
        "shadow_action": shadow_action,
        "would_change_route": would_change_route,
    }
    return trace.model_copy(
        deep=True,
        update={
            "actual_action": observed_action,
            "would_change_route": would_change_route,
            "comparison": comparison,
        },
    )


__all__ = [
    "RuntimeDecisionAccountingResult",
    "finalize_scientific_decision",
    "record_pending_scientific_decision",
    "should_capture_decision_trace",
]
