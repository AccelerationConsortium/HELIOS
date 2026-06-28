"""Trace DTOs for shadow campaign decisions.

This layer records the normalized decision context, the shadow decision plan,
and the runtime action it is compared against. It is pure serialization and
comparison logic; it does not call live services or mutate campaign runtime.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from app.services.decision_models import (
    CampaignDecisionAction,
    CampaignDecisionEvidence,
    CampaignDecisionPlan,
    CampaignRoundContext,
)

__all__ = [
    "CampaignDecisionTrace",
    "CampaignDecisionTraceBuilder",
    "build_campaign_decision_trace",
]


class CampaignDecisionTrace(BaseModel):
    """JSON-serializable audit record for one shadow campaign decision."""

    trace_id: str
    campaign_id: str
    round_index: int = Field(ge=0)
    context: CampaignRoundContext
    decision_plan: CampaignDecisionPlan
    actual_stage: str | None = None
    actual_action: str | None = None
    shadow_action: CampaignDecisionAction
    would_change_route: bool = False
    comparison: dict[str, Any] = Field(default_factory=dict)
    evidence: list[CampaignDecisionEvidence] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class CampaignDecisionTraceBuilder:
    """Build shadow decision traces from context, plan, and actual runtime path."""

    def build(
        self,
        *,
        context: CampaignRoundContext,
        decision_plan: CampaignDecisionPlan,
        actual_stage: str | None = None,
        actual_action: str | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> CampaignDecisionTrace:
        shadow_action = decision_plan.action_type
        would_change_route = (
            False if actual_action is None else actual_action != shadow_action.value
        )
        comparison = {
            "actual_action": actual_action,
            "shadow_action": shadow_action.value,
            "would_change_route": would_change_route,
        }
        return CampaignDecisionTrace(
            trace_id=trace_id or f"cdt-{uuid4().hex}",
            campaign_id=context.campaign_id,
            round_index=context.round_index,
            context=context.model_copy(deep=True),
            decision_plan=decision_plan.model_copy(deep=True),
            actual_stage=actual_stage,
            actual_action=actual_action,
            shadow_action=shadow_action,
            would_change_route=would_change_route,
            comparison=comparison,
            evidence=deepcopy(decision_plan.evidence),
            metadata=deepcopy(dict(metadata or {})),
        )


def build_campaign_decision_trace(**kwargs: Any) -> CampaignDecisionTrace:
    """Build a CampaignDecisionTrace with the default builder."""
    return CampaignDecisionTraceBuilder().build(**kwargs)
