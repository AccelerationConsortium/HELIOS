"""Pydantic models for the campaign-level decision envelope.

Phase 1 keeps these models independent from the dynamic strategy selector.
The existing selector result remains a plain dict so persisted strategy
payloads and resume paths stay compatible.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CampaignDecisionAction(StrEnum):
    """Top-level campaign action proposed by the decision layer."""

    PROPOSE_CANDIDATES = "propose_candidates"
    REVISE_OBJECTIVE = "revise_objective"
    TIGHTEN_CONSTRAINTS = "tighten_constraints"
    QUERY_LITERATURE = "query_literature"
    REQUEST_HUMAN_OBSERVATION = "request_human_observation"
    RUN_VALIDATION = "run_validation"
    RECOVER_FAILURE = "recover_failure"
    STOP_CAMPAIGN = "stop_campaign"


class CampaignDecisionEvidence(BaseModel):
    """Evidence attached to a campaign decision envelope."""

    source: str
    kind: str
    summary: str
    weight: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class CampaignContextRequest(BaseModel):
    """Additional context the decision layer would like before execution."""

    request_type: str
    reason: str
    priority: str = "normal"
    target: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ObjectivePatch(BaseModel):
    """Shadow-only objective update proposal."""

    reason: str
    proposed_changes: dict[str, Any] = Field(default_factory=dict)
    shadow_only: bool = True


class ConstraintPatch(BaseModel):
    """Shadow-only constraint update proposal."""

    reason: str
    proposed_changes: dict[str, Any] = Field(default_factory=dict)
    shadow_only: bool = True


class CampaignDecisionPlan(BaseModel):
    """Campaign-level decision envelope around strategy-selection output."""

    action_type: CampaignDecisionAction
    campaign_intent: str | None = None
    optimization_mode: str | None = None
    candidate_generation_backend: str | None = None
    strategy_trace: dict[str, Any] = Field(default_factory=dict)
    objective_patch: ObjectivePatch | None = None
    constraint_patch: ConstraintPatch | None = None
    context_requests: list[CampaignContextRequest] = Field(default_factory=list)
    evidence: list[CampaignDecisionEvidence] = Field(default_factory=list)
    route_target: str | None = None
    rationale: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    fallback_action: CampaignDecisionAction | None = None
    shadow_only: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class CampaignRoundContext(BaseModel):
    """Inputs available to the campaign decision layer for one round."""

    campaign_id: str
    round_index: int = Field(ge=0)
    stop_requested: bool = False
    failure_summary: dict[str, Any] = Field(default_factory=dict)
    safety_summary: dict[str, Any] = Field(default_factory=dict)
    objective_summary: dict[str, Any] = Field(default_factory=dict)
    constraint_summary: dict[str, Any] = Field(default_factory=dict)
    nexus_diagnostics: dict[str, Any] = Field(default_factory=dict)
    backend_memory_summary: dict[str, Any] = Field(default_factory=dict)
    bo_mcp_summary: dict[str, Any] = Field(default_factory=dict)
    learning_policy_summary: dict[str, Any] = Field(default_factory=dict)
    validation_summary: dict[str, Any] = Field(default_factory=dict)
    human_observations: list[str] = Field(default_factory=list)
    literature_summary: dict[str, Any] = Field(default_factory=dict)
    strategy_selection_result: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
