from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any


class CampaignCreate(BaseModel):
    name: str
    cadence_seconds: int = Field(gt=0)
    protocol: dict[str, Any]
    inputs: dict[str, Any] = Field(default_factory=dict)
    policy_snapshot: dict[str, Any] | None = None
    created_by: str = "system"


class TriggerRequest(BaseModel):
    campaign_id: str | None = None
    protocol: dict[str, Any] | None = None
    inputs: dict[str, Any] | None = None
    policy_snapshot: dict[str, Any] | None = None
    session_key: str | None = None
    actor: str = "trigger"
    payload: dict[str, Any] = Field(default_factory=dict)


class ApprovalRequest(BaseModel):
    approver: str
    reason: str | None = None


class RunSummary(BaseModel):
    id: str
    campaign_id: str | None
    trigger_type: str
    session_key: str
    status: str
    graph_hash: str | None
    created_at: str
    updated_at: str


class RunDetail(BaseModel):
    id: str
    campaign_id: str | None
    trigger_type: str
    trigger_payload: dict[str, Any]
    session_key: str
    status: str
    protocol: dict[str, Any]
    inputs: dict[str, Any]
    compiled_graph: dict[str, Any]
    graph_hash: str | None
    policy_snapshot: dict[str, Any]
    rejection_reason: str | None
    created_by: str
    created_at: str
    updated_at: str
    started_at: str | None
    ended_at: str | None
    steps: list[dict[str, Any]]


class BatchGenerateRequest(BaseModel):
    protocol_template: dict[str, Any]
    dimensions: list[dict[str, Any]]
    strategy: str = "lhs"
    n_candidates: int = Field(default=10, ge=1, le=1000)
    seed: int | None = None
    campaign_id: str | None = None
    created_by: str = "system"


class EventEntry(BaseModel):
    id: str
    run_id: str | None
    actor: str
    action: str
    details: dict[str, Any]
    created_at: str


class TemplateCreateRequest(BaseModel):
    name: str
    protocol: dict[str, Any]
    parent_template_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_by: str = "system"


class ProposalDecisionRequest(BaseModel):
    reviewer: str
    reason: str | None = None
