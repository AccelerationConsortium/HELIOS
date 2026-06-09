"""Agent-led mission contracts.

These contracts model the more autonomous HELIOS path: humans define the
mission and governance envelope, agents propose actions, and an arbiter decides
what can run without turning every proposal into a human-authored workflow.
"""
from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def new_mission_id() -> str:
    return f"mission-{uuid.uuid4().hex[:12]}"


def new_proposal_id() -> str:
    return f"proposal-{uuid.uuid4().hex[:12]}"


class AutonomyLevel(BaseModel):
    """Human-approved operating level for a mission."""

    level: Literal[0, 1, 2, 3, 4, 5] = Field(
        default=3,
        description=(
            "0=human scripted, 1=agent plans, 2=agent executes low-risk runs, "
            "3=agent runs within budget, 4=agent may modify strategy/protocol, "
            "5=fully autonomous under constitution"
        ),
    )

    @property
    def can_auto_execute(self) -> bool:
        return self.level >= 2

    @property
    def can_modify_strategy(self) -> bool:
        return self.level >= 4


class GovernanceEnvelope(BaseModel):
    """Mission-level autonomy boundaries set by the human supervisor."""

    autonomy: AutonomyLevel = Field(default_factory=AutonomyLevel)
    max_rounds: int = Field(default=1, ge=1, le=1000)
    max_total_runs: int = Field(default=1, ge=1, le=10000)
    max_parallel_runs: int = Field(default=1, ge=1, le=100)
    risk_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    require_human_for: list[str] = Field(
        default_factory=lambda: [
            "safety_boundary_change",
            "strategy_change",
            "protocol_mutation",
            "live_hardware",
        ]
    )
    allowed_primitives: list[str] = Field(default_factory=list)
    max_temp_c: float = Field(default=95.0, ge=0.0, le=1200.0)
    max_volume_ul: float = Field(default=1000.0, ge=1.0, le=10000.0)
    dry_run: bool = True

    def as_policy_snapshot(self) -> dict[str, Any]:
        allowed_primitives = self.allowed_primitives
        if not allowed_primitives:
            from app.services.safety import BATTERY_LAB_PRIMITIVES

            allowed_primitives = list(BATTERY_LAB_PRIMITIVES)
        return {
            "max_temp_c": self.max_temp_c,
            "max_volume_ul": self.max_volume_ul,
            "allowed_primitives": allowed_primitives,
            "require_human_approval": False,
        }


class MissionObjective(BaseModel):
    """What the agent collective is trying to accomplish."""

    primary_kpi: str
    direction: Literal["minimize", "maximize"]
    target_value: float | None = None
    hypothesis: str = ""
    success_criteria: list[str] = Field(default_factory=list)


class MissionContract(BaseModel):
    """Human-authored mission: intent, constraints, and starting substrate."""

    mission_id: str = Field(default_factory=new_mission_id)
    objective: MissionObjective
    governance: GovernanceEnvelope = Field(default_factory=GovernanceEnvelope)
    parameter_space: list[dict[str, Any]] = Field(default_factory=list)
    protocol_seed: dict[str, Any]
    inputs: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    created_by: str = "human-supervisor"


class EvidenceItem(BaseModel):
    source: str
    claim: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    data: dict[str, Any] = Field(default_factory=dict)


class AgentProposal(BaseModel):
    """An agent-generated candidate action for a mission."""

    proposal_id: str = Field(default_factory=new_proposal_id)
    mission_id: str
    proposer: str
    action_type: Literal[
        "run_protocol",
        "modify_strategy",
        "request_measurement",
        "pause",
        "abort",
        "recover",
    ]
    title: str
    rationale: str
    expected_value: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    protocol: dict[str, Any] | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    counterarguments: list[str] = Field(default_factory=list)
    rollback_plan: str = ""
    required_permissions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _protocol_required_for_run(self) -> AgentProposal:
        if self.action_type == "run_protocol" and not self.protocol:
            raise ValueError("run_protocol proposals require protocol")
        return self


class ArbiterDecision(BaseModel):
    """Arbiter decision over a single proposal."""

    proposal_id: str
    decision: Literal["approved", "needs_human", "rejected"]
    reason: str
    risk_score: float = Field(ge=0.0, le=1.0)
    expected_value: float = Field(ge=0.0, le=1.0)
    required_permissions: list[str] = Field(default_factory=list)
    run_id: str | None = None


class MissionLaunchResult(BaseModel):
    mission_id: str
    status: Literal["launched", "awaiting_human", "rejected"]
    proposals: list[AgentProposal]
    decisions: list[ArbiterDecision]
    run_ids: list[str] = Field(default_factory=list)
