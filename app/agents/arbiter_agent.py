"""ArbiterAgent for agent-led missions."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import AgentCapability, BaseAgent, DecisionNode
from app.contracts.mission import AgentProposal, ArbiterDecision, GovernanceEnvelope


class ArbiterInput(BaseModel):
    mission_id: str
    governance: GovernanceEnvelope
    proposals: list[AgentProposal]


class ArbiterOutput(BaseModel):
    mission_id: str
    decisions: list[ArbiterDecision]
    decision_nodes: list[dict[str, Any]] = Field(default_factory=list)


class ArbiterAgent(BaseAgent[ArbiterInput, ArbiterOutput]):
    """Evaluate agent proposals against the human-approved governance envelope."""

    name = "arbiter_agent"
    description = "Governance-aware arbiter for agent-led missions"
    layer = "cross-cutting"
    capabilities = [
        AgentCapability(
            name="governance.arbitrate",
            description="Approve, reject, or escalate agent proposals",
            min_confidence=0.85,
        )
    ]

    def validate_input(self, input_data: ArbiterInput) -> list[str]:
        errors: list[str] = []
        if not input_data.mission_id:
            errors.append("mission_id is required")
        if not input_data.proposals:
            errors.append("at least one proposal is required")
        for proposal in input_data.proposals:
            if proposal.mission_id != input_data.mission_id:
                errors.append(f"proposal {proposal.proposal_id} belongs to another mission")
        return errors

    async def process(self, input_data: ArbiterInput) -> ArbiterOutput:
        decisions: list[ArbiterDecision] = []
        nodes: list[dict[str, Any]] = []

        for proposal in input_data.proposals:
            decision, reason = self._decide(input_data.governance, proposal)
            decisions.append(
                ArbiterDecision(
                    proposal_id=proposal.proposal_id,
                    decision=decision,
                    reason=reason,
                    risk_score=proposal.risk_score,
                    expected_value=proposal.expected_value,
                    required_permissions=proposal.required_permissions,
                )
            )
            nodes.append(
                DecisionNode(
                    id=f"arbiter:{proposal.proposal_id}",
                    label=f"Arbitrate {proposal.action_type}",
                    options=["approved", "needs_human", "rejected"],
                    selected=decision,
                    reason=reason,
                    outcome=proposal.title,
                ).to_dict()
            )

        return ArbiterOutput(
            mission_id=input_data.mission_id,
            decisions=decisions,
            decision_nodes=nodes,
        )

    def _decide(
        self,
        governance: GovernanceEnvelope,
        proposal: AgentProposal,
    ) -> tuple[str, str]:
        if proposal.action_type == "abort":
            return "approved", "Abort proposals are allowed to stop unsafe or low-value work."

        if proposal.action_type == "pause":
            return "needs_human", "Proposal explicitly requests human attention."

        blocked_permissions = [
            permission
            for permission in proposal.required_permissions
            if permission in governance.require_human_for
        ]
        if blocked_permissions:
            return (
                "needs_human",
                "Proposal requires human-governed permission: "
                + ", ".join(blocked_permissions),
            )

        if proposal.risk_score > governance.risk_threshold:
            return (
                "needs_human",
                f"Risk score {proposal.risk_score:.2f} exceeds threshold "
                f"{governance.risk_threshold:.2f}.",
            )

        if proposal.action_type == "modify_strategy" and not governance.autonomy.can_modify_strategy:
            return "needs_human", "Strategy changes require autonomy level 4 or higher."

        if proposal.action_type == "run_protocol" and not governance.autonomy.can_auto_execute:
            return "needs_human", "Protocol execution requires autonomy level 2 or higher."

        if proposal.expected_value <= 0.0:
            return "rejected", "Proposal has no positive expected value."

        if proposal.confidence < 0.2:
            return "rejected", "Proposal confidence is too low for execution."

        return "approved", "Proposal is within governance envelope."

