"""Mission launch service for agent-led HELIOS operation."""
from __future__ import annotations

from app.agents.arbiter_agent import ArbiterAgent, ArbiterInput
from app.contracts.mission import (
    AgentProposal,
    ArbiterDecision,
    EvidenceItem,
    MissionContract,
    MissionLaunchResult,
)
from app.services.run_service import DomainError, create_run


async def launch_mission(contract: MissionContract) -> MissionLaunchResult:
    """Launch an agent-led mission and create approved first actions.

    The first implementation intentionally keeps persistence on the existing
    run/audit path: approved protocol proposals become runs, while proposal and
    arbiter context is stored in the run trigger payload.
    """
    proposals = _seed_proposals(contract)
    arbiter = ArbiterAgent()
    arbiter_result = await arbiter.run(
        ArbiterInput(
            mission_id=contract.mission_id,
            governance=contract.governance,
            proposals=proposals,
        )
    )
    if not arbiter_result.success or arbiter_result.output is None:
        errors = "; ".join(arbiter_result.errors) or "arbiter failed"
        raise DomainError(errors)

    decisions = arbiter_result.output.decisions
    run_ids: list[str] = []
    decisions_by_id = {decision.proposal_id: decision for decision in decisions}
    updated_decisions: list[ArbiterDecision] = []

    for proposal in proposals:
        decision = decisions_by_id[proposal.proposal_id]
        if decision.decision == "approved" and proposal.action_type == "run_protocol":
            run = create_run(
                trigger_type="mission_agent",
                trigger_payload={
                    "mission_id": contract.mission_id,
                    "proposal": proposal.model_dump(mode="json"),
                    "arbiter_decision": decision.model_dump(mode="json"),
                    "governance": contract.governance.model_dump(mode="json"),
                },
                campaign_id=None,
                protocol=proposal.protocol or contract.protocol_seed,
                inputs=proposal.inputs or contract.inputs,
                policy_snapshot=contract.governance.as_policy_snapshot(),
                actor=proposal.proposer,
                session_key=contract.mission_id,
            )
            run_ids.append(run["id"])
            decision = decision.model_copy(update={"run_id": run["id"]})
        updated_decisions.append(decision)

    if run_ids:
        status = "launched"
    elif any(decision.decision == "needs_human" for decision in updated_decisions):
        status = "awaiting_human"
    else:
        status = "rejected"

    return MissionLaunchResult(
        mission_id=contract.mission_id,
        status=status,
        proposals=proposals,
        decisions=updated_decisions,
        run_ids=run_ids,
    )


def _seed_proposals(contract: MissionContract) -> list[AgentProposal]:
    """Create the first autonomous proposal set from the mission seed.

    Later versions can source these from PlannerAgent, ScientistSwarm, memory,
    or live observations. This first cut still makes the agent-led contract
    concrete: the proposal carries evidence, risk, expected value, and rollback.
    """
    steps = contract.protocol_seed.get("steps", [])
    n_steps = len(steps) if isinstance(steps, list) else 0
    required_permissions: list[str] = []
    if not contract.governance.dry_run:
        required_permissions.append("live_hardware")

    return [
        AgentProposal(
            mission_id=contract.mission_id,
            proposer="planner_agent",
            action_type="run_protocol",
            title=f"Execute first mission probe for {contract.objective.primary_kpi}",
            rationale=(
                "Start with the provided protocol seed to gather evidence before "
                "changing strategy or expanding the search space."
            ),
            expected_value=0.65 if n_steps else 0.0,
            risk_score=0.15 + min(n_steps, 10) * 0.02,
            confidence=0.7,
            protocol=contract.protocol_seed,
            inputs=contract.inputs,
            evidence=[
                EvidenceItem(
                    source="mission_contract",
                    claim="Human supervisor supplied this protocol as the initial executable seed.",
                    confidence=0.8,
                    data={"step_count": n_steps},
                )
            ],
            counterarguments=[
                "The seed protocol may not be globally optimal; it is used to collect the first observation."
            ],
            rollback_plan="Stop after the first run and keep the mission in human-reviewable state.",
            required_permissions=required_permissions,
            metadata={
                "primary_kpi": contract.objective.primary_kpi,
                "direction": contract.objective.direction,
                "autonomy_level": contract.governance.autonomy.level,
            },
        )
    ]

