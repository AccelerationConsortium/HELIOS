"""Safety Agent -- cross-cutting veto authority.

Wraps existing safety.py, action_contracts.py, and error_policy.py
into a single agent with veto power over any operation.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent, DecisionNode


class SafetyCheckInput(BaseModel):
    """Input for safety validation."""
    compiled_graph: dict[str, Any]
    policy_snapshot: dict[str, Any]
    interlock_state: dict[str, Any] = Field(default_factory=lambda: {
        "hardware_interlock_ok": True,
        "cooling_ok": True,
    })


class SafetyCheckOutput(BaseModel):
    """Output from safety validation."""
    allowed: bool
    violations: list[str] = Field(default_factory=list)
    requires_approval: bool = False
    safety_score: float = 1.0  # 0.0 = dangerous, 1.0 = safe
    decision_nodes: list[dict[str, Any]] = Field(default_factory=list)


class SafetyAgent(BaseAgent[SafetyCheckInput, SafetyCheckOutput]):
    name = "safety_agent"
    description = "Cross-cutting safety veto authority"
    layer = "cross-cutting"

    def validate_input(self, input_data: SafetyCheckInput) -> list[str]:
        errors: list[str] = []
        if not input_data.policy_snapshot:
            errors.append("policy_snapshot is required")
        return errors

    async def process(self, input_data: SafetyCheckInput) -> SafetyCheckOutput:
        from app.services.safety import evaluate_preflight

        result = evaluate_preflight(
            compiled_graph=input_data.compiled_graph,
            policy_snapshot=input_data.policy_snapshot,
        )

        n_steps = len(input_data.compiled_graph.get("steps", []))
        n_violations = len(result.violations)
        safety_score = max(0.0, 1.0 - (n_violations / max(n_steps, 1)))

        # Decision node 1: preflight verdict
        preflight_node = DecisionNode(
            id="preflight_check",
            label="Safety preflight check",
            options=["Approved", "Denied (safety veto)"],
            selected="Approved" if result.allowed else "Denied (safety veto)",
            reason=f"{n_violations} violation(s) across {n_steps} step(s), score={safety_score:.2f}",
            outcome="; ".join(result.violations) if result.violations else "No violations",
        )

        # Decision node 2: escalation
        escalation_node = DecisionNode(
            id="escalation",
            label="Escalation required?",
            options=["Auto-approved", "Requires human review"],
            selected="Requires human review" if result.requires_approval else "Auto-approved",
            reason="safety_class=CAREFUL primitives present" if result.requires_approval else "All primitives within auto-approval threshold",
        )

        return SafetyCheckOutput(
            allowed=result.allowed,
            violations=result.violations,
            requires_approval=result.requires_approval,
            safety_score=safety_score,
            decision_nodes=[preflight_node.to_dict(), escalation_node.to_dict()],
        )
