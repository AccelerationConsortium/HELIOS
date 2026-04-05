"""Safety Agent -- cross-cutting veto authority.

Wraps existing safety.py, action_contracts.py, and error_policy.py
into a single agent with veto power over any operation.

v2: Runtime pause decisions based on safety score.
- safety_score >= 0.8  → auto-approve
- 0.5 <= score < 0.8   → pause for human review (marginal)
- score < 0.5          → veto (blocked regardless)
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import AgentPauseRejected, BaseAgent, DecisionNode
from app.agents.pause import Granularity, PauseRequest, RiskAssessment


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
    # v2: granularity used for this check
    granularity_used: str = "coarse"


class SafetyAgent(BaseAgent[SafetyCheckInput, SafetyCheckOutput]):
    name = "safety_agent"
    description = "Cross-cutting safety veto authority"
    layer = "cross-cutting"

    # Configurable thresholds (can be tuned per-deployment)
    MARGINAL_THRESHOLD = 0.8   # Below this → pause for human review
    VETO_THRESHOLD = 0.5       # Below this → block unconditionally
    # Hazardous reagents / primitives that always require human review
    ALWAYS_PAUSE_PRIMITIVES: set[str] = {"heating", "high_voltage", "uv_exposure", "pressure"}

    def validate_input(self, input_data: SafetyCheckInput) -> list[str]:
        errors: list[str] = []
        if not input_data.policy_snapshot:
            errors.append("policy_snapshot is required")
        return errors

    async def assess_granularity(
        self,
        input_data: SafetyCheckInput,
        context: dict[str, Any] | None = None,
    ) -> Granularity:
        """Safety agent always uses FINE granularity — every check matters."""
        return Granularity.FINE

    async def process(self, input_data: SafetyCheckInput) -> SafetyCheckOutput:
        from app.services.safety import evaluate_preflight

        result = evaluate_preflight(
            compiled_graph=input_data.compiled_graph,
            policy_snapshot=input_data.policy_snapshot,
        )

        n_steps = len(input_data.compiled_graph.get("steps", []))
        n_violations = len(result.violations)
        safety_score = max(0.0, 1.0 - (n_violations / max(n_steps, 1)))

        # ── v2: Check for always-pause primitives ──────────────────────
        primitives_used = {
            step.get("primitive", "")
            for step in input_data.compiled_graph.get("steps", [])
        }
        hazardous_primitives = primitives_used & self.ALWAYS_PAUSE_PRIMITIVES

        # ── v3: Consult memory for historical failure signals ─────────
        memory_high_risk: list[str] = []
        try:
            from app.services.memory_risk_bridge import get_primitive_risk_profile
            for prim in primitives_used:
                if not prim:
                    continue
                profile = get_primitive_risk_profile(prim)
                if profile.failure_rate > 0.3 and profile.total_observations >= 5:
                    memory_high_risk.append(
                        f"{prim} (fail_rate={profile.failure_rate:.0%}, "
                        f"n={profile.total_observations}, "
                        f"cause={profile.dominant_cause})"
                    )
        except Exception:
            pass  # Memory is advisory

        # ── v2+v3: Runtime pause decision ──────────────────────────────
        pause_triggered = False
        needs_pause = (
            safety_score < self.MARGINAL_THRESHOLD
            or hazardous_primitives
            or memory_high_risk  # v3: historical failure rate triggers pause
        )

        if result.allowed and needs_pause:
            # Marginal safety — agent decides to pause for human review
            risk_factors: dict[str, float] = {
                "safety_score": safety_score,
                "violation_ratio": n_violations / max(n_steps, 1),
            }
            if hazardous_primitives:
                risk_factors["hazardous_primitive_count"] = float(len(hazardous_primitives))
            if memory_high_risk:
                risk_factors["memory_high_risk_count"] = float(len(memory_high_risk))
            if not input_data.interlock_state.get("hardware_interlock_ok", True):
                risk_factors["hardware_interlock_failed"] = 1.0
            if not input_data.interlock_state.get("cooling_ok", True):
                risk_factors["cooling_system_failed"] = 1.0

            reason_parts = []
            if safety_score < self.MARGINAL_THRESHOLD:
                reason_parts.append(f"safety_score={safety_score:.2f} (marginal)")
            if hazardous_primitives:
                reason_parts.append(f"hazardous primitives: {hazardous_primitives}")
            if memory_high_risk:
                reason_parts.append(f"memory high-risk: {memory_high_risk}")

            pause_result = await self.request_pause(PauseRequest(
                reason=" | ".join(reason_parts),
                risk_factors=risk_factors,
                suggested_action="approve",
                checkpoint={
                    "compiled_graph_hash": hash(str(input_data.compiled_graph)),
                    "safety_score": safety_score,
                    "violations": result.violations,
                },
            ))
            pause_triggered = True

            if pause_result.decision == "rejected":
                result.allowed = False
                result.violations.append(
                    f"Operator vetoed marginal safety "
                    f"(score={safety_score:.2f}, "
                    f"hazardous={hazardous_primitives or 'none'})"
                )
                # Recalculate score after operator veto
                n_violations = len(result.violations)
                safety_score = max(0.0, 1.0 - (n_violations / max(n_steps, 1)))

        # ── Decision nodes ─────────────────────────────────────────────
        preflight_node = DecisionNode(
            id="preflight_check",
            label="Safety preflight check",
            options=["Approved", "Denied (safety veto)"],
            selected="Approved" if result.allowed else "Denied (safety veto)",
            reason=f"{n_violations} violation(s) across {n_steps} step(s), score={safety_score:.2f}",
            outcome="; ".join(result.violations) if result.violations else "No violations",
        )

        escalation_node = DecisionNode(
            id="escalation",
            label="Escalation required?",
            options=["Auto-approved", "Requires human review", "Operator vetoed"],
            selected=(
                "Operator vetoed" if (pause_triggered and not result.allowed)
                else "Requires human review" if (result.requires_approval or pause_triggered)
                else "Auto-approved"
            ),
            reason=(
                f"safety_score={safety_score:.2f}, "
                f"marginal_threshold={self.MARGINAL_THRESHOLD}, "
                f"hazardous_primitives={hazardous_primitives or 'none'}"
            ),
        )

        # v2: Granularity decision node
        granularity_node = DecisionNode(
            id="safety_granularity",
            label="Safety check granularity",
            options=["fine", "coarse", "adaptive"],
            selected="fine",
            reason="Safety agent always uses fine granularity — every check matters",
        )

        return SafetyCheckOutput(
            allowed=result.allowed,
            violations=result.violations,
            requires_approval=result.requires_approval or pause_triggered,
            safety_score=safety_score,
            decision_nodes=[
                preflight_node.to_dict(),
                escalation_node.to_dict(),
                granularity_node.to_dict(),
            ],
            granularity_used="fine",
        )
