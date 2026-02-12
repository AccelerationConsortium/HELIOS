"""
Recovery Agent - delegates to policy-driven decision engine.

This module provides backward compatibility while using the new policy system.

Phase 2 Safety Integration:
- Chemical safety events trigger SafetyAgent veto power
- For chemical safety errors, RecoveryAgent can only choose:
  SAFE_SHUTDOWN, EVACUATE, or ASK_HUMAN
- SafetyPacket constraints inform recovery decisions
"""
from typing import List, Dict, Optional, TYPE_CHECKING
from ..core.types import (
    DeviceState, HardwareError, Decision, Action,
    CHEMICAL_SAFETY_ERRORS,
)
from ..llm.advisor import LLMAdvisor
from ..llm.null_advisor import NullLLMAdvisor
from ..llm.types import LLMDecisionProposal
from .policy import (
    decide_recovery,
    analyze_signature,
    RecoveryConfig,
    RECOVERY_CONFIG,
)

if TYPE_CHECKING:
    from ..core.safety_types import SafetyPacket


# Actions allowed for chemical safety events (SafetyAgent veto)
CHEMICAL_SAFETY_ALLOWED_ACTIONS = {"safe_shutdown", "evacuate", "ask_human"}


class RecoveryAgent:
    """
    Agent responsible for recovery decisions.

    Phase 1: deterministic policy-driven decisions.
    Phase 2: optional LLM advisor can *propose* decisions, but policy remains
    the gatekeeper. Chemical safety events trigger SafetyAgent veto power.

    Delegates to the policy module for all decisions.
    Maintains retry counts and provides the familiar interface.

    Safety Integration:
        When a SafetyPacket is provided and the error is a chemical safety event,
        the agent enforces the SafetyAgent's veto power:
        - Only SAFE_SHUTDOWN, EVACUATE, or ASK_HUMAN decisions are allowed
        - Normal recovery strategies (RETRY, SKIP, DEGRADE) are blocked
        - This implements plan.md section 5's veto rules
    """

    def __init__(
        self,
        config: Optional[RecoveryConfig] = None,
        llm_advisor: Optional[LLMAdvisor] = None,
        safety_packet: Optional["SafetyPacket"] = None,
    ):
        """Initialize RecoveryAgent.

        Args:
            config: Recovery configuration (uses default if not provided).
            llm_advisor: Optional LLM advisor for Phase 2 proposals.
            safety_packet: Optional SafetyPacket for chemical safety constraints.
        """
        self.config = config or RECOVERY_CONFIG
        self.retry_counts: Dict[str, int] = {}
        self.safety_packet: Optional["SafetyPacket"] = safety_packet

        # LLM advisor is optional. Default is a no-op advisor.
        self.llm_advisor: LLMAdvisor = llm_advisor or NullLLMAdvisor()
        self.last_llm_proposal: Optional[LLMDecisionProposal] = None

    def set_safety_packet(self, packet: Optional["SafetyPacket"]) -> None:
        """Update the SafetyPacket for runtime constraint checking."""
        self.safety_packet = packet

    def decide(
        self,
        state: DeviceState,
        error: HardwareError,
        history: List[DeviceState] = None,
        last_action: Optional[Action] = None,
        stage: Optional[str] = None
    ) -> Decision:
        """
        Make a recovery decision based on current state and error.

        Args:
            state: Current device state
            error: The hardware error that occurred
            history: Recent telemetry history (optional)
            last_action: The action that failed (optional)
            stage: Current workflow stage (optional)

        Returns:
            Decision with kind, rationale, and actions

        Safety Integration:
            For chemical safety events (spill, fire, exposure, etc.),
            the SafetyAgent has veto power. The decision will be forced
            to SAFE_SHUTDOWN or ABORT with evacuation/human intervention.
        """
        # Update retry count
        err_key = error.type
        self.retry_counts[err_key] = self.retry_counts.get(err_key, 0) + 1

        # --- Check for chemical safety event (SafetyAgent veto) ---
        if self._is_chemical_safety_event(error, state):
            return self._handle_chemical_safety_event(error, state, history)

        # Delegate to policy (the gatekeeper)
        decision = decide_recovery(
            state=state,
            error=error,
            history=history or [],
            retry_counts=self.retry_counts,
            last_action=last_action,
            stage=stage,
            config=self.config,
        )

        # --- Apply SafetyPacket constraints to decision ---
        if self.safety_packet is not None:
            decision = self._apply_safety_constraints(decision, error, state)

        # Phase 2 hook: LLM can propose, but policy still decides.
        # We store the last proposal for logging/UI, but do not execute it directly.
        try:
            self.last_llm_proposal = self.llm_advisor.propose_recovery(
                state=state,
                error=error,
                history=history or [],
                retry_counts=self.retry_counts,
                last_action=last_action,
                stage=stage,
                baseline_decision=decision,
            )
        except Exception:
            # Advisor failures should never break recovery.
            self.last_llm_proposal = None

        # Reset retry count on abort (we're done with this error)
        if decision.kind == "abort":
            self.retry_counts[err_key] = 0

        return decision

    def _is_chemical_safety_event(
        self,
        error: HardwareError,
        state: DeviceState
    ) -> bool:
        """Check if error is a chemical safety event requiring veto.

        Chemical safety events give SafetyAgent FINAL VETO POWER.
        """
        # Check error type
        if error.type in CHEMICAL_SAFETY_ERRORS:
            return True

        # Check telemetry for chemical safety indicators
        telemetry = state.telemetry or {}
        chemical_indicators = [
            "spill_detected", "leak_detected", "fire_detected",
            "smoke_detected", "exposure_detected", "off_gas_detected",
        ]
        for indicator in chemical_indicators:
            if telemetry.get(indicator):
                return True

        # Check for threshold violations from SafetyPacket
        if self.safety_packet is not None:
            from ..safety.checker import check_chemical_safety_event
            required_action = check_chemical_safety_event(
                error.type,
                telemetry,
                self.safety_packet
            )
            if required_action is not None:
                return True

        return False

    def _handle_chemical_safety_event(
        self,
        error: HardwareError,
        state: DeviceState,
        history: Optional[List[DeviceState]]
    ) -> Decision:
        """Handle chemical safety event with SafetyAgent veto power.

        For chemical safety events, only these actions are allowed:
        - SAFE_SHUTDOWN
        - EVACUATE
        - ASK_HUMAN

        This implements plan.md section 5's veto rules.
        """
        from ..safety.checker import check_chemical_safety_event

        telemetry = state.telemetry or {}

        # Determine required action based on error type and telemetry
        required_action = "safe_shutdown"
        if self.safety_packet is not None:
            action = check_chemical_safety_event(
                error.type,
                telemetry,
                self.safety_packet
            )
            if action:
                required_action = action

        # Override based on error severity
        evacuation_errors = {
            "fire_detected", "smoke_detected", "exposure_detected",
            "off_gas_detected", "thermal_runaway",
        }
        if error.type in evacuation_errors:
            required_action = "evacuate"

        # Build decision
        rationale_parts = [
            f"CHEMICAL SAFETY EVENT: {error.type}",
            "SafetyAgent veto active - normal recovery strategies blocked",
            f"Required action: {required_action.upper()}",
        ]

        # Add emergency playbook info if available
        if self.safety_packet and self.safety_packet.emergency_playbooks:
            for playbook in self.safety_packet.emergency_playbooks:
                # Match playbook to error type
                if self._playbook_matches_error(playbook.scenario, error.type):
                    rationale_parts.append(f"Emergency procedure: {playbook.scenario}")
                    if playbook.immediate_actions:
                        rationale_parts.append(
                            f"Actions: {'; '.join(playbook.immediate_actions[:2])}"
                        )
                    break

        # Build recovery actions
        actions = []
        if required_action == "safe_shutdown":
            actions = [
                Action(
                    name="emergency_stop",
                    effect="write",
                    device=state.name,
                    params={"reason": error.type},
                    safety_constraints=["chemical_safety_event"],
                ),
                Action(
                    name="cool_down",
                    effect="write",
                    device=state.name,
                    postconditions=["telemetry.heating == False"],
                ),
            ]
        elif required_action == "evacuate":
            actions = [
                Action(
                    name="emergency_stop",
                    effect="write",
                    device=state.name,
                    params={"reason": error.type},
                    safety_constraints=["evacuation_required"],
                ),
                Action(
                    name="activate_alarm",
                    effect="write",
                    device=state.name,
                    params={"type": "evacuation"},
                ),
            ]

        print(f"\n  🚨 CHEMICAL SAFETY EVENT DETECTED: {error.type}")
        print(f"     SafetyAgent veto active - forcing {required_action.upper()}")

        return Decision(
            kind="abort",  # Chemical safety always aborts
            rationale=" | ".join(rationale_parts),
            actions=actions,
        )

    def _playbook_matches_error(self, scenario: str, error_type: str) -> bool:
        """Check if a playbook scenario matches an error type."""
        mappings = {
            "fire": ["fire_detected", "thermal_runaway"],
            "spill": ["spill_detected", "leak_detected", "containment_breach"],
            "exposure": ["exposure_detected"],
            "skin_contact": ["exposure_detected"],
            "eye_contact": ["exposure_detected"],
            "overheat": ["thermal_runaway", "chemical_threshold_exceeded"],
        }
        return error_type in mappings.get(scenario, [])

    def _apply_safety_constraints(
        self,
        decision: Decision,
        error: HardwareError,
        state: DeviceState
    ) -> Decision:
        """Apply SafetyPacket constraints to modify decision if needed.

        This checks if the proposed decision actions would violate
        safety constraints and adjusts accordingly.
        """
        if self.safety_packet is None:
            return decision

        from ..safety.checker import check_action_safety

        # Check each action in the decision
        for action in decision.actions:
            result = check_action_safety(action, self.safety_packet, state)

            if result.result == "block":
                # Action would violate safety - escalate to abort
                print(f"  ⚠ Recovery action '{action.name}' blocked by safety constraints")
                print(f"    Reason: {result.rationale}")

                # If we're trying to retry/degrade but it's blocked, escalate
                if decision.kind in ["retry", "degrade"]:
                    return Decision(
                        kind="abort",
                        rationale=f"Recovery blocked by safety: {result.rationale}",
                        actions=[
                            Action(
                                name="safe_shutdown",
                                effect="write",
                                device=state.name,
                                postconditions=["telemetry.heating == False"],
                            )
                        ],
                    )

        return decision

    def reset_retry_counts(self):
        """Reset all retry counts (e.g., after successful action)."""
        self.retry_counts.clear()

    def get_retry_count(self, error_type: str) -> int:
        """Get current retry count for an error type."""
        return self.retry_counts.get(error_type, 0)

    def analyze_fault_signature(self, history: List[DeviceState]) -> str:
        """
        Analyze fault signature from telemetry history.

        Returns: drift, oscillation, stall, noisy, stable, or unknown
        """
        result = analyze_signature(history)
        return result.mode
