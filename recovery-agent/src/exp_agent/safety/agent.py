"""SafetyAgent protocol definition.

This module defines the interface that all safety agent implementations
must follow. The interface is minimal and designed for easy integration
with external safety systems like Safety SDL Agent.

Interface design based on plan.md section 4:
- assess(plan) -> SafetyPacket: Pre-flight safety assessment
- answer(question, context) -> SafetyGuidance: Runtime safety queries
"""

from typing import Protocol, Optional, Dict, Any, runtime_checkable
from exp_agent.core.safety_types import (
    SafetyPacket,
    SafetyGuidance,
    ExperimentSummary,
)


@runtime_checkable
class SafetyAgent(Protocol):
    """Protocol for chemical safety agent implementations.

    This is the minimal interface required for integrating a chemical
    safety agent with the disaster recovery system.

    Implementations:
    - MockSafetyAgent: Testing implementation with configurable responses
    - SafetySDLAgentAdapter: Adapter for Alan's Safety SDL Agent (Phase 2)

    Example usage:
        ```python
        # Pre-flight assessment
        experiment = ExperimentSummary(
            title="Ethanol evaporation",
            chemicals=[ChemicalInfo(name="Ethanol", cas_number="64-17-5")],
            parameters={"temperature": 78, "duration": "2 hours"},
        )
        packet = await safety_agent.assess(experiment)

        if packet.gate_decision == "deny":
            raise SafetyGateError(packet.gate_rationale)

        # Runtime query
        guidance = await safety_agent.answer(
            question="Temperature exceeded 85°C, what should I do?",
            context={"current_temp": 87, "target_temp": 78}
        )
        ```

    Integration with Recovery Agent:
        The SafetyPacket returned by assess() is used by:
        1. WorkflowSupervisor: Pre-flight gate decision
        2. GuardedExecutor: Runtime constraint checking
        3. RecoveryAgent: Action selection constraints

    Veto Rules (from plan.md section 5):
        Chemical safety events (spill/exposure/fire/overheat beyond threshold)
        trigger SafetyAgent or its rule layer with FINAL VETO POWER.
        RecoveryAgent can only choose SAFE_SHUTDOWN/EVACUATE/ASK_HUMAN.
    """

    async def assess(self, experiment: ExperimentSummary) -> SafetyPacket:
        """Perform pre-flight safety assessment.

        This method is called before a workflow starts to evaluate
        the safety of the planned experiment.

        Args:
            experiment: Summary of the experiment including chemicals,
                       procedure steps, and parameters.

        Returns:
            SafetyPacket containing:
            - gate_decision: "allow", "allow_with_constraints", or "deny"
            - hazards: Identified chemical hazards
            - ppe: Required personal protective equipment
            - monitoring: Variables to monitor with thresholds
            - thresholds: Safety thresholds for runtime checking
            - emergency_playbooks: Emergency response procedures
            - constraints: Runtime constraints for recovery actions

        Raises:
            SafetyAssessmentError: If assessment fails (e.g., network error)

        Note:
            This is an async method to support external API calls.
            Implementations should handle timeouts and retries internally.
        """
        ...

    async def answer(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SafetyGuidance:
        """Query safety agent about a specific situation.

        This method is used during runtime when the recovery agent
        needs guidance about a specific situation or action.

        Args:
            question: Natural language question about safety.
                     Examples:
                     - "Temperature exceeded 85°C, what should I do?"
                     - "Is it safe to continue heating after a 5 minute pause?"
                     - "What PPE is needed for cleaning a small ethanol spill?"
            context: Optional context dictionary with current state.
                    Useful keys: current_temp, pressure, chemicals_present,
                    last_action, error_type, etc.

        Returns:
            SafetyGuidance containing:
            - guidance: Human-readable safety advice
            - recommended_actions: Actions to take
            - prohibited_actions: Actions to avoid
            - requires_human: Whether human intervention needed
            - confidence: Confidence in the guidance (0-1)

        Note:
            This method should be used sparingly during normal operation
            as it may involve LLM calls. For common scenarios, use the
            emergency_playbooks from the SafetyPacket instead.
        """
        ...


class SafetyAgentError(Exception):
    """Base exception for safety agent errors."""
    pass


class SafetyAssessmentError(SafetyAgentError):
    """Error during safety assessment."""
    pass


class SafetyGateError(SafetyAgentError):
    """Experiment blocked by safety gate."""

    def __init__(self, rationale: str, packet: Optional[SafetyPacket] = None):
        self.rationale = rationale
        self.packet = packet
        super().__init__(f"Safety gate denied: {rationale}")


class SafetyConstraintViolation(SafetyAgentError):
    """Action blocked due to safety constraint violation."""

    def __init__(
        self,
        action_name: str,
        violated_constraints: list,
        rationale: str
    ):
        self.action_name = action_name
        self.violated_constraints = violated_constraints
        self.rationale = rationale
        super().__init__(
            f"Action '{action_name}' blocked: {rationale}"
        )
