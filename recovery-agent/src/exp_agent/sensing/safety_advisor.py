"""
SafetyAdvisor integration for evidence-enhanced explanations.

When DEGRADED/INTERLOCKED occurs, this module:
1. Packages snapshot + incident + chemical context
2. Calls SafetyAgent.answer() for explanation
3. Returns explanation + suggestions + questions (never executable actions)

Key principle: SafetyAdvisor is proposal-only. It provides:
- Human-readable explanations for logs/notifications
- Suggested questions for human confirmation
- NEVER directly executable actions
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, Optional, Any, runtime_checkable

from exp_agent.sensing.safety_state import (
    SafetyStateUpdate,
    SafetyState,
    InterlockReason,
    EvidenceChain,
)
from exp_agent.sensing.protocol.snapshot import SystemSnapshot
from exp_agent.core.safety_types import SafetyPacket


@dataclass
class AdvisorQuery:
    """Query to SafetyAdvisor for explanation."""

    # Current state
    state_update: SafetyStateUpdate

    # Snapshot at time of incident
    snapshot: Optional[SystemSnapshot] = None

    # Chemical context from pre-flight assessment
    safety_packet: Optional[SafetyPacket] = None

    # Incident description
    incident_summary: str = ""

    # Specific question (optional)
    question: Optional[str] = None

    def to_prompt(self) -> str:
        """Generate a prompt for the safety advisor."""
        parts = []

        # State summary
        parts.append(f"Current safety state: {self.state_update.state.name}")
        parts.append(f"Reason: {self.state_update.reason.value}")

        # Interlocks
        if self.state_update.interlocks:
            parts.append("\nActive interlocks:")
            for interlock in self.state_update.interlocks:
                parts.append(f"  - {interlock.reason.value}: {interlock.message}")
                if interlock.current_value is not None:
                    parts.append(f"    Current value: {interlock.current_value}")
                if interlock.threshold is not None:
                    parts.append(f"    Threshold: {interlock.threshold}")

        # Evidence
        if self.state_update.evidence:
            parts.append(f"\nSnapshot ID: {self.state_update.evidence.snapshot_id}")
            parts.append(f"Trigger values: {self.state_update.evidence.trigger_values}")

        # Chemical context
        if self.safety_packet:
            parts.append("\nChemical context:")
            if self.safety_packet.hazards:
                parts.append(f"  GHS hazards: {', '.join(self.safety_packet.hazards)}")
            if self.safety_packet.constraints:
                parts.append("  Active constraints:")
                for c in self.safety_packet.constraints[:3]:  # Limit to first 3
                    parts.append(f"    - {c.constraint_type}: {c.description}")

        # Incident
        if self.incident_summary:
            parts.append(f"\nIncident: {self.incident_summary}")

        # Question
        if self.question:
            parts.append(f"\nQuestion: {self.question}")
        else:
            parts.append("\nPlease explain the situation and recommend next steps.")

        return "\n".join(parts)


@dataclass
class AdvisorResponse:
    """Response from SafetyAdvisor - explanation only, no executable actions."""

    # Human-readable explanation
    explanation: str

    # Suggested follow-up questions for human
    questions_for_human: list[str] = field(default_factory=list)

    # Risk assessment (for logging)
    risk_level: str = "unknown"  # low, medium, high, critical

    # Chemical-specific notes
    chemical_notes: Optional[str] = None

    # Suggested monitoring (what to watch)
    monitoring_suggestions: list[str] = field(default_factory=list)

    # Timestamp
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Source model (for audit)
    model_source: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "explanation": self.explanation,
            "questions_for_human": self.questions_for_human,
            "risk_level": self.risk_level,
            "chemical_notes": self.chemical_notes,
            "monitoring_suggestions": self.monitoring_suggestions,
            "ts": self.ts.isoformat(),
            "model_source": self.model_source,
        }


@runtime_checkable
class SafetyAdvisorProtocol(Protocol):
    """Protocol for SafetyAdvisor implementations."""

    async def explain(self, query: AdvisorQuery) -> AdvisorResponse:
        """
        Get explanation for a safety state.

        IMPORTANT: This should NEVER return executable actions.
        It provides explanation and questions only.
        """
        ...


class NullSafetyAdvisor:
    """No-op advisor for when no external advisor is available."""

    async def explain(self, query: AdvisorQuery) -> AdvisorResponse:
        """Return a basic explanation without external advisor."""
        state = query.state_update

        # Build basic explanation
        if state.state == SafetyState.SAFE:
            explanation = "System is operating normally."
        elif state.state == SafetyState.DEGRADED:
            explanation = (
                f"System is in degraded mode due to: {state.reason.value}. "
                "High-risk operations are blocked until the issue is resolved."
            )
        elif state.state == SafetyState.INTERLOCKED:
            explanation = (
                f"System is interlocked due to: {state.reason.value}. "
                "Operations have been stopped. Manual intervention may be required."
            )
        else:  # EMERGENCY
            explanation = (
                f"EMERGENCY: {state.reason.value}. "
                "Immediate human intervention required. "
                "Consider evacuation if personnel are at risk."
            )

        # Add interlock details
        if state.interlocks:
            details = []
            for i in state.interlocks:
                if i.current_value is not None:
                    details.append(f"{i.sensor_id}: {i.current_value} ({i.message})")
            if details:
                explanation += "\n\nActive conditions:\n" + "\n".join(f"  - {d}" for d in details)

        # Suggest questions
        questions = []
        if state.state >= SafetyState.INTERLOCKED:
            questions.append("Is it safe for personnel to approach the equipment?")
            questions.append("Should we initiate emergency shutdown procedure?")
        if state.reason in (InterlockReason.TEMPERATURE_HIGH, InterlockReason.TEMPERATURE_RUNAWAY):
            questions.append("Is active cooling available and should it be activated?")
        if state.reason in (InterlockReason.PRESSURE_HIGH, InterlockReason.PRESSURE_CRITICAL):
            questions.append("Is the pressure relief system functioning correctly?")

        return AdvisorResponse(
            explanation=explanation,
            questions_for_human=questions,
            risk_level="critical" if state.state >= SafetyState.EMERGENCY else (
                "high" if state.state >= SafetyState.INTERLOCKED else "medium"
            ),
            model_source="null_advisor",
        )


class SafetyAdvisorBridge:
    """
    Bridge between SafetyStateMachine and SafetyAgent.answer().

    This wraps the existing SafetyAgent to provide evidence-enhanced explanations.
    """

    def __init__(
        self,
        safety_agent=None,  # SafetyAgentProtocol from safety/agent.py
        fallback: Optional[SafetyAdvisorProtocol] = None,
    ):
        self._safety_agent = safety_agent
        self._fallback = fallback or NullSafetyAdvisor()

    async def explain(self, query: AdvisorQuery) -> AdvisorResponse:
        """
        Get explanation using SafetyAgent.answer() if available.

        Falls back to NullSafetyAdvisor if no agent configured.
        """
        if self._safety_agent is None:
            return await self._fallback.explain(query)

        try:
            # Build context for answer()
            context = {
                "state": query.state_update.state.name,
                "reason": query.state_update.reason.value,
                "interlocks": [i.to_dict() for i in query.state_update.interlocks],
                "evidence": query.state_update.evidence.to_dict() if query.state_update.evidence else None,
            }

            # Call safety agent
            question = query.question or query.to_prompt()
            guidance = await self._safety_agent.answer(question, context)

            # Convert guidance to AdvisorResponse
            return AdvisorResponse(
                explanation=guidance.answer if hasattr(guidance, 'answer') else str(guidance),
                questions_for_human=guidance.follow_up_questions if hasattr(guidance, 'follow_up_questions') else [],
                risk_level=guidance.risk_level if hasattr(guidance, 'risk_level') else "unknown",
                chemical_notes=guidance.chemical_notes if hasattr(guidance, 'chemical_notes') else None,
                model_source="safety_agent",
            )

        except Exception as e:
            # Log error and fall back
            print(f"SafetyAgent.answer() failed: {e}")
            response = await self._fallback.explain(query)
            response.explanation = f"[Fallback] {response.explanation}"
            return response

    def attach_to_state_machine(self, state_machine) -> None:
        """
        Attach to SafetyStateMachine to automatically explain state changes.

        Explanation is added to SafetyStateUpdate.explanation field.
        """
        import asyncio

        async def on_state_change(update: SafetyStateUpdate):
            if update.state >= SafetyState.DEGRADED:
                query = AdvisorQuery(state_update=update)
                response = await self.explain(query)
                update.explanation = response.explanation

        def sync_callback(update: SafetyStateUpdate):
            # Run async explain in background
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(on_state_change(update))
            except RuntimeError:
                # No running loop, run synchronously
                asyncio.run(on_state_change(update))

        state_machine.set_state_callback(sync_callback)
