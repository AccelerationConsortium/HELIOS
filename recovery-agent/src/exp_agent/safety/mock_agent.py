"""Mock SafetyAgent implementation for testing.

This module provides a configurable mock implementation of the SafetyAgent
protocol for unit testing and development without external dependencies.

Features:
- Configurable gate decisions and safety packets
- Predefined scenarios for common chemicals
- Support for custom response injection
"""

from typing import Optional, Dict, Any, List
from datetime import datetime

from exp_agent.core.safety_types import (
    SafetyPacket,
    SafetyGuidance,
    ExperimentSummary,
    GHSHazard,
    PPERequirement,
    MonitoringItem,
    SafetyThreshold,
    EmergencyPlaybook,
    SafetyConstraint,
    GateDecision,
    HazardSeverity,
)


class MockSafetyAgent:
    """Mock implementation of SafetyAgent for testing.

    This class provides configurable responses for testing the safety
    integration without requiring external API calls.

    Example usage:
        ```python
        # Default behavior (allow all)
        agent = MockSafetyAgent()

        # Deny all experiments
        agent = MockSafetyAgent(default_decision="deny")

        # Use predefined chemical profiles
        agent = MockSafetyAgent(use_chemical_profiles=True)

        # Inject custom response
        agent = MockSafetyAgent()
        agent.set_next_response(custom_packet)
        ```
    """

    # Predefined chemical hazard profiles
    CHEMICAL_PROFILES: Dict[str, GHSHazard] = {
        "ethanol": GHSHazard(
            cas_number="64-17-5",
            chemical_name="Ethanol",
            ghs_codes=["H225", "H319"],
            precautionary_codes=["P210", "P233", "P240", "P241", "P242", "P243"],
            hazard_summary="Highly flammable liquid and vapor. Causes serious eye irritation.",
            toxicity_summary="Low acute toxicity. May cause drowsiness or dizziness.",
            stability_reactivity="Stable under normal conditions. Avoid heat, sparks, open flames.",
            emergency_response="In case of fire: Use CO2, dry chemical, or alcohol-resistant foam.",
            source="PubChem",
        ),
        "acetone": GHSHazard(
            cas_number="67-64-1",
            chemical_name="Acetone",
            ghs_codes=["H225", "H319", "H336"],
            precautionary_codes=["P210", "P233", "P240", "P241", "P242", "P243", "P261", "P271"],
            hazard_summary="Highly flammable liquid and vapor. Causes serious eye irritation. May cause drowsiness.",
            toxicity_summary="Low acute toxicity. Central nervous system effects at high concentrations.",
            stability_reactivity="Stable under normal conditions. Keep away from oxidizers.",
            emergency_response="In case of fire: Use CO2, dry chemical, or alcohol-resistant foam.",
            source="PubChem",
        ),
        "sulfuric_acid": GHSHazard(
            cas_number="7664-93-9",
            chemical_name="Sulfuric Acid",
            ghs_codes=["H314", "H290"],
            precautionary_codes=["P260", "P264", "P280", "P301+P330+P331", "P303+P361+P353", "P304+P340", "P305+P351+P338"],
            hazard_summary="Causes severe skin burns and eye damage. May be corrosive to metals.",
            toxicity_summary="Highly corrosive. Inhalation may cause respiratory irritation.",
            stability_reactivity="Strong oxidizer. Reacts violently with water (exothermic). Incompatible with bases, metals, organics.",
            emergency_response="Skin contact: Immediately flush with plenty of water for at least 15 minutes. Seek medical attention.",
            source="PubChem",
        ),
        "hydrogen_peroxide": GHSHazard(
            cas_number="7722-84-1",
            chemical_name="Hydrogen Peroxide (30%)",
            ghs_codes=["H271", "H302", "H314", "H332"],
            precautionary_codes=["P210", "P220", "P280", "P303+P361+P353", "P305+P351+P338"],
            hazard_summary="May cause fire or explosion; strong oxidizer. Harmful if swallowed. Causes severe skin burns.",
            toxicity_summary="Corrosive to tissues. May cause systemic effects if ingested.",
            stability_reactivity="Strong oxidizer. May decompose explosively if heated. Incompatible with reducing agents, combustibles.",
            emergency_response="Fire: Flood with water. Do NOT use dry chemicals. Evacuate if large spill.",
            source="PubChem",
        ),
    }

    # Default PPE for different risk levels
    DEFAULT_PPE: Dict[HazardSeverity, List[PPERequirement]] = {
        "low": [
            PPERequirement(
                category="eye_face",
                item="Safety glasses",
                standard="ANSI Z87.1",
                mandatory=True,
            ),
            PPERequirement(
                category="body",
                item="Lab coat",
                standard="ASTM F1001",
                mandatory=True,
            ),
        ],
        "medium": [
            PPERequirement(
                category="eye_face",
                item="Safety goggles",
                standard="ANSI Z87.1 D3",
                specification="Splash protection",
                mandatory=True,
            ),
            PPERequirement(
                category="hand",
                item="Nitrile gloves",
                standard="EN 374-1 Type B",
                specification="Chemical resistant, min 0.1mm",
                mandatory=True,
            ),
            PPERequirement(
                category="body",
                item="Lab coat",
                standard="ASTM F1001",
                mandatory=True,
            ),
        ],
        "high": [
            PPERequirement(
                category="eye_face",
                item="Face shield + safety goggles",
                standard="ANSI Z87.1 D3 + Z87+",
                specification="Full face protection",
                mandatory=True,
            ),
            PPERequirement(
                category="hand",
                item="Nitrile gloves (double)",
                standard="EN 374-1 Type A",
                specification="Chemical resistant, min 0.2mm, breakthrough >60 min",
                mandatory=True,
            ),
            PPERequirement(
                category="body",
                item="Chemical-resistant apron",
                standard="ISO 16602 Type 6",
                mandatory=True,
            ),
            PPERequirement(
                category="respiratory",
                item="Half-face respirator",
                standard="NIOSH N95",
                specification="Organic vapor cartridge",
                mandatory=False,
            ),
        ],
        "critical": [
            PPERequirement(
                category="eye_face",
                item="Face shield + chemical goggles",
                standard="ANSI Z87.1 D3 + Z87+",
                specification="Full face, indirect vent",
                mandatory=True,
            ),
            PPERequirement(
                category="hand",
                item="Butyl rubber gloves",
                standard="EN 374-1 Type A",
                specification="Acid resistant, min 0.4mm",
                mandatory=True,
            ),
            PPERequirement(
                category="body",
                item="Full chemical suit",
                standard="ISO 16602 Type 3",
                specification="Liquid tight",
                mandatory=True,
            ),
            PPERequirement(
                category="respiratory",
                item="Full-face respirator or SCBA",
                standard="NIOSH P100 or SCBA",
                specification="Multi-gas cartridge",
                mandatory=True,
            ),
        ],
    }

    def __init__(
        self,
        default_decision: GateDecision = "allow",
        default_risk_level: HazardSeverity = "medium",
        use_chemical_profiles: bool = True,
        latency_ms: int = 0,
    ):
        """Initialize MockSafetyAgent.

        Args:
            default_decision: Default gate decision when no specific rules match.
            default_risk_level: Default risk level for experiments.
            use_chemical_profiles: If True, use predefined chemical hazard profiles.
            latency_ms: Simulated latency in milliseconds (for testing async behavior).
        """
        self.default_decision = default_decision
        self.default_risk_level = default_risk_level
        self.use_chemical_profiles = use_chemical_profiles
        self.latency_ms = latency_ms

        # For response injection
        self._next_packet: Optional[SafetyPacket] = None
        self._next_guidance: Optional[SafetyGuidance] = None

        # For call tracking
        self.assess_calls: List[ExperimentSummary] = []
        self.answer_calls: List[Dict[str, Any]] = []

    def set_next_response(
        self,
        packet: Optional[SafetyPacket] = None,
        guidance: Optional[SafetyGuidance] = None
    ) -> None:
        """Inject custom response for next call.

        Args:
            packet: Custom SafetyPacket for next assess() call.
            guidance: Custom SafetyGuidance for next answer() call.
        """
        self._next_packet = packet
        self._next_guidance = guidance

    def reset(self) -> None:
        """Reset mock state."""
        self._next_packet = None
        self._next_guidance = None
        self.assess_calls.clear()
        self.answer_calls.clear()

    async def assess(self, experiment: ExperimentSummary) -> SafetyPacket:
        """Perform mock safety assessment."""
        import asyncio
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)

        self.assess_calls.append(experiment)

        # Return injected response if available
        if self._next_packet is not None:
            packet = self._next_packet
            self._next_packet = None
            return packet

        # Build response based on experiment
        hazards: List[GHSHazard] = []
        risk_level = self.default_risk_level

        # Identify chemicals and get hazard profiles
        if self.use_chemical_profiles:
            for chem in experiment.chemicals:
                name_lower = chem.name.lower().replace(" ", "_").replace("-", "_")
                if name_lower in self.CHEMICAL_PROFILES:
                    hazards.append(self.CHEMICAL_PROFILES[name_lower])
                    # Escalate risk based on hazards
                    profile = self.CHEMICAL_PROFILES[name_lower]
                    if any(code in profile.ghs_codes for code in ["H271", "H314", "H330"]):
                        risk_level = "critical"
                    elif any(code in profile.ghs_codes for code in ["H225", "H300", "H310"]):
                        if risk_level in ["low", "medium"]:
                            risk_level = "high"

        # Check temperature constraints
        temp = experiment.parameters.get("temperature")
        constraints: List[SafetyConstraint] = []
        thresholds: List[SafetyThreshold] = []
        monitoring: List[MonitoringItem] = []

        if temp is not None:
            # Add temperature monitoring
            monitoring.append(MonitoringItem(
                variable="temperature",
                unit="°C",
                frequency="continuous",
                normal_range=f"{temp-5} - {temp+5}°C",
                warning_threshold=temp + 10,
                critical_threshold=temp + 20,
                action_on_warning="Reduce heating rate",
                action_on_critical="Stop heating immediately",
            ))

            # Add temperature threshold
            thresholds.append(SafetyThreshold(
                variable="temperature",
                operator=">",
                value=temp + 20,
                unit="°C",
                severity="critical",
                action="stop_heating",
                rationale=f"Temperature must not exceed {temp + 20}°C",
            ))

            # Check for flammable solvents
            if any(h.ghs_codes and "H225" in h.ghs_codes for h in hazards):
                # Add constraint for flammable materials
                constraints.append(SafetyConstraint(
                    type="temperature_limit",
                    description=f"Temperature limited due to flammable solvent",
                    parameter="temperature",
                    value=80,  # Below common solvent flash points
                    unit="°C",
                    mandatory=True,
                    rationale="Flammable solvent present - limit temperature",
                    source="GHS H225",
                ))

        # Add ventilation constraint for volatile chemicals
        if any(h.ghs_codes and any(c in h.ghs_codes for c in ["H225", "H336"]) for h in hazards):
            constraints.append(SafetyConstraint(
                type="ventilation_required",
                description="Fume hood required for volatile/flammable materials",
                mandatory=True,
                rationale="Volatile or flammable chemicals present",
                source="SOP",
            ))

        # Build emergency playbooks
        playbooks: List[EmergencyPlaybook] = [
            EmergencyPlaybook(
                scenario="skin_contact",
                severity="high",
                immediate_actions=[
                    "Remove contaminated clothing",
                    "Flush skin with water for at least 15 minutes",
                    "Seek medical attention if irritation persists",
                ],
                requires_evacuation=False,
                requires_human=True,
                recovery_possible=True,
            ),
            EmergencyPlaybook(
                scenario="eye_contact",
                severity="high",
                immediate_actions=[
                    "Immediately flush eyes with water for at least 15 minutes",
                    "Hold eyelids open during flushing",
                    "Seek immediate medical attention",
                ],
                requires_evacuation=False,
                requires_human=True,
                recovery_possible=False,
            ),
            EmergencyPlaybook(
                scenario="spill",
                severity="medium",
                immediate_actions=[
                    "Alert personnel in the area",
                    "Evacuate if large spill or toxic fumes",
                    "Use appropriate absorbent material",
                    "Dispose as hazardous waste",
                ],
                requires_evacuation=False,  # Depends on size
                requires_human=True,
                recovery_possible=True,
            ),
        ]

        # Add fire playbook for flammable materials
        if any(h.ghs_codes and "H225" in h.ghs_codes for h in hazards):
            playbooks.append(EmergencyPlaybook(
                scenario="fire",
                severity="critical",
                immediate_actions=[
                    "Activate fire alarm",
                    "Evacuate the area immediately",
                    "Use CO2 or dry chemical extinguisher if small fire",
                    "Do NOT use water on chemical fires",
                    "Call emergency services",
                ],
                requires_evacuation=True,
                requires_human=True,
                recovery_possible=False,
            ))

        # Determine gate decision
        decision = self.default_decision
        rationale = "Assessment completed successfully."

        if risk_level == "critical" and decision == "allow":
            decision = "allow_with_constraints"
            rationale = "High-risk experiment allowed with strict safety constraints."

        if any(h.ghs_codes and "H330" in h.ghs_codes for h in hazards):
            # Fatal if inhaled - deny without proper containment
            if experiment.environment not in ["glovebox", "fume_hood", "closed_system"]:
                decision = "deny"
                rationale = "Highly toxic chemical requires glovebox or closed system."

        return SafetyPacket(
            gate_decision=decision,
            gate_rationale=rationale,
            hazards=hazards,
            overall_risk_level=risk_level,
            ppe=self.DEFAULT_PPE.get(risk_level, self.DEFAULT_PPE["medium"]),
            monitoring=monitoring,
            thresholds=thresholds,
            emergency_playbooks=playbooks,
            constraints=constraints,
            assessed_at=datetime.now().isoformat(),
            assessment_source="mock_safety_agent",
        )

    async def answer(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SafetyGuidance:
        """Provide mock safety guidance."""
        import asyncio
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)

        self.answer_calls.append({"question": question, "context": context})

        # Return injected response if available
        if self._next_guidance is not None:
            guidance = self._next_guidance
            self._next_guidance = None
            return guidance

        # Simple keyword-based responses
        question_lower = question.lower()
        context = context or {}

        if "temperature" in question_lower and "exceed" in question_lower:
            return SafetyGuidance(
                query=question,
                guidance="Temperature exceedance detected. Reduce heating immediately and allow system to stabilize.",
                recommended_actions=[
                    "Turn off or reduce heating",
                    "Monitor temperature closely",
                    "Wait for temperature to return to safe range",
                    "Investigate cause of exceedance",
                ],
                prohibited_actions=[
                    "Continue heating",
                    "Increase heating rate",
                    "Add reactive chemicals",
                ],
                requires_human=False,
                confidence=0.9,
                sources=["SOP", "Safety best practices"],
            )

        if "spill" in question_lower:
            return SafetyGuidance(
                query=question,
                guidance="Spill detected. Evacuate non-essential personnel and contain the spill.",
                recommended_actions=[
                    "Alert others in the area",
                    "Put on appropriate PPE",
                    "Use absorbent material to contain",
                    "Dispose as hazardous waste",
                ],
                prohibited_actions=[
                    "Ignore the spill",
                    "Use inappropriate cleaning materials",
                    "Pour down drain",
                ],
                requires_human=True,
                confidence=0.85,
                sources=["Emergency response SOP"],
            )

        # Default response
        return SafetyGuidance(
            query=question,
            guidance="Please consult the SOP or contact the safety officer for specific guidance.",
            recommended_actions=["Review SOP", "Contact safety officer if uncertain"],
            prohibited_actions=[],
            requires_human=True,
            confidence=0.5,
            sources=["General safety guidelines"],
        )
