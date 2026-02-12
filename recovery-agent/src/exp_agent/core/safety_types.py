"""Safety types for chemical safety integration.

This module defines schemas for integrating with external safety agents
(e.g., Alan's Safety SDL Agent) to provide chemical hazard awareness
and safety constraints for disaster recovery decisions.

Key concepts:
- SafetyPacket: Pre-flight assessment result containing hazards, PPE, thresholds, etc.
- SafetyConstraint: Runtime constraints that recovery actions must respect
- EmergencyPlaybook: Scenario-based emergency response procedures
- GateDecision: Pre-flight gate decision (ALLOW, ALLOW_WITH_CONSTRAINTS, DENY)
"""

from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# Gate Decision Types
# =============================================================================

GateDecision = Literal["allow", "allow_with_constraints", "deny"]
"""Pre-flight gate decision for experiment safety assessment."""

SafetyCheckResult = Literal["allow", "block", "require_human"]
"""Runtime safety check result for recovery actions."""

HazardSeverity = Literal["low", "medium", "high", "critical"]
"""Severity level for chemical hazards."""

ConstraintType = Literal[
    "temperature_limit",      # Max/min temperature bounds
    "pressure_limit",         # Max/min pressure bounds
    "rate_limit",             # Max addition/heating rate
    "ventilation_required",   # Fume hood or ventilation requirement
    "incompatibility",        # Chemical incompatibility warning
    "no_ignition_source",     # No open flames or sparks
    "no_heating",             # Heating prohibited
    "no_water",               # Water contact prohibited
    "inert_atmosphere",       # Requires inert gas blanket
    "cooling_required",       # Active cooling required
    "time_limit",             # Maximum operation duration
    "proximity_limit",        # Minimum safe distance
    "ppe_required",           # Specific PPE requirement
    "supervision_required",   # Human supervision required
    "evacuation_ready",       # Evacuation route must be clear
    "custom",                 # Custom constraint
]


# =============================================================================
# GHS Hazard Information
# =============================================================================

class GHSHazard(BaseModel):
    """GHS (Globally Harmonized System) hazard classification.

    Maps to ChemicalHazardInfo from Safety SDL Agent.
    """
    model_config = ConfigDict(frozen=False)

    cas_number: Optional[str] = Field(
        default=None,
        description="CAS Registry Number of the chemical"
    )
    chemical_name: Optional[str] = Field(
        default=None,
        description="Common or IUPAC name of the chemical"
    )
    ghs_codes: List[str] = Field(
        default_factory=list,
        description="GHS hazard codes (e.g., H225, H315, H318)"
    )
    precautionary_codes: List[str] = Field(
        default_factory=list,
        description="GHS precautionary codes (e.g., P264, P280)"
    )
    hazard_summary: Optional[str] = Field(
        default=None,
        description="Human-readable hazard summary"
    )
    toxicity_summary: Optional[str] = Field(
        default=None,
        description="Toxicity information summary"
    )
    stability_reactivity: Optional[str] = Field(
        default=None,
        description="Stability and reactivity information"
    )
    emergency_response: Optional[str] = Field(
        default=None,
        description="Emergency response guidance"
    )
    source: Optional[str] = Field(
        default=None,
        description="Data source (e.g., PubChem, SDS)"
    )


# =============================================================================
# PPE Requirements
# =============================================================================

class PPERequirement(BaseModel):
    """Personal Protective Equipment requirement with standards.

    Based on Safety SDL Agent's PPE output format with ANSI/EN/ISO codes.
    """
    model_config = ConfigDict(frozen=False)

    category: Literal[
        "eye_face", "hand", "respiratory", "foot", "head", "body", "hearing"
    ] = Field(description="PPE category")

    item: str = Field(
        description="Specific PPE item (e.g., 'Safety goggles', 'Nitrile gloves')"
    )
    standard: str = Field(
        description="Standard code (e.g., 'ANSI Z87.1 D3', 'EN 374-1 Type A')"
    )
    specification: Optional[str] = Field(
        default=None,
        description="Additional specs (e.g., 'breakthrough >60 min for acetone')"
    )
    mandatory: bool = Field(
        default=True,
        description="Whether this PPE is mandatory vs recommended"
    )


# =============================================================================
# Monitoring and Thresholds
# =============================================================================

class MonitoringItem(BaseModel):
    """Variable to monitor during experiment with thresholds.

    Based on SOP monitoring section from Safety SDL Agent.
    """
    model_config = ConfigDict(frozen=False)

    variable: str = Field(
        description="Variable to monitor (e.g., 'temperature', 'pressure')"
    )
    unit: str = Field(
        description="Unit of measurement (e.g., '°C', 'bar', 'mL/min')"
    )
    frequency: Optional[str] = Field(
        default=None,
        description="Monitoring frequency (e.g., 'continuous', 'every 5 min')"
    )
    normal_range: Optional[str] = Field(
        default=None,
        description="Expected normal range (e.g., '20-25°C')"
    )
    warning_threshold: Optional[float] = Field(
        default=None,
        description="Warning threshold value"
    )
    critical_threshold: Optional[float] = Field(
        default=None,
        description="Critical threshold value - requires immediate action"
    )
    action_on_warning: Optional[str] = Field(
        default=None,
        description="Action when warning threshold exceeded"
    )
    action_on_critical: Optional[str] = Field(
        default=None,
        description="Action when critical threshold exceeded"
    )


class SafetyThreshold(BaseModel):
    """Safety threshold with severity and actions.

    Used for runtime enforcement in recovery decisions.
    """
    model_config = ConfigDict(frozen=False)

    variable: str = Field(description="Variable name (e.g., 'temperature')")
    operator: Literal["<", "<=", ">", ">=", "==", "!=", "in_range"] = Field(
        description="Comparison operator"
    )
    value: float = Field(description="Threshold value")
    value_max: Optional[float] = Field(
        default=None,
        description="Upper bound for 'in_range' operator"
    )
    unit: str = Field(description="Unit of measurement")
    severity: HazardSeverity = Field(description="Severity if violated")
    action: str = Field(
        description="Required action if violated (e.g., 'stop_heating', 'evacuate')"
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Why this threshold exists"
    )


# =============================================================================
# Emergency Response
# =============================================================================

class EmergencyPlaybook(BaseModel):
    """Scenario-based emergency response procedure.

    Based on SOP emergency_response section from Safety SDL Agent.
    """
    model_config = ConfigDict(frozen=False)

    scenario: str = Field(
        description="Emergency scenario (e.g., 'skin_contact', 'spill', 'fire')"
    )
    severity: HazardSeverity = Field(
        default="high",
        description="Severity of this scenario"
    )
    immediate_actions: List[str] = Field(
        description="Immediate steps to take (short, actionable)"
    )
    follow_up_actions: List[str] = Field(
        default_factory=list,
        description="Follow-up actions after immediate response"
    )
    requires_evacuation: bool = Field(
        default=False,
        description="Whether this scenario requires evacuation"
    )
    requires_human: bool = Field(
        default=True,
        description="Whether human intervention is required"
    )
    recovery_possible: bool = Field(
        default=False,
        description="Whether automated recovery is possible after this"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional safety notes"
    )


# =============================================================================
# Safety Constraints
# =============================================================================

class SafetyConstraint(BaseModel):
    """Runtime safety constraint for recovery actions.

    These constraints are checked before executing any recovery action.
    """
    model_config = ConfigDict(frozen=False)

    type: ConstraintType = Field(description="Type of constraint")
    description: str = Field(description="Human-readable constraint description")
    parameter: Optional[str] = Field(
        default=None,
        description="Parameter this constraint applies to (e.g., 'temperature')"
    )
    value: Optional[Any] = Field(
        default=None,
        description="Constraint value (type depends on constraint type)"
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit for the value"
    )
    mandatory: bool = Field(
        default=True,
        description="If True, violation blocks action; if False, warning only"
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Safety rationale for this constraint"
    )
    source: Optional[str] = Field(
        default=None,
        description="Source of constraint (e.g., 'SDS', 'SOP', 'GHS')"
    )


# =============================================================================
# Experiment Summary (Input to Safety Agent)
# =============================================================================

class ChemicalInfo(BaseModel):
    """Chemical information for safety assessment."""
    model_config = ConfigDict(frozen=False)

    name: str = Field(description="Chemical name")
    cas_number: Optional[str] = Field(default=None, description="CAS number")
    smiles: Optional[str] = Field(default=None, description="SMILES notation")
    amount: Optional[str] = Field(default=None, description="Amount with unit")
    role: Optional[str] = Field(
        default=None,
        description="Role in experiment (e.g., 'solvent', 'reagent', 'catalyst')"
    )


class ExperimentSummary(BaseModel):
    """Experiment summary for pre-flight safety assessment.

    This is sent to the Safety Agent for evaluation.
    """
    model_config = ConfigDict(frozen=False)

    title: Optional[str] = Field(default=None, description="Experiment title")
    chemicals: List[ChemicalInfo] = Field(
        default_factory=list,
        description="List of chemicals involved"
    )
    procedure_steps: List[str] = Field(
        default_factory=list,
        description="High-level procedure steps"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key parameters (temperature, pressure, duration, etc.)"
    )
    equipment: List[str] = Field(
        default_factory=list,
        description="Equipment used (fume hood, closed system, etc.)"
    )
    environment: Optional[str] = Field(
        default=None,
        description="Environment description (e.g., 'fume hood', 'glovebox')"
    )


# =============================================================================
# Safety Packet (Main Output)
# =============================================================================

class SafetyPacket(BaseModel):
    """Complete safety assessment result from Safety Agent.

    This is the main integration point between the Safety Agent and
    the Disaster Recovery Agent. It contains all safety information
    needed for pre-flight gating and runtime constraint checking.

    Structure based on Safety SDL Agent outputs:
    - ChemicalHazardSummary → hazards
    - PersonalProtectiveEquipment → ppe
    - SOP.monitoring → monitoring
    - SOP.emergency_response → emergency_playbooks
    - Constraints derived from all above
    """
    model_config = ConfigDict(frozen=False)

    # Gate decision
    gate_decision: GateDecision = Field(
        description="Pre-flight gate decision"
    )
    gate_rationale: str = Field(
        default="",
        description="Rationale for gate decision"
    )

    # Hazard information
    hazards: List[GHSHazard] = Field(
        default_factory=list,
        description="Chemical hazards identified (GHS/H statements)"
    )
    overall_risk_level: HazardSeverity = Field(
        default="medium",
        description="Overall risk level of the experiment"
    )

    # PPE requirements
    ppe: List[PPERequirement] = Field(
        default_factory=list,
        description="Required personal protective equipment"
    )

    # Monitoring requirements
    monitoring: List[MonitoringItem] = Field(
        default_factory=list,
        description="Variables to monitor with thresholds"
    )

    # Safety thresholds for runtime enforcement
    thresholds: List[SafetyThreshold] = Field(
        default_factory=list,
        description="Safety thresholds for runtime checking"
    )

    # Emergency procedures
    emergency_playbooks: List[EmergencyPlaybook] = Field(
        default_factory=list,
        description="Emergency response procedures by scenario"
    )

    # Hard constraints
    constraints: List[SafetyConstraint] = Field(
        default_factory=list,
        description="Runtime constraints for recovery actions"
    )

    # SOP reference (if generated)
    sop_title: Optional[str] = Field(
        default=None,
        description="Title of generated SOP"
    )
    sop_procedure: List[str] = Field(
        default_factory=list,
        description="SOP procedure steps with safety limits"
    )

    # Metadata
    assessed_at: Optional[str] = Field(
        default=None,
        description="ISO timestamp of assessment"
    )
    assessment_source: Optional[str] = Field(
        default=None,
        description="Source of assessment (e.g., 'safety_sdl_agent')"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-blocking warnings from assessment"
    )


# =============================================================================
# Safety Guidance (Runtime Query Response)
# =============================================================================

class SafetyGuidance(BaseModel):
    """Response from Safety Agent for runtime queries.

    Used when the recovery agent needs to ask the safety agent
    about a specific situation or action.
    """
    model_config = ConfigDict(frozen=False)

    query: str = Field(description="Original query")
    guidance: str = Field(description="Safety guidance response")
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended actions"
    )
    prohibited_actions: List[str] = Field(
        default_factory=list,
        description="Actions that should NOT be taken"
    )
    requires_human: bool = Field(
        default=False,
        description="Whether human intervention is required"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in guidance"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Sources for this guidance"
    )


# =============================================================================
# Safety Check Result (Runtime Action Validation)
# =============================================================================

class ActionSafetyCheck(BaseModel):
    """Result of checking a recovery action against safety constraints.

    Used by GuardedExecutor.check_safety() with SafetyPacket.
    """
    model_config = ConfigDict(frozen=False)

    action_name: str = Field(description="Name of the action being checked")
    result: SafetyCheckResult = Field(description="Check result")
    violated_constraints: List[SafetyConstraint] = Field(
        default_factory=list,
        description="Constraints that would be violated"
    )
    violated_thresholds: List[SafetyThreshold] = Field(
        default_factory=list,
        description="Thresholds that would be violated"
    )
    triggered_playbooks: List[str] = Field(
        default_factory=list,
        description="Emergency playbook scenarios that apply"
    )
    rationale: str = Field(
        default="",
        description="Explanation of check result"
    )
    alternative_actions: List[str] = Field(
        default_factory=list,
        description="Suggested alternative actions if blocked"
    )
