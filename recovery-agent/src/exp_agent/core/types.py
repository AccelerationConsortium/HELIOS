"""Core types for exp-agent using Pydantic v2."""

from typing import Literal, Optional, Any, List, Dict
from pydantic import BaseModel, Field, model_validator, ConfigDict

# Type aliases
Effect = Literal["read", "write"]
DecisionType = Literal["retry", "skip", "abort", "degrade"]
Severity = Literal["low", "medium", "high"]
Criticality = Literal["critical", "optional"]
OnFailure = Literal["abort", "retry", "skip"]
DeviceStatus = Literal["idle", "running", "error"]
ErrorType = Literal[
    # Core simulated/hardware-ish errors
    "overshoot",
    "timeout",
    "sensor_fail",
    "safety_violation",
    "postcondition_failed",
    "flow_blocked",
    "collision",
    "signal_saturated",
    "motor_stall",
    "encoder_error",
    "limit_exceeded",
    "lamp_failure",
    "low_signal",

    # Infrastructure / integration errors
    "communication_error",
    "driver_error",
    "invalid_action",
    "external_error",
    "execution_error",
    "degraded_execution_failed",
    "missing_dependency",
    "connection_failed",
    "not_connected",
    "command_failed",
    "protocol_error",
    "read_error",

    # Workflow/domain-ish errors
    "sample_contamination",

    # Chemical safety errors (Phase 2 - Safety Integration)
    # These errors trigger SafetyAgent veto power - RecoveryAgent
    # can only choose SAFE_SHUTDOWN, EVACUATE, or ASK_HUMAN
    "spill_detected",           # Chemical spill detected by sensors
    "leak_detected",            # Leak in system detected
    "exposure_detected",        # Personnel exposure to hazardous material
    "fire_detected",            # Fire detected
    "smoke_detected",           # Smoke detected
    "thermal_runaway",          # Uncontrolled temperature increase
    "incompatible_mix",         # Incompatible chemicals may have mixed
    "pressure_buildup",         # Unexpected pressure increase
    "off_gas_detected",         # Unexpected gas release
    "ventilation_failure",      # Required ventilation not working
    "containment_breach",       # Containment system compromised
    "chemical_threshold_exceeded",  # Chemical safety threshold violated
]

# Chemical safety error types that trigger SafetyAgent veto
CHEMICAL_SAFETY_ERRORS: set[str] = {
    "spill_detected",
    "leak_detected",
    "exposure_detected",
    "fire_detected",
    "smoke_detected",
    "thermal_runaway",
    "incompatible_mix",
    "pressure_buildup",
    "off_gas_detected",
    "ventilation_failure",
    "containment_breach",
    "chemical_threshold_exceeded",
}


class DeviceState(BaseModel):
    """State of a device at a point in time."""
    model_config = ConfigDict(frozen=False)

    name: str
    status: DeviceStatus = "idle"
    telemetry: Dict[str, Any] = Field(default_factory=dict)


class HardwareError(Exception):
    """Hardware error with context for recovery decisions.

    Note: This inherits from Exception (not BaseModel) to allow raising.
    Provides Pydantic-compatible serialization via model_dump().
    """

    def __init__(
        self,
        device: str,
        type: str,
        severity: Severity,
        message: str,
        when: str = "",
        action: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.device = device
        self.type = type
        self.message = message
        self.when = when
        self.action = action
        self.context = context or {}

        # Auto-escalate critical errors
        critical_errors = {"collision", "safety_violation"}
        if type in critical_errors and severity != "high":
            severity = "high"
        self.severity = severity

        super().__init__(self.message)

    def __str__(self) -> str:
        return f"HardwareError({self.device}): [{self.severity}] {self.type} - {self.message}"

    def model_dump(self) -> Dict[str, Any]:
        """Pydantic-compatible serialization."""
        return {
            "device": self.device,
            "type": self.type,
            "severity": self.severity,
            "message": self.message,
            "when": self.when,
            "action": self.action,
            "context": self.context
        }


class Action(BaseModel):
    """An action to be executed on a device."""
    model_config = ConfigDict(frozen=False)

    name: str
    effect: Effect
    params: Dict[str, Any] = Field(default_factory=dict)
    irreversible: bool = False
    preconditions: List[str] = Field(default_factory=list)
    postconditions: List[str] = Field(default_factory=list)
    safety_constraints: List[str] = Field(default_factory=list)
    device: Optional[str] = None


class Decision(BaseModel):
    """A recovery decision with rationale and actions."""
    model_config = ConfigDict(frozen=False)

    kind: DecisionType
    rationale: str
    actions: List[Action] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_abort_has_rationale(self) -> 'Decision':
        """Ensure ABORT decisions have clear rationale."""
        if self.kind == "abort" and len(self.rationale) < 10:
            raise ValueError("ABORT decisions require detailed rationale (min 10 chars)")
        return self


class ExecutionState(BaseModel):
    """Current state of experiment execution."""
    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    devices: Dict[str, DeviceState] = Field(default_factory=dict)
    irreversible_actions: List[str] = Field(default_factory=list)
    hazards: List[str] = Field(default_factory=list)
    last_error: Optional[Any] = None  # HardwareError (not a BaseModel)


# ============================================================================
# Workflow Plan Types
# ============================================================================

class PlanStep(BaseModel):
    """A step in a workflow plan with criticality and failure semantics."""
    model_config = ConfigDict(frozen=False)

    step_id: str
    stage: str  # e.g. "setup", "heating", "hold", "measure", "cooldown"
    action: Action
    criticality: Criticality = "critical"
    on_failure: OnFailure = "abort"
    max_retries: int = Field(default=2, ge=0, le=10)
    description: str = ""


class PlanPatch(BaseModel):
    """
    Produced when a degrade decision occurs.
    Tells downstream steps how to adjust their parameters and postconditions.
    """
    model_config = ConfigDict(frozen=False)

    # Parameter overrides: step_id -> {param_name: new_value}
    overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # Postcondition relaxations: step_id -> list of new postcondition strings
    relaxations: Dict[str, List[str]] = Field(default_factory=dict)
    # Human-readable notes about the degradation
    notes: List[str] = Field(default_factory=list)
    # The original target that was degraded
    original_target: Optional[float] = None
    # The new degraded target
    degraded_target: Optional[float] = None


# ============================================================================
# Recovery Policy Types (new)
# ============================================================================

class ErrorProfile(BaseModel):
    """Classification profile for an error type."""
    model_config = ConfigDict(frozen=False)

    unsafe: bool
    recoverable: bool
    default_strategy: DecisionType
    safe_shutdown_required: bool = False
    diagnostics: List[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_safety_consistency(self) -> 'ErrorProfile':
        """Ensure unrecoverable errors require safe shutdown."""
        if not self.recoverable and self.unsafe:
            object.__setattr__(self, 'safe_shutdown_required', True)
        return self


class SignatureResult(BaseModel):
    """Result of telemetry signature analysis."""
    model_config = ConfigDict(frozen=False)

    mode: Literal["drift", "oscillation", "stall", "noisy", "stable", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default_factory=dict)


class RecoveryDecision(BaseModel):
    """Full recovery decision with context."""
    model_config = ConfigDict(frozen=False)

    kind: DecisionType
    rationale: str
    actions: List[Action] = Field(default_factory=list)
    error_profile: Optional[ErrorProfile] = None
    signature: Optional[SignatureResult] = None
    degraded_target: Optional[float] = None
    sample_status: Literal["intact", "compromised", "destroyed", "anomalous"] = "intact"
