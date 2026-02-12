from dataclasses import dataclass, field
from typing import Literal, List, Optional
from ..core.types import HardwareError, Action

Recoverability = Literal["recoverable", "non_recoverable", "unsafe"]

@dataclass
class ErrorProfile:
    recoverability: Recoverability
    recommended_actions: List[str] # names of actions e.g. "cool_down", "retry"
    safe_shutdown_required: bool = False
    diagnostics: List[str] = field(default_factory=list) # names of diagnostic actions

class ErrorClassifier:
    def classify(self, error: HardwareError) -> ErrorProfile:
        """Maps raw hardware errors to actionable profiles."""
        
        # 1. Safety Violations (Critical)
        if error.type in ["safety_violation", "overshoot"]:
            return ErrorProfile(
                recoverability="recoverable", # Technically we can recover by cooling, but it's a "violation"
                # Actually, overshoot usually requires degrade or abort. 
                # Let's say it's recoverable-via-degrade.
                recommended_actions=["cool_down", "degrade"],
                safe_shutdown_required=True # If recovery fails, must shutdown
            )

        # 2. Sensor Failures (Hardware Broken)
        if error.type == "sensor_fail":
            return ErrorProfile(
                recoverability="non_recoverable",
                recommended_actions=["cool_down", "abort"],
                safe_shutdown_required=True
            )

        # 3. Transients (Timeout, Comms)
        if error.type in ["timeout", "communication_error"]:
            return ErrorProfile(
                recoverability="recoverable",
                recommended_actions=["retry", "wait"],
                safe_shutdown_required=False,
                diagnostics=["read_state"] # Check if it was just a blip
            )
            
        # 4. Logic Errors
        if error.type == "postcondition_failed":
            # Action didn't work. Try again?
            return ErrorProfile(
                recoverability="recoverable",
                recommended_actions=["retry"],
                safe_shutdown_required=False,
                diagnostics=["read_state"]
            )

        # Default
        return ErrorProfile(
            recoverability="unsafe", 
            recommended_actions=["cool_down", "abort"], 
            safe_shutdown_required=True
        )
