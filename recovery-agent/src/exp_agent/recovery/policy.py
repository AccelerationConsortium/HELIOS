"""
Policy-driven recovery decision engine.

All recovery decisions are made here - single point of truth.
Replaces scattered if-else logic in RecoveryAgent and ErrorClassifier.

Now using Pydantic v2 for type-safe configuration and validation.
"""
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from ..core.types import (
    DeviceState, HardwareError, Action,
    ErrorProfile, SignatureResult, RecoveryDecision,
    DecisionType
)


# ============================================================================
# Configuration (Pydantic models - ready for YAML loading)
# ============================================================================

class SignatureConfig(BaseModel):
    """Thresholds for fault signature detection."""
    model_config = ConfigDict(frozen=False)

    # Drift detection
    drift_slope_threshold: float = Field(default=0.5, description="Slope threshold for drift detection")

    # Stall detection
    stall_epsilon: float = Field(default=0.1, description="No-change threshold for stall")
    stall_min_samples: int = Field(default=3, ge=2)

    # Oscillation detection
    oscillation_amplitude_threshold: float = Field(default=2.0, gt=0)
    oscillation_min_peaks: int = Field(default=2, ge=1)

    # Noise detection
    noise_variance_threshold: float = Field(default=1.5, gt=0)

    # General
    min_history_samples: int = Field(default=3, ge=2)


class RecoveryConfig(BaseModel):
    """Recovery behavior configuration."""
    model_config = ConfigDict(frozen=False)

    # Backoff timing (seconds)
    backoff_schedule: List[float] = Field(default=[0, 2, 5, 10])
    max_backoff: float = Field(default=10.0, ge=0)

    # Degradation
    default_degraded_temp: float = Field(default=110.0)
    drift_degrade_delta: float = Field(default=10.0, gt=0)
    default_degrade_delta: float = Field(default=15.0, gt=0)
    ambient_temp: float = Field(default=25.0)

    # Stabilization time (seconds)
    stabilize_time_oscillation: float = Field(default=10.0, ge=0)
    stabilize_time_drift: float = Field(default=5.0, ge=0)
    stabilize_time_default: float = Field(default=2.0, ge=0)

    # Retry limits
    max_retries_per_error: int = Field(default=3, ge=1, le=10)


# Global configs (will load from YAML later)
SIGNATURE_CONFIG = SignatureConfig()
RECOVERY_CONFIG = RecoveryConfig()


# ============================================================================
# Fault Signature Analysis
# ============================================================================

SignatureMode = Literal["drift", "oscillation", "stall", "noisy", "stable", "unknown"]


def analyze_signature(
    history: List[DeviceState],
    config: SignatureConfig = SIGNATURE_CONFIG,
    metric: Optional[str] = None
) -> SignatureResult:
    """
    Analyze telemetry history to detect fault signature.

    Args:
        history: List of device states
        config: Signature detection thresholds
        metric: Specific telemetry key to analyze (auto-detected if None)

    Returns one of: drift, oscillation, stall, noisy, stable, unknown
    """
    if not history or len(history) < config.min_history_samples:
        return SignatureResult(mode="unknown", confidence=0.0)

    # Auto-detect metric from first state with telemetry
    if metric is None:
        for s in history:
            if "temperature" in s.telemetry:
                metric = "temperature"
                break
            elif "flow_rate" in s.telemetry:
                metric = "flow_rate"
                break
            elif "pressure" in s.telemetry:
                metric = "pressure"
                break
            elif "x" in s.telemetry:
                metric = "x"  # Position
                break
            elif "signal_intensity" in s.telemetry:
                metric = "signal_intensity"
                break

    if metric is None:
        return SignatureResult(mode="unknown", confidence=0.0)

    # Extract readings for the chosen metric
    values = [
        s.telemetry.get(metric, 0)
        for s in history
        if metric in s.telemetry
    ]

    if len(values) < config.min_history_samples:
        return SignatureResult(mode="unknown", confidence=0.0)

    # Use 'values' instead of 'temps' for the rest of the analysis
    temps = values  # Keep variable name for minimal changes below

    details: Dict[str, Any] = {}

    # Calculate basic statistics
    n = len(temps)
    avg_temp = sum(temps) / n
    variance = sum((t - avg_temp) ** 2 for t in temps) / n
    details["variance"] = variance
    details["avg_temp"] = avg_temp

    # Calculate slope (linear trend)
    if n >= 2:
        avg_slope = (temps[-1] - temps[0]) / (n - 1) if n > 1 else 0
        details["avg_slope"] = avg_slope
    else:
        avg_slope = 0

    # Calculate differences for oscillation/stall detection
    diffs = [temps[i+1] - temps[i] for i in range(n-1)]
    details["diffs"] = diffs

    # 1. Check for STALL (no change)
    if all(abs(d) < config.stall_epsilon for d in diffs):
        return SignatureResult(
            mode="stall",
            confidence=0.9,
            details=details
        )

    # 2. Check for OSCILLATION first (sign changes with amplitude)
    sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
    max_amplitude = max(temps) - min(temps) if temps else 0
    details["sign_changes"] = sign_changes
    details["max_amplitude"] = max_amplitude

    if (sign_changes >= config.oscillation_min_peaks and
        max_amplitude > config.oscillation_amplitude_threshold):
        return SignatureResult(
            mode="oscillation",
            confidence=min(sign_changes / 4.0, 1.0),
            details=details
        )

    # 3. Check for DRIFT (consistent directional change)
    if abs(avg_slope) > config.drift_slope_threshold:
        return SignatureResult(
            mode="drift",
            confidence=min(abs(avg_slope) / (config.drift_slope_threshold * 2), 1.0),
            details=details
        )

    # 4. Check for NOISY (high variance but no clear pattern)
    if variance > config.noise_variance_threshold:
        return SignatureResult(
            mode="noisy",
            confidence=min(variance / (config.noise_variance_threshold * 2), 1.0),
            details=details
        )

    # 5. Otherwise STABLE
    return SignatureResult(
        mode="stable",
        confidence=0.8,
        details=details
    )


# ============================================================================
# Error Classification
# ============================================================================

def classify_error(error: HardwareError) -> ErrorProfile:
    """
    Classify error into actionable profile.

    Error types by device category:

    Heater:
    - safety_violation, overshoot: unsafe, may degrade
    - sensor_fail: non-recoverable, must abort

    Pump:
    - flow_blocked: unsafe, may recover via prime/reduce
    - pressure_drop: unsafe, may recover via reduce flow
    - leak_detected: unsafe, must abort
    - cavitation: recoverable, retry with reduce flow

    Positioner:
    - collision: unsafe, non-recoverable, abort
    - position_drift: recoverable, retry with recalibrate
    - motor_stall: unsafe, may recover via retract
    - limit_exceeded: recoverable, retry with reduced range
    - encoder_error: unsafe, must abort

    Spectrometer:
    - signal_saturated: recoverable, reduce integration
    - baseline_drift: recoverable, dark subtract
    - calibration_lost: recoverable, recalibrate
    - low_signal: recoverable, increase integration
    - lamp_failure: non-recoverable, abort

    General:
    - timeout, communication_error: transient, retry
    - postcondition_failed: retry with escalation
    - unknown: unsafe, abort
    """
    error_type = error.type

    # ========================================================================
    # Heater errors
    # ========================================================================
    if error_type in ["safety_violation", "overshoot"]:
        return ErrorProfile(
            unsafe=True,
            recoverable=True,
            default_strategy="degrade",
            safe_shutdown_required=True
        )

    if error_type == "sensor_fail":
        return ErrorProfile(
            unsafe=True,
            recoverable=False,
            default_strategy="abort",
            safe_shutdown_required=True
        )

    # ========================================================================
    # Pump errors
    # ========================================================================
    if error_type == "flow_blocked":
        return ErrorProfile(
            unsafe=True,
            recoverable=True,
            default_strategy="degrade",
            safe_shutdown_required=True,
            diagnostics=["check_pressure", "check_valve"]
        )

    if error_type == "pressure_drop":
        return ErrorProfile(
            unsafe=True,
            recoverable=True,
            default_strategy="degrade",
            safe_shutdown_required=False,
            diagnostics=["check_reservoir", "check_lines"]
        )

    if error_type == "leak_detected":
        return ErrorProfile(
            unsafe=True,
            recoverable=False,
            default_strategy="abort",
            safe_shutdown_required=True
        )

    if error_type == "cavitation":
        return ErrorProfile(
            unsafe=False,
            recoverable=True,
            default_strategy="retry",
            safe_shutdown_required=False,
            diagnostics=["prime_pump"]
        )

    # ========================================================================
    # Positioner errors
    # ========================================================================
    if error_type == "collision":
        return ErrorProfile(
            unsafe=True,
            recoverable=False,
            default_strategy="abort",
            safe_shutdown_required=True
        )

    if error_type == "position_drift":
        return ErrorProfile(
            unsafe=False,
            recoverable=True,
            default_strategy="retry",
            safe_shutdown_required=False,
            diagnostics=["home", "recalibrate"]
        )

    if error_type == "motor_stall":
        return ErrorProfile(
            unsafe=True,
            recoverable=True,
            default_strategy="degrade",
            safe_shutdown_required=True,
            diagnostics=["retract", "reduce_speed"]
        )

    if error_type == "limit_exceeded":
        return ErrorProfile(
            unsafe=False,
            recoverable=True,
            default_strategy="retry",
            safe_shutdown_required=False,
            diagnostics=["check_limits"]
        )

    if error_type == "encoder_error":
        return ErrorProfile(
            unsafe=True,
            recoverable=False,
            default_strategy="abort",
            safe_shutdown_required=True
        )

    # ========================================================================
    # Spectrometer errors
    # ========================================================================
    if error_type == "signal_saturated":
        return ErrorProfile(
            unsafe=False,
            recoverable=True,
            default_strategy="degrade",
            safe_shutdown_required=False,
            diagnostics=["reduce_integration"]
        )

    if error_type == "baseline_drift":
        return ErrorProfile(
            unsafe=False,
            recoverable=True,
            default_strategy="retry",
            safe_shutdown_required=False,
            diagnostics=["dark_subtract"]
        )

    if error_type == "calibration_lost":
        return ErrorProfile(
            unsafe=False,
            recoverable=True,
            default_strategy="retry",
            safe_shutdown_required=False,
            diagnostics=["recalibrate"]
        )

    if error_type == "low_signal":
        return ErrorProfile(
            unsafe=False,
            recoverable=True,
            default_strategy="degrade",
            safe_shutdown_required=False,
            diagnostics=["increase_integration", "check_lamp"]
        )

    if error_type == "lamp_failure":
        return ErrorProfile(
            unsafe=False,
            recoverable=False,
            default_strategy="abort",
            safe_shutdown_required=False
        )

    # ========================================================================
    # General/Transient errors
    # ========================================================================
    if error_type in ["timeout", "communication_error"]:
        return ErrorProfile(
            unsafe=False,
            recoverable=True,
            default_strategy="retry",
            safe_shutdown_required=False,
            diagnostics=["read_state"],
        )

    # Device/executor integration errors (usually transient-ish or infrastructure)
    if error_type in [
        "driver_error",
        "command_failed",
        "protocol_error",
        "read_error",
        "not_connected",
        "connection_failed",
        "missing_dependency",
        "external_error",
    ]:
        return ErrorProfile(
            unsafe=False,
            recoverable=True,
            default_strategy="retry",
            safe_shutdown_required=False,
            diagnostics=["read_state"],
        )

    # Pure execution/orchestration issues (likely code/config bugs)
    if error_type in ["invalid_action", "execution_error", "degraded_execution_failed"]:
        return ErrorProfile(
            unsafe=False,
            recoverable=False,
            default_strategy="abort",
            safe_shutdown_required=False,
        )

    # Domain-ish workflow errors (sample is compromised; continuing may be pointless)
    if error_type in ["sample_contamination"]:
        return ErrorProfile(
            unsafe=False,
            recoverable=False,
            default_strategy="abort",
            safe_shutdown_required=False,
        )

    if error_type == "postcondition_failed":
        return ErrorProfile(
            unsafe=False,
            recoverable=True,
            default_strategy="retry",
            safe_shutdown_required=False,
            diagnostics=["read_state"]
        )

    # Unknown - treat as unsafe
    return ErrorProfile(
        unsafe=True,
        recoverable=False,
        default_strategy="abort",
        safe_shutdown_required=True
    )


# ============================================================================
# Recovery Actions
# ============================================================================

def cool_down() -> Action:
    """Create a cool_down action."""
    return Action(name="cool_down", effect="write")


def set_temperature(temp: float) -> Action:
    """Create a set_temperature action."""
    return Action(name="set_temperature", effect="write", params={"temperature": temp})


def wait_action(seconds: float) -> Action:
    """Create a wait action."""
    return Action(name="wait", effect="write", params={"duration": seconds})


def compute_degraded_target(
    target: Optional[float],
    mode: SignatureMode,
    config: RecoveryConfig = RECOVERY_CONFIG
) -> float:
    """Compute degraded temperature target based on mode."""
    if target is None:
        return config.default_degraded_temp

    if mode in ["drift", "noisy"]:
        delta = config.drift_degrade_delta
    else:
        delta = config.default_degrade_delta

    return max(config.ambient_temp, target - delta)


def stabilize_time(
    mode: SignatureMode,
    config: RecoveryConfig = RECOVERY_CONFIG
) -> float:
    """Get stabilization wait time based on mode."""
    if mode == "oscillation":
        return config.stabilize_time_oscillation
    elif mode == "drift":
        return config.stabilize_time_drift
    else:
        return config.stabilize_time_default


def backoff(
    retry_count: int,
    config: RecoveryConfig = RECOVERY_CONFIG
) -> float:
    """Calculate backoff time based on retry count."""
    schedule = config.backoff_schedule
    idx = min(retry_count, len(schedule) - 1)
    return min(schedule[idx], config.max_backoff)


# ============================================================================
# Sample Status
# ============================================================================

SampleStatus = Literal["intact", "compromised", "destroyed", "anomalous"]


def determine_sample_status(
    profile: ErrorProfile,
    mode: SignatureMode,
    decision_kind: str,
    stage: Optional[str] = None
) -> SampleStatus:
    """Determine sample status after recovery decision."""
    if decision_kind == "abort" and profile.unsafe:
        return "destroyed"
    elif decision_kind == "abort":
        return "compromised"
    elif decision_kind == "degrade":
        return "compromised"
    else:
        return "intact"


# ============================================================================
# Main Decision Function
# ============================================================================

def decide_recovery(
    state: DeviceState,
    error: HardwareError,
    history: List[DeviceState],
    retry_counts: Dict[str, int],
    last_action: Optional[Action] = None,
    stage: Optional[str] = None,
    config: RecoveryConfig = RECOVERY_CONFIG
) -> RecoveryDecision:
    """
    Main recovery decision function - single point of truth.

    Args:
        state: Current device state
        error: The hardware error that occurred
        history: Recent telemetry history
        retry_counts: Count of retries per error type
        last_action: The action that failed (if any)
        stage: Current workflow stage (optional)
        config: Recovery configuration

    Returns:
        RecoveryDecision with kind, rationale, actions, and sample_status
    """
    # 1. Classify the error
    profile = classify_error(error)

    # 2. Analyze fault signature
    sig = analyze_signature(history)
    mode = sig.mode

    # 3. Extract target from last action if available
    target = None
    if last_action and last_action.params:
        target = last_action.params.get("temperature")

    # 4. Get retry count for this error type
    r = retry_counts.get(error.type, 0)

    print(f"[Policy] Error: {error.type} | Profile: unsafe={profile.unsafe}, recoverable={profile.recoverable}")
    print(f"[Policy] Signature: {mode} (confidence={sig.confidence:.2f})")
    print(f"[Policy] Retry count: {r} | Target: {target}")

    # ========================================================================
    # Decision Logic
    # ========================================================================

    # 0) UNSAFE PREEMPTION - always handle unsafe first
    if profile.unsafe:
        actions = [cool_down()]

        # Can we recover via degradation?
        if profile.recoverable and mode in ["drift", "noisy"] and stage != "cleanup":
            degraded = compute_degraded_target(target, mode, config)
            actions.append(set_temperature(degraded))
            actions.append(wait_action(stabilize_time(mode, config)))

            return RecoveryDecision(
                kind="degrade",
                rationale=f"Unsafe condition ({error.type}) with {mode} signature. Degrading to {degraded}°C.",
                actions=actions,
                error_profile=profile,
                signature=sig,
                degraded_target=degraded,
                sample_status="compromised"
            )

        # Cannot recover - abort
        return RecoveryDecision(
            kind="abort",
            rationale=f"Unsafe condition ({error.type}), cannot recover. Aborting.",
            actions=actions,
            error_profile=profile,
            signature=sig,
            sample_status="destroyed"
        )

    # 1) NON-RECOVERABLE
    if not profile.recoverable:
        return RecoveryDecision(
            kind="abort",
            rationale=f"Non-recoverable error ({error.type}). Aborting.",
            actions=[cool_down()],
            error_profile=profile,
            signature=sig,
            sample_status="compromised"
        )

    # 2) RECOVERABLE - choose strategy based on error type and mode

    # 2a) Transient errors (timeout, communication)
    if error.type in ["timeout", "communication_error"]:
        wait_time = backoff(r, config)
        return RecoveryDecision(
            kind="retry",
            rationale=f"Transient error ({error.type}). Retry after {wait_time}s backoff.",
            actions=[wait_action(wait_time)] if wait_time > 0 else [],
            error_profile=profile,
            signature=sig,
            sample_status="intact"
        )

    # 2b) Postcondition failures - escalation logic
    if error.type == "postcondition_failed":
        # Stall means device isn't responding - abort
        if mode == "stall":
            return RecoveryDecision(
                kind="abort",
                rationale="Postcondition failed with stall signature. Device unresponsive.",
                actions=[cool_down()],
                error_profile=profile,
                signature=sig,
                sample_status="compromised"
            )

        # First retry - immediate
        if r == 0:
            return RecoveryDecision(
                kind="retry",
                rationale="Postcondition failed. First retry attempt.",
                actions=[],
                error_profile=profile,
                signature=sig,
                sample_status="intact"
            )

        # Second retry - with short wait
        if r == 1:
            return RecoveryDecision(
                kind="retry",
                rationale="Postcondition failed again. Retry with 2s wait.",
                actions=[wait_action(2)],
                error_profile=profile,
                signature=sig,
                sample_status="intact"
            )

        # Third+ retry - degrade
        if r >= 2 and target is not None:
            degraded = compute_degraded_target(target, mode, config)
            return RecoveryDecision(
                kind="degrade",
                rationale=f"Repeated postcondition failures. Degrading to {degraded}°C.",
                actions=[cool_down(), set_temperature(degraded)],
                error_profile=profile,
                signature=sig,
                degraded_target=degraded,
                sample_status="compromised"
            )

    # 2c) Safety violations that are technically recoverable
    if error.type in ["safety_violation", "overshoot"]:
        if target is not None:
            degraded = compute_degraded_target(target, mode, config)
            return RecoveryDecision(
                kind="degrade",
                rationale=f"Safety condition ({error.type}). Degrading to {degraded}°C.",
                actions=[cool_down(), set_temperature(degraded)],
                error_profile=profile,
                signature=sig,
                degraded_target=degraded,
                sample_status="compromised"
            )

    # 3) FALLBACK - abort safely
    return RecoveryDecision(
        kind="abort",
        rationale=f"No matching recovery strategy for {error.type}. Aborting safely.",
        actions=[cool_down()],
        error_profile=profile,
        signature=sig,
        sample_status="compromised"
    )
