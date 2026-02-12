"""Error type mapping for RecoveryAgent integration.

Maps Python exceptions and OTbot errors to recovery-agent error types
for consistent error recovery strategies.
"""
from typing import Any


# Python exception to recovery-agent error type mapping
EXCEPTION_TYPE_MAP = {
    # Connection/Network errors
    "ConnectionError": "connection_lost",
    "ConnectionRefusedError": "connection_lost",
    "ConnectionResetError": "connection_lost",
    "ConnectionAbortedError": "connection_lost",
    "BrokenPipeError": "connection_lost",
    "TimeoutError": "timeout",
    "asyncio.TimeoutError": "timeout",

    # Hardware/Device errors
    "IOError": "sensor_fail",
    "OSError": "sensor_fail",
    "RuntimeError": "actuator_jam",
    "DeviceError": "sensor_fail",

    # Data/Validation errors
    "ValueError": "postcondition_failed",
    "TypeError": "postcondition_failed",
    "AssertionError": "safety_violation",

    # Resource errors
    "MemoryError": "resource_exhausted",
    "OverflowError": "overshoot",

    # Safety errors
    "SafetyError": "safety_violation",
    "ChemicalSafetyError": "spill_detected",

    # Generic errors
    "Exception": "unknown_error",
}


# OTbot-specific error codes to recovery-agent types
OTBOT_ERROR_MAP = {
    # Execution errors
    "execution_failed": "unknown_error",
    "protocol_failed": "postcondition_failed",
    "validation_failed": "postcondition_failed",

    # Hardware errors
    "robot_error": "actuator_jam",
    "pipette_error": "actuator_jam",
    "deck_error": "postcondition_failed",
    "tip_error": "actuator_jam",

    # Safety errors
    "safety_veto": "safety_violation",
    "threshold_exceeded": "chemical_threshold_exceeded",

    # QC errors
    "qc_failed": "postcondition_failed",
    "sensing_failed": "sensor_fail",

    # Timeout/Connection
    "timeout": "timeout",
    "connection_error": "connection_lost",

    # Resource errors
    "out_of_tips": "resource_exhausted",
    "out_of_labware": "resource_exhausted",
}


# Error severity mapping based on error type
ERROR_SEVERITY_MAP = {
    # High severity (critical/unsafe)
    "spill_detected": "high",
    "fire_detected": "high",
    "exposure_detected": "high",
    "thermal_runaway": "high",
    "containment_breach": "high",
    "safety_violation": "high",
    "chemical_threshold_exceeded": "high",

    # Medium severity (problematic but recoverable)
    "sensor_fail": "medium",
    "actuator_jam": "medium",
    "connection_lost": "medium",
    "postcondition_failed": "medium",
    "overshoot": "medium",

    # Low severity (transient/minor)
    "timeout": "low",
    "sensor_drift": "low",
    "unknown_error": "low",
}


def map_exception_to_error_type(exc: Exception) -> str:
    """Map Python exception to recovery-agent error type.

    Args:
        exc: The exception to map

    Returns:
        Recovery-agent error type string
    """
    exc_type = type(exc).__name__

    # Try exact match
    if exc_type in EXCEPTION_TYPE_MAP:
        return EXCEPTION_TYPE_MAP[exc_type]

    # Try parent class match
    for base_class in type(exc).__mro__[1:]:  # Skip the class itself
        base_name = base_class.__name__
        if base_name in EXCEPTION_TYPE_MAP:
            return EXCEPTION_TYPE_MAP[base_name]

    # Default to unknown_error
    return "unknown_error"


def map_otbot_error_to_type(error_code: str) -> str:
    """Map OTbot error code to recovery-agent error type.

    Args:
        error_code: OTbot error code string

    Returns:
        Recovery-agent error type string
    """
    return OTBOT_ERROR_MAP.get(error_code, "unknown_error")


def get_error_severity(error_type: str) -> str:
    """Get severity level for an error type.

    Args:
        error_type: Recovery-agent error type

    Returns:
        Severity level: "low", "medium", or "high"
    """
    return ERROR_SEVERITY_MAP.get(error_type, "medium")


def extract_error_context(exc: Exception) -> dict[str, Any]:
    """Extract relevant context from exception for recovery decision.

    Args:
        exc: The exception to extract context from

    Returns:
        Dictionary with error context (telemetry, traceback info, etc.)
    """
    context = {
        "exception_type": type(exc).__name__,
        "message": str(exc),
    }

    # Extract specific attributes if available
    if hasattr(exc, "__cause__") and exc.__cause__:
        context["cause"] = str(exc.__cause__)

    if hasattr(exc, "__context__") and exc.__context__:
        context["context"] = str(exc.__context__)

    # Extract custom attributes (e.g., from OTbot custom exceptions)
    for attr in ["device", "stage", "run_id", "candidate_id"]:
        if hasattr(exc, attr):
            context[attr] = getattr(exc, attr)

    return context


def should_emit_chemical_safety_alert(error_type: str, telemetry: dict[str, Any]) -> bool:
    """Check if error requires chemical safety alert.

    Args:
        error_type: Recovery-agent error type
        telemetry: Current telemetry data

    Returns:
        True if chemical safety alert should be emitted
    """
    # Chemical safety error types
    chemical_errors = {
        "spill_detected",
        "fire_detected",
        "exposure_detected",
        "thermal_runaway",
        "containment_breach",
        "chemical_threshold_exceeded",
        "off_gas_detected",
    }

    if error_type in chemical_errors:
        return True

    # Check telemetry for chemical safety indicators
    chemical_indicators = [
        "spill_detected",
        "leak_detected",
        "fire_detected",
        "smoke_detected",
        "exposure_detected",
        "off_gas_detected",
    ]

    for indicator in chemical_indicators:
        if telemetry.get(indicator):
            return True

    # Check for threshold violations
    temp = telemetry.get("temperature")
    if temp and temp > 80.0:  # High temperature threshold
        return True

    pressure = telemetry.get("pressure")
    if pressure and pressure > 2.0:  # High pressure threshold
        return True

    return False


# OTbot device name normalization
DEVICE_NAME_MAP = {
    "opentrons": "opentrons_ot2",
    "ot2": "opentrons_ot2",
    "ot-2": "opentrons_ot2",
    "plc": "plc_controller",
    "relay": "relay_controller",
    "squidstat": "squidstat_controller",
    "temperature": "plc_controller",
    "heating": "plc_controller",
}


def normalize_device_name(device: str) -> str:
    """Normalize device name for consistency.

    Args:
        device: Raw device name

    Returns:
        Normalized device name
    """
    device_lower = device.lower().strip()
    return DEVICE_NAME_MAP.get(device_lower, device_lower)
