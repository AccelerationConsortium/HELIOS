"""Sensing/QA Agent — mid-run quality control.

Monitors instrument readings during execution and performs
quality gate checks after each step completes. Can flag
suspect data, detect anomalies, and recommend pause/abort.

Sits in L0 (execution layer) as a runtime monitor.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QC check definitions
# ---------------------------------------------------------------------------

class QCCheck(BaseModel):
    """A single quality control check definition."""
    name: str
    metric: str  # which metric to check
    min_value: float | None = None
    max_value: float | None = None
    expected_value: float | None = None
    tolerance_pct: float = 10.0  # acceptable deviation from expected


class QCResult(BaseModel):
    """Result of a single QC check."""
    check_name: str
    passed: bool
    actual_value: float | None = None
    expected_value: float | None = None
    threshold_min: float | None = None
    threshold_max: float | None = None
    message: str = ""
    severity: Literal["info", "warning", "critical"] = "info"


# ---------------------------------------------------------------------------
# Step-level QC input/output
# ---------------------------------------------------------------------------

class SensingInput(BaseModel):
    """Input for post-step quality check."""
    step_key: str
    primitive: str
    params: dict[str, Any] = Field(default_factory=dict)
    step_result: dict[str, Any] = Field(default_factory=dict)
    policy_snapshot: dict[str, Any] = Field(default_factory=dict)
    qc_checks: list[QCCheck] = Field(default_factory=list)
    # Running stats from previous steps
    step_history: list[dict[str, Any]] = Field(default_factory=list)


class SensingOutput(BaseModel):
    """Output from post-step quality check."""
    overall_quality: Literal["good", "suspect", "failed"] = "good"
    checks: list[QCResult] = Field(default_factory=list)
    anomalies_detected: list[str] = Field(default_factory=list)
    recommendation: Literal["continue", "pause", "abort"] = "continue"

    @property
    def has_failures(self) -> bool:
        return any(not c.passed for c in self.checks)


# ---------------------------------------------------------------------------
# Built-in QC checks
# ---------------------------------------------------------------------------

def _check_volume_accuracy(
    params: dict[str, Any],
    result: dict[str, Any],
    tolerance_pct: float = 10.0,
) -> QCResult | None:
    """Check if dispensed volume matches requested volume."""
    requested = params.get("volume_ul", params.get("volume"))
    actual = result.get("actual_volume_ul")

    if requested is None or actual is None:
        return None

    requested = float(requested)
    actual = float(actual)

    if requested == 0:
        return None

    deviation_pct = abs(actual - requested) / requested * 100
    passed = deviation_pct <= tolerance_pct

    return QCResult(
        check_name="volume_accuracy",
        passed=passed,
        actual_value=actual,
        expected_value=requested,
        message=f"Volume deviation: {deviation_pct:.1f}% ({'OK' if passed else 'FAIL'})",
        severity="info" if passed else "warning",
    )


def _check_temperature_stability(
    params: dict[str, Any],
    result: dict[str, Any],
    tolerance_c: float = 2.0,
) -> QCResult | None:
    """Check if temperature reached and held at target."""
    target = params.get("temp_c", params.get("temperature"))
    actual = result.get("actual_temp_c", result.get("temperature"))

    if target is None or actual is None:
        return None

    target = float(target)
    actual = float(actual)
    deviation = abs(actual - target)
    passed = deviation <= tolerance_c

    return QCResult(
        check_name="temperature_stability",
        passed=passed,
        actual_value=actual,
        expected_value=target,
        threshold_min=target - tolerance_c,
        threshold_max=target + tolerance_c,
        message=f"Temperature deviation: {deviation:.1f}°C ({'OK' if passed else 'FAIL'})",
        severity="info" if passed else "critical",
    )


def _check_current_range(
    result: dict[str, Any],
    max_current_ma: float = 100.0,
) -> QCResult | None:
    """Check if measured current is within safe range."""
    current = result.get("current_ma", result.get("peak_current_ma"))

    if current is None:
        return None

    current = float(current)
    passed = abs(current) <= max_current_ma

    return QCResult(
        check_name="current_range",
        passed=passed,
        actual_value=current,
        threshold_max=max_current_ma,
        message=f"Current: {current:.2f} mA (max: {max_current_ma})",
        severity="info" if passed else "critical",
    )


def _check_step_duration(
    result: dict[str, Any],
    max_duration_s: float = 300.0,
) -> QCResult | None:
    """Check if step took unreasonably long (may indicate a hang)."""
    duration = result.get("duration_s", result.get("elapsed_s"))

    if duration is None:
        return None

    duration = float(duration)
    passed = duration <= max_duration_s

    return QCResult(
        check_name="step_duration",
        passed=passed,
        actual_value=duration,
        threshold_max=max_duration_s,
        message=f"Step took {duration:.1f}s (max: {max_duration_s}s)",
        severity="info" if passed else "warning",
    )


def _detect_anomalies(
    step_result: dict[str, Any],
    step_history: list[dict[str, Any]],
) -> list[str]:
    """Simple anomaly detection: flag if any numeric value is >3 stddev from running mean.

    Only works after at least 3 prior steps for statistics.
    """
    anomalies: list[str] = []

    if len(step_history) < 3:
        return anomalies

    # For each numeric value in the current step result
    for key, value in step_result.items():
        if not isinstance(value, (int, float)) or math.isnan(value):
            continue

        # Gather history for this key
        hist_values = []
        for prev in step_history:
            prev_val = prev.get(key)
            if isinstance(prev_val, (int, float)) and not math.isnan(prev_val):
                hist_values.append(float(prev_val))

        if len(hist_values) < 3:
            continue

        # Compute mean and stddev
        mean = sum(hist_values) / len(hist_values)
        variance = sum((v - mean) ** 2 for v in hist_values) / len(hist_values)
        stddev = math.sqrt(variance) if variance > 0 else 0

        if stddev == 0:
            # All historical values are identical; flag if the new value
            # deviates at all (any deviation from a constant series is notable).
            if abs(float(value) - mean) > 0:
                anomalies.append(
                    f"{key}={value:.4f} deviates from constant mean={mean:.4f}"
                )
            continue

        z_score = abs(float(value) - mean) / stddev
        if z_score > 3.0:
            anomalies.append(
                f"{key}={value:.4f} is {z_score:.1f} stddev from mean={mean:.4f}"
            )

    return anomalies


# ---------------------------------------------------------------------------
# Primitive -> QC check mapping
# ---------------------------------------------------------------------------

# Which built-in checks to run for each primitive category
_PRIMITIVE_QC_MAP: dict[str, list[str]] = {
    "robot.aspirate": ["volume_accuracy", "step_duration"],
    "robot.dispense": ["volume_accuracy", "step_duration"],
    "heat": ["temperature_stability", "step_duration"],
    "squidstat.run_experiment": ["current_range", "step_duration"],
    "squidstat.get_data": ["step_duration"],
}


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------


class SensingAgent(BaseAgent[SensingInput, SensingOutput]):
    name = "sensing_agent"
    description = "Mid-run quality control and anomaly detection"
    layer = "L0"

    def validate_input(self, input_data: SensingInput) -> list[str]:
        errors: list[str] = []
        if not input_data.step_key:
            errors.append("step_key is required")
        if not input_data.primitive:
            errors.append("primitive is required")
        return errors

    async def process(self, input_data: SensingInput) -> SensingOutput:
        checks: list[QCResult] = []

        # 1. Run built-in checks based on primitive type
        builtin_checks = _PRIMITIVE_QC_MAP.get(input_data.primitive, [])

        for check_name in builtin_checks:
            result = None
            if check_name == "volume_accuracy":
                result = _check_volume_accuracy(
                    input_data.params, input_data.step_result,
                )
            elif check_name == "temperature_stability":
                result = _check_temperature_stability(
                    input_data.params, input_data.step_result,
                )
            elif check_name == "current_range":
                max_current = float(
                    input_data.policy_snapshot.get("max_current_ma", 100.0)
                )
                result = _check_current_range(
                    input_data.step_result, max_current,
                )
            elif check_name == "step_duration":
                max_dur = float(
                    input_data.policy_snapshot.get("max_step_duration_s", 300.0)
                )
                result = _check_step_duration(
                    input_data.step_result, max_dur,
                )

            if result is not None:
                checks.append(result)

        # 2. Run user-defined QC checks
        for qc in input_data.qc_checks:
            val = input_data.step_result.get(qc.metric)
            if val is None:
                continue

            val = float(val)
            passed = True
            msg_parts = []

            if qc.min_value is not None and val < qc.min_value:
                passed = False
                msg_parts.append(f"below min {qc.min_value}")
            if qc.max_value is not None and val > qc.max_value:
                passed = False
                msg_parts.append(f"above max {qc.max_value}")
            if qc.expected_value is not None:
                tolerance = abs(qc.expected_value * qc.tolerance_pct / 100)
                if abs(val - qc.expected_value) > tolerance:
                    passed = False
                    msg_parts.append(f"outside {qc.tolerance_pct}% tolerance of {qc.expected_value}")

            checks.append(QCResult(
                check_name=qc.name,
                passed=passed,
                actual_value=val,
                expected_value=qc.expected_value,
                threshold_min=qc.min_value,
                threshold_max=qc.max_value,
                message=f"{qc.metric}={val}: {'; '.join(msg_parts)}" if msg_parts else f"{qc.metric}={val}: OK",
                severity="warning" if not passed else "info",
            ))

        # 3. Anomaly detection
        anomalies = _detect_anomalies(
            input_data.step_result,
            input_data.step_history,
        )

        # 4. Determine overall quality
        critical_fails = [c for c in checks if not c.passed and c.severity == "critical"]
        warning_fails = [c for c in checks if not c.passed and c.severity == "warning"]

        if critical_fails:
            quality = "failed"
            recommendation = "abort"
        elif warning_fails or anomalies:
            quality = "suspect"
            recommendation = "continue"  # warn but don't stop
        else:
            quality = "good"
            recommendation = "continue"

        return SensingOutput(
            overall_quality=quality,
            checks=checks,
            anomalies_detected=anomalies,
            recommendation=recommendation,
        )
