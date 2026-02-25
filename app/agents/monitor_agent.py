"""Monitor Agent — per-candidate QC, instrument health, and trend analysis.

Replaces SensingAgent as the L0 mid-run monitor.  In addition to QC checks
and 3σ anomaly detection (delegated to the existing sensing_agent helpers),
it adds:
  - Time-series trend analysis (5-step sliding window, per numeric metric)
  - Instrument health inference (from primitive name + QC outcome)
  - SSE ``instrument_status`` events for the frontend status bar

Emits:
  - ``agent_thinking``  — one message per QC phase
  - ``instrument_status`` — one event per inferred instrument
  - ``agent_result``    — MonitorOutput summary
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent
from app.agents.sensing_agent import (
    QCCheck,
    QCResult,
    _PRIMITIVE_QC_MAP,
    _check_current_range,
    _check_step_duration,
    _check_temperature_stability,
    _check_volume_accuracy,
    _detect_anomalies,
)

logger = logging.getLogger(__name__)

# Type alias for the SSE emit callback passed in from the orchestrator
EmitCallback = Callable[[dict[str, Any]], None]

# ---------------------------------------------------------------------------
# Primitive → instrument name mapping
# ---------------------------------------------------------------------------

_PRIMITIVE_TO_INSTRUMENT: dict[str, str] = {
    "robot.aspirate": "liquid_handler",
    "robot.dispense": "liquid_handler",
    "robot.mix": "liquid_handler",
    "heat": "thermocycler",
    "cool": "thermocycler",
    "squidstat.run_experiment": "squidstat",
    "squidstat.get_data": "squidstat",
    "plate_reader.read": "plate_reader",
    "centrifuge.spin": "centrifuge",
    "shaker.mix": "shaker",
}

# Sliding window size for trend analysis
_TREND_WINDOW = 5


# ---------------------------------------------------------------------------
# I/O models
# ---------------------------------------------------------------------------


class MonitorInput(BaseModel):
    """Input for the per-candidate Monitor agent."""

    step_key: str
    primitive: str
    params: dict[str, Any] = Field(default_factory=dict)
    step_result: dict[str, Any] = Field(default_factory=dict)
    policy_snapshot: dict[str, Any] = Field(default_factory=dict)
    qc_checks: list[QCCheck] = Field(default_factory=list)
    step_history: list[dict[str, Any]] = Field(default_factory=list)
    round_number: int = 0
    # Optional SSE emit callback — set by orchestrator before calling run()
    emit: EmitCallback | None = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}


class MonitorOutput(BaseModel):
    """Output from the per-candidate Monitor agent."""

    overall_quality: Literal["good", "suspect", "failed"] = "good"
    checks: list[QCResult] = Field(default_factory=list)
    anomalies_detected: list[str] = Field(default_factory=list)
    recommendation: Literal["continue", "pause", "abort"] = "continue"
    instrument_health: dict[str, dict[str, str]] = Field(default_factory=dict)
    trend_report: dict[str, str] = Field(default_factory=dict)

    @property
    def has_failures(self) -> bool:
        return any(not c.passed for c in self.checks)


# ---------------------------------------------------------------------------
# Trend analysis helpers
# ---------------------------------------------------------------------------


def _compute_trends(
    step_result: dict[str, Any],
    step_history: list[dict[str, Any]],
    window: int = _TREND_WINDOW,
) -> dict[str, str]:
    """Sliding-window slope-based trend analysis for numeric metrics.

    For each numeric key present in ``step_result`` and at least
    ``window`` prior steps, compute the linear slope direction:
      - |slope| < noise_floor → "stable"
      - slope > noise_floor   → "rising"
      - slope < -noise_floor  → "falling"

    Returns a dict mapping metric name → trend label.
    """
    trends: dict[str, str] = {}

    for key, value in step_result.items():
        if not isinstance(value, (int, float)) or math.isnan(float(value)):
            continue

        # Collect history for this key (most recent ``window`` entries)
        series: list[float] = []
        for prev in step_history[-window:]:
            prev_val = prev.get(key)
            if isinstance(prev_val, (int, float)) and not math.isnan(float(prev_val)):
                series.append(float(prev_val))

        series.append(float(value))  # include current value

        if len(series) < 3:
            continue

        # Simple linear slope: y = a + b*x, x = 0,1,...,n-1
        n = len(series)
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(series) / n
        num = sum((xs[i] - mean_x) * (series[i] - mean_y) for i in range(n))
        den = sum((xs[i] - mean_x) ** 2 for i in range(n))
        slope = num / den if abs(den) > 1e-12 else 0.0

        # Noise floor: 2% of the mean absolute value
        noise_floor = max(abs(mean_y) * 0.02, 1e-9)

        if slope > noise_floor:
            trends[key] = "rising"
        elif slope < -noise_floor:
            trends[key] = "falling"
        else:
            trends[key] = "stable"

    return trends


# ---------------------------------------------------------------------------
# Instrument health inference
# ---------------------------------------------------------------------------


def _infer_instrument_health(
    primitive: str,
    quality: Literal["good", "suspect", "failed"],
    anomalies: list[str],
) -> dict[str, dict[str, str]]:
    """Map QC outcome to instrument health status.

    Rules:
      - QC good + no anomalies → ready
      - QC good + anomalies    → busy (possible transient issue)
      - QC suspect             → busy
      - QC failed              → error
    """
    instrument = _PRIMITIVE_TO_INSTRUMENT.get(primitive)
    if instrument is None:
        # Try prefix match (e.g. "robot.custom_step" → "liquid_handler")
        for prefix, name in _PRIMITIVE_TO_INSTRUMENT.items():
            if primitive.startswith(prefix.split(".")[0]):
                instrument = name
                break

    if instrument is None:
        instrument = "unknown_instrument"

    if quality == "failed":
        status = "error"
        message = f"QC failed on {primitive}"
    elif quality == "suspect" or anomalies:
        status = "busy"
        first_anomaly = anomalies[0] if anomalies else "suspect QC"
        message = f"Anomaly detected: {first_anomaly[:80]}"
    else:
        status = "ready"
        message = f"{primitive} completed normally"

    return {instrument: {"status": status, "message": message}}


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------


class MonitorAgent(BaseAgent[MonitorInput, MonitorOutput]):
    """Per-candidate quality monitor.

    Replaces SensingAgent.  Runs built-in QC checks, user-defined QC
    checks, 3σ anomaly detection, 5-step sliding window trend analysis,
    and infers instrument health status.

    When an ``emit`` callback is set on the input, emits ``agent_thinking``
    and ``instrument_status`` SSE events during processing.
    """

    name = "monitor_agent"
    description = "Per-candidate QC, instrument health, and trend analysis"
    layer = "L0"

    def validate_input(self, input_data: MonitorInput) -> list[str]:
        errors: list[str] = []
        if not input_data.step_key:
            errors.append("step_key is required")
        if not input_data.primitive:
            errors.append("primitive is required")
        return errors

    async def process(self, input_data: MonitorInput) -> MonitorOutput:  # noqa: C901
        emit = input_data.emit
        checks: list[QCResult] = []

        # ── Phase 1: Built-in QC checks ──────────────────────────────────
        if emit:
            emit({
                "type": "agent_thinking",
                "agent": "monitor",
                "round": input_data.round_number,
                "message": f"Running built-in QC checks for {input_data.primitive}...",
            })

        builtin_checks = _PRIMITIVE_QC_MAP.get(input_data.primitive, [])
        for check_name in builtin_checks:
            result: QCResult | None = None
            if check_name == "volume_accuracy":
                result = _check_volume_accuracy(input_data.params, input_data.step_result)
            elif check_name == "temperature_stability":
                result = _check_temperature_stability(input_data.params, input_data.step_result)
            elif check_name == "current_range":
                max_current = float(input_data.policy_snapshot.get("max_current_ma", 100.0))
                result = _check_current_range(input_data.step_result, max_current)
            elif check_name == "step_duration":
                max_dur = float(input_data.policy_snapshot.get("max_step_duration_s", 300.0))
                result = _check_step_duration(input_data.step_result, max_dur)

            if result is not None:
                checks.append(result)
                if emit:
                    status_str = "✓" if result.passed else "✗"
                    emit({
                        "type": "agent_thinking",
                        "agent": "monitor",
                        "round": input_data.round_number,
                        "message": f"  {status_str} {result.check_name}: {result.message}",
                    })

        # ── Phase 2: User-defined QC checks ──────────────────────────────
        if input_data.qc_checks and emit:
            emit({
                "type": "agent_thinking",
                "agent": "monitor",
                "round": input_data.round_number,
                "message": f"Running {len(input_data.qc_checks)} user-defined QC check(s)...",
            })

        for qc in input_data.qc_checks:
            val = input_data.step_result.get(qc.metric)
            if val is None:
                continue

            val = float(val)
            passed = True
            msg_parts: list[str] = []

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
                    msg_parts.append(
                        f"outside {qc.tolerance_pct}% tolerance of {qc.expected_value}"
                    )

            check_result = QCResult(
                check_name=qc.name,
                passed=passed,
                actual_value=val,
                expected_value=qc.expected_value,
                threshold_min=qc.min_value,
                threshold_max=qc.max_value,
                message=(
                    f"{qc.metric}={val}: {'; '.join(msg_parts)}"
                    if msg_parts
                    else f"{qc.metric}={val}: OK"
                ),
                severity="warning" if not passed else "info",
            )
            checks.append(check_result)

        # ── Phase 3: Anomaly detection ────────────────────────────────────
        if emit:
            emit({
                "type": "agent_thinking",
                "agent": "monitor",
                "round": input_data.round_number,
                "message": "Running 3σ anomaly detection...",
            })

        anomalies = _detect_anomalies(input_data.step_result, input_data.step_history)

        if anomalies and emit:
            for anomaly in anomalies:
                emit({
                    "type": "agent_thinking",
                    "agent": "monitor",
                    "round": input_data.round_number,
                    "message": f"  ⚠ Anomaly: {anomaly}",
                })

        # ── Phase 4: Trend analysis ───────────────────────────────────────
        if emit:
            emit({
                "type": "agent_thinking",
                "agent": "monitor",
                "round": input_data.round_number,
                "message": "Analyzing metric trends (5-step window)...",
            })

        trend_report = _compute_trends(input_data.step_result, input_data.step_history)

        if trend_report and emit:
            summary = ", ".join(f"{k}: {v}" for k, v in list(trend_report.items())[:4])
            emit({
                "type": "agent_thinking",
                "agent": "monitor",
                "round": input_data.round_number,
                "message": f"  Trends → {summary}",
            })

        # ── Phase 5: Overall quality & recommendation ─────────────────────
        critical_fails = [c for c in checks if not c.passed and c.severity == "critical"]
        warning_fails = [c for c in checks if not c.passed and c.severity == "warning"]

        if critical_fails:
            quality: Literal["good", "suspect", "failed"] = "failed"
            recommendation: Literal["continue", "pause", "abort"] = "abort"
        elif warning_fails or anomalies:
            quality = "suspect"
            recommendation = "continue"
        else:
            quality = "good"
            recommendation = "continue"

        # ── Phase 6: Instrument health + SSE ─────────────────────────────
        instrument_health = _infer_instrument_health(
            input_data.primitive, quality, anomalies
        )

        if emit:
            for inst_name, health in instrument_health.items():
                emit({
                    "type": "instrument_status",
                    "instrument": inst_name,
                    "status": health["status"],
                    "message": health["message"],
                    "round": input_data.round_number,
                })

        return MonitorOutput(
            overall_quality=quality,
            checks=checks,
            anomalies_detected=anomalies,
            recommendation=recommendation,
            instrument_health=instrument_health,
            trend_report=trend_report,
        )
