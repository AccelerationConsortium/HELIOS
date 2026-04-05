"""ObservationAgent — unified observation facade.

Merges outputs from MonitorAgent, SensingAgent, and AnalyzerAgent into a
single ObservationPacket with KPI metrics, QC results, anomalies, and analysis.

Layer: L0 (post-execution)
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)


# ── Output Model ───────────────────────────────────────────────────────────

class ObservationPacket(BaseModel):
    """Unified observation output after execution completes."""

    run_id: str
    step_id: str
    campaign_id: str
    round_number: int
    created_at: str

    # KPI values
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="KPI values (metric_name → value)",
    )

    # Artifact references
    artifacts: list[str] = Field(
        default_factory=list,
        description="Artifact store keys/URIs",
    )

    # Anomalies detected
    anomalies: list[str] = Field(
        default_factory=list,
        description="Detected anomalies from 3σ detection",
    )

    # QC results
    qc_passed: bool = True
    qc_findings: list[str] = Field(default_factory=list)

    # Sensor state
    sensor_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Current instrument health and status",
    )

    # Analysis
    analysis_summary: str = ""


# ── Input Model ────────────────────────────────────────────────────────────

class ObservationInput(BaseModel):
    """Input for ObservationAgent."""

    run_id: str
    campaign_id: str
    round_number: int
    step_id: str | None = None

    # Optional: raw result from execution
    result_packet: dict[str, Any] | None = None

    # Optional: monitor agent output (for QC + instrument health)
    monitor_output: dict[str, Any] | None = None

    # Optional: analyzer output (for metrics + narrative)
    analyzer_output: dict[str, Any] | None = None


# ── Agent Implementation ───────────────────────────────────────────────────

class ObservationAgent(BaseAgent[ObservationInput, ObservationPacket]):
    """Unified observation facade over Monitor, Sensing, and Analyzer agents.

    Collects raw metrics, QC results, anomalies, and analysis into a single
    ObservationPacket for downstream consumption by OptimizationAgent.
    """

    name = "observation_agent"
    description = "Unified observation facade (Monitor + Sensing + Analyzer)"
    layer = "L0"

    def validate_input(self, input_data: ObservationInput) -> list[str]:
        errors: list[str] = []
        if not input_data.run_id:
            errors.append("run_id is required")
        if not input_data.campaign_id:
            errors.append("campaign_id is required")
        if input_data.round_number < 0:
            errors.append("round_number must be >= 0")
        return errors

    async def process(self, input_data: ObservationInput) -> ObservationPacket:
        from app.core.db import utcnow_iso

        # ── Phase 1: Extract metrics from result_packet ─────────────────────
        metrics: dict[str, float] = {}
        artifacts: list[str] = []

        if input_data.result_packet:
            metrics.update(input_data.result_packet.get("kpi_values", {}))
            artifacts.extend(input_data.result_packet.get("artifact_uris", []))

        # ── Phase 2: Extract QC results from monitor_output ────────────────
        qc_passed = True
        qc_findings: list[str] = []

        if input_data.monitor_output:
            monitor = input_data.monitor_output
            qc_passed = monitor.get("overall_quality") != "failed"

            # Collect check findings
            for check in monitor.get("checks", []):
                if not check.get("passed", True):
                    qc_findings.append(check.get("message", "unknown check failed"))

        # ── Phase 3: Extract anomalies from monitor_output ────────────────
        anomalies: list[str] = []

        if input_data.monitor_output:
            anomalies.extend(input_data.monitor_output.get("anomalies_detected", []))

        # ── Phase 4: Extract sensor state and analysis ──────────────────────
        sensor_state: dict[str, Any] = {}
        analysis_summary: str = ""

        if input_data.monitor_output:
            sensor_state.update(input_data.monitor_output.get("instrument_health", {}))
            trend_report = input_data.monitor_output.get("trend_report", {})
            if trend_report:
                sensor_state["trends"] = trend_report

        if input_data.analyzer_output:
            analysis_summary = input_data.analyzer_output.get("narrative", "")
            # Extract round-level metrics from analyzer
            if "round_best_kpi" in input_data.analyzer_output:
                metrics["round_best_kpi"] = input_data.analyzer_output["round_best_kpi"]
            if "round_mean_kpi" in input_data.analyzer_output:
                metrics["round_mean_kpi"] = input_data.analyzer_output["round_mean_kpi"]
            if "aleatoric_std" in input_data.analyzer_output:
                metrics["aleatoric_std"] = input_data.analyzer_output["aleatoric_std"]

        step_id = input_data.step_id or f"step-{input_data.round_number}"

        logger.info(
            "observation_agent: run=%s campaign=%s round=%d qc_passed=%s anomalies=%d",
            input_data.run_id, input_data.campaign_id, input_data.round_number,
            qc_passed, len(anomalies),
            extra={"campaign_id": input_data.campaign_id},
        )

        return ObservationPacket(
            run_id=input_data.run_id,
            step_id=step_id,
            campaign_id=input_data.campaign_id,
            round_number=input_data.round_number,
            created_at=utcnow_iso(),
            metrics=metrics,
            artifacts=artifacts,
            anomalies=anomalies,
            qc_passed=qc_passed,
            qc_findings=qc_findings,
            sensor_state=sensor_state,
            analysis_summary=analysis_summary,
        )
