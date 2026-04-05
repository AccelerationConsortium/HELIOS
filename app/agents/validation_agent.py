"""ValidationAgent — 3-tier validation before execution.

Performs semantic, capability, and simulation-level validation of a
WorkflowGraph before sending to ExecutionAgent.

Layer: L1
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent
from app.agents.capability_agent import CapabilitySnapshot
from app.contracts.workflow_ir import WorkflowGraph

logger = logging.getLogger(__name__)


# ── Sub-models ─────────────────────────────────────────────────────────────

@dataclass
class TierResult:
    """Result from one validation tier."""

    tier_number: int
    tier_name: str
    passed: bool
    findings: list[str] = field(default_factory=list)


# ── Input/Output Models ────────────────────────────────────────────────────

class ValidationInput(BaseModel):
    """Input for ValidationAgent."""

    workflow: WorkflowGraph
    capability: CapabilitySnapshot
    compiled_protocol: dict[str, Any] | None = None


class ValidationOutput(BaseModel):
    """Output from ValidationAgent."""

    passed: bool
    tier_results: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    simulation_error_raw: str | None = None


# ── Agent Implementation ───────────────────────────────────────────────────

class ValidationAgent(BaseAgent[ValidationInput, ValidationOutput]):
    """3-tier validation: Semantic → Capability → Opentrons Simulation.

    Tier 1 (Semantic): Check workflow step parameters make sense
    Tier 2 (Capability): Check primitives exist in CapabilitySnapshot
    Tier 3 (Simulation): POST to robot HTTP API if reachable
    """

    name = "validation_agent"
    description = "3-tier workflow validation (semantic, capability, simulation)"
    layer = "L1"

    def validate_input(self, input_data: ValidationInput) -> list[str]:
        errors: list[str] = []
        if not input_data.workflow or not input_data.workflow.steps:
            errors.append("workflow must contain at least one step")
        return errors

    async def process(self, input_data: ValidationInput) -> ValidationOutput:
        workflow = input_data.workflow
        capability = input_data.capability

        tier_results: list[TierResult] = []
        all_errors: list[str] = []
        all_warnings: list[str] = []
        sim_error_raw: str | None = None

        # ── Tier 1: Semantic validation ────────────────────────────────────
        tier1 = self._validate_semantic(workflow)
        tier_results.append(tier1)
        if not tier1.passed:
            all_errors.extend(tier1.findings)

        # ── Tier 2: Capability validation ──────────────────────────────────
        tier2 = self._validate_capability(workflow, capability)
        tier_results.append(tier2)
        if not tier2.passed:
            all_errors.extend(tier2.findings)
        all_warnings.extend([f for f in tier2.findings if "constrained" in f.lower()])

        # ── Tier 3: Opentrons simulation validation ────────────────────────
        tier3_passed = True
        tier3_findings: list[str] = []

        if capability.robot_reachable and input_data.compiled_protocol:
            try:
                tier3_passed, findings, error_raw = await self._validate_opentrons_simulation(
                    capability, input_data.compiled_protocol
                )
                tier3_findings = findings
                sim_error_raw = error_raw
            except Exception as exc:
                tier3_passed = False
                sim_error_raw = str(exc)
                tier3_findings = [f"Simulation error: {exc}"]
                logger.warning(
                    "validation_agent: opentrons simulation failed: %s",
                    exc, exc_info=True,
                )
        else:
            tier3_findings.append(
                "Skipped (robot unreachable or no compiled_protocol)"
            )

        tier3 = TierResult(
            tier_number=3,
            tier_name="Opentrons Simulation",
            passed=tier3_passed,
            findings=tier3_findings,
        )
        tier_results.append(tier3)

        if not tier3_passed:
            all_errors.extend(tier3_findings)

        # ── Overall result ─────────────────────────────────────────────────
        passed = all([t.passed for t in tier_results])

        logger.info(
            "validation_agent: workflow=%s passed=%s tier1=%s tier2=%s tier3=%s",
            workflow.graph_id, passed, tier_results[0].passed,
            tier_results[1].passed, tier_results[2].passed,
        )

        return ValidationOutput(
            passed=passed,
            tier_results=[
                {
                    "tier_number": t.tier_number,
                    "tier_name": t.tier_name,
                    "passed": t.passed,
                    "findings": t.findings,
                }
                for t in tier_results
            ],
            errors=all_errors,
            warnings=all_warnings,
            simulation_error_raw=sim_error_raw,
        )

    # ── Tier 1: Semantic Validation ────────────────────────────────────────

    def _validate_semantic(self, workflow: WorkflowGraph) -> TierResult:
        """Check that step parameters make sense."""
        findings: list[str] = []

        for step in workflow.steps:
            # Check volume parameters
            volume = step.parameters.get("volume_ul") or step.parameters.get("volume")
            if volume is not None:
                try:
                    vol_f = float(volume)
                    if vol_f <= 0:
                        findings.append(
                            f"{step.step_id}: volume_ul={vol_f} must be > 0"
                        )
                except (ValueError, TypeError):
                    findings.append(
                        f"{step.step_id}: volume_ul={volume} is not a valid number"
                    )

            # Check well names (basic format: A1-H12 for 96-well plate)
            for well_key in ["source", "destination", "well"]:
                well_name = step.parameters.get(well_key)
                if well_name and isinstance(well_name, str):
                    if not self._is_valid_well_name(well_name):
                        findings.append(
                            f"{step.step_id}: {well_key}={well_name} is not a valid well name"
                        )

            # Check temperature parameters
            temp = step.parameters.get("temp_c") or step.parameters.get("temperature")
            if temp is not None:
                try:
                    temp_f = float(temp)
                    if temp_f < -273.15 or temp_f > 1500:  # absolute zero to plasma
                        findings.append(
                            f"{step.step_id}: temperature={temp_f}°C is outside physical bounds"
                        )
                except (ValueError, TypeError):
                    findings.append(
                        f"{step.step_id}: temperature={temp} is not a valid number"
                    )

            # Check step dependencies exist
            for dep_id in step.dependencies:
                if not workflow.step_by_id(dep_id):
                    findings.append(
                        f"{step.step_id}: dependency {dep_id} does not exist"
                    )

        passed = len(findings) == 0

        return TierResult(
            tier_number=1,
            tier_name="Semantic Validation",
            passed=passed,
            findings=findings,
        )

    # ── Tier 2: Capability Validation ──────────────────────────────────────

    def _validate_capability(
        self, workflow: WorkflowGraph, capability: CapabilitySnapshot
    ) -> TierResult:
        """Check that every primitive is available."""
        findings: list[str] = []

        primitives_used = workflow.primitives_used()
        available = set(capability.available_primitives)
        constrained = set(capability.constrained_primitives)

        for prim in primitives_used:
            if prim not in available:
                if prim in constrained:
                    findings.append(
                        f"Primitive '{prim}' is constrained by current deck/instrument state"
                    )
                else:
                    findings.append(
                        f"Primitive '{prim}' is not available"
                    )

        passed = len(findings) == 0

        return TierResult(
            tier_number=2,
            tier_name="Capability Validation",
            passed=passed,
            findings=findings,
        )

    # ── Tier 3: Opentrons Simulation ───────────────────────────────────────

    async def _validate_opentrons_simulation(
        self, capability: CapabilitySnapshot, compiled_protocol: dict[str, Any]
    ) -> tuple[bool, list[str], str | None]:
        """POST compiled_protocol to robot for simulation.

        Returns:
            (passed, findings, error_raw)
        """
        import httpx

        findings: list[str] = []
        error_raw: str | None = None

        if not compiled_protocol or not compiled_protocol.get("steps"):
            return True, ["No steps to simulate"], None

        url = f"http://{capability.robot_ip}:31950/protocols"

        try:
            timeout = 15.0
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    url,
                    json=compiled_protocol,
                    headers={"Opentrons-Version": "3"},
                )

                # Any non-2xx response is a simulation error
                if resp.status_code >= 300:
                    error_raw = resp.text
                    findings.append(
                        f"Opentrons simulation returned {resp.status_code}: {error_raw[:200]}"
                    )
                    return False, findings, error_raw

            findings.append("Opentrons simulation passed")
            return True, findings, None

        except (httpx.HTTPError, httpx.TimeoutException, OSError) as exc:
            error_raw = str(exc)
            findings.append(f"Simulation HTTP error: {exc}")
            logger.warning(
                "validation_agent: opentrons simulation request failed: %s",
                exc,
            )
            return False, findings, error_raw

    # ── Helper methods ─────────────────────────────────────────────────────

    @staticmethod
    def _is_valid_well_name(well: str) -> bool:
        """Check if well name matches A1-H12 pattern (96-well plate).

        Also accepts 384-well (A1-P24) and single-column names.
        """
        if not well or not isinstance(well, str):
            return False

        well_upper = well.upper().strip()

        # Row letter
        if len(well_upper) < 2:
            return False

        row = well_upper[0]
        if not ("A" <= row <= "P"):  # A-P covers 96 and 384 well plates
            return False

        # Column number
        try:
            col = int(well_upper[1:])
            if col < 1 or col > 24:  # 1-24 covers both 96 and 384
                return False
            return True
        except ValueError:
            return False
