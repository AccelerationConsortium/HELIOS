"""ExecutionAgent — L1 execution routing and compilation.

Routes a WorkflowGraph to the appropriate execution backend (opentrons_mcp,
python_api, or simulation) based on device capability snapshot and workflow
preferences. Delegates compilation to backend-specific compilers.

v2: Runtime risk assessment and pause decisions.
- Novel primitives → pause
- Large volumes → pause
- Low instrument health → pause
- High recent failure rate → pause

Layer: L1
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import AgentPauseRejected, BaseAgent, DecisionNode
from app.agents.capability_agent import CapabilitySnapshot
from app.agents.pause import Granularity, PauseRequest, RiskAssessment
from app.contracts.run_bundle import RunBundle, new_run_bundle_id
from app.contracts.workflow_ir import ExecutionBackend, WorkflowGraph

logger = logging.getLogger(__name__)


# ── Input/Output Models ────────────────────────────────────────────────────

class ExecutionInput(BaseModel):
    """Input for ExecutionAgent."""

    workflow: WorkflowGraph
    capability: CapabilitySnapshot
    campaign_id: str
    round_number: int = Field(ge=0)
    # v2: historical context for risk assessment
    recent_failure_rate: float = 0.0     # 0.0–1.0, from campaign metrics
    proven_primitives: list[str] = Field(
        default_factory=list,
        description="Primitives that have succeeded in previous rounds",
    )


class ExecutionOutput(BaseModel):
    """Output from ExecutionAgent."""

    run_bundle: RunBundle
    backend_used: ExecutionBackend
    routing_decision: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    # v2
    granularity_used: str = "coarse"
    risk_assessment: dict[str, Any] = Field(default_factory=dict)


# ── Risk thresholds ───────────────────────────────────────────────────────

VOLUME_PAUSE_THRESHOLD_UL = 500.0   # Volumes above this trigger pause
HEALTH_PAUSE_THRESHOLD = 0.8        # Instrument health below this triggers pause
FAILURE_RATE_THRESHOLD = 0.2        # Recent failure rate above this triggers pause


# ── Agent Implementation ───────────────────────────────────────────────────

class ExecutionAgent(BaseAgent[ExecutionInput, ExecutionOutput]):
    """Routes WorkflowGraph to appropriate backend and compiles to RunBundle.

    Decision logic:
    1. Check which primitives are available in CapabilitySnapshot
    2. If robot unreachable → fall back to simulation
    3. If all required primitives available → use preferred_backend
    4. Otherwise → use simulation

    v2: Before execution, assess risk and optionally pause for human review.
    """

    name = "execution_agent"
    description = "WorkflowGraph routing and backend compilation"
    layer = "L1"

    def validate_input(self, input_data: ExecutionInput) -> list[str]:
        errors: list[str] = []
        if not input_data.workflow or not input_data.workflow.steps:
            errors.append("workflow must contain at least one step")
        if not input_data.campaign_id:
            errors.append("campaign_id is required")
        return errors

    # ── v2+v3: Granularity assessment (memory-enriched) ─────────────

    async def assess_granularity(
        self,
        input_data: ExecutionInput,
        context: dict[str, Any] | None = None,
    ) -> Granularity:
        """Decide granularity based on execution risk + historical memory."""
        risk = self._assess_risk(input_data)

        # v3: Also consult memory-based workflow risk
        try:
            from app.services.memory_risk_bridge import assess_workflow_risk
            steps_with_params = [
                (step.abstract_primitive, step.parameters)
                for step in input_data.workflow.steps
            ]
            mem_report = assess_workflow_risk(steps_with_params)
            # Use the stricter of the two recommendations
            if mem_report.recommended_granularity == "fine":
                return Granularity.FINE
        except Exception:
            pass  # Memory is advisory

        if risk.risk_score > 0.7:
            return Granularity.FINE
        elif risk.risk_score < 0.3:
            return Granularity.COARSE
        return Granularity.ADAPTIVE

    # ── v2+v3: Risk assessment (memory-enriched) ──────────────────

    def _assess_risk(self, input_data: ExecutionInput) -> RiskAssessment:
        """Evaluate execution risk based on multiple signals + historical memory."""
        workflow = input_data.workflow
        capability = input_data.capability
        factors: dict[str, float] = {}

        # Factor 1: Novel primitives (never successfully run before)
        primitives_used = workflow.primitives_used()
        proven = set(input_data.proven_primitives)
        novel = primitives_used - proven - {"simulation"}
        novel_ratio = len(novel) / max(len(primitives_used), 1)
        factors["novel_primitive_ratio"] = novel_ratio

        # Factor 2: Volume risk
        max_vol = 0.0
        for step in workflow.steps:
            vol = step.parameters.get("volume_ul", 0.0)
            if isinstance(vol, (int, float)):
                max_vol = max(max_vol, float(vol))
        vol_ratio = min(max_vol / VOLUME_PAUSE_THRESHOLD_UL, 1.0) if VOLUME_PAUSE_THRESHOLD_UL > 0 else 0.0
        factors["volume_risk"] = vol_ratio

        # Factor 3: Instrument health
        health = getattr(capability, "health_score", 1.0) or 1.0
        health_risk = max(0.0, 1.0 - health)
        factors["instrument_health_risk"] = health_risk

        # Factor 4: Recent failure rate
        failure_risk = min(input_data.recent_failure_rate, 1.0)
        factors["recent_failure_rate"] = failure_risk

        # ── v3: Factor 5 — Memory-based historical risk ──────────
        memory_risk = 0.0
        memory_details: list[str] = []
        try:
            from app.services.memory_risk_bridge import assess_workflow_risk
            steps_with_params = [
                (step.abstract_primitive, step.parameters)
                for step in workflow.steps
            ]
            mem_report = assess_workflow_risk(steps_with_params)
            memory_risk = mem_report.overall_risk
            factors["memory_historical_risk"] = memory_risk
            if mem_report.high_risk_primitives:
                memory_details.append(
                    f"memory high-risk: {mem_report.high_risk_primitives}"
                )
            if mem_report.novel_primitives:
                memory_details.append(
                    f"memory novel: {mem_report.novel_primitives}"
                )
            # Check param deviation signals
            for profile in mem_report.primitive_profiles:
                for param, dev in profile.param_risk_signals.items():
                    if dev > 0.5:  # >1.5 sigma deviation
                        factors[f"param_deviation_{profile.primitive}.{param}"] = dev
                        memory_details.append(
                            f"{profile.primitive}.{param} deviates {dev:.0%} from historical"
                        )
        except Exception:
            factors["memory_historical_risk"] = 0.0

        # Weighted composite score (v3: include memory factor)
        base_weights = {
            "novel_primitive_ratio": 0.25,
            "volume_risk": 0.15,
            "instrument_health_risk": 0.20,
            "recent_failure_rate": 0.15,
            "memory_historical_risk": 0.25,
        }
        risk_score = sum(factors.get(k, 0.0) * w for k, w in base_weights.items())

        # Determine if we should pause
        should_pause = (
            novel_ratio > 0                              # Any novel primitive
            or max_vol > VOLUME_PAUSE_THRESHOLD_UL       # Large volume
            or health < HEALTH_PAUSE_THRESHOLD            # Degraded instrument
            or failure_risk > FAILURE_RATE_THRESHOLD       # High recent failures
            or memory_risk > 0.5                          # v3: memory says high risk
        )

        reason_parts = []
        if novel:
            reason_parts.append(f"novel primitives: {novel}")
        if max_vol > VOLUME_PAUSE_THRESHOLD_UL:
            reason_parts.append(f"high volume: {max_vol:.0f}µL")
        if health < HEALTH_PAUSE_THRESHOLD:
            reason_parts.append(f"instrument health: {health:.2f}")
        if failure_risk > FAILURE_RATE_THRESHOLD:
            reason_parts.append(f"failure rate: {failure_risk:.2f}")
        reason_parts.extend(memory_details)

        return RiskAssessment(
            should_pause=should_pause,
            risk_score=risk_score,
            reason=" | ".join(reason_parts) if reason_parts else "nominal",
            factors=factors,
        )

    # ── Main process ──────────────────────────────────────────────────

    async def process(self, input_data: ExecutionInput) -> ExecutionOutput:
        workflow = input_data.workflow
        capability = input_data.capability
        campaign_id = input_data.campaign_id
        round_number = input_data.round_number

        # ── v2: Risk assessment and optional pause ────────────────────
        risk = self._assess_risk(input_data)
        granularity = await self.assess_granularity(input_data)

        if risk.should_pause:
            pause_result = await self.request_pause(PauseRequest(
                reason=risk.reason,
                risk_factors=risk.factors,
                suggested_action="approve",
                checkpoint={
                    "workflow_id": workflow.graph_id,
                    "campaign_id": campaign_id,
                    "round_number": round_number,
                    "risk_score": risk.risk_score,
                },
            ))
            if pause_result.decision == "rejected":
                raise AgentPauseRejected(
                    f"Operator rejected execution: {risk.reason} "
                    f"(risk_score={risk.risk_score:.2f})"
                )
            # If modified, caller can apply changes via pause_result.modifications

        # ── Phase 1: Determine which backend to use ───────────────────
        backend, routing_decision, warnings = self._select_backend(
            workflow, capability, campaign_id
        )

        logger.info(
            "execution_agent: backend=%s campaign=%s round=%d workflow=%s risk=%.2f granularity=%s",
            backend, campaign_id, round_number, workflow.graph_id,
            risk.risk_score, granularity.value,
            extra={"campaign_id": campaign_id},
        )

        # ── Phase 2: Compile WorkflowGraph to RunBundle ──────────────
        run_bundle: RunBundle | None = None
        compile_warnings: list[str] = []

        if backend == "opentrons_mcp":
            run_bundle, compile_warnings = await self._compile_opentrons(
                workflow, campaign_id, round_number
            )
        elif backend == "python_api":
            run_bundle, compile_warnings = await self._compile_python_api(
                workflow, campaign_id, round_number
            )
        else:  # simulation
            run_bundle, compile_warnings = await self._compile_simulation(
                workflow, campaign_id, round_number
            )

        if compile_warnings:
            warnings.extend(compile_warnings)

        return ExecutionOutput(
            run_bundle=run_bundle,
            backend_used=backend,
            routing_decision=routing_decision,
            warnings=warnings,
            granularity_used=granularity.value,
            risk_assessment=risk.model_dump(),
        )

    # ── Backend selection logic ────────────────────────────────────────────

    def _select_backend(
        self,
        workflow: WorkflowGraph,
        capability: CapabilitySnapshot,
        campaign_id: str,
    ) -> tuple[ExecutionBackend, dict[str, Any], list[str]]:
        """Select the appropriate execution backend.

        Returns:
            (backend, routing_decision_dict, warnings)
        """
        warnings: list[str] = []
        routing_decision: dict[str, Any] = {
            "workflow_id": workflow.graph_id,
            "campaign_id": campaign_id,
            "robot_reachable": capability.robot_reachable,
            "preferred_backend": workflow.preferred_backend,
            "primitives_used": list(workflow.primitives_used()),
        }

        # Step 1: Check if robot is reachable
        if not capability.robot_reachable:
            routing_decision["reason"] = "robot_unreachable → simulation"
            routing_decision["selected_backend"] = "simulation"
            warnings.append(
                f"Robot at {capability.robot_ip} is unreachable; falling back to simulation"
            )
            logger.info(
                "execution_agent: robot unreachable, using simulation for %s",
                campaign_id,
            )
            return "simulation", routing_decision, warnings

        # Step 2: Check if all required primitives are available
        primitives_used = workflow.primitives_used()
        available = set(capability.available_primitives)
        missing = primitives_used - available

        if missing:
            routing_decision["missing_primitives"] = list(missing)
            routing_decision["reason"] = f"primitives unavailable: {missing} → simulation"
            routing_decision["selected_backend"] = "simulation"
            warnings.append(
                f"Missing primitives: {missing}; falling back to simulation"
            )
            logger.info(
                "execution_agent: missing primitives %s, using simulation for %s",
                missing, campaign_id,
            )
            return "simulation", routing_decision, warnings

        # Step 3: All primitives available; use preferred backend
        preferred = workflow.preferred_backend
        if preferred == "opentrons_mcp":
            routing_decision["reason"] = "all primitives available, using preferred backend"
            routing_decision["selected_backend"] = "opentrons_mcp"
            return "opentrons_mcp", routing_decision, warnings
        elif preferred == "python_api":
            routing_decision["reason"] = "all primitives available, using preferred backend"
            routing_decision["selected_backend"] = "python_api"
            return "python_api", routing_decision, warnings
        else:  # simulation
            routing_decision["reason"] = "all primitives available, using preferred backend (simulation)"
            routing_decision["selected_backend"] = "simulation"
            return "simulation", routing_decision, warnings

    # ── Backend-specific compilers ─────────────────────────────────────────

    async def _compile_opentrons(
        self,
        workflow: WorkflowGraph,
        campaign_id: str,
        round_number: int,
    ) -> tuple[RunBundle, list[str]]:
        """Compile WorkflowGraph using the Opentrons MCP backend.

        Wraps CompilerAgent to produce a RunBundle.
        """
        from app.agents.compiler_agent import CompileInput, CompilerAgent
        import hashlib

        try:
            compiler = CompilerAgent()

            # Convert workflow to protocol dict (simplified)
            protocol_dict = self._workflow_to_protocol(workflow)

            compile_input = CompileInput(
                protocol=protocol_dict,
                inputs={"batch_size": 1, "campaign_id": campaign_id},
                policy_snapshot={},
            )

            compile_result = await compiler.process(compile_input)

            # Build RunBundle
            run_bundle = RunBundle(
                bundle_id=new_run_bundle_id(),
                plan_id=workflow.graph_id,
                contract_id=campaign_id,
                round_number=round_number,
                candidate_index=0,
                created_at=self._get_timestamp(),
                compiled_protocol=compile_result.compiled_graph,
                graph_hash=compile_result.graph_hash,
                deck_layout=self._build_deck_layout(compile_result.deck_plan),
                params={"campaign_id": campaign_id, "round": round_number},
                policy_snapshot={},
                protocol_pattern_id=workflow.graph_id,
                protocol_version="1.0",
            )

            warnings: list[str] = compile_result.layout_warnings

            logger.debug(
                "execution_agent: opentrons compilation succeeded for %s",
                campaign_id,
            )

            return run_bundle, warnings

        except Exception as exc:
            logger.error(
                "execution_agent: opentrons compilation failed: %s",
                exc, exc_info=True,
            )
            raise

    async def _compile_python_api(
        self,
        workflow: WorkflowGraph,
        campaign_id: str,
        round_number: int,
    ) -> tuple[RunBundle, list[str]]:
        """Compile WorkflowGraph using the Python API backend."""
        import hashlib

        try:
            # Convert workflow to protocol dict
            protocol_dict = self._workflow_to_protocol(workflow)

            # Compute hash
            protocol_str = str(protocol_dict)
            graph_hash = hashlib.sha256(protocol_str.encode()).hexdigest()[:12]

            run_bundle = RunBundle(
                bundle_id=new_run_bundle_id(),
                plan_id=workflow.graph_id,
                contract_id=campaign_id,
                round_number=round_number,
                candidate_index=0,
                created_at=self._get_timestamp(),
                compiled_protocol=protocol_dict,
                graph_hash=graph_hash,
                deck_layout=self._build_deck_layout({}),
                params={"campaign_id": campaign_id, "round": round_number},
                policy_snapshot={},
                protocol_pattern_id=workflow.graph_id,
                protocol_version="1.0",
            )

            logger.debug(
                "execution_agent: python_api compilation succeeded for %s",
                campaign_id,
            )

            return run_bundle, []

        except Exception as exc:
            logger.error(
                "execution_agent: python_api compilation failed: %s",
                exc, exc_info=True,
            )
            raise

    async def _compile_simulation(
        self,
        workflow: WorkflowGraph,
        campaign_id: str,
        round_number: int,
    ) -> tuple[RunBundle, list[str]]:
        """Compile WorkflowGraph to a simulation-mode RunBundle (dry-run).

        Produces a compiled_protocol with mode='simulation' containing
        the abstract steps from the workflow.
        """
        import hashlib

        try:
            # Build simulation protocol
            simulation_protocol = {
                "mode": "simulation",
                "workflow_id": workflow.graph_id,
                "steps": [
                    {
                        "step_id": step.step_id,
                        "intent": step.intent,
                        "abstract_primitive": step.abstract_primitive,
                        "parameters": step.parameters,
                        "constraints": step.constraints,
                        "estimated_duration_s": step.estimated_duration_s,
                    }
                    for step in workflow.topological_order()
                ],
                "estimated_total_duration_s": workflow.estimated_total_duration_s(),
            }

            # Compute hash
            protocol_str = str(simulation_protocol)
            graph_hash = hashlib.sha256(protocol_str.encode()).hexdigest()[:12]

            run_bundle = RunBundle(
                bundle_id=new_run_bundle_id(),
                plan_id=workflow.graph_id,
                contract_id=campaign_id,
                round_number=round_number,
                candidate_index=0,
                created_at=self._get_timestamp(),
                compiled_protocol=simulation_protocol,
                graph_hash=graph_hash,
                deck_layout=self._build_deck_layout({}),
                params={"campaign_id": campaign_id, "round": round_number},
                policy_snapshot={},
                protocol_pattern_id=workflow.graph_id,
                protocol_version="1.0",
            )

            warnings = [
                f"Simulation mode: {len(workflow.steps)} abstract steps will be dry-run only"
            ]

            logger.debug(
                "execution_agent: simulation compilation succeeded for %s",
                campaign_id,
            )

            return run_bundle, warnings

        except Exception as exc:
            logger.error(
                "execution_agent: simulation compilation failed: %s",
                exc, exc_info=True,
            )
            raise

    # ── Helper methods ─────────────────────────────────────────────────────

    def _workflow_to_protocol(self, workflow: WorkflowGraph) -> dict[str, Any]:
        """Convert a WorkflowGraph to a protocol dict (simplified)."""
        steps = []
        for step in workflow.topological_order():
            steps.append({
                "step_key": step.step_id,
                "intent": step.intent,
                "primitive": step.abstract_primitive,
                "parameters": step.parameters,
                "constraints": step.constraints,
                "labware_refs": step.labware_refs,
            })

        return {
            "workflow_id": workflow.graph_id,
            "campaign_id": workflow.campaign_id,
            "round": workflow.round_number,
            "steps": steps,
            "metadata": workflow.metadata,
        }

    def _build_deck_layout(self, deck_plan: dict[str, Any]) -> Any:
        """Build a DeckLayout from deck plan (minimal implementation)."""
        from app.contracts.run_bundle import DeckLayout

        return DeckLayout(slot_assignments={}, pipette_mounts={})

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from app.core.db import utcnow_iso

        return utcnow_iso()
