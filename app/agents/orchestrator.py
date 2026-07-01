"""Main Orchestrator Agent — routes tasks to sub-agents.

This is the top-level agent that scientists interact with (via the API).
It receives a TaskContract and drives the full campaign lifecycle:
  L3 (intake) → L2 (planning) → L1 (compilation) → L0 (execution)

The orchestrator does NOT do work itself — it delegates to sub-agents
and handles the flow of contracts between layers.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import AgentResult, BaseAgent
from app.agents.pause import PauseRequest, PauseResult
from app.core.config import get_settings
from app.services.adaptive_campaign_substrate import (
    build_adaptive_campaign_substrate_snapshot,
)
from app.services.adaptive_substrate_inputs import (
    action_specs_from_registry,
    available_capabilities,
    campaign_instruments_from_protocol,
    failure_attribution_from_events,
    objective_state_from_input,
)
from app.services.decision_layer import CampaignDecisionLayer
from app.services.decision_trace import CampaignDecisionTraceBuilder
from app.services.primitives_registry import get_registry
from app.services.round_context import build_campaign_round_context

logger = logging.getLogger(__name__)

#: Cap on instrumented (experiment) actions mapped into the substrate snapshot
#: each round, to bound shadow-log payload size. Instrument-less actions
#: (report / informational / diagnostic) are always retained.
_ADAPTIVE_SUBSTRATE_ACTION_CAP = 24


def _maybe_record_contextual_shadow_decision(
    *,
    campaign_id: str,
    round_index: int,
    strategy_selection_result: dict[str, Any] | None = None,
    stop_requested: bool = False,
    failure_summary: dict[str, Any] | None = None,
    safety_summary: dict[str, Any] | None = None,
    objective_summary: dict[str, Any] | None = None,
    constraint_summary: dict[str, Any] | None = None,
    nexus_diagnostics: dict[str, Any] | None = None,
    backend_memory_summary: dict[str, Any] | None = None,
    bo_mcp_summary: dict[str, Any] | None = None,
    learning_policy_summary: dict[str, Any] | None = None,
    validation_summary: dict[str, Any] | None = None,
    human_observations: list[str] | None = None,
    literature_summary: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    actual_stage: str | None = None,
    actual_action: str | None = None,
) -> Any | None:
    """Record a contextual decision trace without affecting live routing."""
    try:
        if not get_settings().contextual_decision_shadow_enabled:
            return None

        context = build_campaign_round_context(
            campaign_id=campaign_id,
            round_index=round_index,
            strategy_selection_result=strategy_selection_result,
            stop_requested=stop_requested,
            failure_summary=failure_summary,
            safety_summary=safety_summary,
            objective_summary=objective_summary,
            constraint_summary=constraint_summary,
            nexus_diagnostics=nexus_diagnostics,
            backend_memory_summary=backend_memory_summary,
            bo_mcp_summary=bo_mcp_summary,
            learning_policy_summary=learning_policy_summary,
            validation_summary=validation_summary,
            human_observations=human_observations,
            literature_summary=literature_summary,
            metadata=metadata,
        )
        decision_plan = CampaignDecisionLayer().decide(context)
        trace = CampaignDecisionTraceBuilder().build(
            context=context,
            decision_plan=decision_plan,
            actual_stage=actual_stage,
            actual_action=actual_action or "propose_candidates",
            metadata=metadata,
        )
        logger.info(
            "contextual_shadow_decision_trace %s",
            json.dumps(trace.model_dump(mode="json"), sort_keys=True),
        )
        return trace
    except Exception:
        logger.warning(
            "Contextual shadow decision hook failed; continuing live campaign",
            exc_info=True,
        )
        return None


def _maybe_record_adaptive_campaign_substrate_snapshot(
    *,
    campaign_id: str,
    round_index: int,
    objective_kpi: str,
    max_rounds: int | None = None,
    failure_event_dicts: list[dict[str, Any]] | None = None,
    protocol_template: dict[str, Any] | None = None,
    safety_summary: dict[str, Any] | None = None,
    now: Any | None = None,
) -> Any | None:
    """Record the adaptive campaign substrate snapshot as a parallel shadow track.

    Strictly observational: it assembles the Phase 1-5 substrate artifact from
    read-only round inputs and logs it. It never affects routing, strategy
    selection, candidate selection, or execution, and its return value is
    intentionally ignored by the round loop. The VoI ranking inside the artifact
    is advisory only. Failures are swallowed so the live campaign is unaffected.
    """
    try:
        if not get_settings().adaptive_substrate_shadow_enabled:
            return None

        registry = get_registry()
        objective_state = objective_state_from_input(
            campaign_id=campaign_id,
            objective_kpi=objective_kpi,
            max_rounds=max_rounds,
        )
        failure_attribution = failure_attribution_from_events(
            list(failure_event_dicts or [])
        )
        campaign_instruments = campaign_instruments_from_protocol(
            protocol_template, registry
        )
        actions = action_specs_from_registry(
            registry,
            instruments=campaign_instruments or None,
            cap=_ADAPTIVE_SUBSTRATE_ACTION_CAP,
        )
        capabilities, capabilities_source = available_capabilities(
            campaign_instruments=campaign_instruments, registry=registry
        )

        snapshot = build_adaptive_campaign_substrate_snapshot(
            campaign_id=campaign_id,
            round_index=round_index,
            objective_state=objective_state,
            failure_attribution=failure_attribution,
            actions=actions,
            available_capabilities=capabilities,
            value_signals=[],
            safety_summary=safety_summary,
            now=now,
        )
        snapshot.metadata["available_capabilities_source"] = capabilities_source

        logger.info(
            "adaptive_campaign_substrate_snapshot %s",
            json.dumps(snapshot.model_dump(mode="json"), sort_keys=True),
        )
        return snapshot
    except Exception:
        logger.warning(
            "Adaptive substrate shadow hook failed; continuing live campaign",
            exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Orchestrator I/O
# ---------------------------------------------------------------------------

class OrchestratorInput(BaseModel):
    """Input to the orchestrator — a task contract plus execution options."""
    # Task contract fields (flattened for simplicity)
    contract_id: str
    objective_kpi: str
    direction: str  # "minimize" | "maximize"
    max_rounds: int
    batch_size: int
    strategy: str = "lhs"
    target_value: float | None = None

    # Parameter space
    dimensions: list[dict[str, Any]]
    protocol_template: dict[str, Any]

    # Safety
    policy_snapshot: dict[str, Any] = Field(default_factory=dict)

    # Protocol pattern
    protocol_pattern_id: str = ""

    # Options
    dry_run: bool = False
    plan_only: bool = False  # if True, only produce the plan, don't execute
    require_manual_confirmation: bool = Field(
        default=False,
        description=(
            "When True, every candidate execution requires operator approval "
            "via POST /api/v1/runs/{run_id}/approve before hardware runs. "
            "SSE events are emitted for each pending approval. "
            "Cleaning steps also require confirmation."
        ),
    )

    # External campaign ID (if provided by the API layer)
    campaign_id: str = ""

    # --- Enhancement: Auto deck layout from NL description ---
    deck_description: str = Field(
        default="",
        description="NL deck layout description (parsed by DeckLayoutAgent)",
    )

    # --- Enhancement: Tool holder configs ---
    tool_holders: list[str] = Field(
        default_factory=list,
        description="Paths to tool holder config JSON files to load",
    )

    # --- Enhancement: NLP code generation ---
    nl_intent: str = Field(
        default="",
        description="NL experiment description for code generation via NLPCodeAgent",
    )
    nl_auto_approve: bool = Field(
        default=False,
        description="Auto-approve NLP-generated code (skip user confirmation)",
    )

    # --- Enhancement: Cleaning workflows ---
    pre_clean_workflow: str = Field(
        default="",
        description="Cleaning workflow ID to run before each candidate execution",
    )
    post_clean_workflow: str = Field(
        default="",
        description="Cleaning workflow ID to run after each candidate execution",
    )


class RankedRecipe(BaseModel):
    """A recipe ranked by KPI performance with uncertainty estimate."""
    rank: int
    params: dict[str, Any]
    kpi_value: float
    kpi_uncertainty: float | None = None  # std from replicates, None if single obs
    n_observations: int = 1  # how many times this recipe was observed
    round_numbers: list[int] = Field(default_factory=list)


class OrchestratorOutput(BaseModel):
    """Output from the orchestrator — campaign results."""
    campaign_id: str
    status: str  # "planned" | "running" | "completed" | "failed"
    plan_summary: dict[str, Any] = Field(default_factory=dict)
    rounds_completed: int = 0
    best_kpi: float | None = None
    stop_reason: str = ""
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    # Top-K ranking with uncertainty
    top_k_recipes: list[RankedRecipe] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class OrchestratorAgent(BaseAgent[OrchestratorInput, OrchestratorOutput]):
    """Main orchestrator that coordinates all sub-agents.

    Lifecycle:
    1. Plan: PlannerAgent → CampaignPlan
    2. For each round:
       a. DesignAgent → candidate parameters
       b. CompilerAgent → executable DAG
       c. SafetyAgent → preflight check
       d. Execute (via worker) → with RecoveryAgent retry logic
       e. SensingAgent → QC check
       f. StopAgent → continue/stop decision
    """
    name = "orchestrator"
    description = "Main campaign orchestrator"
    layer = "top"

    def __init__(self):
        """Initialize orchestrator with recovery agent, strategy router, and control plane."""
        super().__init__()
        # Import here to avoid circular dependency
        from app.agents.recovery_agent import RecoveryAgent
        self.recovery = RecoveryAgent()

        # RL strategy router (optional — defaults to rule-based mode)
        try:
            from app.services.strategy_router import StrategyRouter
            self._strategy_router = StrategyRouter()
        except Exception:
            self._strategy_router = None
            logger.debug("Strategy router not available, using rule-based only", exc_info=True)

        # v2: ControlPlane for cross-agent routing and audit
        from app.agents.control_plane import ControlPlane
        from app.agents.stage import AgentStageRunner
        self._control_plane = ControlPlane()
        self._control_plane.set_pause_handler(self._handle_pause)
        self._stage_runner = AgentStageRunner(self._control_plane)

    def _emit(self, campaign_id: str, event: dict[str, Any]) -> None:
        """Persist event to DB, then publish to SSE subscribers (best-effort)."""
        # 1. Persist to campaign_events table
        try:
            from app.services.campaign_events import log_event
            seq = log_event(campaign_id, event.get("type", "agent_event"), event)
            event["_seq"] = seq  # attach seq for SSE id: field
        except Exception:
            pass  # DB write is best-effort
        # 2. Publish to in-memory SSE queues
        try:
            from app.api.v1.endpoints.orchestrate_events import publish_campaign_event
            publish_campaign_event(campaign_id, event)
        except Exception:
            pass  # SSE is best-effort; don't break orchestrator on publish failure

    @staticmethod
    def _build_campaign_context(input_data: OrchestratorInput) -> dict[str, Any]:
        """Derive the initial scientific context from the orchestrator contract."""
        constraints = []
        for dim in input_data.dimensions:
            if dim.get("choices") is not None:
                constraints.append({
                    "type": "categorical_domain",
                    "parameter": dim.get("param_name") or dim.get("name"),
                    "choices": dim.get("choices"),
                })
            if dim.get("min_value") is not None or dim.get("max_value") is not None:
                constraints.append({
                    "type": "bounds",
                    "parameter": dim.get("param_name") or dim.get("name"),
                    "min": dim.get("min_value"),
                    "max": dim.get("max_value"),
                })

        return {
            "scientific_goal": input_data.objective_kpi,
            "objective_hierarchy": [
                {
                    "level": "performance",
                    "name": input_data.objective_kpi,
                    "metric": input_data.objective_kpi,
                    "direction": input_data.direction,
                    "target": input_data.target_value,
                    "weight": 1.0,
                }
            ],
            "current_objective_level": "performance",
            "known_constraints": constraints,
            "measurement_protocols": [input_data.protocol_template]
            if input_data.protocol_template else [],
            "budget_remaining": {
                "max_rounds": input_data.max_rounds,
                "batch_size": input_data.batch_size,
            },
            "human_preferences": dict(input_data.policy_snapshot or {}),
            "synthesis_routes": [],
            "domain_hypotheses": [],
            "literature_priors": [],
            "human_observations": [],
        }

    @staticmethod
    def _campaign_context_from_dict(raw: dict[str, Any] | None):
        """Convert persisted context JSON into the selector dataclass."""
        from app.services.strategy_models import CampaignContext, ObjectiveSpec

        if not raw:
            return None
        objectives = tuple(
            ObjectiveSpec(
                level=obj.get("level", "performance"),
                name=obj.get("name", obj.get("metric", "")),
                metric=obj.get("metric"),
                direction=obj.get("direction", "maximize"),
                target=obj.get("target"),
                weight=float(obj.get("weight", 1.0)),
            )
            for obj in raw.get("objective_hierarchy", [])
            if isinstance(obj, dict)
        )
        return CampaignContext(
            scientific_goal=raw.get("scientific_goal", ""),
            objective_hierarchy=objectives,
            current_objective_level=raw.get("current_objective_level", "performance"),
            domain_hypotheses=tuple(raw.get("domain_hypotheses", []) or []),
            known_constraints=tuple(raw.get("known_constraints", []) or []),
            synthesis_routes=tuple(raw.get("synthesis_routes", []) or []),
            measurement_protocols=tuple(raw.get("measurement_protocols", []) or []),
            instrument_state=dict(raw.get("instrument_state", {}) or {}),
            material_family=raw.get("material_family"),
            prior_campaigns=tuple(raw.get("prior_campaigns", []) or []),
            literature_priors=tuple(raw.get("literature_priors", []) or []),
            budget_remaining=dict(raw.get("budget_remaining", {}) or {}),
            human_preferences=dict(raw.get("human_preferences", {}) or {}),
            human_observations=tuple(raw.get("human_observations", []) or []),
        )

    @staticmethod
    def _failure_events_from_dicts(raw: list[dict[str, Any]] | None):
        """Convert persisted failure-event JSON into selector dataclasses."""
        from app.services.strategy_models import FailureEvent

        return tuple(
            FailureEvent(
                failure_type=event.get("failure_type", "backend"),
                reason=event.get("reason", ""),
                backend_name=event.get("backend_name"),
                round_number=event.get("round_number"),
                candidate_index=event.get("candidate_index"),
                params=dict(event.get("params", {}) or {}),
                penalize_backend=bool(event.get("penalize_backend", False)),
            )
            for event in (raw or [])
            if isinstance(event, dict)
        )

    def _register_campaign_agents(self, campaign_id: str) -> None:
        """Register the campaign's agent runtime with the ControlPlane."""
        from app.agents.analyzer_agent import AnalyzerAgent
        from app.agents.cleaning_agent import CleaningAgent
        from app.agents.code_writer_agent import CodeWriterAgent
        from app.agents.compiler_agent import CompilerAgent
        from app.agents.deck_layout_agent import DeckLayoutAgent
        from app.agents.design_agent import DesignAgent
        from app.agents.monitor_agent import MonitorAgent
        from app.agents.nlp_code_agent import NLPCodeAgent
        from app.agents.planner_agent import PlannerAgent
        from app.agents.query_agent import QueryAgent
        from app.agents.safety_agent import SafetyAgent
        from app.agents.sensing_agent import SensingAgent
        from app.agents.simulation_agent import SimulationAgent
        from app.agents.stop_agent import StopAgent
        from app.agents.validation_agent import ValidationAgent

        self._control_plane.set_campaign_id(campaign_id)
        for agent in [
            PlannerAgent(),
            DeckLayoutAgent(),
            NLPCodeAgent(),
            CleaningAgent(),
            DesignAgent(),
            QueryAgent(),
            CompilerAgent(),
            CodeWriterAgent(),
            SafetyAgent(),
            SensingAgent(),
            ValidationAgent(),
            SimulationAgent(),
            StopAgent(),
            MonitorAgent(),
            AnalyzerAgent(),
            self.recovery,
        ]:
            self._control_plane.register(agent)

    def _campaign_stage_graph(self) -> dict[str, Any]:
        """Return the explicit stage graph used by the main campaign loop."""
        from app.agents.stage import AgentStageGraph

        graph = AgentStageGraph.linear(
            "orchestrator_main",
            [
                ("planning", ("planner_agent",)),
                ("design", ("design_agent",)),
                ("compile", ("compiler_agent",)),
                ("safety", ("safety_agent",)),
                ("analyze", ("analyzer_agent",)),
            ],
        )
        return graph.to_dict()

    async def _call_stage(
        self,
        *,
        campaign_id: str,
        stage_name: str,
        agent_name: str,
        input_data: BaseModel,
        trace_id: str | None = None,
        timeout_s: float = 300.0,
    ) -> AgentResult[Any]:
        """Run a single-agent stage through the shared stage runner."""
        return await self._stage_runner.run_single(
            stage_name,
            agent_name,
            input_data,
            caller="orchestrator",
            trace_id=trace_id,
            timeout_s=timeout_s,
            emit=lambda event: self._emit(campaign_id, event),
        )

    # ------------------------------------------------------------------
    # v2: Pause handler — persists to DB, emits SSE, polls for decision
    # ------------------------------------------------------------------

    async def _handle_pause(self, agent_name: str, request: PauseRequest) -> PauseResult:
        """Unified pause handler for all agents routed through ControlPlane.

        1. Persist to pause_requests table (survives restart)
        2. Emit SSE event for operator UI
        3. Poll DB until operator responds or timeout
        4. Return PauseResult to the requesting agent
        """
        from app.agents.pause import PauseResult
        from app.services.pause_store import get_pause_status, save_pause

        campaign_id = getattr(self, "_current_campaign_id", "")
        pause_id = request.pause_id

        # 1. Persist
        try:
            save_pause(
                campaign_id=campaign_id,
                agent_name=agent_name,
                pause_id=pause_id,
                reason=request.reason,
                risk_factors=request.risk_factors,
                suggested_action=request.suggested_action,
                checkpoint=request.checkpoint,
                metadata=request.metadata,
                expires_in_s=request.expires_in_s,
            )
        except Exception:
            logger.debug("Failed to persist pause request", exc_info=True)

        # 2. Emit SSE
        self._emit(campaign_id, {
            "type": "agent_pause_requested",
            "pause_id": pause_id,
            "agent": agent_name,
            "reason": request.reason,
            "risk_factors": request.risk_factors,
            "suggested_action": request.suggested_action,
            "expires_in_s": request.expires_in_s,
            "message": (
                f"⏸ {agent_name} requests human review: {request.reason} "
                f"— resolve via POST /api/v1/pauses/{pause_id}/resolve"
            ),
        })

        if getattr(self, "_auto_approve_pauses", False):
            self._emit(campaign_id, {
                "type": "agent_pause_resolved",
                "pause_id": pause_id,
                "agent": agent_name,
                "decision": "approved",
                "decided_by": "auto",
                "message": f"Auto-approved {agent_name} pause for dry-run/benchmark execution",
            })
            return PauseResult(decision="approved", decided_by="auto")

        # 3. Poll for decision
        _POLL_INTERVAL = 2.0
        _elapsed = 0.0

        while _elapsed < request.expires_in_s:
            await asyncio.sleep(_POLL_INTERVAL)
            _elapsed += _POLL_INTERVAL

            status = get_pause_status(pause_id)
            if status and status["status"] == "resolved":
                decision = status["decision"] or "approved"
                self._emit(campaign_id, {
                    "type": "agent_pause_resolved",
                    "pause_id": pause_id,
                    "agent": agent_name,
                    "decision": decision,
                    "decided_by": status.get("decided_by", ""),
                    "message": f"{'✅' if decision == 'approved' else '❌'} "
                               f"{agent_name} pause {decision}",
                })
                return PauseResult(
                    decision=decision,
                    modifications=status.get("modifications", {}),
                    decided_by=status.get("decided_by", ""),
                    decided_at=status.get("decided_at", ""),
                )

        # 4. Timeout
        self._emit(campaign_id, {
            "type": "agent_pause_resolved",
            "pause_id": pause_id,
            "agent": agent_name,
            "decision": "timeout",
            "message": f"⏰ {agent_name} pause timed out after {request.expires_in_s}s",
        })
        return PauseResult(decision="timeout")

    def validate_input(self, input_data: OrchestratorInput) -> list[str]:
        errors = []
        if not input_data.contract_id:
            errors.append("contract_id is required")
        if not input_data.objective_kpi:
            errors.append("objective_kpi is required")
        if input_data.max_rounds < 1:
            errors.append("max_rounds must be >= 1")
        if not input_data.dimensions:
            errors.append("At least one dimension is required")
        return errors

    async def process(
        self,
        input_data: OrchestratorInput,
        *,
        resume_from_round: int | None = None,
        restored_state: dict[str, Any] | None = None,
    ) -> OrchestratorOutput:
        campaign_id = input_data.campaign_id or f"camp-{uuid.uuid4().hex[:12]}"
        agent_trace: list[dict[str, Any]] = []
        # Store manual confirmation flag for _execute_real_run and cleaning hooks
        self._require_manual_confirmation = input_data.require_manual_confirmation
        self._auto_approve_pauses = bool(
            input_data.dry_run or input_data.policy_snapshot.get("auto_approve_pauses", False)
        )
        # v2: Track campaign_id for pause handler
        self._current_campaign_id = campaign_id
        self._register_campaign_agents(campaign_id)
        campaign_context_dict = self._build_campaign_context(input_data)
        failure_event_dicts: list[dict[str, Any]] = []

        # --- Checkpoint: create campaign in DB ---
        from app.services.campaign_state import (
            append_failure_event,
            append_space_revision,
            checkpoint_kpi,
            complete_candidate,
            complete_round,
            create_campaign,
            get_completed_rounds,
            is_candidate_done,
            save_plan,
            start_candidate,
            start_round,
            update_campaign_status,
            update_candidate_graph_hash,
        )
        if resume_from_round is None:
            # Fresh campaign
            create_campaign(
                campaign_id,
                input_data.model_dump(mode="json"),
                direction=input_data.direction,
                campaign_context=campaign_context_dict,
            )

        def _note_failure_event(event: dict[str, Any]) -> None:
            failure_event_dicts.append(event)
            try:
                append_failure_event(campaign_id, event)
            except Exception:
                logger.debug("Failed to persist failure event", exc_info=True)

        self._emit(campaign_id, {
            "type": "campaign_start",
            "campaign_id": campaign_id,
            "phase": "planning" if resume_from_round is None else "resuming",
            "message": "Starting campaign planning..." if resume_from_round is None else f"Resuming from round {resume_from_round}...",
        })
        self._emit(campaign_id, {
            "type": "agent_stage_graph",
            "graph": self._campaign_stage_graph(),
        })

        # ---- Phase 1: Planning (skip if resuming) ----
        from app.agents.planner_agent import PlannerInput

        if resume_from_round is not None:
            # Skip planning on resume — reload plan from DB
            from app.services.campaign_state import load_campaign as _load_campaign
            _saved = _load_campaign(campaign_id)
            if _saved is None or _saved.get("plan") is None:
                return OrchestratorOutput(
                    campaign_id=campaign_id,
                    status="failed",
                    errors=["Cannot resume: no saved plan found"],
                    agent_trace=agent_trace,
                )
            campaign_context_dict = dict(
                _saved.get("campaign_context") or campaign_context_dict
            )
            failure_event_dicts = list(_saved.get("failure_events") or [])
            # Re-run planner with same input to get CampaignPlan object
            plan_input = PlannerInput(
                contract_id=input_data.contract_id,
                objective_kpi=input_data.objective_kpi,
                direction=input_data.direction,
                max_rounds=input_data.max_rounds,
                batch_size=input_data.batch_size,
                strategy=input_data.strategy,
                target_value=input_data.target_value,
                dimensions=input_data.dimensions,
                protocol_template=input_data.protocol_template,
            )
            plan_result = await self._call_stage(
                campaign_id=campaign_id,
                stage_name="planning",
                agent_name="planner_agent",
                input_data=plan_input,
            )
            if not plan_result.success:
                return OrchestratorOutput(
                    campaign_id=campaign_id,
                    status="failed",
                    errors=["Resume re-planning failed: " + str(plan_result.errors)],
                    agent_trace=agent_trace,
                )
            plan = plan_result.output
            plan_summary = {
                "plan_id": plan.plan_id,
                "total_planned_runs": plan.total_planned_runs,
                "n_rounds": len(plan.planned_rounds),
                "strategy_schedule": plan.strategy_schedule,
                "estimated_tip_usage": plan.estimated_tip_usage,
            }
        else:
            self._emit(campaign_id, {
                "type": "agent_thinking",
                "agent": "planner",
                "message": "Analyzing parameter space and generating campaign plan...",
            })

            plan_input = PlannerInput(
                contract_id=input_data.contract_id,
                objective_kpi=input_data.objective_kpi,
                direction=input_data.direction,
                max_rounds=input_data.max_rounds,
                batch_size=input_data.batch_size,
                strategy=input_data.strategy,
                target_value=input_data.target_value,
                dimensions=input_data.dimensions,
                protocol_template=input_data.protocol_template,
            )

            plan_result = await self._call_stage(
                campaign_id=campaign_id,
                stage_name="planning",
                agent_name="planner_agent",
                input_data=plan_input,
            )
            agent_trace.append({
                "agent": "planner_agent",
                "success": plan_result.success,
                "duration_ms": plan_result.duration_ms,
                "errors": plan_result.errors,
            })

            self._emit(campaign_id, {
                "type": "agent_result",
                "agent": "planner",
                "success": plan_result.success,
                "duration_ms": plan_result.duration_ms,
                "message": "Plan generated" if plan_result.success else f"Planning failed: {plan_result.errors}",
            })

            if plan_result.success and plan_result.output and plan_result.output.decision_nodes:
                self._emit(campaign_id, {
                    "type": "agent_decision_tree",
                    "agent": "planner",
                    "round": 0,
                    "nodes": plan_result.output.decision_nodes,
                })

            if not plan_result.success:
                update_campaign_status(campaign_id, "failed", error=str(plan_result.errors))
                return OrchestratorOutput(
                    campaign_id=campaign_id,
                    status="failed",
                    errors=plan_result.errors,
                    agent_trace=agent_trace,
                )

            plan = plan_result.output
            plan_summary = {
                "plan_id": plan.plan_id,
                "total_planned_runs": plan.total_planned_runs,
                "n_rounds": len(plan.planned_rounds),
                "strategy_schedule": plan.strategy_schedule,
                "estimated_tip_usage": plan.estimated_tip_usage,
            }

            # --- Checkpoint: save plan ---
            try:
                save_plan(
                    campaign_id,
                    plan_summary,
                    total_rounds=len(plan.planned_rounds),
                )
                update_campaign_status(campaign_id, "running")
            except Exception:
                logger.debug("Failed to checkpoint plan", exc_info=True)

            if input_data.plan_only:
                update_campaign_status(campaign_id, "completed", stop_reason="plan_only")
                return OrchestratorOutput(
                    campaign_id=campaign_id,
                    status="planned",
                    plan_summary=plan_summary,
                    agent_trace=agent_trace,
                )

        # ---- Phase 1.5: Enhancement pre-processing ----

        # Enhancement 1: Auto deck layout from NL description
        if input_data.deck_description and resume_from_round is None:
            try:
                from app.agents.deck_layout_agent import DeckLayoutInput
                deck_input = DeckLayoutInput(
                    phase="parse",
                    deck_text=input_data.deck_description,
                    protocol_steps=input_data.protocol_template.get("steps", []),
                )
                deck_result = await self._control_plane.call(
                    "deck_layout",
                    deck_input,
                    caller="orchestrator",
                )
                if deck_result.success and deck_result.output:
                    agent_trace.append({
                        "agent": "deck_layout",
                        "success": True,
                        "status": deck_result.output.status,
                        "slot_count": deck_result.output.slot_count,
                    })
                    self._emit(campaign_id, {
                        "type": "agent_result",
                        "agent": "deck_layout",
                        "status": deck_result.output.status,
                        "message": deck_result.output.chat_message,
                    })
                    # Register custom labware
                    if deck_result.output.custom_labware_definitions:
                        from app.services.custom_labware_registry import (
                            get_custom_labware_definition,
                            register_custom_labware,
                        )
                        for cld in deck_result.output.custom_labware_definitions:
                            load_name = cld.get("load_name", "")
                            defn = get_custom_labware_definition(load_name)
                            if defn:
                                register_custom_labware(defn)
            except Exception:
                logger.debug("Deck layout enhancement failed", exc_info=True)

        # Enhancement 2: NLP code generation (with confirmation)
        if input_data.nl_intent and resume_from_round is None:
            try:
                from app.agents.nlp_code_agent import NLPCodeInput
                nlp_input = NLPCodeInput(
                    phase="generate",
                    intent=input_data.nl_intent,
                    context={
                        "dimensions": input_data.dimensions,
                        "objective_kpi": input_data.objective_kpi,
                    },
                    auto_approve=input_data.nl_auto_approve,
                    campaign_id=campaign_id,
                )
                nlp_result = await self._control_plane.call(
                    "nlp_code",
                    nlp_input,
                    caller="orchestrator",
                )
                if nlp_result.success and nlp_result.output:
                    agent_trace.append({
                        "agent": "nlp_code",
                        "success": True,
                        "status": nlp_result.output.status,
                    })
                    self._emit(campaign_id, {
                        "type": "agent_result",
                        "agent": "nlp_code",
                        "status": nlp_result.output.status,
                        "message": nlp_result.output.chat_message,
                    })
                    # If auto-approved, inject generated steps into protocol template
                    if nlp_result.output.status in ("auto_approved", "confirmed"):
                        if nlp_result.output.protocol_steps:
                            existing_steps = input_data.protocol_template.get("steps", [])
                            input_data.protocol_template["steps"] = (
                                existing_steps + nlp_result.output.protocol_steps
                            )
            except Exception:
                logger.debug("NLP code enhancement failed", exc_info=True)

        # Enhancement 3: Load tool holder configs
        if input_data.tool_holders:
            try:
                from app.services.tool_holder_config import load_tool_holder_config
                for th_path in input_data.tool_holders:
                    config = load_tool_holder_config(th_path)
                    agent_trace.append({
                        "agent": "tool_holder_loader",
                        "holder_name": config.holder_name,
                        "slot": config.slot_number,
                        "positions": config.position_names(),
                    })
                    self._emit(campaign_id, {
                        "type": "tool_holder_loaded",
                        "holder_name": config.holder_name,
                        "slot": config.slot_number,
                        "message": f"Tool holder '{config.holder_name}' loaded ({len(config.positions)} positions)",
                    })
            except Exception:
                logger.debug("Tool holder loading failed", exc_info=True)

        # ---- Phase 2: Execute rounds ----
        from app.agents.analyzer_agent import AnalyzerInput
        from app.agents.compiler_agent import CompileInput
        from app.agents.design_agent import DesignInput
        from app.agents.monitor_agent import MonitorInput
        from app.agents.safety_agent import SafetyCheckInput
        from app.agents.simulation_agent import SimulationInput
        from app.agents.stop_agent import StopInput
        from app.services.deck_layout import (
            WellExhaustedError,
            create_well_allocator_from_deck_plan,
            plan_deck_layout,
        )

        # --- Restore or init in-memory state ---
        if restored_state is not None:
            kpi_history = list(restored_state.get("kpi_history", []))
            all_kpis = list(restored_state.get("all_kpis", []))
            all_params = list(restored_state.get("all_params", []))
            all_rounds = list(restored_state.get("all_rounds", []))
            best_kpi = restored_state.get("best_kpi")
            total_runs = restored_state.get("total_runs", 0)
            # Per-backend recent-failure history (best-effort restore).
            backend_failure_counts: dict[str, int] = dict(
                restored_state.get("backend_failure_counts", {})
            )
            # Failed experiment coordinates (Dim 9 / P3b: avoid in generation).
            all_failed_params: list[dict[str, Any]] = list(
                restored_state.get("all_failed_params", [])
            )
            # (c) bomcp TuRBO trust-region state, threaded across rounds.
            bomcp_backend_state: dict[str, Any] | None = restored_state.get(
                "bomcp_backend_state"
            )
            latest_strategy_trace: dict[str, Any] | None = restored_state.get(
                "latest_strategy_trace"
            )
            backend_performance_records: dict[str, Any] = dict(
                restored_state.get("backend_performance", {}) or {}
            )
        else:
            kpi_history: list[float] = []
            all_kpis: list[float] = []
            all_params: list[dict[str, Any]] = []
            all_rounds: list[int] = []
            best_kpi: float | None = None
            total_runs = 0
            backend_failure_counts = {}
            all_failed_params = []
            bomcp_backend_state = None
            latest_strategy_trace = None
            backend_performance_records = {}

        step_history: list[dict[str, Any]] = []

        # Batch-level data for data-driven strategy selector (v3)
        last_batch_kpis: list[float] = []
        last_batch_params: list[dict[str, Any]] = []
        # QC tracking for replicate_need_score (v3)
        total_qc_checks: int = 0
        total_qc_fails: int = 0

        # Determine which rounds to skip on resume
        _completed_round_nums: set[int] = set()
        if resume_from_round is not None:
            _completed_round_nums = set(get_completed_rounds(campaign_id))

        # ---- Well allocation for destination labware ----
        # Build a minimal deck plan to initialise the well allocator.
        # The allocator tracks which destination well each (round, candidate)
        # pair is assigned to, preventing double-use across rounds.
        well_allocator = None
        try:
            _deck = plan_deck_layout(
                protocol_steps=input_data.protocol_template.get("steps", []),
                batch_size=input_data.batch_size,
            )
            well_allocator = create_well_allocator_from_deck_plan(_deck, role="destination")
            if well_allocator:
                self._emit(campaign_id, {
                    "type": "well_allocator_init",
                    "labware": well_allocator.labware_name,
                    "slot": well_allocator.slot_number,
                    "capacity": well_allocator.capacity,
                    "message": (
                        f"Well allocator: {well_allocator.capacity} wells "
                        f"in '{well_allocator.labware_name}' (slot {well_allocator.slot_number})"
                    ),
                })
        except Exception:
            logger.debug("Could not initialise well allocator", exc_info=True)

        for planned_round in plan.planned_rounds:
            round_num = planned_round.round_number

            # Skip completed rounds on resume
            if round_num in _completed_round_nums:
                logger.info("Skipping completed round %d on resume", round_num)
                continue

            self._emit(campaign_id, {
                "type": "round_start",
                "round": round_num,
                "total_rounds": len(plan.planned_rounds),
                "strategy": planned_round.strategy,
                "message": f"Starting round {round_num}/{len(plan.planned_rounds)} (strategy: {planned_round.strategy})",
            })

            # 2a. Design parameters — if "adaptive", re-select strategy
            #     using real-time KPI history AND batch-level data for
            #     data-driven switching.
            round_strategy = planned_round.strategy
            strategy_decision_info: dict[str, Any] = {}
            strategy_decision: Any = None  # holds StrategyDecision if adaptive

            if round_strategy == "adaptive" and kpi_history:
                try:
                    from app.services.optimization_backends import list_backends
                    from app.services.strategy_selector import (
                        CampaignSnapshot,
                        select_strategy,
                        strategy_trace_to_dict,
                    )

                    qc_fail_rate = (
                        total_qc_fails / total_qc_checks
                        if total_qc_checks > 0
                        else 0.0
                    )
                    snapshot = CampaignSnapshot(
                        round_number=round_num,
                        max_rounds=input_data.max_rounds,
                        n_observations=total_runs,
                        n_dimensions=len(input_data.dimensions),
                        has_categorical=any(
                            d.get("choices") is not None
                            for d in input_data.dimensions
                        ),
                        has_log_scale=any(
                            d.get("log_scale", False)
                            for d in input_data.dimensions
                        ),
                        kpi_history=tuple(kpi_history),
                        direction=input_data.direction,
                        user_strategy_hint=input_data.strategy if input_data.strategy != "adaptive" else "",
                        available_backends=list_backends(),
                        # Data-driven fields from last round
                        last_batch_kpis=tuple(last_batch_kpis),
                        last_batch_params=tuple(last_batch_params),
                        best_kpi_so_far=best_kpi,
                        # v3: full history for kNN signals
                        all_params=tuple(all_params),
                        all_kpis=tuple(all_kpis),
                        qc_fail_rate=qc_fail_rate,
                        # Δ2: recent per-backend failures bias backend ranking.
                        backend_failure_counts=dict(backend_failure_counts),
                        # Dim 9 / P3b: failed coordinates -> failure-region avoidance.
                        failed_params=tuple(all_failed_params),
                        campaign_context=self._campaign_context_from_dict(
                            campaign_context_dict
                        ),
                        failure_events=self._failure_events_from_dicts(
                            failure_event_dicts
                        ),
                        campaign_id=campaign_id,
                        previous_intent=(
                            latest_strategy_trace or {}
                        ).get("selected_intent"),
                        backend_performance_records=backend_performance_records,
                    )
                    # Route through RL or rule-based strategy router
                    if self._strategy_router is not None:
                        decision = self._strategy_router.select_strategy(
                            snapshot, campaign_id,
                        )
                    else:
                        decision = select_strategy(snapshot)
                    strategy_decision = decision  # store for stabilize check
                    _router_action = None  # track RL action for post-round hook
                    if hasattr(decision, 'reason') and '[RL-' in (decision.reason or ''):
                        # Extract action id from RL decision reason string
                        try:
                            import re
                            _action_match = re.search(r'action=(\d+)', decision.reason)
                            _router_action = int(_action_match.group(1)) if _action_match else None
                        except Exception:
                            _router_action = None
                    round_strategy = decision.backend_name

                    # Map backend names that aren't in candidate_gen._STRATEGIES
                    _BACKEND_PASSTHROUGH = {
                        "lhs", "random", "bayesian", "prior_guided", "grid",
                        "adaptive",
                    }
                    if round_strategy not in _BACKEND_PASSTHROUGH:
                        round_strategy = "adaptive"  # use adaptive path in candidate_gen

                    strategy_decision_info = {
                        "backend": decision.backend_name,
                        "phase": decision.phase,
                        "reason": decision.reason,
                        "confidence": decision.confidence,
                        "strategy_trace": strategy_trace_to_dict(
                            decision.strategy_trace
                        ),
                    }
                    latest_strategy_trace = strategy_decision_info.get(
                        "strategy_trace"
                    )
                    _space_revision = (
                        strategy_decision_info.get("strategy_trace") or {}
                    ).get("space_revision")
                    if _space_revision:
                        try:
                            append_space_revision(campaign_id, _space_revision)
                        except Exception:
                            logger.debug(
                                "Failed to persist space revision",
                                exc_info=True,
                            )

                    # Include diagnostic signals in SSE event
                    diag_info = {}
                    if decision.diagnostics:
                        diag_info = {
                            "space_coverage": decision.diagnostics.space_coverage,
                            "model_uncertainty": decision.diagnostics.model_uncertainty,
                            "noise_ratio": decision.diagnostics.noise_ratio,
                            "replicate_need_score": decision.diagnostics.replicate_need_score,
                            "local_smoothness": decision.diagnostics.local_smoothness,
                            "improvement_velocity": decision.diagnostics.improvement_velocity,
                            "ei_decay_proxy": decision.diagnostics.ei_decay_proxy,
                            "batch_kpi_cv": decision.diagnostics.batch_kpi_cv,
                            "batch_param_spread": decision.diagnostics.batch_param_spread,
                            "convergence_status": decision.diagnostics.convergence_status,
                            "convergence_confidence": decision.diagnostics.convergence_confidence,
                        }

                    # v3: phase posterior + actions + explanation
                    posterior_info = {}
                    if decision.phase_posterior:
                        posterior_info = {
                            "explore": decision.phase_posterior.explore,
                            "exploit": decision.phase_posterior.exploit,
                            "refine": decision.phase_posterior.refine,
                            "stabilize": decision.phase_posterior.stabilize,
                            "entropy": decision.phase_posterior.entropy,
                        }

                    actions_info = [
                        {
                            "name": a.name,
                            "backend": a.backend_name,
                            "utility": a.utility,
                            "reason": a.reason,
                        }
                        for a in decision.actions_considered[:4]  # top 4
                    ]

                    # v4: adaptive weights, drift, evidence, stabilize spec
                    weights_info = {}
                    if decision.weights_used:
                        weights_info = {
                            "w_improvement": decision.weights_used.w_improvement,
                            "w_info_gain": decision.weights_used.w_info_gain,
                            "w_risk": decision.weights_used.w_risk,
                            "reason": decision.weights_used.reason,
                        }

                    evidence_info = [
                        {
                            "signal": e.signal_name,
                            "value": e.signal_value,
                            "action": e.target_action,
                            "contribution": e.contribution,
                            "description": e.description,
                        }
                        for e in decision.evidence[:5]  # top 5
                    ]

                    stabilize_info = {}
                    if decision.stabilize_spec:
                        stabilize_info = {
                            "strategy": decision.stabilize_spec.strategy,
                            "n_points": len(decision.stabilize_spec.points_to_replicate),
                            "n_replicates": decision.stabilize_spec.n_replicates,
                            "reason": decision.stabilize_spec.reason,
                        }

                    self._emit(campaign_id, {
                        "type": "strategy_decision",
                        "round": round_num,
                        "backend": decision.backend_name,
                        "phase": decision.phase,
                        "reason": decision.reason,
                        "confidence": decision.confidence,
                        "diagnostics": diag_info,
                        "phase_posterior": posterior_info,
                        "actions": actions_info,
                        "explanation": decision.explanation,
                        # v4 fields
                        "weights_used": weights_info,
                        "drift_score": decision.drift_score,
                        "evidence": evidence_info,
                        "stabilize_spec": stabilize_info,
                        "strategy_trace": strategy_trace_to_dict(
                            decision.strategy_trace
                        ),
                        "message": f"Strategy: {decision.backend_name} ({decision.phase}) — {decision.reason}",
                    })
                except Exception:
                    logger.debug(
                        "Adaptive strategy selection failed, using planned strategy",
                        exc_info=True,
                    )

            _maybe_record_contextual_shadow_decision(
                campaign_id=campaign_id,
                round_index=round_num,
                strategy_selection_result=strategy_decision_info,
                failure_summary={
                    "events": list(failure_event_dicts),
                    "requires_recovery": any(
                        event.get("failure_type") in {"hardware", "backend"}
                        for event in failure_event_dicts
                        if isinstance(event, dict)
                    ),
                },
                safety_summary=dict(input_data.policy_snapshot or {}),
                objective_summary={
                    "objective_kpi": input_data.objective_kpi,
                    "direction": input_data.direction,
                    "target_value": input_data.target_value,
                    "max_rounds": input_data.max_rounds,
                },
                constraint_summary={
                    "dimensions": list(input_data.dimensions),
                    "policy_snapshot": dict(input_data.policy_snapshot or {}),
                },
                backend_memory_summary=dict(backend_performance_records),
                bo_mcp_summary={
                    "backend_state": bomcp_backend_state,
                },
                human_observations=list(
                    (campaign_context_dict or {}).get("human_observations", []) or []
                ),
                literature_summary={
                    "literature_priors": list(
                        (campaign_context_dict or {}).get("literature_priors", []) or []
                    ),
                },
                metadata={
                    "round_strategy": round_strategy,
                    "planned_strategy": planned_round.strategy,
                    "shadow_only": True,
                },
                actual_stage="candidate_generation",
            )

            # Parallel shadow track: adaptive campaign substrate snapshot.
            # Independent of the contextual decision trace above; observational
            # only, its return value is intentionally not consumed.
            _maybe_record_adaptive_campaign_substrate_snapshot(
                campaign_id=campaign_id,
                round_index=round_num,
                objective_kpi=input_data.objective_kpi,
                max_rounds=input_data.max_rounds,
                failure_event_dicts=failure_event_dicts,
                protocol_template=input_data.protocol_template,
                safety_summary=dict(input_data.policy_snapshot or {}),
            )

            # Reset per-round batch collectors
            round_batch_kpis: list[float] = []
            round_batch_params: list[dict[str, Any]] = []

            # --- Checkpoint: round start (after strategy decided) ---
            # n_candidates is not known yet — will be updated after design
            try:
                start_round(
                    campaign_id, round_num, round_strategy,
                    n_candidates=planned_round.batch_size,
                    strategy_decision=strategy_decision_info or None,
                )
            except Exception:
                logger.debug("Failed to checkpoint round start", exc_info=True)

            # --- Stabilize spec execution: bypass DesignAgent ---
            # When strategy selector returns stabilize + concrete spec,
            # replicate the specified points instead of generating new ones.
            stabilize_candidates: list[dict[str, Any]] | None = None
            if (
                strategy_decision is not None
                and strategy_decision.stabilize_spec is not None
                and strategy_decision.stabilize_spec.points_to_replicate
            ):
                spec = strategy_decision.stabilize_spec
                stabilize_candidates = []
                for pt in spec.points_to_replicate:
                    for _rep in range(spec.n_replicates):
                        stabilize_candidates.append(dict(pt))

                self._emit(campaign_id, {
                    "type": "stabilize_execution",
                    "round": round_num,
                    "strategy": spec.strategy,
                    "n_points": len(spec.points_to_replicate),
                    "n_replicates": spec.n_replicates,
                    "total_candidates": len(stabilize_candidates),
                    "message": (
                        f"Stabilize: replicating {len(spec.points_to_replicate)} points "
                        f"× {spec.n_replicates} replicates = {len(stabilize_candidates)} runs"
                    ),
                })

                agent_trace.append({
                    "agent": "stabilize_spec",
                    "round": round_num,
                    "strategy": spec.strategy,
                    "n_points": len(spec.points_to_replicate),
                    "n_replicates": spec.n_replicates,
                })

            if stabilize_candidates is not None:
                # Use stabilize-generated candidates directly
                design_candidates = stabilize_candidates
            else:
                # Normal path: generate candidates via DesignAgent
                design_input = DesignInput(
                    dimensions=input_data.dimensions,
                    protocol_template=input_data.protocol_template,
                    strategy=round_strategy,
                    batch_size=planned_round.batch_size,
                    seed=round_num,
                    campaign_id=campaign_id,
                    kpi_name=input_data.objective_kpi,
                    store=not input_data.dry_run,  # skip DB in dry_run mode
                    # Dim 9 / P3b: steer candidates away from failed coordinates.
                    failed_params=list(all_failed_params),
                    # (c) thread the bomcp TuRBO trust region into this round.
                    backend_state=bomcp_backend_state,
                )

                self._emit(campaign_id, {
                    "type": "agent_thinking",
                    "agent": "design",
                    "round": round_num,
                    "strategy": round_strategy,
                    "message": f"Designing {planned_round.batch_size} candidate parameters (strategy: {round_strategy})...",
                })

                design_result = await self._call_stage(
                    campaign_id=campaign_id,
                    stage_name="design",
                    agent_name="design_agent",
                    input_data=design_input,
                )
                agent_trace.append({
                    "agent": "design_agent",
                    "round": round_num,
                    "success": design_result.success,
                    "duration_ms": design_result.duration_ms,
                })

                self._emit(campaign_id, {
                    "type": "agent_result",
                    "agent": "design",
                    "round": round_num,
                    "success": design_result.success,
                    "n_candidates": len(design_result.output.candidates) if design_result.success else 0,
                    "message": f"Generated {len(design_result.output.candidates)} candidates" if design_result.success else f"Design failed: {design_result.errors}",
                })

                if not design_result.success:
                    logger.warning(
                        "Round %d: design failed: %s", round_num, design_result.errors
                    )
                    continue
                design_candidates = list(design_result.output.candidates)
                # (c) carry the bomcp TuRBO trust region into the next round.
                if design_result.output.backend_state is not None:
                    bomcp_backend_state = design_result.output.backend_state

            # ⑤: attach read-only memory recall (similar past candidates +
            # cross-campaign failure zones) to this round's persisted decision
            # trace. Evidence-only — it never changes candidate selection — and
            # fail-open so a recall error can never stop the round.
            try:
                from app.optimization.decision_evidence import (
                    evidence_for_candidates,
                )

                _round_evidence = evidence_for_candidates(
                    campaign_id,
                    design_candidates,
                    input_data.dimensions,
                    input_data.protocol_template,
                )
                _evidence_trace = (strategy_decision_info or {}).get("strategy_trace")
                if _round_evidence is not None and isinstance(_evidence_trace, dict):
                    _evidence_trace["evidence"] = _round_evidence
            except Exception:
                logger.debug(
                    "decision evidence recall failed; continuing", exc_info=True
                )

            # 2b. For each candidate, compile protocol
            for i, candidate_params in enumerate(design_candidates):
                # --- Checkpoint: candidate start + idempotent skip ---
                try:
                    start_candidate(campaign_id, round_num, i, candidate_params)
                except Exception:
                    logger.debug("Failed to checkpoint candidate start", exc_info=True)

                # Allocate destination well for this candidate
                dest_well: str | None = None
                if well_allocator is not None:
                    try:
                        dest_well = well_allocator.allocate(
                            round_number=round_num,
                            candidate_index=i,
                        )
                    except WellExhaustedError as wee:
                        logger.warning(
                            "Round %d candidate %d: %s", round_num, i, wee,
                        )
                        self._emit(campaign_id, {
                            "type": "well_exhausted",
                            "round": round_num,
                            "candidate": i,
                            "message": str(wee),
                        })
                        # Cannot proceed — break out of candidate loop
                        break

                # Build protocol with candidate params
                from app.services.protocol_patterns import get_pattern

                pattern = get_pattern(input_data.protocol_pattern_id)
                if pattern is None:
                    # Use the template as-is
                    protocol = input_data.protocol_template
                else:
                    protocol = pattern.to_protocol_json(candidate_params)

                compile_input = CompileInput(
                    protocol=protocol,
                    inputs={
                        "candidate_index": i,
                        "round": round_num,
                        "destination_well": dest_well,
                    },
                    policy_snapshot=input_data.policy_snapshot,
                )

                self._emit(campaign_id, {
                    "type": "agent_thinking",
                    "agent": "compiler",
                    "round": round_num,
                    "candidate": i,
                    "message": f"Compiling protocol for candidate {i}...",
                })

                compile_result = await self._call_stage(
                    campaign_id=campaign_id,
                    stage_name="compile",
                    agent_name="compiler_agent",
                    input_data=compile_input,
                )
                if not compile_result.success:
                    self._emit(campaign_id, {
                        "type": "agent_result",
                        "agent": "compiler",
                        "round": round_num,
                        "candidate": i,
                        "success": False,
                        "message": f"Compilation failed for candidate {i}",
                    })
                    try:
                        complete_candidate(campaign_id, round_num, i, status="failed", error="compilation_failed")
                    except Exception:
                        pass
                    _note_failure_event({
                        "failure_type": "protocol",
                        "reason": "Compilation failed before execution",
                        "backend_name": strategy_decision.backend_name
                        if strategy_decision is not None else None,
                        "round_number": round_num,
                        "candidate_index": i,
                        "params": candidate_params,
                        "penalize_backend": False,
                    })
                    continue

                # --- Idempotent skip: check graph_hash ---
                _graph_hash = getattr(compile_result.output, "graph_hash", None)
                if _graph_hash:
                    try:
                        update_candidate_graph_hash(campaign_id, round_num, i, _graph_hash)
                    except Exception:
                        pass
                    if is_candidate_done(campaign_id, round_num, i, _graph_hash):
                        logger.info(
                            "Round %d candidate %d: idempotent skip (hash=%s)",
                            round_num, i, _graph_hash,
                        )
                        self._emit(campaign_id, {
                            "type": "candidate_skipped",
                            "round": round_num,
                            "candidate": i,
                            "graph_hash": _graph_hash,
                            "message": f"Candidate {i} already completed (idempotent skip)",
                        })
                        continue

                # 2c. Safety check
                safety_input = SafetyCheckInput(
                    compiled_graph=compile_result.output.compiled_graph,
                    policy_snapshot=input_data.policy_snapshot,
                )

                self._emit(campaign_id, {
                    "type": "agent_thinking",
                    "agent": "safety",
                    "round": round_num,
                    "candidate": i,
                    "message": f"Running safety preflight for candidate {i}...",
                })

                safety_result = await self._call_stage(
                    campaign_id=campaign_id,
                    stage_name="safety",
                    agent_name="safety_agent",
                    input_data=safety_input,
                )
                if safety_result.success and not safety_result.output.allowed:
                    logger.warning(
                        "Round %d candidate %d: safety veto: %s",
                        round_num, i, safety_result.output.violations,
                    )
                    try:
                        complete_candidate(campaign_id, round_num, i, status="failed", error="safety_veto")
                    except Exception:
                        pass
                    _note_failure_event({
                        "failure_type": "constraint",
                        "reason": "Safety preflight vetoed the candidate",
                        "backend_name": strategy_decision.backend_name
                        if strategy_decision is not None else None,
                        "round_number": round_num,
                        "candidate_index": i,
                        "params": candidate_params,
                        "penalize_backend": True,
                    })
                    continue

                self._emit(campaign_id, {
                    "type": "agent_result",
                    "agent": "safety",
                    "round": round_num,
                    "candidate": i,
                    "allowed": safety_result.output.allowed if safety_result.success else True,
                    "message": "Safety check passed" if (not safety_result.success or safety_result.output.allowed) else f"Safety veto: {safety_result.output.violations}",
                })

                if safety_result.success and safety_result.output and safety_result.output.decision_nodes:
                    self._emit(campaign_id, {
                        "type": "agent_decision_tree",
                        "agent": "safety",
                        "round": round_num,
                        "candidate": i,
                        "nodes": safety_result.output.decision_nodes,
                    })

                # 2c-bis. Simulation — dry-run verification before physical execution
                sim_input = SimulationInput(
                    compiled_graph=compile_result.output.compiled_graph,
                    deck_plan=compile_result.output.deck_plan,
                    policy_snapshot=input_data.policy_snapshot,
                    candidate_params=candidate_params if isinstance(candidate_params, dict) else {},
                    round_number=round_num,
                    candidate_index=i,
                    emit=lambda ev: self._emit(campaign_id, ev),
                )
                sim_result = await self._control_plane.call(
                    "simulation_agent",
                    sim_input,
                    caller="orchestrator",
                )

                if sim_result.success:
                    sim_out = sim_result.output
                    self._emit(campaign_id, {
                        "type": "agent_result",
                        "agent": "simulation",
                        "round": round_num,
                        "candidate": i,
                        "success": sim_out.verdict != "fail",
                        "verdict": sim_out.verdict,
                        "n_errors": sim_out.n_errors,
                        "n_warnings": sim_out.n_warnings,
                        "estimated_duration_s": sim_out.estimated_duration_s,
                        "summary": sim_out.summary,
                        "message": sim_out.summary,
                    })
                    if sim_out.decision_nodes:
                        self._emit(campaign_id, {
                            "type": "agent_decision_tree",
                            "agent": "simulation",
                            "round": round_num,
                            "candidate": i,
                            "nodes": sim_out.decision_nodes,
                        })
                    if sim_out.verdict == "fail":
                        logger.warning(
                            "Round %d candidate %d: simulation veto (%d errors)",
                            round_num, i, sim_out.n_errors,
                        )
                        try:
                            complete_candidate(campaign_id, round_num, i, status="failed", error="simulation_fail")
                        except Exception:
                            pass
                        _note_failure_event({
                            "failure_type": "model",
                            "reason": "Simulation vetoed the candidate before execution",
                            "backend_name": strategy_decision.backend_name
                            if strategy_decision is not None else None,
                            "round_number": round_num,
                            "candidate_index": i,
                            "params": candidate_params,
                            "penalize_backend": False,
                        })
                        continue
                else:
                    logger.warning(
                        "Round %d candidate %d: simulation agent failed: %s",
                        round_num, i, sim_result.errors,
                    )

                # Enhancement 4: Pre-execution cleaning
                if input_data.pre_clean_workflow and not input_data.dry_run:
                    # Manual confirmation gate for cleaning
                    _clean_approved = True
                    if getattr(self, "_require_manual_confirmation", False):
                        _clean_approved = await self._await_cleaning_approval(
                            campaign_id=campaign_id,
                            workflow_id=input_data.pre_clean_workflow,
                            phase="pre",
                            round_num=round_num,
                            candidate_idx=i,
                        )
                    if _clean_approved:
                        try:
                            from app.agents.cleaning_agent import CleaningInput as _CleanInput
                            _clean_in = _CleanInput(
                                workflow_id=input_data.pre_clean_workflow,
                                step_prefix=f"round_{round_num}_cand_{i}_pre_",
                            )
                            _clean_res = await self._control_plane.call(
                                "cleaning",
                                _clean_in,
                                caller="orchestrator",
                            )
                            if _clean_res.success and _clean_res.output:
                                self._emit(campaign_id, {
                                    "type": "cleaning_complete",
                                    "phase": "pre",
                                    "round": round_num,
                                    "candidate": i,
                                    "message": _clean_res.output.chat_message,
                                })
                        except Exception:
                            logger.debug("Pre-clean failed", exc_info=True)
                    else:
                        self._emit(campaign_id, {
                            "type": "cleaning_approval_resolved",
                            "phase": "pre",
                            "round": round_num,
                            "candidate": i,
                            "decision": "skipped",
                            "message": f"Pre-clean skipped (not approved) for candidate {i}",
                        })

                # 2d. Execute
                run_kpi: float | None = None
                run_step_result: dict[str, Any] = {}

                well_info = f" → well {dest_well}" if dest_well else ""
                self._emit(campaign_id, {
                    "type": "agent_thinking",
                    "agent": "executor",
                    "round": round_num,
                    "candidate": i,
                    "destination_well": dest_well,
                    "message": f"Executing candidate {i}{well_info} ({'dry run' if input_data.dry_run else 'real hardware'})...",
                })

                if input_data.dry_run:
                    if input_data.policy_snapshot.get("simulation_oracle") == "electrochem":
                        from app.services.electrochem_surrogate import (
                            ElectrochemSurrogateConfig,
                            evaluate_electrochem_candidate,
                        )

                        oracle_seed = int(input_data.policy_snapshot.get("simulation_seed", 0))
                        oracle_noise = float(input_data.policy_snapshot.get("simulation_noise_std_mv", 2.0))
                        fault_profile = str(input_data.policy_snapshot.get("simulation_fault_profile", "none"))
                        observation = evaluate_electrochem_candidate(
                            candidate_params if isinstance(candidate_params, dict) else {},
                            config=ElectrochemSurrogateConfig(
                                seed=oracle_seed + round_num * 1000 + i,
                                noise_std_mv=oracle_noise,
                                fault_profile=fault_profile,
                            ),
                        )
                        run_kpi = observation.overpotential_mv
                        run_step_result = observation.to_step_result()
                    else:
                        # Simulate a KPI value for generic dry-run campaigns.
                        import random
                        run_kpi = random.gauss(100, 20)
                        run_step_result = {"simulated_kpi": run_kpi}
                else:
                    # Real execution with recovery: create run → dispatch to worker → collect results
                    # RecoveryAgent provides retry/abort/degrade strategies on failure
                    run_kpi, run_step_result = await self._execute_candidate_with_recovery(
                        campaign_id=campaign_id,
                        protocol=protocol,
                        inputs={"candidate_index": i, "round": round_num},
                        policy_snapshot=input_data.policy_snapshot,
                        objective_kpi=input_data.objective_kpi,
                        candidate_params=candidate_params,
                        agent_trace=agent_trace,
                        round_num=round_num,
                        candidate_idx=i,
                    )

                self._emit(campaign_id, {
                    "type": "agent_result",
                    "agent": "executor",
                    "round": round_num,
                    "candidate": i,
                    "kpi": run_kpi,
                    "message": f"Execution complete — KPI={run_kpi}" if run_kpi is not None else "Execution complete — no KPI",
                })

                # Enhancement 4: Post-execution cleaning
                if input_data.post_clean_workflow and not input_data.dry_run:
                    _clean_approved = True
                    if getattr(self, "_require_manual_confirmation", False):
                        _clean_approved = await self._await_cleaning_approval(
                            campaign_id=campaign_id,
                            workflow_id=input_data.post_clean_workflow,
                            phase="post",
                            round_num=round_num,
                            candidate_idx=i,
                        )
                    if _clean_approved:
                        try:
                            from app.agents.cleaning_agent import CleaningInput as _CleanInput
                            _clean_in = _CleanInput(
                                workflow_id=input_data.post_clean_workflow,
                                step_prefix=f"round_{round_num}_cand_{i}_post_",
                            )
                            _clean_res = await self._control_plane.call(
                                "cleaning",
                                _clean_in,
                                caller="orchestrator",
                            )
                            if _clean_res.success and _clean_res.output:
                                self._emit(campaign_id, {
                                    "type": "cleaning_complete",
                                    "phase": "post",
                                    "round": round_num,
                                    "candidate": i,
                                    "message": _clean_res.output.chat_message,
                                })
                        except Exception:
                            logger.debug("Post-clean failed", exc_info=True)
                    else:
                        self._emit(campaign_id, {
                            "type": "cleaning_approval_resolved",
                            "phase": "post",
                            "round": round_num,
                            "candidate": i,
                            "decision": "skipped",
                            "message": f"Post-clean skipped (not approved) for candidate {i}",
                        })

                # 2e. Quality check via MonitorAgent
                step_result = run_step_result
                monitor_input = MonitorInput(
                    step_key=f"round_{round_num}_candidate_{i}",
                    primitive="robot.dispense",
                    params=candidate_params,
                    step_result=step_result,
                    policy_snapshot=input_data.policy_snapshot,
                    step_history=step_history,
                    round_number=round_num,
                    emit=lambda event: self._emit(campaign_id, event),
                )

                monitor_result = await self._control_plane.call(
                    "monitor_agent",
                    monitor_input,
                    caller="orchestrator",
                )

                self._emit(campaign_id, {
                    "type": "agent_result",
                    "agent": "monitor",
                    "round": round_num,
                    "candidate": i,
                    "quality": monitor_result.output.overall_quality if monitor_result.success else "unknown",
                    "recommendation": monitor_result.output.recommendation if monitor_result.success else "unknown",
                    "message": f"QC: {monitor_result.output.overall_quality} — {monitor_result.output.recommendation}" if monitor_result.success else "QC check failed",
                })

                agent_trace.append({
                    "agent": "monitor_agent",
                    "round": round_num,
                    "candidate": i,
                    "quality": (
                        monitor_result.output.overall_quality
                        if monitor_result.success
                        else "unknown"
                    ),
                    "recommendation": (
                        monitor_result.output.recommendation
                        if monitor_result.success
                        else "unknown"
                    ),
                })

                # Accumulate step history for anomaly detection
                step_history.append(step_result)

                # Track QC outcomes for v3 qc_fail_rate
                total_qc_checks += 1

                # If monitor recommends abort, skip this candidate's KPI
                qc_quality = monitor_result.output.overall_quality if monitor_result.success else "unknown"
                if (
                    monitor_result.success
                    and monitor_result.output.recommendation == "abort"
                ):
                    total_qc_fails += 1
                    # Dim 9 / P3b: this coordinate produced a failed experiment;
                    # feed it to the failure-region model so future candidates
                    # steer around it.
                    all_failed_params.append(candidate_params)
                    logger.warning(
                        "Round %d candidate %d: QC failed, skipping",
                        round_num, i,
                    )
                    try:
                        complete_candidate(
                            campaign_id, round_num, i,
                            qc=qc_quality, status="failed", error="qc_abort",
                        )
                        checkpoint_kpi(
                            campaign_id, kpi_history, all_kpis, all_params,
                            all_rounds, best_kpi, total_runs,
                            backend_failure_counts=backend_failure_counts,
                            all_failed_params=all_failed_params,
                            bomcp_backend_state=bomcp_backend_state,
                            latest_strategy_trace=(strategy_decision_info or {}).get(
                                "strategy_trace"
                            ),
                        )
                    except Exception:
                        pass
                    _note_failure_event({
                        "failure_type": "measurement",
                        "reason": f"QC monitor recommended abort: {qc_quality}",
                        "backend_name": strategy_decision.backend_name
                        if strategy_decision is not None else None,
                        "round_number": round_num,
                        "candidate_index": i,
                        "params": candidate_params,
                        "penalize_backend": False,
                    })
                    continue

                # Record KPI (after QC pass)
                _candidate_run_id = step_result.get("run_id") if isinstance(step_result, dict) else None
                if run_kpi is not None:
                    kpi_history.append(run_kpi)
                    total_runs += 1

                    if best_kpi is None:
                        best_kpi = run_kpi
                    elif input_data.direction == "maximize":
                        best_kpi = max(best_kpi, run_kpi)
                    else:
                        best_kpi = min(best_kpi, run_kpi)

                    # Collect per-round batch data for strategy selector
                    round_batch_kpis.append(run_kpi)
                    round_batch_params.append(candidate_params)

                    # v3: accumulate full history for kNN signals
                    all_kpis.append(run_kpi)
                    all_params.append(candidate_params)
                    all_rounds.append(round_num)

                # --- Checkpoint: candidate completion + KPI snapshot ---
                try:
                    complete_candidate(
                        campaign_id, round_num, i,
                        kpi=run_kpi, run_id=_candidate_run_id,
                        qc=qc_quality, status="completed",
                    )
                    checkpoint_kpi(
                        campaign_id, kpi_history, all_kpis, all_params,
                        all_rounds, best_kpi, total_runs,
                        backend_failure_counts=backend_failure_counts,
                        all_failed_params=all_failed_params,
                        bomcp_backend_state=bomcp_backend_state,
                        latest_strategy_trace=(strategy_decision_info or {}).get(
                            "strategy_trace"
                        ),
                    )
                except Exception:
                    logger.debug("Failed to checkpoint candidate", exc_info=True)

            # Update shared batch data for next round's strategy decision
            if round_batch_kpis:
                last_batch_kpis = round_batch_kpis
                last_batch_params = round_batch_params

            # --- Checkpoint: round completion ---
            try:
                complete_round(campaign_id, round_num, round_batch_kpis, round_batch_params)
            except Exception:
                logger.debug("Failed to checkpoint round completion", exc_info=True)

            # --- RL post-round hook (online learning) ---
            if self._strategy_router is not None:
                try:
                    from app.services.strategy_selector import compute_diagnostics
                    _rl_diag = compute_diagnostics(snapshot) if strategy_decision and strategy_decision.diagnostics is None else (strategy_decision.diagnostics if strategy_decision else None)
                    _round_qc_fails = sum(1 for k in round_batch_kpis if k is None) if round_batch_kpis else 0
                    _prev_best = kpi_history[-2] if len(kpi_history) >= 2 else None
                    _is_terminal = (round_num >= input_data.max_rounds)
                    _target_reached = (
                        input_data.target_value is not None
                        and best_kpi is not None
                        and (
                            (input_data.direction == "maximize" and best_kpi >= input_data.target_value)
                            or (input_data.direction == "minimize" and best_kpi <= input_data.target_value)
                        )
                    )
                    self._strategy_router.on_round_complete(
                        campaign_id=campaign_id,
                        snapshot=snapshot,
                        diagnostics=_rl_diag,
                        action=_router_action if '_router_action' in dir() else None,
                        kpi_prev=_prev_best,
                        kpi_curr=best_kpi,
                        n_qc_failures=_round_qc_fails,
                        is_terminal=_is_terminal,
                        target_reached=_target_reached,
                    )
                except Exception:
                    logger.debug("RL post-round hook failed", exc_info=True)

            # --- Per-backend recent-failure history (feeds rank_backends) ---
            # Attribute this round's execution outcome (any QC failure) to the
            # backend the strategy selector chose for it.  Only adaptive rounds
            # set strategy_decision, so other rounds are skipped.
            if round_batch_kpis and strategy_decision is not None:
                from app.optimization.failure_history import update_backend_failures

                _round_had_failure = any(k is None for k in round_batch_kpis)
                backend_failure_counts = update_backend_failures(
                    backend_failure_counts,
                    strategy_decision.backend_name,
                    round_had_failure=_round_had_failure,
                )
                try:
                    checkpoint_kpi(
                        campaign_id, kpi_history, all_kpis, all_params,
                        all_rounds, best_kpi, total_runs,
                        backend_failure_counts=backend_failure_counts,
                        all_failed_params=all_failed_params,
                        bomcp_backend_state=bomcp_backend_state,
                        latest_strategy_trace=(strategy_decision_info or {}).get(
                            "strategy_trace"
                        ),
                    )
                except Exception:
                    logger.debug("Failed to checkpoint strategy state", exc_info=True)

            # 2e-ext. Causal model update (async, non-blocking)
            # After each round, update the causal graph with new observations.
            # Results are advisory only — failures never stall the campaign loop.
            if round_num >= 3 and round_num % 3 == 0:
                try:
                    from app.services.causal_engine import CausalReasoningService
                    _causal_svc = CausalReasoningService()
                    asyncio.create_task(
                        _causal_svc.update_causal_model(campaign_id, round_num),
                        name=f"causal-update-{campaign_id}-r{round_num}",
                    )
                except Exception:
                    logger.debug("Causal model update skipped", exc_info=True)

            # 2e-ext2. Knowledge bus: post-round harvest from strategy router
            try:
                from app.services.knowledge_bus import get_bus
                _kbus = get_bus(campaign_id)
                await _kbus.clear_round(str(round_num - 3))  # GC stale events
            except Exception:
                pass

            # 2f. AnalyzerAgent — per-round analysis and narrative
            _round_qc_fail_rate = (
                total_qc_fails / total_qc_checks if total_qc_checks > 0 else 0.0
            )
            analyzer_input = AnalyzerInput(
                round_number=round_num,
                direction=input_data.direction,
                kpi_name=input_data.objective_kpi,
                round_kpis=list(round_batch_kpis),
                round_params=list(round_batch_params),
                all_kpis=list(all_kpis),
                all_params=list(all_params),
                all_rounds=list(all_rounds),
                qc_fail_rate=_round_qc_fail_rate,
                max_rounds=input_data.max_rounds,
                n_dimensions=len(input_data.dimensions),
                has_categorical=any(
                    d.get("choices") is not None for d in input_data.dimensions
                ),
                has_log_scale=any(
                    d.get("log_scale", False) for d in input_data.dimensions
                ),
                step_history=list(step_history),
                emit=lambda event: self._emit(campaign_id, event),
            )
            analyzer_result = await self._call_stage(
                campaign_id=campaign_id,
                stage_name="analyze",
                agent_name="analyzer_agent",
                input_data=analyzer_input,
            )
            if analyzer_result.success:
                self._emit(campaign_id, {
                    "type": "agent_result",
                    "agent": "analyzer",
                    "round": round_num,
                    "narrative": analyzer_result.output.narrative,
                    "convergence": analyzer_result.output.convergence_status,
                    "best_kpi": analyzer_result.output.round_best_kpi,
                    "message": analyzer_result.output.narrative,
                })
                if analyzer_result.output.decision_nodes:
                    self._emit(campaign_id, {
                        "type": "agent_decision_tree",
                        "agent": "analyzer",
                        "round": round_num,
                        "nodes": analyzer_result.output.decision_nodes,
                    })
                agent_trace.append({
                    "agent": "analyzer_agent",
                    "round": round_num,
                    "narrative": analyzer_result.output.narrative,
                    "convergence": analyzer_result.output.convergence_status,
                })
            else:
                logger.warning(
                    "AnalyzerAgent failed for round %d: %s",
                    round_num, analyzer_result.errors,
                )

            # 2g. Stop decision
            stop_input = StopInput(
                kpi_history=kpi_history,
                current_round=round_num,
                max_rounds=input_data.max_rounds,
                target_value=input_data.target_value,
                direction=input_data.direction,
                total_runs_so_far=total_runs,
            )

            self._emit(campaign_id, {
                "type": "agent_thinking",
                "agent": "stop",
                "round": round_num,
                "message": f"Evaluating stop condition (best KPI so far: {best_kpi})...",
            })

            stop_result = await self._control_plane.call(
                "stop_agent",
                stop_input,
                caller="orchestrator",
            )
            agent_trace.append({
                "agent": "stop_agent",
                "round": round_num,
                "decision": stop_result.output.decision if stop_result.success else "error",
            })

            decision = stop_result.output.decision if stop_result.success else "error"
            self._emit(campaign_id, {
                "type": "agent_result",
                "agent": "stop",
                "round": round_num,
                "decision": decision,
                "best_kpi": best_kpi,
                "message": f"Stop decision: {decision}" + (f" (best KPI: {best_kpi})" if best_kpi is not None else ""),
            })

            if stop_result.success and stop_result.output.decision != "continue":
                top_k = self._compute_top_k_ranking(
                    all_params, all_kpis, all_rounds, input_data.direction,
                )
                # --- RL campaign complete hook (early stop) ---
                if self._strategy_router is not None:
                    try:
                        self._strategy_router.on_campaign_complete(campaign_id)
                    except Exception:
                        logger.debug("RL campaign hook failed", exc_info=True)
                # --- Checkpoint: campaign completed (early stop) ---
                try:
                    update_campaign_status(
                        campaign_id, "completed",
                        stop_reason=stop_result.output.decision,
                        best_kpi=best_kpi,
                    )
                except Exception:
                    logger.debug("Failed to checkpoint campaign completion", exc_info=True)
                self._emit(campaign_id, {
                    "type": "campaign_complete",
                    "campaign_id": campaign_id,
                    "status": "completed",
                    "rounds_completed": round_num,
                    "best_kpi": best_kpi,
                    "stop_reason": stop_result.output.decision,
                    "top_k_recipes": [r.model_dump() for r in top_k],
                    "message": f"Campaign completed — {stop_result.output.decision} (best KPI: {best_kpi})",
                })
                return OrchestratorOutput(
                    campaign_id=campaign_id,
                    status="completed",
                    plan_summary=plan_summary,
                    rounds_completed=round_num,
                    best_kpi=best_kpi,
                    stop_reason=stop_result.output.decision,
                    agent_trace=agent_trace,
                    top_k_recipes=top_k,
                )

        top_k = self._compute_top_k_ranking(
            all_params, all_kpis, all_rounds, input_data.direction,
        )
        # --- RL campaign complete hook (budget exhausted) ---
        if self._strategy_router is not None:
            try:
                self._strategy_router.on_campaign_complete(campaign_id)
            except Exception:
                logger.debug("RL campaign hook failed", exc_info=True)
        # --- Checkpoint: campaign completed (budget exhausted) ---
        try:
            update_campaign_status(
                campaign_id, "completed",
                stop_reason="budget_exhausted",
                best_kpi=best_kpi,
            )
        except Exception:
            logger.debug("Failed to checkpoint campaign completion", exc_info=True)
        self._emit(campaign_id, {
            "type": "campaign_complete",
            "campaign_id": campaign_id,
            "status": "completed",
            "rounds_completed": len(plan.planned_rounds),
            "best_kpi": best_kpi,
            "stop_reason": "budget_exhausted",
            "top_k_recipes": [r.model_dump() for r in top_k],
            "message": f"Campaign completed — budget exhausted after {len(plan.planned_rounds)} rounds (best KPI: {best_kpi})",
        })

        return OrchestratorOutput(
            campaign_id=campaign_id,
            status="completed",
            plan_summary=plan_summary,
            rounds_completed=len(plan.planned_rounds),
            best_kpi=best_kpi,
            stop_reason="budget_exhausted",
            agent_trace=agent_trace,
            top_k_recipes=top_k,
        )

    @staticmethod
    def _compute_top_k_ranking(
        all_params: list[dict[str, Any]],
        all_kpis: list[float],
        all_rounds: list[int],
        direction: str,
        k: int = 5,
    ) -> list[RankedRecipe]:
        """Rank unique recipes by KPI, with uncertainty from replicates.

        Recipes with identical param dicts are grouped.  For grouped recipes,
        the KPI is the mean and uncertainty is the standard deviation.
        """
        import math as _math

        # Group by param fingerprint
        groups: dict[str, list[tuple[float, int]]] = {}
        param_by_key: dict[str, dict[str, Any]] = {}
        for params, kpi, rnd in zip(all_params, all_kpis, all_rounds, strict=False):
            # Deterministic key from sorted param items
            key = str(sorted(
                (k, round(v, 8) if isinstance(v, float) else v)
                for k, v in params.items()
            ))
            if key not in groups:
                groups[key] = []
                param_by_key[key] = params
            groups[key].append((kpi, rnd))

        # Compute mean KPI and uncertainty per group
        scored: list[tuple[float, str]] = []
        for key, obs_list in groups.items():
            kpis = [o[0] for o in obs_list]
            mean_kpi = sum(kpis) / len(kpis)
            scored.append((mean_kpi, key))

        # Sort: best first
        is_minimize = direction == "minimize"
        scored.sort(key=lambda x: x[0], reverse=not is_minimize)

        recipes: list[RankedRecipe] = []
        for rank_idx, (mean_kpi, key) in enumerate(scored[:k]):
            obs_list = groups[key]
            kpis = [o[0] for o in obs_list]
            rounds = [o[1] for o in obs_list]
            n = len(kpis)
            if n >= 2:
                m = sum(kpis) / n
                variance = sum((x - m) ** 2 for x in kpis) / (n - 1)
                uncertainty = _math.sqrt(max(variance, 0.0))
            else:
                uncertainty = None
            recipes.append(RankedRecipe(
                rank=rank_idx + 1,
                params=param_by_key[key],
                kpi_value=round(mean_kpi, 6),
                kpi_uncertainty=round(uncertainty, 6) if uncertainty is not None else None,
                n_observations=n,
                round_numbers=sorted(rounds),
            ))
        return recipes

    async def resume_campaign(self, campaign_id: str) -> OrchestratorOutput:
        """Resume an incomplete campaign from its last checkpoint."""
        from app.services.campaign_state import (
            load_campaign,
            load_completed_candidates,
        )

        state = load_campaign(campaign_id)
        if state is None:
            raise ValueError(f"Campaign {campaign_id} not found in DB")

        if state["status"] in ("completed", "failed", "cancelled"):
            raise ValueError(
                f"Campaign {campaign_id} already {state['status']}, cannot resume"
            )

        # Rebuild OrchestratorInput from stored input_json
        input_data = OrchestratorInput(**state["input"])
        input_data.campaign_id = campaign_id

        # Rebuild accumulated state from DB
        restored = load_completed_candidates(campaign_id)

        # Find first incomplete round
        start_round_num = state["current_round"] or 1

        return await self.process(
            input_data,
            resume_from_round=start_round_num,
            restored_state=restored,
        )

    async def _execute_real_run(
        self,
        *,
        campaign_id: str,
        protocol: dict[str, Any],
        inputs: dict[str, Any],
        policy_snapshot: dict[str, Any],
        objective_kpi: str,
        candidate_params: dict[str, Any],
        agent_trace: list[dict[str, Any]],
        round_num: int,
        candidate_idx: int,
    ) -> tuple[float | None, dict[str, Any]]:
        """Execute a single candidate run on real hardware.

        Creates a run via run_service, dispatches to the worker in a thread,
        then extracts the KPI from the run artifacts.

        Returns (kpi_value, step_result_dict).
        """
        import asyncio

        kpi_value: float | None = None
        step_result: dict[str, Any] = {}

        try:
            from app.services.run_service import create_run, worker_load_run
            from app.worker import execute_run

            # Create DB run entry (compiles protocol, runs safety preflight)
            # When manual confirmation is enabled, force human approval
            effective_policy = dict(policy_snapshot)
            if getattr(self, "_require_manual_confirmation", False):
                effective_policy["require_human_approval"] = True

            run = await asyncio.to_thread(
                create_run,
                trigger_type="orchestrator",
                trigger_payload={
                    "campaign_id": campaign_id,
                    "round": round_num,
                    "candidate": candidate_idx,
                    "candidate_params": candidate_params,
                },
                campaign_id=campaign_id,
                protocol=protocol,
                inputs=inputs,
                policy_snapshot=effective_policy,
                actor="orchestrator_agent",
            )

            run_id = run["id"]
            run_status = run["status"]

            agent_trace.append({
                "agent": "worker",
                "round": round_num,
                "candidate": candidate_idx,
                "run_id": run_id,
                "initial_status": run_status,
            })

            # Only execute if run was scheduled (not rejected/awaiting approval)
            if run_status == "scheduled":
                returncode = await asyncio.to_thread(execute_run, run_id)

                if returncode == 0:
                    # Load completed run and extract KPI
                    completed_run = await asyncio.to_thread(worker_load_run, run_id)
                    step_result = {
                        "run_id": run_id,
                        "status": "succeeded",
                        "candidate_params": candidate_params,
                    }

                    # Extract KPI from run artifacts/results
                    # The KPI is typically stored in step results or computed
                    # from instrument data. For now, extract from the
                    # completed run's output if available.
                    run_outputs = completed_run.get("outputs", {})
                    if objective_kpi in run_outputs:
                        kpi_value = float(run_outputs[objective_kpi])
                    step_result["kpi"] = kpi_value
                else:
                    step_result = {
                        "run_id": run_id,
                        "status": "failed",
                        "returncode": returncode,
                    }
                    logger.warning(
                        "Round %d candidate %d: worker returned code %d",
                        round_num, candidate_idx, returncode,
                    )
            elif run_status == "rejected":
                step_result = {
                    "run_id": run_id,
                    "status": "rejected",
                    "reason": run.get("rejection_reason", ""),
                }
                logger.warning(
                    "Round %d candidate %d: run rejected: %s",
                    round_num, candidate_idx, run.get("rejection_reason"),
                )
            elif run_status == "awaiting_approval":
                # Emit SSE event for operator to review and approve
                self._emit(campaign_id, {
                    "type": "candidate_awaiting_approval",
                    "run_id": run_id,
                    "round": round_num,
                    "candidate": candidate_idx,
                    "candidate_params": candidate_params,
                    "message": (
                        f"⏳ Candidate {candidate_idx} (round {round_num}) "
                        f"ready — approve via POST /api/v1/runs/{run_id}/approve"
                    ),
                })
                logger.info(
                    "Round %d candidate %d: awaiting operator approval (run %s)",
                    round_num, candidate_idx, run_id,
                )

                # Poll DB until operator approves/rejects (or timeout)
                from app.services.run_service import get_run as _get_run
                _POLL_INTERVAL = 3.0   # seconds between polls
                _POLL_TIMEOUT = 1800.0  # 30 minutes max wait
                _elapsed = 0.0
                _resolved = False

                while _elapsed < _POLL_TIMEOUT:
                    await asyncio.sleep(_POLL_INTERVAL)
                    _elapsed += _POLL_INTERVAL
                    _current = await asyncio.to_thread(_get_run, run_id)
                    _cur_status = _current["status"] if _current else "unknown"

                    if _cur_status == "scheduled":
                        # Operator approved — execute now
                        self._emit(campaign_id, {
                            "type": "candidate_approval_resolved",
                            "run_id": run_id,
                            "round": round_num,
                            "candidate": candidate_idx,
                            "decision": "approved",
                            "message": f"✅ Candidate {candidate_idx} approved — executing",
                        })
                        returncode = await asyncio.to_thread(execute_run, run_id)
                        if returncode == 0:
                            completed_run = await asyncio.to_thread(worker_load_run, run_id)
                            step_result = {
                                "run_id": run_id,
                                "status": "succeeded",
                                "candidate_params": candidate_params,
                            }
                            run_outputs = completed_run.get("outputs", {})
                            if objective_kpi in run_outputs:
                                kpi_value = float(run_outputs[objective_kpi])
                            step_result["kpi"] = kpi_value
                        else:
                            step_result = {
                                "run_id": run_id,
                                "status": "failed",
                                "returncode": returncode,
                            }
                        _resolved = True
                        break
                    elif _cur_status == "rejected":
                        self._emit(campaign_id, {
                            "type": "candidate_approval_resolved",
                            "run_id": run_id,
                            "round": round_num,
                            "candidate": candidate_idx,
                            "decision": "rejected",
                            "message": f"❌ Candidate {candidate_idx} rejected by operator",
                        })
                        step_result = {
                            "run_id": run_id,
                            "status": "rejected",
                            "reason": _current.get("rejection_reason", "operator rejected"),
                        }
                        _resolved = True
                        break

                if not _resolved:
                    # Timeout — skip this candidate
                    self._emit(campaign_id, {
                        "type": "candidate_approval_resolved",
                        "run_id": run_id,
                        "round": round_num,
                        "candidate": candidate_idx,
                        "decision": "timeout",
                        "message": f"⏰ Candidate {candidate_idx} approval timed out after {_POLL_TIMEOUT}s",
                    })
                    step_result = {
                        "run_id": run_id,
                        "status": "approval_timeout",
                    }
                    logger.warning(
                        "Round %d candidate %d: approval timed out after %.0fs",
                        round_num, candidate_idx, _POLL_TIMEOUT,
                    )
            else:
                # Unknown status — skip
                step_result = {
                    "run_id": run_id,
                    "status": run_status,
                }

        except Exception as exc:
            logger.error(
                "Round %d candidate %d: execution failed: %s",
                round_num, candidate_idx, exc,
                exc_info=True,
            )
            step_result = {"status": "error", "error": str(exc)}

        return kpi_value, step_result

    async def _await_cleaning_approval(
        self,
        *,
        campaign_id: str,
        workflow_id: str,
        phase: str,
        round_num: int,
        candidate_idx: int,
    ) -> bool:
        """Wait for operator to approve a cleaning operation.

        Creates a confirmation request in the in-memory store, emits an SSE
        event, and polls until the operator responds or timeout.

        Returns True if approved, False if rejected or timed out.
        """
        import asyncio

        from app.services.code_confirmation import (
            CodeConfirmationRequest,
            CodeConfirmationStatus,
            get_confirmation_status,
            request_code_confirmation,
        )

        req = CodeConfirmationRequest(
            confirmation_type="cleaning",
            workflow_id=workflow_id,
            description=f"{phase}-clean for round {round_num} candidate {candidate_idx}",
            campaign_id=campaign_id,
        )
        req_id = request_code_confirmation(req)

        self._emit(campaign_id, {
            "type": "cleaning_awaiting_approval",
            "request_id": req_id,
            "workflow_id": workflow_id,
            "phase": phase,
            "round": round_num,
            "candidate": candidate_idx,
            "message": (
                f"🫧 {phase.capitalize()}-clean ({workflow_id}) for candidate {candidate_idx} "
                f"— approve via POST /api/v1/confirmations/{req_id}/respond"
            ),
        })

        _POLL_INTERVAL = 2.0
        _POLL_TIMEOUT = 600.0  # 10 minutes for cleaning approval
        _elapsed = 0.0

        while _elapsed < _POLL_TIMEOUT:
            await asyncio.sleep(_POLL_INTERVAL)
            _elapsed += _POLL_INTERVAL
            status = get_confirmation_status(req_id)
            if status == CodeConfirmationStatus.APPROVED:
                self._emit(campaign_id, {
                    "type": "cleaning_approval_resolved",
                    "request_id": req_id,
                    "phase": phase,
                    "round": round_num,
                    "candidate": candidate_idx,
                    "decision": "approved",
                    "message": f"✅ {phase.capitalize()}-clean approved",
                })
                return True
            elif status in (CodeConfirmationStatus.REJECTED, CodeConfirmationStatus.MODIFIED):
                self._emit(campaign_id, {
                    "type": "cleaning_approval_resolved",
                    "request_id": req_id,
                    "phase": phase,
                    "round": round_num,
                    "candidate": candidate_idx,
                    "decision": "rejected",
                    "message": f"❌ {phase.capitalize()}-clean rejected",
                })
                return False

        # Timeout
        self._emit(campaign_id, {
            "type": "cleaning_approval_resolved",
            "request_id": req_id,
            "phase": phase,
            "round": round_num,
            "candidate": candidate_idx,
            "decision": "timeout",
            "message": f"⏰ {phase.capitalize()}-clean approval timed out",
        })
        return False

    async def _execute_candidate_with_recovery(
        self,
        *,
        campaign_id: str,
        protocol: dict[str, Any],
        inputs: dict[str, Any],
        policy_snapshot: dict[str, Any],
        objective_kpi: str,
        candidate_params: dict[str, Any],
        agent_trace: list[dict[str, Any]],
        round_num: int,
        candidate_idx: int,
    ) -> tuple[float | None, dict[str, Any]]:
        """Execute candidate with recovery logic on failure.

        Wraps _execute_real_run with retry/abort/degrade recovery strategies.
        Returns (kpi_value, step_result) after recovery attempts.
        """
        from app.agents.recovery_agent import RecoveryInput
        from app.services.error_mapping import (
            extract_error_context,
            get_error_severity,
            map_exception_to_error_type,
        )

        max_retries = 3
        retry_count = 0
        retry_history: list[dict[str, Any]] = []
        recovery_episode: dict[str, Any] | None = None
        last_attempt_result: dict[str, Any] | None = None

        while retry_count <= max_retries:
            try:
                # Attempt execution
                kpi_value, step_result = await self._execute_real_run(
                    campaign_id=campaign_id,
                    protocol=protocol,
                    inputs=inputs,
                    policy_snapshot=policy_snapshot,
                    objective_kpi=objective_kpi,
                    candidate_params=candidate_params,
                    agent_trace=agent_trace,
                    round_num=round_num,
                    candidate_idx=candidate_idx,
                )

                # Check if execution failed
                if step_result.get("status") in ["failed", "error", "rejected"]:
                    error_msg = step_result.get("error", "Execution failed")

                    # Map error to recovery-agent type
                    error_type = map_exception_to_error_type(Exception(error_msg))
                    error_severity = get_error_severity(error_type)

                    # Build recovery input
                    recovery_input = RecoveryInput(
                        error_type=error_type,
                        error_message=error_msg,
                        device_name="campaign_execution",
                        device_status="error",
                        error_severity=error_severity,
                        telemetry={
                            "round": round_num,
                            "candidate": candidate_idx,
                            "retry_count": retry_count,
                            "run_id": step_result.get("run_id"),
                        },
                        history=retry_history,
                        retry_count=retry_count,
                        stage=f"round_{round_num}_candidate_{candidate_idx}",
                        episode=recovery_episode,
                        last_attempt_result=last_attempt_result,
                    )

                    # Get recovery decision
                    recovery_result = await self._call_recovery_agent(recovery_input)

                    if not recovery_result.success:
                        logger.error(
                            "Recovery agent failed: %s",
                            recovery_result.errors,
                        )
                        raise Exception(error_msg)

                    decision = recovery_result.output.decision
                    rationale = recovery_result.output.rationale
                    recovery_episode = recovery_result.output.episode.model_dump()
                    self._emit_recovery_phase(
                        campaign_id=campaign_id,
                        round_num=round_num,
                        candidate_idx=candidate_idx,
                        error_type=error_type,
                        error_severity=error_severity,
                        recovery_output=recovery_result.output,
                    )

                    # Emit recovery event
                    self._emit(campaign_id, {
                        "type": "recovery_decision",
                        "agent": "recovery",
                        "round": round_num,
                        "candidate": candidate_idx,
                        "error_type": error_type,
                        "error_severity": error_severity,
                        "decision": decision,
                        "rationale": rationale,
                        "retry_count": retry_count,
                        "chemical_safety_event": recovery_result.output.chemical_safety_event,
                        "episode_id": recovery_result.output.episode.episode_id,
                        "phase": recovery_result.output.phase,
                        "next_action": recovery_result.output.next_action,
                    })

                    # Check for chemical safety escalation
                    if recovery_result.output.chemical_safety_event:
                        logger.warning(
                            "🚨 Chemical safety event: %s - aborting",
                            error_type,
                        )
                        self._emit(campaign_id, {
                            "type": "chemical_safety_alert",
                            "round": round_num,
                            "candidate": candidate_idx,
                            "error_type": error_type,
                            "message": "Chemical safety event detected - SafetyAgent veto active",
                        })
                        # Force abort on chemical safety
                        raise Exception(f"Chemical safety event: {error_msg}")

                    # Execute recovery decision
                    if decision == "retry":
                        retry_count += 1
                        retry_delay = recovery_result.output.retry_delay_seconds

                        logger.info(
                            "Recovery: retry attempt %d/%d after %.1fs delay",
                            retry_count,
                            max_retries,
                            retry_delay,
                        )

                        # Add to history for fault signature analysis
                        retry_history.append({
                            "device_name": "campaign_execution",
                            "status": "error",
                            "telemetry": {
                                "error_type": error_type,
                                "retry_count": retry_count - 1,
                            },
                        })
                        last_attempt_result = {
                            "result": "failed",
                            "error_type": error_type,
                            "error_message": error_msg,
                            "retry_count": retry_count - 1,
                        }

                        if retry_delay > 0:
                            await asyncio.sleep(retry_delay)

                        continue  # Retry execution

                    elif decision == "abort":
                        logger.warning(
                            "Recovery: abort execution (rationale: %s)",
                            rationale[:100],
                        )
                        raise Exception(f"Recovery abort: {error_msg}")

                    elif decision == "skip":
                        logger.info(
                            "Recovery: skip candidate (rationale: %s)",
                            rationale[:100],
                        )
                        return None, {
                            "status": "skipped",
                            "reason": "recovery_skip",
                            "rationale": rationale,
                        }

                    elif decision == "degrade":
                        logger.info(
                            "Recovery: continue in degraded mode (rationale: %s)",
                            rationale[:100],
                        )
                        # Mark as degraded but return results
                        step_result["degraded"] = True
                        step_result["recovery_rationale"] = rationale
                        return kpi_value, step_result

                # Success path
                if retry_count > 0:
                    logger.info(
                        "Execution succeeded after %d retries",
                        retry_count,
                    )
                    # Emit success after recovery
                    self._emit(campaign_id, {
                        "type": "recovery_success",
                        "round": round_num,
                        "candidate": candidate_idx,
                        "retries": retry_count,
                        "episode_id": recovery_episode.get("episode_id") if recovery_episode else None,
                        "message": f"Execution succeeded after {retry_count} retries",
                    })

                return kpi_value, step_result

            except Exception as exc:
                # Exception during execution
                error_type = map_exception_to_error_type(exc)
                error_severity = get_error_severity(error_type)
                error_context = extract_error_context(exc)

                logger.error(
                    "Round %d candidate %d: execution exception: %s",
                    round_num,
                    candidate_idx,
                    exc,
                    exc_info=True,
                )

                # Build recovery input
                recovery_input = RecoveryInput(
                    error_type=error_type,
                    error_message=str(exc),
                    device_name="campaign_execution",
                    device_status="error",
                    error_severity=error_severity,
                    telemetry={
                        "round": round_num,
                        "candidate": candidate_idx,
                        "retry_count": retry_count,
                        **error_context,
                    },
                    history=retry_history,
                    retry_count=retry_count,
                    stage=f"round_{round_num}_candidate_{candidate_idx}",
                    episode=recovery_episode,
                    last_attempt_result=last_attempt_result,
                )

                # Get recovery decision
                recovery_result = await self._call_recovery_agent(recovery_input)

                if not recovery_result.success:
                    logger.error(
                        "Recovery agent failed: %s",
                        recovery_result.errors,
                    )
                    raise exc

                decision = recovery_result.output.decision
                rationale = recovery_result.output.rationale
                recovery_episode = recovery_result.output.episode.model_dump()
                self._emit_recovery_phase(
                    campaign_id=campaign_id,
                    round_num=round_num,
                    candidate_idx=candidate_idx,
                    error_type=error_type,
                    error_severity=error_severity,
                    recovery_output=recovery_result.output,
                )

                # Emit recovery event
                self._emit(campaign_id, {
                    "type": "recovery_decision",
                    "agent": "recovery",
                    "round": round_num,
                    "candidate": candidate_idx,
                    "error_type": error_type,
                    "error_severity": error_severity,
                    "decision": decision,
                    "rationale": rationale,
                    "retry_count": retry_count,
                    "chemical_safety_event": recovery_result.output.chemical_safety_event,
                    "episode_id": recovery_result.output.episode.episode_id,
                    "phase": recovery_result.output.phase,
                    "next_action": recovery_result.output.next_action,
                })

                # Check for chemical safety escalation
                if recovery_result.output.chemical_safety_event:
                    logger.warning(
                        "🚨 Chemical safety event: %s - aborting",
                        error_type,
                    )
                    self._emit(campaign_id, {
                        "type": "chemical_safety_alert",
                        "round": round_num,
                        "candidate": candidate_idx,
                        "error_type": error_type,
                        "message": "Chemical safety event detected - SafetyAgent veto active",
                    })
                    raise exc

                # Execute recovery decision
                if decision == "retry":
                    retry_count += 1
                    retry_delay = recovery_result.output.retry_delay_seconds

                    logger.info(
                        "Recovery: retry attempt %d/%d after %.1fs delay",
                        retry_count,
                        max_retries,
                        retry_delay,
                    )

                    # Add to history
                    retry_history.append({
                        "device_name": "campaign_execution",
                        "status": "error",
                        "telemetry": {
                            "error_type": error_type,
                            "error_message": str(exc),
                            "retry_count": retry_count - 1,
                        },
                    })
                    last_attempt_result = {
                        "result": "failed",
                        "error_type": error_type,
                        "error_message": str(exc),
                        "retry_count": retry_count - 1,
                    }

                    if retry_delay > 0:
                        await asyncio.sleep(retry_delay)

                    continue  # Retry execution

                elif decision == "abort":
                    logger.warning(
                        "Recovery: abort execution (rationale: %s)",
                        rationale[:100],
                    )
                    raise exc

                elif decision == "skip":
                    logger.info(
                        "Recovery: skip candidate (rationale: %s)",
                        rationale[:100],
                    )
                    return None, {
                        "status": "skipped",
                        "reason": "recovery_skip",
                        "rationale": rationale,
                    }

                elif decision == "degrade":
                    logger.info(
                        "Recovery: continue in degraded mode (rationale: %s)",
                        rationale[:100],
                    )
                    # Return degraded result
                    return None, {
                        "status": "degraded",
                        "reason": "recovery_degrade",
                        "rationale": rationale,
                        "error": str(exc),
                    }

        # Max retries exceeded
        logger.error(
            "Max retries (%d) exceeded for round %d candidate %d",
            max_retries,
            round_num,
            candidate_idx,
        )
        self._emit(campaign_id, {
            "type": "recovery_failed",
            "round": round_num,
            "candidate": candidate_idx,
            "retries": retry_count,
            "episode_id": recovery_episode.get("episode_id") if recovery_episode else None,
            "message": f"Max retries ({max_retries}) exceeded",
        })

        return None, {
            "status": "failed",
            "reason": "max_retries_exceeded",
            "retries": retry_count,
            "recovery_episode": recovery_episode,
        }

    def _emit_recovery_phase(
        self,
        *,
        campaign_id: str,
        round_num: int,
        candidate_idx: int,
        error_type: str,
        error_severity: str,
        recovery_output: Any,
    ) -> None:
        """Emit the current recovery episode phase for UI/audit visibility."""
        episode = recovery_output.episode
        latest_attempt = episode.attempts[-1].model_dump() if episode.attempts else None
        latest_observation = episode.observations[-1] if episode.observations else None
        self._emit(campaign_id, {
            "type": "recovery_phase",
            "agent": "recovery",
            "round": round_num,
            "candidate": candidate_idx,
            "episode_id": episode.episode_id,
            "phase": recovery_output.phase,
            "decision": recovery_output.decision,
            "error_type": error_type,
            "error_severity": error_severity,
            "latest_observation": latest_observation,
            "latest_attempt": latest_attempt,
            "rejected_hypotheses": episode.rejected_hypotheses,
            "next_action": recovery_output.next_action,
            "terminal": recovery_output.terminal,
        })

    async def _call_recovery_agent(self, recovery_input: BaseModel) -> AgentResult:
        """Route recovery decisions through the ControlPlane."""
        return await self._control_plane.call(
            "recovery_agent",
            recovery_input,
            caller="orchestrator.recovery",
            timeout_s=120.0,
        )
