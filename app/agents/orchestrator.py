"""Main Orchestrator Agent — routes tasks to sub-agents.

This is the top-level agent that scientists interact with (via the API).
It receives a TaskContract and drives the full campaign lifecycle:
  L3 (intake) → L2 (planning) → L1 (compilation) → L0 (execution)

The orchestrator does NOT do work itself — it delegates to sub-agents
and handles the flow of contracts between layers.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent, AgentResult

logger = logging.getLogger(__name__)


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

    # External campaign ID (if provided by the API layer)
    campaign_id: str = ""


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
        """Initialize orchestrator with recovery agent."""
        super().__init__()
        # Import here to avoid circular dependency
        from app.agents.recovery_agent import RecoveryAgent
        self.recovery = RecoveryAgent()

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

        # --- Checkpoint: create campaign in DB ---
        from app.services.campaign_state import (
            create_campaign,
            save_plan,
            start_round,
            complete_round,
            start_candidate,
            complete_candidate,
            update_candidate_graph_hash,
            is_candidate_done,
            checkpoint_kpi,
            update_campaign_status,
            get_completed_rounds,
        )
        if resume_from_round is None:
            # Fresh campaign
            create_campaign(
                campaign_id,
                input_data.model_dump(mode="json"),
                direction=input_data.direction,
            )

        self._emit(campaign_id, {
            "type": "campaign_start",
            "campaign_id": campaign_id,
            "phase": "planning" if resume_from_round is None else "resuming",
            "message": "Starting campaign planning..." if resume_from_round is None else f"Resuming from round {resume_from_round}...",
        })

        # ---- Phase 1: Planning (skip if resuming) ----
        from app.agents.planner_agent import PlannerAgent, PlannerInput

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
            # Re-run planner with same input to get CampaignPlan object
            planner = PlannerAgent()
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
            plan_result = await planner.run(plan_input)
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

            planner = PlannerAgent()
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

            plan_result = await planner.run(plan_input)
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

        # ---- Phase 2: Execute rounds ----
        from app.agents.design_agent import DesignAgent, DesignInput
        from app.agents.compiler_agent import CompilerAgent, CompileInput
        from app.agents.safety_agent import SafetyAgent, SafetyCheckInput
        from app.agents.stop_agent import StopAgent, StopInput
        from app.agents.sensing_agent import SensingAgent, SensingInput
        from app.services.deck_layout import (
            create_well_allocator_from_deck_plan,
            plan_deck_layout,
            WellExhaustedError,
        )

        design_agent = DesignAgent()
        compiler = CompilerAgent()
        safety = SafetyAgent()
        stop_agent = StopAgent()
        sensing = SensingAgent()

        # --- Restore or init in-memory state ---
        if restored_state is not None:
            kpi_history = list(restored_state.get("kpi_history", []))
            all_kpis = list(restored_state.get("all_kpis", []))
            all_params = list(restored_state.get("all_params", []))
            all_rounds = list(restored_state.get("all_rounds", []))
            best_kpi = restored_state.get("best_kpi")
            total_runs = restored_state.get("total_runs", 0)
        else:
            kpi_history: list[float] = []
            all_kpis: list[float] = []
            all_params: list[dict[str, Any]] = []
            all_rounds: list[int] = []
            best_kpi: float | None = None
            total_runs = 0

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
                    from app.services.strategy_selector import (
                        CampaignSnapshot,
                        select_strategy,
                    )
                    from app.services.optimization_backends import list_backends

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
                    )
                    decision = select_strategy(snapshot)
                    strategy_decision = decision  # store for stabilize check
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
                    }

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
                        "message": f"Strategy: {decision.backend_name} ({decision.phase}) — {decision.reason}",
                    })
                except Exception:
                    logger.debug(
                        "Adaptive strategy selection failed, using planned strategy",
                        exc_info=True,
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
                )

                self._emit(campaign_id, {
                    "type": "agent_thinking",
                    "agent": "design",
                    "round": round_num,
                    "strategy": round_strategy,
                    "message": f"Designing {planned_round.batch_size} candidate parameters (strategy: {round_strategy})...",
                })

                design_result = await design_agent.run(design_input)
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

                compile_result = await compiler.run(compile_input)
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

                safety_result = await safety.run(safety_input)
                if safety_result.success and not safety_result.output.allowed:
                    logger.warning(
                        "Round %d candidate %d: safety veto: %s",
                        round_num, i, safety_result.output.violations,
                    )
                    try:
                        complete_candidate(campaign_id, round_num, i, status="failed", error="safety_veto")
                    except Exception:
                        pass
                    continue

                self._emit(campaign_id, {
                    "type": "agent_result",
                    "agent": "safety",
                    "round": round_num,
                    "candidate": i,
                    "allowed": safety_result.output.allowed if safety_result.success else True,
                    "message": "Safety check passed" if (not safety_result.success or safety_result.output.allowed) else f"Safety veto: {safety_result.output.violations}",
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
                    # Simulate a KPI value
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

                # 2e. Quality check via SensingAgent
                step_result = run_step_result
                sensing_input = SensingInput(
                    step_key=f"round_{round_num}_candidate_{i}",
                    primitive="robot.dispense",
                    params=candidate_params,
                    step_result=step_result,
                    policy_snapshot=input_data.policy_snapshot,
                    step_history=step_history,
                )
                self._emit(campaign_id, {
                    "type": "agent_thinking",
                    "agent": "sensing",
                    "round": round_num,
                    "candidate": i,
                    "message": f"Quality checking candidate {i} results...",
                })

                sensing_result = await sensing.run(sensing_input)

                self._emit(campaign_id, {
                    "type": "agent_result",
                    "agent": "sensing",
                    "round": round_num,
                    "candidate": i,
                    "quality": sensing_result.output.overall_quality if sensing_result.success else "unknown",
                    "recommendation": sensing_result.output.recommendation if sensing_result.success else "unknown",
                    "message": f"QC: {sensing_result.output.overall_quality} — {sensing_result.output.recommendation}" if sensing_result.success else "QC check failed",
                })

                agent_trace.append({
                    "agent": "sensing_agent",
                    "round": round_num,
                    "candidate": i,
                    "quality": (
                        sensing_result.output.overall_quality
                        if sensing_result.success
                        else "unknown"
                    ),
                    "recommendation": (
                        sensing_result.output.recommendation
                        if sensing_result.success
                        else "unknown"
                    ),
                })

                # Accumulate step history for anomaly detection
                step_history.append(step_result)

                # Track QC outcomes for v3 qc_fail_rate
                total_qc_checks += 1

                # If sensing recommends abort, skip this candidate's KPI
                qc_quality = sensing_result.output.overall_quality if sensing_result.success else "unknown"
                if (
                    sensing_result.success
                    and sensing_result.output.recommendation == "abort"
                ):
                    total_qc_fails += 1
                    logger.warning(
                        "Round %d candidate %d: QC failed, skipping",
                        round_num, i,
                    )
                    try:
                        complete_candidate(
                            campaign_id, round_num, i,
                            qc=qc_quality, status="failed", error="qc_abort",
                        )
                    except Exception:
                        pass
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

            # 2f. Stop decision
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

            stop_result = await stop_agent.run(stop_input)
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
        for params, kpi, rnd in zip(all_params, all_kpis, all_rounds):
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
                policy_snapshot=policy_snapshot,
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
            else:
                # awaiting_approval or other — skip for autonomous mode
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
            map_exception_to_error_type,
            get_error_severity,
            extract_error_context,
            should_emit_chemical_safety_alert,
        )

        max_retries = 3
        retry_count = 0
        retry_history: list[dict[str, Any]] = []

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
                    )

                    # Get recovery decision
                    recovery_result = await self.recovery.run(recovery_input)

                    if not recovery_result.success:
                        logger.error(
                            "Recovery agent failed: %s",
                            recovery_result.errors,
                        )
                        raise Exception(error_msg)

                    decision = recovery_result.output.decision
                    rationale = recovery_result.output.rationale

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
                )

                # Get recovery decision
                recovery_result = await self.recovery.run(recovery_input)

                if not recovery_result.success:
                    logger.error(
                        "Recovery agent failed: %s",
                        recovery_result.errors,
                    )
                    raise exc

                decision = recovery_result.output.decision
                rationale = recovery_result.output.rationale

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
            "message": f"Max retries ({max_retries}) exceeded",
        })

        return None, {
            "status": "failed",
            "reason": "max_retries_exceeded",
            "retries": retry_count,
        }
