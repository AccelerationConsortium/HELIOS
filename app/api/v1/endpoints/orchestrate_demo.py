"""Demo endpoint for the frontend UI visualization.

Sends a detailed, realistic agent execution trace for screen-recording
demos and live walkthroughs. The 8-round trajectory covers:

  - Round 1–3: LHS exploration (space-filling)
  - Round 4:   Strategy switch → Bayesian kNN
  - Round 5:   Recovery event (a primitive fails, gets fixed-forward)
  - Round 6:   Human-in-the-loop pause + auto-approve
  - Round 7:   BO posterior update, refined search
  - Round 8:   Convergence detected → stop

Plus ancillary events the lab UI hooks need to render:
  - agent_thinking / agent_result / agent_decision_tree
  - strategy_decision with phase posterior
  - safety_check pass/fail
  - tool_call / hardware_action
  - recovery_decision / recovery_success
  - approval_requested / approval_received
  - analyzer_result with narrative
  - campaign_complete with top_k_recipes
  - memory_recall_event (triggers the Memory Recall banner)

The KPI trajectory is hand-curated to look like a real Bayesian
optimization run: 198 → 165 → 142 → 118 → 105 (after recovery) → 88
→ 72 → 58 mV (73% reduction over 8 rounds).
"""
from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orchestrate", tags=["orchestrate"])

# Registry of running demo tasks. Holds a strong reference so tasks aren't
# garbage-collected mid-flight; entries remove themselves on completion.
_demo_tasks: dict[str, asyncio.Task] = {}


def is_demo_campaign(campaign_id: str) -> bool:
    """True if this id belongs to a demo campaign (running or recently run)."""
    return campaign_id.startswith("demo-")


# Hand-curated KPI trajectory — looks like a real BO run
ETA10_TRAJECTORY = [198.0, 165.0, 142.0, 118.0, 105.0, 88.0, 72.0, 58.0]


class DemoRequest(BaseModel):
    """Demo campaign request."""
    objective_kpi: str = "overpotential_eta10"
    max_rounds: int = 8   # 8 is the realistic default; client can override


class DemoResponse(BaseModel):
    """Demo campaign response."""
    campaign_id: str
    status: str = "started"


def _strategy_for_round(n: int) -> str:
    if n <= 3:
        return "lhs"
    if n == 4:
        return "bayesian_knn"  # switch point
    return "bayesian_bo"


def _phase_posterior(round_num: int) -> dict:
    """Bayesian phase posterior for the strategy decision tree."""
    if round_num <= 2:
        return {"explore": 0.85, "exploit": 0.05, "refine": 0.05, "stabilize": 0.05}
    if round_num <= 4:
        return {"explore": 0.55, "exploit": 0.25, "refine": 0.15, "stabilize": 0.05}
    if round_num <= 6:
        return {"explore": 0.20, "exploit": 0.45, "refine": 0.30, "stabilize": 0.05}
    return {"explore": 0.05, "exploit": 0.25, "refine": 0.40, "stabilize": 0.30}


def _dominant_phase(posterior: dict) -> str:
    """Return the phase name with the highest posterior mass."""
    best_phase, _ = max(posterior.items(), key=lambda item: item[1])
    return best_phase


async def _emit_round(campaign_id: str, emitter, round_num: int,
                     eta10: float, prev_eta10: float | None,
                     simulate_failure: bool = False) -> None:
    """Emit a full round's events: start → decisions → actions → result."""
    from app.api.v1.endpoints.orchestrate_events import publish_campaign_event

    strategy = _strategy_for_round(round_num)
    is_recovery_round = simulate_failure

    # 1. Round start
    publish_campaign_event(campaign_id, {
        "type": "round_start",
        "round": round_num,
        "total_rounds": 8,
        "strategy": strategy,
        "message": f"Starting round {round_num}/8 (strategy: {strategy})",
    })
    await asyncio.sleep(0.25)

    # 2. Strategy decision (with phase posterior — feeds Live tab strategy pill)
    posterior = _phase_posterior(round_num)
    publish_campaign_event(campaign_id, {
        "type": "strategy_decision",
        "round": round_num,
        "backend": strategy,
        "phase": _dominant_phase(posterior),
        "phase_posterior": posterior,
        "confidence": 0.55 + (round_num * 0.05),
        "drift_score": max(0.0, 0.25 - (round_num * 0.03)),
        "reason": (
            "LHS space-filling to map response surface"
            if round_num <= 3 else
            "Switching to Bayesian kNN — posterior mass has shifted toward exploitation"
            if round_num == 4 else
            "Refining around current best — GP posterior uncertainty narrowing"
        ),
    })
    await asyncio.sleep(0.2)

    # 3. Safety check (one of: tip-volume, KOH-concentration, current-density)
    safety_check = [
        ("Tip volume limit", 3.0, "mL max",  True),
        ("KOH concentration range", 0.5, "M",  True),
        ("Current density safe window", 12.0, "mA/cm²",  True),
    ][round_num % 3]
    publish_campaign_event(campaign_id, {
        "type": "safety_check",
        "round": round_num,
        "check_name": safety_check[0],
        "passed": safety_check[3],
        "details": f"{safety_check[1]} {safety_check[2]}",
    })
    await asyncio.sleep(0.15)

    # 4. Tool call: design
    publish_campaign_event(campaign_id, {
        "type": "tool_call",
        "round": round_num,
        "tool": "design_agent",
        "operation": "generate_candidates",
        "params": {"n_points": 4, "strategy": strategy},
        "agent": "design",
    })
    await asyncio.sleep(0.2)

    # 5. Tool call: compile (only on every other round to vary pacing)
    if round_num % 2 == 0:
        publish_campaign_event(campaign_id, {
            "type": "tool_call",
            "round": round_num,
            "tool": "compiler",
            "operation": "emit_protocol",
            "params": {"lines": 380 + round_num * 12},
            "agent": "compiler",
        })
        await asyncio.sleep(0.15)

    # 6. Hardware action: aspirate + dispense + (sometimes) fail
    if is_recovery_round:
        # Fail the aspirate — this triggers the recovery flow
        publish_campaign_event(campaign_id, {
            "type": "hardware_action",
            "round": round_num,
            "hardware": "opentrons_ot2",
            "action": "aspirate",
            "details": {"volume_ul": 200, "tip_idx": 3},
            "agent": "executor",
        })
        await asyncio.sleep(0.2)
        publish_campaign_event(campaign_id, {
            "type": "agent_result",
            "round": round_num,
            "agent": "executor",
            "success": False,
            "message": "Aspirate failed: tip 3 not attached (sensor P12 LOW)",
            "error_type": "tip_not_attached",
            "duration_ms": 8500,
        })
        await asyncio.sleep(0.3)
        # Recovery agent kicks in
        publish_campaign_event(campaign_id, {
            "type": "recovery_decision",
            "round": round_num,
            "decision": "retry",
            "retry_count": 1,
            "error_type": "tip_not_attached",
            "error_severity": "medium",
            "reason": "Per memory: 'if robot.aspirate fails with tip → drop_tip + pick_up_tip'",
        })
        await asyncio.sleep(0.4)
        publish_campaign_event(campaign_id, {
            "type": "tool_call",
            "round": round_num,
            "tool": "robot.drop_tip",
            "operation": "drop_tip",
            "params": {},
            "agent": "executor",
        })
        await asyncio.sleep(0.2)
        publish_campaign_event(campaign_id, {
            "type": "tool_call",
            "round": round_num,
            "tool": "robot.pick_up_tip",
            "operation": "pick_up_tip",
            "params": {"tip_idx": 7},
            "agent": "executor",
        })
        await asyncio.sleep(0.3)
        publish_campaign_event(campaign_id, {
            "type": "recovery_success",
            "round": round_num,
            "retries": 1,
            "message": "Recovery succeeded after 1 retry — re-aspirating with tip 7",
        })
        await asyncio.sleep(0.2)
        # Now the actual dispense completes
        publish_campaign_event(campaign_id, {
            "type": "hardware_action",
            "round": round_num,
            "hardware": "opentrons_ot2",
            "action": "dispense",
            "details": {"volume_ul": 200, "well": f"A{round_num}"},
            "agent": "executor",
        })
        await asyncio.sleep(0.3)
    else:
        publish_campaign_event(campaign_id, {
            "type": "hardware_action",
            "round": round_num,
            "hardware": "opentrons_ot2",
            "action": "aspirate+dispense",
            "details": {"volume_ul": 200, "well": f"A{round_num}"},
            "agent": "executor",
        })
        await asyncio.sleep(0.3)

    # 7. Monitor/QC
    publish_campaign_event(campaign_id, {
        "type": "agent_result",
        "round": round_num,
        "agent": "monitor",
        "success": True,
        "message": "Photo quality: good · Volume accuracy: ±5% · HER curve shape: valid",
        "duration_ms": 1200,
    })
    await asyncio.sleep(0.15)

    # 8. EIS / electrochemistry measurement
    publish_campaign_event(campaign_id, {
        "type": "hardware_action",
        "round": round_num,
        "hardware": "squidstat",
        "action": "eis_spectrum",
        "details": {"freq_lo_hz": 0.1, "freq_hi_hz": 1e5, "amplitude_mv": 10},
        "agent": "sensing",
    })
    await asyncio.sleep(0.25)

    # 9. Round complete + KPI (the chart-feeding events)
    improvement = 0.0 if prev_eta10 is None else (prev_eta10 - eta10) / prev_eta10 * 100
    publish_campaign_event(campaign_id, {
        "type": "round_complete",
        "round": round_num,
        "eta10": eta10,
        "improvement_pct": improvement,
        "message": f"Round {round_num} complete: η10 = {eta10:.1f} mV"
                    + (f" (↓{improvement:.1f}%)" if improvement else ""),
    })
    publish_campaign_event(campaign_id, {
        "type": "agent_result",
        "round": round_num,
        "agent": "analyzer",
        "success": True,
        "kpi": eta10,
        "message": f"Round {round_num} analyzed: η10 = {eta10:.1f} mV",
        "duration_ms": 1500,
        "narrative": (
            f"Cumulative improvement {improvement:.1f}% this round. "
            f"Posterior mass shifting toward lower-Fe compositions; "
            f"next round should narrow the search."
            if improvement > 5 else
            "Marginal improvement this round — variance dominates, "
            "acquisition function should favor exploration next."
        ),
    })

    # 10. Strategy pill — updates Live tab
    if round_num in (4, 7):
        await asyncio.sleep(0.1)
        publish_campaign_event(campaign_id, {
            "type": "strategy_decision",
            "round": round_num,
            "backend": _strategy_for_round(round_num),
            "phase": "refine" if round_num == 7 else "exploit",
            "phase_posterior": _phase_posterior(round_num),
            "reason": "Strategy refinement based on posterior narrowing",
        })


async def _run_demo_campaign(campaign_id: str, max_rounds: int = 8) -> None:
    """Run a realistic-looking demo campaign with full agent trace."""
    from app.api.v1.endpoints.orchestrate_events import publish_campaign_event

    emit = lambda evt: publish_campaign_event(campaign_id, evt)

    try:
        # ---- 0. Memory recall banner (UI picks this up as a top-right toast) ----
        emit({
            "type": "memory_recall_event",
            "campaign_id": campaign_id,
            "primitives": ["robot.aspirate", "robot.dispense", "potentiostat.chrono"],
            "summary": "robot.aspirate.volume_ul prior: mean=180.0, σ=25.0, n=6 (1 prior failure: 'tip')",
        })
        await asyncio.sleep(0.3)

        # ---- 1. Campaign start ----
        emit({
            "type": "campaign_start",
            "campaign_id": campaign_id,
            "phase": "demo",
            "message": "Starting demo campaign — natural language → autonomous execution",
        })
        await asyncio.sleep(0.4)

        # ---- 2. Planner thinking + decision tree ----
        emit({
            "type": "agent_thinking",
            "agent": "planner",
            "round": 0,
            "message": "Parsing objective: minimize η10. 3-factor composition sweep (Fe/Co/Ni), "
                       "deposition time 5-60 min, KOH conc 0.1-1.0 M. "
                       "Estimating: 14D search space, expect 8-12 rounds to converge.",
        })
        await asyncio.sleep(0.6)
        emit({
            "type": "agent_decision_tree",
            "agent": "planner",
            "nodes": [
                {"id": "n1", "label": "Objective understood",  "selected": "minimize_eta10",   "options": ["minimize_eta10", "maximize_current_density"], "reason": "User said 'minimize overpotential'"},
                {"id": "n2", "label": "Strategy seed",          "selected": "lhs_first",         "options": ["lhs_first", "random_first", "manual_first"],        "reason": "Wide bounds, no prior data → space-filling first"},
                {"id": "n3", "label": "Parameter space",       "selected": "3_factors_3_kw",   "options": ["3_factors_3_kw", "5_factors_5_kw", "constrained_1kw"], "reason": "Domain expert said 3 composition factors matter"},
            ],
        })
        await asyncio.sleep(0.4)
        emit({
            "type": "agent_result",
            "agent": "planner",
            "success": True,
            "message": "Plan generated: 8 rounds, 14D search space, LHS→Bayesian strategy",
        })
        await asyncio.sleep(0.3)

        # ---- 3. Execute rounds ----
        prev_eta10 = None
        for round_num in range(1, max_rounds + 1):
            eta10 = ETA10_TRAJECTORY[round_num - 1] if round_num <= len(ETA10_TRAJECTORY) else 50.0
            # Round 5: simulate recovery event
            simulate_failure = (round_num == 5)
            await _emit_round(campaign_id, None, round_num, eta10, prev_eta10,
                              simulate_failure=simulate_failure)
            prev_eta10 = eta10

            # Round 6: human-in-the-loop pause + auto-approve
            if round_num == 6:
                await asyncio.sleep(0.2)
                emit({
                    "type": "approval_requested",
                    "round": round_num,
                    "agent": "design",
                    "risk_factors": {"novel_territory": 0.65, "max_confidence": 0.42},
                    "suggested_action": "approve",
                    "message": "Design agent paused: 65% of next-batch candidates are in "
                              "unexplored parameter territory. Approve to continue?",
                })
                # Simulate operator thinking time. 14s is long enough that
                # the modal is visible across multiple screenshot samples
                # during a 90s recording. The client also has its own
                # 18s auto-approve countdown as a fallback.
                await asyncio.sleep(14.0)
                emit({
                    "type": "approval_received",
                    "round": round_num,
                    "decision": "approved",
                    "decided_by": "operator",
                })
                await asyncio.sleep(0.2)

            await asyncio.sleep(0.4)

        # ---- 4. Stop agent: convergence analysis ----
        emit({
            "type": "agent_thinking",
            "agent": "stop",
            "message": "Analyzing convergence: 3 consecutive rounds with <5% improvement. "
                       "Bayesian GP posterior variance shrinking. Suggest stop.",
        })
        await asyncio.sleep(0.6)
        emit({
            "type": "agent_result",
            "agent": "stop",
            "success": True,
            "message": "Convergence detected — stopping campaign. Final η10 = 58.0 mV (70.7% reduction).",
        })
        await asyncio.sleep(0.3)

        # ---- 5. Campaign complete with top-K recipes ----
        emit({
            "type": "campaign_complete",
            "campaign_id": campaign_id,
            "status": "completed",
            "total_rounds": max_rounds,
            "best_eta10": ETA10_TRAJECTORY[-1],
            "best_recipe": {
                "Fe_ratio": 0.42,
                "Co_ratio": 0.31,
                "Ni_ratio": 0.27,
                "deposition_time_min": 28,
                "KOH_conc_M": 0.6,
            },
            "top_k_recipes": [
                {"round": 8, "kpi": 58.0,  "params": {"Fe": 0.42, "Co": 0.31, "Ni": 0.27, "t": 28, "KOH": 0.6}},
                {"round": 7, "kpi": 72.0,  "params": {"Fe": 0.38, "Co": 0.35, "Ni": 0.27, "t": 32, "KOH": 0.55}},
                {"round": 6, "kpi": 88.0,  "params": {"Fe": 0.45, "Co": 0.30, "Ni": 0.25, "t": 35, "KOH": 0.5}},
            ],
            "message": f"Demo campaign completed: {max_rounds} rounds executed, "
                       f"η10 reduced from {ETA10_TRAJECTORY[0]:.0f} to {ETA10_TRAJECTORY[-1]:.0f} mV",
        })

    except Exception as exc:
        logger.exception("Demo campaign failed")
        emit({
            "type": "campaign_complete",
            "campaign_id": campaign_id,
            "status": "failed",
            "error": str(exc),
            "message": f"Demo campaign failed: {exc}",
        })


@router.post("/demo", response_model=DemoResponse)
async def orquestrate_demo(payload: DemoRequest) -> DemoResponse:
    """Fire a full demo campaign and stream the agent trace via SSE.

    The campaign runs in the background; the response returns immediately
    with a campaign_id that can be used to subscribe to the SSE stream.
    """
    campaign_id = f"demo-{uuid.uuid4().hex[:12]}"

    # Register a campaign_state row first: campaign_events has a FOREIGN KEY
    # on campaign_state, so without this row every demo event silently fails
    # to persist and SSE reconnection replay delivers nothing.
    try:
        from app.services.campaign_state import create_campaign
        create_campaign(
            campaign_id,
            {"demo": True, "objective_kpi": payload.objective_kpi, "max_rounds": payload.max_rounds},
            direction="minimize",
        )
    except Exception:
        logger.debug("Demo campaign_state registration failed", exc_info=True)

    task = asyncio.create_task(
        _run_demo_campaign(campaign_id, payload.max_rounds),
        name=f"demo-{campaign_id}",
    )

    # Keep a strong reference while running; self-clean on completion so
    # demo tasks don't accumulate in memory for the life of the process.
    _demo_tasks[campaign_id] = task
    task.add_done_callback(lambda _t: _demo_tasks.pop(campaign_id, None))

    return DemoResponse(campaign_id=campaign_id, status="started")
