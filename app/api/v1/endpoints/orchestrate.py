"""API endpoints for the orchestrator agent system.

Provides REST endpoints to start, monitor, and stop orchestrator-driven
campaigns. Supports both direct orchestrator input and bridging from
the existing conversation/init flow via session_id.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.orchestrator import OrchestratorAgent, OrchestratorInput, OrchestratorOutput
from app.services.contract_bridge import (
    injection_pack_to_task_contract,
    task_contract_to_orchestrator_input,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orchestrate", tags=["orchestrate"])

# Module-level store for running orchestrator campaign tasks.
_running_campaigns: dict[str, asyncio.Task] = {}
_campaign_results: dict[str, OrchestratorOutput] = {}
_campaign_errors: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class OrchestrateRequest(BaseModel):
    """Request body for POST /orchestrate/start."""

    contract_id: str
    objective_kpi: str
    direction: str = "minimize"
    max_rounds: int = 20
    batch_size: int = 10
    strategy: str = "lhs"
    target_value: float | None = None
    dimensions: list[dict[str, Any]] = Field(default_factory=list)
    protocol_template: dict[str, Any] = Field(default_factory=lambda: {"steps": []})
    policy_snapshot: dict[str, Any] = Field(default_factory=dict)
    protocol_pattern_id: str = ""
    dry_run: bool = False
    plan_only: bool = False


class OrchestrateStartResponse(BaseModel):
    """Response for a successfully started orchestrator campaign."""

    campaign_id: str
    status: str = "started"


class OrchestrateFromSessionResponse(BaseModel):
    """Response for starting an orchestrator campaign from a session."""

    campaign_id: str
    status: str = "started"
    contract_summary: dict[str, Any] = Field(default_factory=dict)


class OrchestrateStatusResponse(BaseModel):
    """Response for campaign status queries."""

    campaign_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_orchestrator(campaign_id: str, orch_input: OrchestratorInput) -> None:
    """Run the orchestrator agent and store results."""
    agent = OrchestratorAgent()
    try:
        result = await agent.run(orch_input)
        if result.success and result.output is not None:
            _campaign_results[campaign_id] = result.output
        else:
            _campaign_errors[campaign_id] = "; ".join(result.errors) if result.errors else "Unknown error"
    except Exception as exc:
        logger.exception("Orchestrator campaign %s failed", campaign_id)
        _campaign_errors[campaign_id] = str(exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/start", response_model=OrchestrateStartResponse)
async def orchestrate_start(payload: OrchestrateRequest) -> OrchestrateStartResponse:
    """Start an orchestrator campaign from direct input.

    Creates an OrchestratorInput from the request, runs the OrchestratorAgent
    asynchronously in the background, and returns a campaign_id immediately.
    """
    campaign_id = f"orch-{uuid.uuid4().hex[:12]}"

    orch_input = OrchestratorInput(
        contract_id=payload.contract_id,
        objective_kpi=payload.objective_kpi,
        direction=payload.direction,
        max_rounds=payload.max_rounds,
        batch_size=payload.batch_size,
        strategy=payload.strategy,
        target_value=payload.target_value,
        dimensions=payload.dimensions,
        protocol_template=payload.protocol_template,
        policy_snapshot=payload.policy_snapshot,
        protocol_pattern_id=payload.protocol_pattern_id,
        dry_run=payload.dry_run,
        plan_only=payload.plan_only,
        campaign_id=campaign_id,
    )

    task = asyncio.create_task(
        _run_orchestrator(campaign_id, orch_input),
        name=f"orchestrator-{campaign_id}",
    )
    _running_campaigns[campaign_id] = task

    return OrchestrateStartResponse(campaign_id=campaign_id, status="started")


@router.post(
    "/from-session/{session_id}",
    response_model=OrchestrateFromSessionResponse,
)
async def orchestrate_from_session(
    session_id: str,
) -> OrchestrateFromSessionResponse:
    """Bridge from an existing conversation session to the orchestrator.

    1. Calls confirm_and_build(session_id) to get an InjectionPack
    2. Converts to TaskContract via contract_bridge
    3. Converts to OrchestratorInput
    4. Starts the orchestrator campaign in the background
    """
    from app.services.conversation_engine import confirm_and_build
    from app.services.injection_pack import validate_injection_pack

    try:
        pack = confirm_and_build(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Validate
    warnings = validate_injection_pack(pack)
    if warnings:
        logger.warning(
            "InjectionPack warnings for session %s: %s", session_id, warnings
        )

    # Convert to TaskContract
    task_contract = injection_pack_to_task_contract(pack)

    # Start campaign
    campaign_id = f"orch-{uuid.uuid4().hex[:12]}"

    # Convert to OrchestratorInput kwargs
    orch_kwargs = task_contract_to_orchestrator_input(task_contract)
    orch_kwargs["campaign_id"] = campaign_id
    orch_input = OrchestratorInput(**orch_kwargs)
    task = asyncio.create_task(
        _run_orchestrator(campaign_id, orch_input),
        name=f"orchestrator-{campaign_id}",
    )
    _running_campaigns[campaign_id] = task

    contract_summary = {
        "contract_id": task_contract.contract_id,
        "objective_kpi": task_contract.objective.primary_kpi,
        "direction": task_contract.objective.direction,
        "max_rounds": task_contract.stop_conditions.max_rounds,
        "batch_size": task_contract.exploration_space.batch_size,
        "n_dimensions": len(task_contract.exploration_space.dimensions),
        "protocol_pattern_id": task_contract.protocol_pattern_id,
    }

    return OrchestrateFromSessionResponse(
        campaign_id=campaign_id,
        status="started",
        contract_summary=contract_summary,
    )


@router.get("/{campaign_id}/status", response_model=OrchestrateStatusResponse)
async def orchestrate_status(campaign_id: str) -> OrchestrateStatusResponse:
    """Check the status of an orchestrator campaign.

    Checks in-memory state first, then falls back to the DB for campaigns
    that survived a server restart.
    """
    # Check if we have a result in memory
    if campaign_id in _campaign_results:
        output = _campaign_results[campaign_id]
        return OrchestrateStatusResponse(
            campaign_id=campaign_id,
            status="completed",
            result=output.model_dump(),
        )

    # Check if there was an error in memory
    if campaign_id in _campaign_errors:
        return OrchestrateStatusResponse(
            campaign_id=campaign_id,
            status="failed",
            error=_campaign_errors[campaign_id],
        )

    # Check if it's still running in memory
    task = _running_campaigns.get(campaign_id)
    if task is not None:
        if task.done():
            # Task finished but no result/error stored -- check for exception
            try:
                task.result()
            except asyncio.CancelledError:
                return OrchestrateStatusResponse(
                    campaign_id=campaign_id,
                    status="cancelled",
                )
            except Exception as exc:
                _campaign_errors[campaign_id] = str(exc)
                return OrchestrateStatusResponse(
                    campaign_id=campaign_id,
                    status="failed",
                    error=str(exc),
                )

            # If done with no exception but no stored result, treat as completed
            return OrchestrateStatusResponse(
                campaign_id=campaign_id,
                status="completed",
            )

        return OrchestrateStatusResponse(
            campaign_id=campaign_id,
            status="running",
        )

    # --- Fallback to DB (campaign survived a restart) ---
    from app.services.campaign_state import load_campaign
    db_state = load_campaign(campaign_id)
    if db_state is not None:
        db_status = db_state["status"]
        if db_status in ("completed", "failed", "cancelled"):
            return OrchestrateStatusResponse(
                campaign_id=campaign_id,
                status=db_status,
                result={
                    "best_kpi": db_state.get("best_kpi"),
                    "current_round": db_state.get("current_round"),
                    "total_rounds": db_state.get("total_rounds"),
                    "stop_reason": db_state.get("stop_reason"),
                },
                error=db_state.get("error"),
            )
        # Still running/planning in DB but not in memory → resumable
        return OrchestrateStatusResponse(
            campaign_id=campaign_id,
            status="resumable",
            result={
                "current_round": db_state.get("current_round"),
                "total_rounds": db_state.get("total_rounds"),
                "best_kpi": db_state.get("best_kpi"),
                "total_runs": db_state.get("total_runs"),
            },
        )

    raise HTTPException(status_code=404, detail=f"Campaign '{campaign_id}' not found")


@router.post("/{campaign_id}/stop")
async def orchestrate_stop(campaign_id: str) -> dict:
    """Cancel a running orchestrator campaign."""
    task = _running_campaigns.get(campaign_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Campaign '{campaign_id}' not found")

    if task.done():
        return {"campaign_id": campaign_id, "status": "already_finished"}

    task.cancel()
    return {"campaign_id": campaign_id, "status": "cancelled"}


@router.post("/{campaign_id}/resume", response_model=OrchestrateStartResponse)
async def orchestrate_resume(campaign_id: str) -> OrchestrateStartResponse:
    """Resume a paused/crashed campaign from its last checkpoint."""
    from app.services.campaign_state import load_campaign

    # Reject if already running in memory
    task = _running_campaigns.get(campaign_id)
    if task is not None and not task.done():
        raise HTTPException(status_code=409, detail="Campaign is already running")

    db_state = load_campaign(campaign_id)
    if db_state is None:
        raise HTTPException(status_code=404, detail=f"Campaign '{campaign_id}' not found")

    if db_state["status"] in ("completed", "failed", "cancelled"):
        raise HTTPException(
            status_code=400,
            detail=f"Campaign already {db_state['status']}, cannot resume",
        )

    async def _resume(cid: str) -> None:
        agent = OrchestratorAgent()
        try:
            result = await agent.resume_campaign(cid)
            _campaign_results[cid] = result
        except Exception as exc:
            logger.exception("Resume campaign %s failed", cid)
            _campaign_errors[cid] = str(exc)

    task = asyncio.create_task(
        _resume(campaign_id),
        name=f"orchestrator-resume-{campaign_id}",
    )
    _running_campaigns[campaign_id] = task

    return OrchestrateStartResponse(campaign_id=campaign_id, status="resuming")


@router.get("/backends/status")
async def backends_status() -> dict:
    """List available optimization backends and their status.

    Returns ``{backend_name: is_available}`` for all registered backends.
    Useful for the frontend to show which advanced optimization methods
    are installed.
    """
    from app.services.optimization_backends import list_backends

    backends = list_backends()
    available_count = sum(1 for v in backends.values() if v)
    return {
        "backends": backends,
        "available_count": available_count,
        "total_count": len(backends),
        "adaptive_enabled": available_count > 2,  # more than just built_in + lhs
    }
