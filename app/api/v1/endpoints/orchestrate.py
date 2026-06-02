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

from app.agents.orchestrator import OrchestratorInput, OrchestratorOutput
from app.services.contract_bridge import (
    injection_pack_to_task_contract,
    task_contract_to_orchestrator_input,
)
from app.services.durable_execution import get_durable_backend

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


def _track_orchestrator_task(campaign_id: str, task: asyncio.Task) -> None:
    """Populate legacy in-memory result stores from a durable backend task."""

    def _done(done: asyncio.Task) -> None:
        try:
            output = done.result()
            if isinstance(output, OrchestratorOutput):
                _campaign_results[campaign_id] = output
            else:
                _campaign_errors[campaign_id] = "Unexpected orchestrator result"
        except asyncio.CancelledError:
            _campaign_errors[campaign_id] = "Campaign cancelled"
        except Exception as exc:
            logger.exception("Orchestrator campaign %s failed", campaign_id)
            _campaign_errors[campaign_id] = str(exc)

    task.add_done_callback(_done)


async def _start_durable_campaign(
    campaign_id: str,
    orch_input: OrchestratorInput,
    *,
    resume_from_round: int | None = None,
    restored_state: dict[str, Any] | None = None,
) -> None:
    backend = get_durable_backend()
    handle = await backend.start_campaign(
        orch_input,
        resume_from_round=resume_from_round,
        restored_state=restored_state,
    )
    get_task = getattr(backend, "get_task", None)
    if callable(get_task):
        task = get_task(campaign_id)
        if task is not None:
            _running_campaigns[campaign_id] = task
            _track_orchestrator_task(campaign_id, task)
    logger.info(
        "orchestrate.durable_started",
        extra={
            "campaign_id": handle.campaign_id,
            "backend": handle.backend,
            "status": handle.status,
        },
    )


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

    await _start_durable_campaign(campaign_id, orch_input)

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
    await _start_durable_campaign(campaign_id, orch_input)

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
    status = await get_durable_backend().get_status(campaign_id)
    if status is not None:
        return OrchestrateStatusResponse(
            campaign_id=campaign_id,
            status=status.status,
            result=status.result,
            error=status.error,
        )

    raise HTTPException(status_code=404, detail=f"Campaign '{campaign_id}' not found")


@router.post("/{campaign_id}/stop")
async def orchestrate_stop(campaign_id: str) -> dict:
    """Cancel a running orchestrator campaign."""
    status = await get_durable_backend().cancel_campaign(campaign_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Campaign '{campaign_id}' not found")

    if status.status in {"completed", "failed"}:
        return {"campaign_id": campaign_id, "status": "already_finished"}

    public_status = "cancelled" if status.status == "cancelling" else status.status
    return {"campaign_id": campaign_id, "status": public_status}


@router.get("/{campaign_id}/durable-events")
async def orchestrate_durable_events(campaign_id: str) -> dict:
    """Return backend-level lifecycle events for debugging and replay."""
    status = await get_durable_backend().get_status(campaign_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Campaign '{campaign_id}' not found")
    events = await get_durable_backend().list_events(campaign_id)
    return {
        "campaign_id": campaign_id,
        "backend": status.backend,
        "events": [
            {
                "type": event.type,
                "payload": event.payload,
                "created_at": event.created_at,
            }
            for event in events
        ],
    }


@router.post("/{campaign_id}/resume", response_model=OrchestrateStartResponse)
async def orchestrate_resume(campaign_id: str) -> OrchestrateStartResponse:
    """Resume a paused/crashed campaign from its last checkpoint."""
    from app.services.campaign_state import load_campaign, load_completed_candidates

    # Reject if already running in memory
    durable_status = await get_durable_backend().get_status(campaign_id)
    if durable_status is not None and durable_status.status == "running":
        raise HTTPException(status_code=409, detail="Campaign is already running")

    db_state = load_campaign(campaign_id)
    if db_state is None:
        raise HTTPException(status_code=404, detail=f"Campaign '{campaign_id}' not found")

    if db_state["status"] in ("completed", "failed", "cancelled"):
        raise HTTPException(
            status_code=400,
            detail=f"Campaign already {db_state['status']}, cannot resume",
        )

    orch_input = OrchestratorInput(**db_state["input"])
    orch_input.campaign_id = campaign_id
    restored_state = load_completed_candidates(campaign_id)
    start_round_num = db_state["current_round"] or 1

    await _start_durable_campaign(
        campaign_id,
        orch_input,
        resume_from_round=start_round_num,
        restored_state=restored_state,
    )

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
