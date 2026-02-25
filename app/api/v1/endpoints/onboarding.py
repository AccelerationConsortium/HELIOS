"""API endpoints for instrument onboarding.

Provides REST endpoints for the multi-turn instrument onboarding flow:
  1. POST /onboarding/generate  — generate integration code from spec
  2. POST /onboarding/confirm   — approve safety/config confirmations
  3. POST /onboarding/write     — write approved files to disk
  4. GET  /onboarding/status    — check current onboarding session state

The flow mirrors the init conversation pattern: structured questions are
returned as pending_confirmations that the frontend (or chat agent) renders
for user review.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.onboarding_agent import (
    OnboardingAgent,
    OnboardingInput,
    OnboardingOutput,
    PrimitiveSpec,
)
from app.services.primitives_registry import refresh_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/onboarding", tags=["onboarding"])

# Session store: onboarding_id → serialised OnboardingResult
_onboarding_sessions: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class OnboardingDiscoverRequest(BaseModel):
    """Request body for POST /onboarding/discover."""

    instrument_name: str
    manufacturer: str = ""
    model: str = ""
    sdk_package: str = ""
    docs_url: str = ""


class OnboardingGenerateRequest(BaseModel):
    """Request body for POST /onboarding/generate."""

    instrument_name: str
    manufacturer: str = ""
    model: str = ""
    communication: str = "usb"
    description: str = ""
    sdk_package: str = ""
    primitives: list[PrimitiveSpec] = Field(default_factory=list)


class OnboardingConfirmRequest(BaseModel):
    """Request body for POST /onboarding/confirm."""

    onboarding_id: str
    confirmations: dict[str, Any] = Field(
        ...,
        description="Mapping of confirmation id → confirmed value",
    )


class OnboardingWriteRequest(BaseModel):
    """Request body for POST /onboarding/write."""

    onboarding_id: str
    force: bool = False


class OnboardingResponse(BaseModel):
    """Unified response for onboarding endpoints."""

    onboarding_id: str
    status: str
    instrument_name: str = ""
    display_name: str = ""
    chat_message: str = ""
    pending_confirmations: list[dict[str, Any]] = Field(default_factory=list)
    confirmed_count: int = 0
    total_confirmations: int = 0
    files: list[dict[str, Any]] = Field(default_factory=list)
    written_paths: list[str] = Field(default_factory=list)
    manual_todo: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # Included for discover phase so the frontend can read discovered_primitives
    serialised_result: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _output_to_response(
    onboarding_id: str, output: OnboardingOutput,
) -> OnboardingResponse:
    """Convert agent output to API response."""
    return OnboardingResponse(
        onboarding_id=onboarding_id,
        status=output.status,
        instrument_name=output.instrument_name,
        display_name=output.display_name,
        chat_message=output.chat_message,
        pending_confirmations=[c.model_dump() for c in output.pending_confirmations],
        confirmed_count=output.confirmed_count,
        total_confirmations=output.total_confirmations,
        files=[f.model_dump() for f in output.files],
        written_paths=output.written_paths,
        manual_todo=output.manual_todo,
        warnings=output.warnings,
        serialised_result=output.serialised_result,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/discover", response_model=OnboardingResponse)
async def onboarding_discover(
    payload: OnboardingDiscoverRequest,
) -> OnboardingResponse:
    """Auto-discover primitives for an instrument via LLM inference.

    Calls the LLM with instrument details and returns a list of proposed
    primitives.  The session is stored so the caller can pass the
    ``onboarding_id`` directly to ``/generate`` with primitives pre-filled.
    """
    import uuid

    onboarding_id = f"onb-{uuid.uuid4().hex[:12]}"

    agent = OnboardingAgent()
    agent_input = OnboardingInput(
        phase="discover",
        instrument_name=payload.instrument_name,
        manufacturer=payload.manufacturer,
        model=payload.model,
        sdk_package=payload.sdk_package,
        docs_url=payload.docs_url,
    )

    result = await agent.run(agent_input)

    if not result.success or result.output is None:
        raise HTTPException(
            status_code=400,
            detail="; ".join(result.errors) if result.errors else "Discovery failed",
        )

    # Store discovered primitives so the wizard can carry them forward
    _onboarding_sessions[onboarding_id] = result.output.serialised_result

    return _output_to_response(onboarding_id, result.output)


@router.post("/generate", response_model=OnboardingResponse)
async def onboarding_generate(
    payload: OnboardingGenerateRequest,
) -> OnboardingResponse:
    """Generate integration code for a new instrument.

    Returns pending confirmations that the user must review before
    the code can be written to disk.
    """
    import uuid

    onboarding_id = f"onb-{uuid.uuid4().hex[:12]}"

    agent = OnboardingAgent()
    agent_input = OnboardingInput(
        phase="generate",
        instrument_name=payload.instrument_name,
        manufacturer=payload.manufacturer,
        model=payload.model,
        communication=payload.communication,
        description=payload.description,
        sdk_package=payload.sdk_package,
        primitives=payload.primitives,
    )

    result = await agent.run(agent_input)

    if not result.success or result.output is None:
        raise HTTPException(
            status_code=400,
            detail="; ".join(result.errors) if result.errors else "Generation failed",
        )

    # Store session state
    _onboarding_sessions[onboarding_id] = result.output.serialised_result

    return _output_to_response(onboarding_id, result.output)


@router.post("/confirm", response_model=OnboardingResponse)
async def onboarding_confirm(
    payload: OnboardingConfirmRequest,
) -> OnboardingResponse:
    """Confirm or adjust onboarding decisions.

    After the user reviews pending_confirmations from /generate, they
    submit a dict mapping confirmation IDs to approved values.

    If all confirmations are resolved, status becomes 'ready_to_write'.
    """
    session = _onboarding_sessions.get(payload.onboarding_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Onboarding session '{payload.onboarding_id}' not found",
        )

    agent = OnboardingAgent()
    agent_input = OnboardingInput(
        phase="confirm",
        confirmations=payload.confirmations,
        previous_result=session,
    )

    result = await agent.run(agent_input)

    if not result.success or result.output is None:
        raise HTTPException(
            status_code=400,
            detail="; ".join(result.errors) if result.errors else "Confirm failed",
        )

    # Update session state
    _onboarding_sessions[payload.onboarding_id] = result.output.serialised_result

    return _output_to_response(payload.onboarding_id, result.output)


@router.post("/write", response_model=OnboardingResponse)
async def onboarding_write(
    payload: OnboardingWriteRequest,
) -> OnboardingResponse:
    """Write approved files to disk.

    Requires all confirmations to be resolved (or force=True).
    """
    session = _onboarding_sessions.get(payload.onboarding_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Onboarding session '{payload.onboarding_id}' not found",
        )

    agent = OnboardingAgent()
    agent_input = OnboardingInput(
        phase="write",
        previous_result=session,
        force_write=payload.force,
    )

    result = await agent.run(agent_input)

    if not result.success or result.output is None:
        raise HTTPException(
            status_code=400,
            detail="; ".join(result.errors) if result.errors else "Write failed",
        )

    # Update session state
    _onboarding_sessions[payload.onboarding_id] = result.output.serialised_result

    # Hot-reload primitives registry so newly written skill files are visible
    # immediately without requiring a server restart.
    if result.output.status == "written":
        try:
            refresh_registry()
            logger.info(
                "Registry refreshed after onboarding write: %s", payload.onboarding_id,
            )
        except Exception as exc:
            logger.warning("refresh_registry failed (non-fatal): %s", exc)

    return _output_to_response(payload.onboarding_id, result.output)


@router.get("/{onboarding_id}/status", response_model=OnboardingResponse)
async def onboarding_status(onboarding_id: str) -> OnboardingResponse:
    """Check the current state of an onboarding session."""
    session = _onboarding_sessions.get(onboarding_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Onboarding session '{onboarding_id}' not found",
        )

    # Reconstruct output from stored state (lightweight — no regeneration)
    agent = OnboardingAgent()
    result_obj = agent._deserialise_result(session)

    pending = [c for c in result_obj.pending_confirmations if not c.confirmed]
    status = "ready_to_write" if not pending else "needs_confirmation"

    return OnboardingResponse(
        onboarding_id=onboarding_id,
        status=status,
        instrument_name=result_obj.spec.name,
        display_name=result_obj.spec.display_name,
        confirmed_count=result_obj.confirmed_count,
        total_confirmations=result_obj.total_confirmations,
        files=[
            {"path": gf.path, "is_patch": gf.is_patch, "description": gf.description}
            for gf in result_obj.files
        ],
        manual_todo=result_obj.manual_todo,
        warnings=result_obj.warnings,
    )
