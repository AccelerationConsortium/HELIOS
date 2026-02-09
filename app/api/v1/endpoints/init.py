"""API endpoints for the Campaign Initialization conversation flow.

Provides a REST API for the structured 5-round conversation that
produces an :class:`InjectionPack` and launches an autonomous campaign.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.v1.schemas_init import (
    InjectionPack,
    RoundPresentation,
    RoundResponse,
    RoundResult,
    SessionStatus,
)
from app.services.conversation_engine import (
    confirm_and_build,
    get_all_kpis,
    get_all_patterns,
    get_current_round,
    get_session_status,
    go_back,
    start_session,
    submit_round,
)
from app.services.injection_pack import (
    build_diff_summary,
    create_campaign_from_pack,
    validate_injection_pack,
)

router = APIRouter(prefix="/init", tags=["init"])


# ---------------------------------------------------------------------------
# Reference data (must be registered BEFORE /{session_id} routes)
# ---------------------------------------------------------------------------


@router.get("/patterns")
async def list_patterns() -> list[dict]:
    """List all available protocol patterns."""
    return get_all_patterns()


@router.get("/kpis")
async def list_kpis() -> list[dict]:
    """List all available KPI definitions."""
    return get_all_kpis()


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


@router.post("/start")
async def start_init_session(
    created_by: str = "user",
) -> dict:
    """Start a new initialization session and return the first round.

    Returns a RoundPresentation with ``session_id`` populated so the
    frontend can use it for all subsequent API calls.
    """
    session = start_session(author=created_by)
    round_pres = get_current_round(session.session_id)
    data = round_pres.model_dump()
    data["session_id"] = session.session_id
    return data


@router.post("/{session_id}/respond", response_model=RoundResult)
async def respond_to_round(
    session_id: str,
    payload: RoundResponse,
) -> RoundResult:
    """Submit responses for the current round.

    Returns next round on success, or same round with errors on failure.
    """
    try:
        return submit_round(session_id, payload.responses)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{session_id}/status", response_model=SessionStatus)
async def get_status(session_id: str) -> SessionStatus:
    """Get the current status of an initialization session."""
    try:
        return get_session_status(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{session_id}/back", response_model=RoundPresentation)
async def go_back_round(session_id: str) -> RoundPresentation:
    """Go back to the previous round."""
    try:
        return go_back(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{session_id}/round", response_model=RoundPresentation)
async def get_round(session_id: str) -> RoundPresentation:
    """Get the current round presentation."""
    try:
        return get_current_round(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Confirmation & campaign creation
# ---------------------------------------------------------------------------


@router.post("/{session_id}/confirm")
async def confirm_session(session_id: str) -> dict:
    """Confirm the session, build the injection pack, and create a campaign.

    Returns the campaign_id, injection_pack, diff_summary, and any
    validation warnings.
    """
    try:
        pack = confirm_and_build(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Validate cross-object constraints
    warnings = validate_injection_pack(pack)

    # Build diff summary
    diff_summary = build_diff_summary(pack)

    # Create campaign
    try:
        campaign = create_campaign_from_pack(pack)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "campaign_id": campaign.get("id"),
        "injection_pack": pack.model_dump(),
        "diff_summary": diff_summary,
        "warnings": warnings,
        "status": "campaign_created",
    }


@router.get("/{session_id}/pack")
async def get_pack(session_id: str) -> dict:
    """Get the injection pack for a completed session (read-only)."""
    try:
        status = get_session_status(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if status.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Session is '{status.status}', not 'completed'",
        )

    if status.injection_pack_preview is None:
        raise HTTPException(status_code=404, detail="No injection pack found")

    return status.injection_pack_preview
