"""Confirmations API — generic approval/rejection for pending operations.

Handles code confirmations, cleaning approvals, and any other
in-memory confirmation requests created by the orchestrator.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.code_confirmation import (
    CodeConfirmationResponse,
    CodeConfirmationStatus,
    get_confirmation_status,
    get_pending_confirmation,
    list_pending_confirmations,
    respond_to_confirmation,
)

router = APIRouter(prefix="/confirmations", tags=["confirmations"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ConfirmationRespondRequest(BaseModel):
    """Payload for approving or rejecting a confirmation."""

    approved: bool = True
    reason: str = ""
    modified_code: str | None = None
    modified_steps: list[dict] | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("")
async def list_confirmations(
    campaign_id: str = "",
    confirmation_type: str = "",
) -> list[dict]:
    """List all pending confirmations, optionally filtered."""
    pending = list_pending_confirmations(
        campaign_id=campaign_id,
        confirmation_type=confirmation_type,
    )
    return [req.model_dump(mode="json") for req in pending]


@router.get("/{request_id}")
async def get_confirmation(request_id: str) -> dict:
    """Get a specific confirmation request."""
    req = get_pending_confirmation(request_id)
    if req is None:
        raise HTTPException(status_code=404, detail=f"Confirmation {request_id} not found")
    status = get_confirmation_status(request_id)
    return {
        **req.model_dump(mode="json"),
        "current_status": status.value if status else "unknown",
    }


@router.post("/{request_id}/respond")
async def respond_confirmation(
    request_id: str,
    payload: ConfirmationRespondRequest,
) -> dict:
    """Approve or reject a pending confirmation.

    For cleaning confirmations, simply set ``approved: true`` or ``false``.
    For code confirmations, optionally provide ``modified_code`` or ``modified_steps``.
    """
    try:
        response = CodeConfirmationResponse(
            request_id=request_id,
            approved=payload.approved,
            modified_code=payload.modified_code,
            modified_steps=payload.modified_steps,
            rejection_reason=payload.reason,
        )
        new_status = respond_to_confirmation(response)
        return {
            "request_id": request_id,
            "status": new_status.value,
            "message": f"Confirmation {request_id} → {new_status.value}",
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
