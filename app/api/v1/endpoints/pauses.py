"""API endpoints for agent pause requests (v2 human-in-the-loop).

Operators use these endpoints to:
- List pending pause requests for a campaign
- Approve / reject / modify a pause request
- View pause history for audit

These replace the binary ``require_manual_confirmation`` flag with
fine-grained, agent-initiated approval gates.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/pauses", tags=["pauses"])


# ── Request / Response models ─────────────────────────────────────────────


class ResolvePauseRequest(BaseModel):
    """Operator decision on a pause request."""
    decision: str = Field(
        ..., description="One of: approved, rejected, modified"
    )
    decided_by: str = Field(default="operator")
    modifications: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional parameter modifications (for 'modified' decisions)",
    )


class PauseSummary(BaseModel):
    pause_id: str
    agent_name: str
    reason: str
    risk_factors: dict[str, float]
    suggested_action: str
    expires_at: str
    created_at: str


class PauseDetail(PauseSummary):
    status: str
    decision: str | None = None
    decided_by: str | None = None
    decided_at: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.get("/pending/{campaign_id}", response_model=list[PauseSummary])
async def list_pending_pauses(campaign_id: str) -> list[dict[str, Any]]:
    """List all pending (unresolved) pause requests for a campaign."""
    from app.services.pause_store import list_pending_pauses as _list

    return _list(campaign_id)


@router.get("/history/{campaign_id}", response_model=list[PauseDetail])
async def get_pause_history(campaign_id: str) -> list[dict[str, Any]]:
    """Get full pause history for a campaign (pending + resolved)."""
    from app.services.pause_store import get_pause_history as _history

    return _history(campaign_id)


@router.post("/{pause_id}/resolve")
async def resolve_pause(pause_id: str, body: ResolvePauseRequest) -> dict[str, Any]:
    """Resolve (approve/reject/modify) a pending pause request.

    This is the primary operator action endpoint — called from the lab UI
    or via CLI when an agent has paused execution and is waiting for
    human oversight.
    """
    if body.decision not in ("approved", "rejected", "modified"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid decision '{body.decision}'; "
                   f"must be one of: approved, rejected, modified",
        )

    from app.services.pause_store import resolve_pause as _resolve

    ok = _resolve(
        pause_id=pause_id,
        decision=body.decision,
        decided_by=body.decided_by,
        modifications=body.modifications,
    )
    if not ok:
        raise HTTPException(
            status_code=404,
            detail=f"Pause request '{pause_id}' not found or already resolved",
        )

    return {
        "pause_id": pause_id,
        "decision": body.decision,
        "status": "resolved",
    }


@router.get("/{pause_id}")
async def get_pause(pause_id: str) -> dict[str, Any]:
    """Get current status of a single pause request."""
    from app.services.pause_store import get_pause_status

    status = get_pause_status(pause_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Pause '{pause_id}' not found")
    return {"pause_id": pause_id, **status}
