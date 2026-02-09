from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.api.v1.schemas import ApprovalRequest
from app.services.run_service import DomainError, approve_run, get_run, list_events, list_locks, list_runs

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("")
async def list_runs_endpoint(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    return list_runs(limit=limit)


@router.get("/{run_id}")
async def get_run_endpoint(run_id: str) -> dict:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@router.get("/{run_id}/events")
async def get_run_events_endpoint(run_id: str) -> list[dict]:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return list_events(run_id)


@router.post("/{run_id}/approve")
async def approve_run_endpoint(run_id: str, payload: ApprovalRequest) -> dict:
    try:
        return approve_run(run_id=run_id, approver=payload.approver, reason=payload.reason)
    except DomainError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/meta/locks")
async def list_locks_endpoint() -> list[dict]:
    return list_locks()
