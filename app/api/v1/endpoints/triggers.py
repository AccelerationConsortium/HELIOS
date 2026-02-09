from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.v1.schemas import TriggerRequest
from app.services.run_service import DomainError, create_run_from_trigger

router = APIRouter(prefix="/triggers", tags=["triggers"])


def _handle_trigger(trigger_type: str, payload: TriggerRequest) -> dict:
    return create_run_from_trigger(
        trigger_type=trigger_type,
        trigger_payload=payload.payload,
        campaign_id=payload.campaign_id,
        protocol=payload.protocol,
        inputs=payload.inputs,
        policy_snapshot=payload.policy_snapshot,
        actor=payload.actor,
        session_key=payload.session_key,
    )


@router.post("/time")
async def trigger_time(payload: TriggerRequest) -> dict:
    try:
        return _handle_trigger("time", payload)
    except DomainError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/event")
async def trigger_event(payload: TriggerRequest) -> dict:
    try:
        return _handle_trigger("event", payload)
    except DomainError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/external")
async def trigger_external(payload: TriggerRequest) -> dict:
    try:
        return _handle_trigger("external", payload)
    except DomainError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
