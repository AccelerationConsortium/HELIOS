from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.v1.schemas import CampaignCreate
from app.services.run_service import DomainError, create_campaign, list_campaigns

router = APIRouter(prefix="/campaigns", tags=["campaigns"])


@router.post("")
async def create_campaign_endpoint(payload: CampaignCreate) -> dict:
    try:
        return create_campaign(
            name=payload.name,
            cadence_seconds=payload.cadence_seconds,
            protocol=payload.protocol,
            inputs=payload.inputs,
            policy_snapshot=payload.policy_snapshot,
            actor=payload.created_by,
        )
    except DomainError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("")
async def list_campaigns_endpoint() -> list[dict]:
    return list_campaigns()
