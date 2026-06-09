from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.contracts.mission import MissionContract, MissionLaunchResult
from app.services.mission_control import launch_mission
from app.services.run_service import DomainError

router = APIRouter(prefix="/missions", tags=["missions"])


@router.post("/launch")
async def launch_mission_endpoint(payload: MissionContract) -> MissionLaunchResult:
    try:
        return await launch_mission(payload)
    except DomainError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

