"""Swarms API — introspection and spawn interface for agent swarms."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.agents.swarm import SwarmFactory, list_swarms

router = APIRouter(tags=["swarms"])


@router.get("/swarms")
async def get_swarms() -> dict[str, Any]:
    """List all registered agent swarms with their composition.

    Returns metadata about each swarm including constituent agents
    and their roles, aligned with the AI4X paper's four specialist groups.
    """
    swarms = list_swarms()
    return {
        "swarms": swarms,
        "count": len(swarms),
        "available": SwarmFactory.available_swarms(),
    }


@router.get("/swarms/{swarm_name}")
async def get_swarm_detail(swarm_name: str) -> dict[str, Any]:
    """Get detailed information about a specific swarm.

    Parameters
    ----------
    swarm_name:
        One of "scientist", "engineer", "analyst", "validator".
    """
    # Validate against known swarms (never echo raw user input)
    available = SwarmFactory.available_swarms()
    if swarm_name not in available:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Swarm not found",
                "available": available,
            },
        )

    all_swarms = list_swarms()
    for s in all_swarms:
        if s["name"] == swarm_name:
            return {"found": True, "swarm": s}

    # Should not reach here if registry is consistent
    raise HTTPException(status_code=500, detail="Registry inconsistency")
