"""L2 -> L1 contract: Campaign Planner agent output.

The CampaignPlan contains the planned rounds with their parameter candidates,
resource requirements, and contingency strategies.
"""
from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.db import utcnow_iso

__all__ = [
    "CampaignPlan",
    "ResourceRequirements",
    "RoundSpec",
    "new_campaign_plan_id",
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def new_campaign_plan_id() -> str:
    """Generate a unique CampaignPlan identifier."""
    return f"cp-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Sub-models (ordered so forward references are resolved)
# ---------------------------------------------------------------------------

class ResourceRequirements(BaseModel):
    """Resources needed for a round."""

    labware_slots: list[int] = Field(default_factory=list)
    tip_racks_needed: int = 0
    reagent_volumes_ul: dict[str, float] = Field(default_factory=dict)
    estimated_duration_minutes: float = 0.0
    instruments_needed: list[str] = Field(default_factory=list)


class RoundSpec(BaseModel):
    """Plan for a single round of experiments."""

    round_number: int
    candidate_params: list[dict[str, Any]]  # parameter sets to try
    strategy_used: str
    resource_requirements: ResourceRequirements
    fallback_strategy: str | None = None  # what to do if this round fails


# ---------------------------------------------------------------------------
# Top-level contract
# ---------------------------------------------------------------------------

class CampaignPlan(BaseModel):
    """L2 output: multi-round campaign plan.

    Produced by the Campaign Planner agent. Contains the planned
    rounds with their parameter candidates, resource requirements,
    and contingency strategies.
    """

    plan_id: str
    contract_id: str  # links back to TaskContract
    created_at: str

    planned_rounds: list[RoundSpec]
    total_estimated_runs: int
    total_estimated_duration_minutes: float

    # Strategy metadata
    initial_strategy: str
    strategy_schedule: dict[int, str] = Field(
        default_factory=dict,
        description="round_number -> strategy override",
    )

    # Contingency
    max_consecutive_failures: int = Field(default=3)
    failure_escalation: Literal["retry", "skip", "halt", "human_review"] = "human_review"
