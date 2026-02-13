"""L1 -> L0 contract: Protocol Compiler + Deck/Layout agent output.

The RunBundle contains everything needed to physically execute one run
on the OT-2. The Runner agent consumes this directly.
"""
from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.db import utcnow_iso

__all__ = [
    "DeckLayout",
    "RunBundle",
    "SlotAssignment",
    "new_run_bundle_id",
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def new_run_bundle_id() -> str:
    """Generate a unique RunBundle identifier."""
    return f"rb-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Sub-models (ordered so forward references are resolved)
# ---------------------------------------------------------------------------

class SlotAssignment(BaseModel):
    """What's in a deck slot."""

    slot_number: int | str  # int for OT-2 (1-11), str for Flex ("A1"-"D3")
    labware_name: str
    labware_definition: str  # Opentrons labware def name
    role: Literal["source", "destination", "tips", "waste", "wash", "reagent", "custom"]
    contents: dict[str, Any] = Field(default_factory=dict)  # well -> content description


class DeckLayout(BaseModel):
    """OT-2 deck configuration for this run."""

    slot_assignments: dict[int, SlotAssignment]  # slot_number -> assignment
    pipette_mounts: dict[str, str] = Field(
        default_factory=dict
    )  # "left"/"right" -> pipette model


# ---------------------------------------------------------------------------
# Top-level contract
# ---------------------------------------------------------------------------

class RunBundle(BaseModel):
    """L1 output: everything needed to physically execute one run.

    Produced by the Protocol Compiler + Deck/Layout agent.
    The Runner agent consumes this directly.
    """

    bundle_id: str
    plan_id: str  # links to CampaignPlan
    contract_id: str  # links to TaskContract
    round_number: int
    candidate_index: int
    created_at: str

    # Compiled protocol (DAG)
    compiled_protocol: dict[str, Any]
    graph_hash: str

    # Deck setup
    deck_layout: DeckLayout

    # Parameters for this specific run
    params: dict[str, Any]

    # Policy
    policy_snapshot: dict[str, Any]

    # Metadata
    protocol_pattern_id: str
    protocol_version: str = "1.0"
