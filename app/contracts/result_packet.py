"""L0 output: Data/Feature agent output after execution completes.

The ResultPacket contains KPI values, quality assessments, failure info,
and artifact references. The Stop/Continue agent and Campaign Planner
consume this to decide next steps.
"""
from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.db import utcnow_iso

__all__ = [
    "QualityLabel",
    "ResultPacket",
    "new_result_packet_id",
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def new_result_packet_id() -> str:
    """Generate a unique ResultPacket identifier."""
    return f"rp-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class QualityLabel(BaseModel):
    """Quality assessment of a measurement."""

    check_name: str
    passed: bool
    value: float | None = None
    threshold: float | None = None
    message: str = ""


# ---------------------------------------------------------------------------
# Top-level contract
# ---------------------------------------------------------------------------

class ResultPacket(BaseModel):
    """L0 output: results from executing one run.

    Produced by the Data/Feature agent after execution completes.
    The Stop/Continue agent and Campaign Planner consume this.
    """

    packet_id: str
    bundle_id: str  # links to RunBundle
    run_id: str
    round_number: int
    candidate_index: int
    created_at: str

    # Status
    status: Literal["succeeded", "failed", "partial"]

    # KPI values
    kpi_values: dict[str, float]  # kpi_name -> value

    # Quality gates
    quality_labels: list[QualityLabel] = Field(default_factory=list)
    overall_quality: Literal["good", "suspect", "failed"] = "good"

    # Failure info (if applicable)
    failure_reason: str | None = None
    failure_step: str | None = None
    recovery_attempted: bool = False
    recovery_succeeded: bool = False

    # Artifacts
    artifact_uris: list[str] = Field(default_factory=list)

    # Raw data summary
    raw_data_summary: dict[str, Any] = Field(default_factory=dict)
