"""L3 -> L2 contract: Intake/Clarifier agent output.

The TaskContract is the single source of truth flowing from the Intake layer
(L3) to the Planning layer (L2). It is a superset of the InjectionPack,
adding explicit stop conditions and exploration space definition.

Schema Version History:
- 1.0.0: Initial version with 'version' field
- 2.0.0: Renamed 'version' to 'schema_version', added protocol_metadata
"""
from __future__ import annotations

import uuid
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from app.core.db import utcnow_iso
from app.contracts.versioning import BaseVersionedContract

__all__ = [
    "DimensionDef",
    "ExplorationSpace",
    "HumanGatePolicy",
    "ObjectiveSpec",
    "SafetyEnvelope",
    "StopCondition",
    "TaskContract",
    "new_task_contract_id",
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def new_task_contract_id() -> str:
    """Generate a unique TaskContract identifier."""
    return f"tc-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Sub-models (ordered so forward references are resolved)
# ---------------------------------------------------------------------------

class StopCondition(BaseModel):
    """When the campaign should stop."""

    max_rounds: int = Field(ge=1, le=1000)
    max_total_runs: int | None = Field(default=None, ge=1)
    target_kpi_value: float | None = None
    target_kpi_direction: Literal["minimize", "maximize"] = "minimize"
    plateau_patience: int = Field(
        default=5, ge=1, description="Rounds without improvement before stopping"
    )
    plateau_threshold: float = Field(default=0.01, ge=0.001)
    max_wall_time_hours: float | None = Field(default=None, ge=0.1)


class DimensionDef(BaseModel):
    """One optimizable parameter dimension."""

    param_name: str
    param_type: Literal["number", "integer", "categorical", "boolean"]
    min_value: float | None = None
    max_value: float | None = None
    log_scale: bool = False
    choices: list[Any] | None = None
    step_key: str | None = None
    primitive: str | None = None
    unit: str = ""


class ExplorationSpace(BaseModel):
    """Defines the parameter search space with constraints."""

    dimensions: list[DimensionDef]
    forbidden_combinations: list[str] = Field(default_factory=list)
    strategy: Literal["lhs", "bayesian", "prior_guided", "random", "grid"] = "lhs"
    batch_size: int = Field(default=10, ge=1, le=100)


class ObjectiveSpec(BaseModel):
    """What to optimize."""

    objective_type: str
    primary_kpi: str
    direction: Literal["minimize", "maximize"]
    secondary_kpis: list[str] = Field(default_factory=list)
    acceptable_range_pct: float = Field(default=10.0, ge=1.0, le=50.0)


class SafetyEnvelope(BaseModel):
    """Hard safety constraints -- the Safety Agent uses this for veto."""

    max_temp_c: float = Field(default=95.0, ge=0.0, le=1200.0)
    max_volume_ul: float = Field(default=1000.0, ge=1.0, le=10000.0)
    allowed_primitives: list[str] = Field(default_factory=list)
    hazardous_reagents: list[str] = Field(default_factory=list)
    require_human_approval: bool = False


class HumanGatePolicy(BaseModel):
    """When human review is required."""

    auto_approve_magnitude: float = Field(default=0.3, ge=0.0, le=1.0)
    triggers: list[str] = Field(
        default_factory=lambda: ["safety_boundary_change"]
    )


# ---------------------------------------------------------------------------
# Top-level contract
# ---------------------------------------------------------------------------

class TaskContract(BaseVersionedContract):
    """L3 output: the complete 'job order' from scientist intent to executable spec.

    This is the single source of truth flowing from the Intake layer (L3)
    to the Planning layer (L2). Everything the campaign needs to know
    is in this contract.

    Schema Version: 2.0.0
    """

    # Class-level version metadata (ClassVar to avoid Pydantic treating as fields)
    SCHEMA_VERSION: ClassVar[str] = "2.0.0"
    CONTRACT_NAME: ClassVar[str] = "TaskContract"

    contract_id: str
    created_at: str
    created_by: str

    # What to optimize
    objective: ObjectiveSpec

    # Where to search
    exploration_space: ExplorationSpace

    # When to stop
    stop_conditions: StopCondition

    # Safety boundaries
    safety_envelope: SafetyEnvelope

    # Human oversight
    human_gate: HumanGatePolicy

    # Protocol template selection
    protocol_pattern_id: str
    protocol_optional_steps: list[str] = Field(default_factory=list)

    # v2.0.0: Protocol metadata (execution hints, custom labware, etc.)
    protocol_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional protocol execution metadata",
    )
    deprecation_warnings: list[str] = Field(
        default_factory=list,
        description="Warnings about deprecated fields or usage patterns",
    )

    # Provenance
    source_session_id: str | None = None
