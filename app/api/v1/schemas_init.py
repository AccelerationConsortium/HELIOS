"""Pydantic schemas for the Campaign Initialization conversation flow.

Defines the 6 injection pack objects (GoalSpec, ProtocolPatternSpec,
ParamSpaceSpec, SafetyRulesSpec, KPIConfigSpec, HumanGatePolicySpec)
plus conversation DTOs (SlotPresentation, RoundPresentation, etc.).

All models are Pydantic v2 BaseModel subclasses for automatic JSON
serialisation and OpenAPI schema generation.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Injection pack: 6 structured objects
# ---------------------------------------------------------------------------


class GoalSpec(BaseModel):
    """What the campaign is optimizing for.

    Maps to :class:`~app.services.campaign_loop.CampaignGoal`.
    """

    objective_type: str = Field(
        ...,
        description="Experiment category: oer_screening | synthesis_optimization | stability_testing | custom",
    )
    objective_kpi: str = Field(
        ..., description="KPI name from KPI_DEFINITIONS_V1 (e.g. overpotential_mv)"
    )
    direction: Literal["minimize", "maximize"]
    target_value: float | None = Field(
        default=None,
        description="Optional absolute target; campaign stops when reached",
    )
    acceptable_range_pct: float = Field(
        default=10.0, ge=1.0, le=50.0, description="Acceptable range around target (%)"
    )


class ProtocolPatternSpec(BaseModel):
    """Selected protocol pattern with step customisation.

    Maps to :func:`~app.services.protocol_patterns.get_pattern` +
    ``to_protocol_json()``.
    """

    pattern_id: str = Field(..., description="Registered pattern id (e.g. oer_screening)")
    optional_steps: list[str] = Field(
        default_factory=list, description="Steps the user opted in/out of"
    )
    mandatory_steps: list[str] = Field(
        default_factory=list, description="Steps that cannot be removed (display-only)"
    )


class DimensionSpec(BaseModel):
    """A single parameter dimension for optimization.

    Maps to :class:`~app.services.candidate_gen.SearchDimension`.
    """

    param_name: str
    param_type: str = Field(
        ..., description="number | integer | categorical"
    )
    min_value: float | None = None
    max_value: float | None = None
    log_scale: bool = False
    choices: list[Any] | None = None
    optimizable: bool = True
    step_key: str | None = None
    primitive: str | None = None
    unit: str = ""
    description: str = ""
    safety_locked: bool = False


class ParamSpaceSpec(BaseModel):
    """Complete parameter space definition.

    Maps to :class:`~app.services.candidate_gen.ParameterSpace`.
    """

    dimensions: list[DimensionSpec] = Field(default_factory=list)
    strategy: str = Field(
        default="lhs",
        description="Sampling strategy: lhs | prior_guided | random | grid",
    )
    batch_size: int = Field(default=10, ge=1, le=100)
    forbidden_combinations: list[str] = Field(
        default_factory=list,
        description="Constraint expressions (e.g. 'annealing_temp > 500 AND ratio > 5')",
    )


class SafetyRulesSpec(BaseModel):
    """Safety constraints for the campaign.

    Maps to ``policy_snapshot`` dict used by
    :func:`~app.services.safety.evaluate_preflight`.
    """

    max_temp_c: float = Field(default=95.0, ge=0.0, le=1200.0)
    max_volume_ul: float = Field(default=1000.0, ge=1.0, le=10000.0)
    allowed_primitives: list[str] = Field(default_factory=list)
    require_human_approval: bool = False
    hazardous_reagents: list[str] = Field(default_factory=list)


class KPIConfigSpec(BaseModel):
    """KPI selection and targets.

    References entries in ``KPI_DEFINITIONS_V1`` and
    ``KPI_DEFINITIONS_V1_RUN``.
    """

    primary_kpi: str
    secondary_kpis: list[str] = Field(default_factory=list)
    target_value: float | None = None
    acceptable_range_pct: float = Field(default=10.0, ge=1.0, le=50.0)


class HumanGatePolicySpec(BaseModel):
    """When and how humans are consulted during autonomous phase.

    Controls evolution engine approval thresholds and convergence
    stopping conditions.
    """

    auto_approve_magnitude: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Evolution proposals below this magnitude are auto-approved",
    )
    human_gate_triggers: list[str] = Field(
        default_factory=lambda: ["safety_boundary_change"],
        description="Events that force human review",
    )
    plateau_threshold: float = Field(
        default=0.01, ge=0.001, le=0.1,
        description="Convergence plateau detection threshold",
    )
    max_rounds: int = Field(default=20, ge=1, le=1000)
    budget_limit_runs: int | None = Field(
        default=None, ge=1,
        description="Hard limit on total runs (None = no limit)",
    )


class InjectionPackMetadata(BaseModel):
    """Audit metadata for the injection pack."""

    session_id: str
    version: str = "1.0"
    created_at: str
    created_by: str
    checksum: str = Field(
        ..., description="SHA-256 of the pack JSON (excluding metadata itself)"
    )


class InjectionPack(BaseModel):
    """The complete output of the initialization conversation.

    Contains all 6 structured objects needed to launch a campaign,
    plus audit metadata.
    """

    goal: GoalSpec
    protocol: ProtocolPatternSpec
    param_space: ParamSpaceSpec
    safety: SafetyRulesSpec
    kpi_config: KPIConfigSpec
    human_gate: HumanGatePolicySpec
    metadata: InjectionPackMetadata


# ---------------------------------------------------------------------------
# Conversation DTOs
# ---------------------------------------------------------------------------


class SlotPresentation(BaseModel):
    """Definition of a single input widget for the frontend."""

    name: str
    widget: str = Field(
        ...,
        description="Widget type: select | multiselect | number | toggle | text | param_editor | display",
    )
    label: str
    hint: str | None = None
    options: list[Any] | None = None
    default: Any | None = None
    min_val: float | None = None
    max_val: float | None = None
    step_val: float | None = None
    required: bool = True
    error: str | None = None
    current_value: Any | None = None
    unit: str = ""


class RoundPresentation(BaseModel):
    """A complete round for the frontend to render."""

    session_id: str | None = Field(default=None, description="Session identifier (included in start response)")
    round_number: int = Field(ge=1, le=5)
    round_name: str
    message: str = Field(..., description="Bot's question / prompt text")
    slots: list[SlotPresentation]
    is_final: bool = False
    completed: bool = False


class RoundResponse(BaseModel):
    """User's structured response to a round."""

    responses: dict[str, Any]


class RoundResult(BaseModel):
    """Result of submitting a round — success or error with re-ask."""

    success: bool
    next_round: RoundPresentation | None = None
    errors: dict[str, list[str]] = Field(default_factory=dict)
    injection_pack_preview: dict[str, Any] | None = None


class SessionStatus(BaseModel):
    """Current status of an initialization session."""

    session_id: str
    status: str = Field(..., description="active | completed | abandoned")
    current_round: int
    completed_rounds: list[int] = Field(default_factory=list)
    filled_slots: dict[str, Any] = Field(default_factory=dict)
    injection_pack_preview: dict[str, Any] | None = None
