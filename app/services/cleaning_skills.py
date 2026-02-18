"""Cleaning Device Skills — per-device cleaning primitives and workflow composition.

Organises cleaning capabilities by **device type** (ultrasonic bath, acid rinse,
water flush, pump flush, electrode clean).  Each skill defines an ordered
primitive sequence, required hardware, safety constraints, and configurable
parameters.

Skills can be composed into higher-level workflows (pre_deposition_clean,
post_deposition_clean, etc.) via ``compose_workflow()``.

All registries are populated at import time with built-in skills/workflows.
"""
from __future__ import annotations

import copy
import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "CleaningDeviceType",
    "CleaningParam",
    "SafetyConstraint",
    "CleaningSkill",
    "CleaningWorkflow",
    "register_cleaning_skill",
    "get_cleaning_skill",
    "list_cleaning_skills",
    "register_cleaning_workflow",
    "get_cleaning_workflow",
    "list_cleaning_workflows",
    "compose_workflow",
]


# ---------------------------------------------------------------------------
# Enums & data models
# ---------------------------------------------------------------------------


class CleaningDeviceType(str, Enum):
    ULTRASONIC_BATH = "ultrasonic_bath"
    ACID_RINSE = "acid_rinse"
    WATER_FLUSH = "water_flush"
    PUMP_FLUSH = "pump_flush"
    ELECTRODE_CLEAN = "electrode_clean"


class CleaningParam(BaseModel):
    """A configurable parameter for a cleaning skill."""

    name: str
    param_type: str = "number"  # "number" | "integer" | "boolean"
    default: Any = None
    min_value: float | None = None
    max_value: float | None = None
    unit: str = ""
    description: str = ""
    configurable: bool = True


class SafetyConstraint(BaseModel):
    """A safety constraint attached to a cleaning skill."""

    constraint_type: str  # "max_duration" | "requires_rinse_after" | "incompatible_with" | "requires_ppe"
    value: Any = None
    description: str = ""


class CleaningSkill(BaseModel):
    """A single device-level cleaning skill."""

    id: str
    name: str
    device_type: CleaningDeviceType
    description: str = ""
    primitive_sequence: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of protocol step dicts (primitive + params)",
    )
    required_hardware: list[str] = Field(
        default_factory=list,
        description="Hardware IDs required, e.g. ['plc', 'ultrasonic_unit_1']",
    )
    safety_constraints: list[SafetyConstraint] = Field(default_factory=list)
    params: list[CleaningParam] = Field(default_factory=list)
    estimated_duration_s: int = 30
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)

    def to_protocol_steps(
        self,
        param_overrides: dict[str, Any] | None = None,
        step_prefix: str = "",
    ) -> list[dict[str, Any]]:
        """Expand this skill into compiler-ready protocol steps.

        Parameters
        ----------
        param_overrides:
            Override default parameter values.
        step_prefix:
            Prefix for step_key generation (e.g. "round_3_pre_").

        Returns
        -------
        list of protocol step dicts with ``step_key``, ``primitive``, ``params``.
        """
        overrides = param_overrides or {}
        steps: list[dict[str, Any]] = []

        for i, step_template in enumerate(self.primitive_sequence):
            step = copy.deepcopy(step_template)
            step_key = f"{step_prefix}{self.id}_step_{i}"
            step.setdefault("step_key", step_key)

            # Apply param overrides
            step_params = step.get("params", {})
            for param_def in self.params:
                if param_def.name in overrides:
                    step_params[param_def.name] = overrides[param_def.name]
                elif param_def.name in step_params:
                    pass  # keep template value
                elif param_def.default is not None:
                    step_params[param_def.name] = param_def.default
            step["params"] = step_params
            steps.append(step)

        return steps


class CleaningWorkflow(BaseModel):
    """Composed sequence of cleaning skills for a specific purpose."""

    id: str
    name: str
    purpose: str  # "pre_deposition" | "post_deposition" | "tool_clean" | "full_cycle" | "custom"
    skill_sequence: list[str] = Field(
        default_factory=list,
        description="Ordered list of skill IDs to execute",
    )
    description: str = ""


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

_CLEANING_SKILL_REGISTRY: dict[str, CleaningSkill] = {}
_CLEANING_WORKFLOW_REGISTRY: dict[str, CleaningWorkflow] = {}


def register_cleaning_skill(skill: CleaningSkill) -> None:
    """Register a cleaning skill."""
    _CLEANING_SKILL_REGISTRY[skill.id] = skill


def get_cleaning_skill(skill_id: str) -> CleaningSkill | None:
    """Get a cleaning skill by ID."""
    return _CLEANING_SKILL_REGISTRY.get(skill_id)


def list_cleaning_skills(
    device_type: CleaningDeviceType | None = None,
) -> list[CleaningSkill]:
    """List all registered cleaning skills, optionally filtered by device type."""
    skills = list(_CLEANING_SKILL_REGISTRY.values())
    if device_type is not None:
        skills = [s for s in skills if s.device_type == device_type]
    return skills


def register_cleaning_workflow(workflow: CleaningWorkflow) -> None:
    """Register a cleaning workflow."""
    _CLEANING_WORKFLOW_REGISTRY[workflow.id] = workflow


def get_cleaning_workflow(workflow_id: str) -> CleaningWorkflow | None:
    """Get a cleaning workflow by ID."""
    return _CLEANING_WORKFLOW_REGISTRY.get(workflow_id)


def list_cleaning_workflows() -> list[CleaningWorkflow]:
    """List all registered cleaning workflows."""
    return list(_CLEANING_WORKFLOW_REGISTRY.values())


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def compose_workflow(
    skill_ids: list[str],
    params_override: dict[str, dict[str, Any]] | None = None,
    step_prefix: str = "",
) -> list[dict[str, Any]]:
    """Expand a sequence of skill IDs into a flat list of compiler-ready steps.

    Parameters
    ----------
    skill_ids:
        Ordered list of cleaning skill IDs.
    params_override:
        Dict keyed by skill_id → param overrides for that skill.
    step_prefix:
        Prefix for all generated step keys.

    Returns
    -------
    list of protocol step dicts ready for the compiler.

    Raises
    ------
    ValueError
        If a skill_id is not found in the registry.
    """
    overrides = params_override or {}
    all_steps: list[dict[str, Any]] = []

    for skill_id in skill_ids:
        skill = get_cleaning_skill(skill_id)
        if skill is None:
            raise ValueError(f"Cleaning skill not found: {skill_id!r}")

        skill_overrides = overrides.get(skill_id, {})
        prefix = f"{step_prefix}{skill_id}_"
        steps = skill.to_protocol_steps(
            param_overrides=skill_overrides,
            step_prefix=prefix,
        )
        all_steps.extend(steps)

    return all_steps


def expand_workflow(
    workflow_id: str,
    params_override: dict[str, dict[str, Any]] | None = None,
    step_prefix: str = "",
) -> list[dict[str, Any]]:
    """Expand a registered workflow into compiler-ready steps.

    Convenience wrapper around ``compose_workflow`` that looks up
    a workflow by ID first.
    """
    wf = get_cleaning_workflow(workflow_id)
    if wf is None:
        raise ValueError(f"Cleaning workflow not found: {workflow_id!r}")
    return compose_workflow(wf.skill_sequence, params_override, step_prefix)


def validate_skill_composition(skill_ids: list[str]) -> list[str]:
    """Validate that a composition of skills is safe.

    Returns a list of warnings/errors (empty = safe).
    """
    issues: list[str] = []
    prev_postconditions: list[str] = []

    for skill_id in skill_ids:
        skill = get_cleaning_skill(skill_id)
        if skill is None:
            issues.append(f"Unknown skill: {skill_id}")
            continue

        # Check preconditions against previous postconditions
        for pre in skill.preconditions:
            if pre not in prev_postconditions and prev_postconditions:
                issues.append(
                    f"Skill '{skill_id}' requires '{pre}' but previous skills "
                    f"only guarantee: {prev_postconditions}"
                )

        # Check safety constraints
        for constraint in skill.safety_constraints:
            if constraint.constraint_type == "incompatible_with":
                if constraint.value in skill_ids:
                    issues.append(
                        f"Skill '{skill_id}' is incompatible with '{constraint.value}'"
                    )
            if constraint.constraint_type == "requires_rinse_after":
                idx = skill_ids.index(skill_id)
                if idx == len(skill_ids) - 1:
                    issues.append(
                        f"Skill '{skill_id}' requires a rinse step after it, "
                        f"but it is the last skill in the sequence"
                    )

        prev_postconditions = list(skill.postconditions)

    return issues


# ---------------------------------------------------------------------------
# Built-in skills
# ---------------------------------------------------------------------------

# -- Ultrasonic bath skills --

register_cleaning_skill(CleaningSkill(
    id="ultrasonic_water_clean",
    name="Ultrasonic Water Clean",
    device_type=CleaningDeviceType.ULTRASONIC_BATH,
    description="Run ultrasonic bath in the water chamber to remove loose particles.",
    primitive_sequence=[
        {
            "primitive": "plc.set_ultrasonic_on_timer",
            "params": {"chamber": "water", "duration_ms": 10000},
        },
    ],
    required_hardware=["plc"],
    safety_constraints=[
        SafetyConstraint(
            constraint_type="max_duration",
            value=60000,
            description="Max ultrasonic duration 60s to prevent cavitation damage",
        ),
    ],
    params=[
        CleaningParam(
            name="duration_ms",
            param_type="integer",
            default=10000,
            min_value=1000,
            max_value=60000,
            unit="ms",
            description="Ultrasonic bath duration",
        ),
    ],
    estimated_duration_s=15,
    preconditions=[],
    postconditions=["surface_particles_removed"],
))

register_cleaning_skill(CleaningSkill(
    id="ultrasonic_acid_clean",
    name="Ultrasonic Acid Clean",
    device_type=CleaningDeviceType.ULTRASONIC_BATH,
    description="Run ultrasonic bath in the acid (H2SO4) chamber for deep cleaning.",
    primitive_sequence=[
        {
            "primitive": "plc.set_ultrasonic_on_timer",
            "params": {"chamber": "acid", "duration_ms": 10000},
        },
    ],
    required_hardware=["plc"],
    safety_constraints=[
        SafetyConstraint(
            constraint_type="max_duration",
            value=30000,
            description="Max acid ultrasonic duration 30s to limit acid exposure",
        ),
        SafetyConstraint(
            constraint_type="requires_rinse_after",
            value=True,
            description="Must rinse with water after acid cleaning",
        ),
    ],
    params=[
        CleaningParam(
            name="duration_ms",
            param_type="integer",
            default=10000,
            min_value=1000,
            max_value=30000,
            unit="ms",
            description="Acid ultrasonic bath duration",
        ),
    ],
    estimated_duration_s=15,
    preconditions=[],
    postconditions=["acid_cleaned"],
))

# -- Water flush skills --

register_cleaning_skill(CleaningSkill(
    id="water_flush_reactor",
    name="Water Flush Reactor",
    device_type=CleaningDeviceType.WATER_FLUSH,
    description="Flush the reactor well with DI water using the peristaltic pump.",
    primitive_sequence=[
        {
            "primitive": "plc.set_pump_on_timer",
            "params": {"pump_id": "water_pump", "duration_ms": 5000},
        },
    ],
    required_hardware=["plc"],
    params=[
        CleaningParam(
            name="duration_ms",
            param_type="integer",
            default=5000,
            min_value=2000,
            max_value=30000,
            unit="ms",
            description="Water flush pump duration",
        ),
    ],
    estimated_duration_s=8,
    preconditions=[],
    postconditions=["reactor_flushed_water"],
))

# -- Acid rinse skills --

register_cleaning_skill(CleaningSkill(
    id="acid_rinse_reactor",
    name="Acid Rinse Reactor",
    device_type=CleaningDeviceType.ACID_RINSE,
    description="Rinse the reactor with dilute acid solution via robot aspirate/dispense.",
    primitive_sequence=[
        {
            "primitive": "robot.aspirate",
            "params": {
                "labware": "acid_reservoir",
                "well": "A1",
                "volume_ul": 200,
            },
        },
        {
            "primitive": "robot.dispense",
            "params": {
                "labware": "reactor",
                "well": "{{destination_well}}",
                "volume_ul": 200,
            },
        },
    ],
    required_hardware=["ot2"],
    safety_constraints=[
        SafetyConstraint(
            constraint_type="requires_rinse_after",
            value=True,
            description="Must rinse with water after acid rinse",
        ),
        SafetyConstraint(
            constraint_type="requires_ppe",
            value="acid_resistant_gloves",
            description="Ensure acid-compatible tips are used",
        ),
    ],
    params=[
        CleaningParam(
            name="volume_ul",
            param_type="number",
            default=200,
            min_value=50,
            max_value=500,
            unit="uL",
            description="Volume of acid rinse",
        ),
    ],
    estimated_duration_s=20,
    preconditions=[],
    postconditions=["reactor_acid_rinsed"],
))

# -- Pump flush skills --

register_cleaning_skill(CleaningSkill(
    id="pump_flush_lines",
    name="Pump Flush Lines",
    device_type=CleaningDeviceType.PUMP_FLUSH,
    description="Flush all tubing/lines with DI water using the peristaltic pump.",
    primitive_sequence=[
        {
            "primitive": "plc.set_pump_on_timer",
            "params": {"pump_id": "flush_pump", "duration_ms": 8000},
        },
    ],
    required_hardware=["plc"],
    params=[
        CleaningParam(
            name="duration_ms",
            param_type="integer",
            default=8000,
            min_value=3000,
            max_value=30000,
            unit="ms",
            description="Flush pump duration",
        ),
    ],
    estimated_duration_s=12,
    preconditions=[],
    postconditions=["lines_flushed"],
))

# -- Electrode cleaning skills --

register_cleaning_skill(CleaningSkill(
    id="electrode_acid_clean",
    name="Electrode Acid Clean",
    device_type=CleaningDeviceType.ELECTRODE_CLEAN,
    description="Move electrode to acid chamber of wash station, ultrasonic clean, then water rinse.",
    primitive_sequence=[
        {
            "primitive": "robot.move_to_well",
            "params": {
                "labware": "ultrasonic_bath",
                "well": "A2",
                "offset_z": -5.0,
            },
            "description": "Move electrode to acid chamber",
        },
        {
            "primitive": "plc.set_ultrasonic_on_timer",
            "params": {"chamber": "acid", "duration_ms": 8000},
        },
        {
            "primitive": "robot.move_to_well",
            "params": {
                "labware": "ultrasonic_bath",
                "well": "A1",
                "offset_z": -5.0,
            },
            "description": "Move electrode to water chamber for rinse",
        },
        {
            "primitive": "plc.set_ultrasonic_on_timer",
            "params": {"chamber": "water", "duration_ms": 5000},
        },
    ],
    required_hardware=["ot2", "plc"],
    safety_constraints=[
        SafetyConstraint(
            constraint_type="max_duration",
            value=30000,
            description="Max acid exposure 30s for electrode",
        ),
    ],
    params=[
        CleaningParam(
            name="acid_duration_ms",
            param_type="integer",
            default=8000,
            min_value=3000,
            max_value=30000,
            unit="ms",
            description="Acid ultrasonic duration for electrode",
        ),
        CleaningParam(
            name="water_rinse_duration_ms",
            param_type="integer",
            default=5000,
            min_value=2000,
            max_value=15000,
            unit="ms",
            description="Water rinse ultrasonic duration after acid",
        ),
    ],
    estimated_duration_s=25,
    preconditions=[],
    postconditions=["electrode_clean", "acid_residue_removed"],
))

register_cleaning_skill(CleaningSkill(
    id="electrode_water_rinse",
    name="Electrode Water Rinse",
    device_type=CleaningDeviceType.ELECTRODE_CLEAN,
    description="Move electrode to water chamber and ultrasonic rinse.",
    primitive_sequence=[
        {
            "primitive": "robot.move_to_well",
            "params": {
                "labware": "ultrasonic_bath",
                "well": "A1",
                "offset_z": -5.0,
            },
            "description": "Move electrode to water chamber",
        },
        {
            "primitive": "plc.set_ultrasonic_on_timer",
            "params": {"chamber": "water", "duration_ms": 5000},
        },
    ],
    required_hardware=["ot2", "plc"],
    params=[
        CleaningParam(
            name="duration_ms",
            param_type="integer",
            default=5000,
            min_value=2000,
            max_value=15000,
            unit="ms",
            description="Water rinse ultrasonic duration",
        ),
    ],
    estimated_duration_s=10,
    preconditions=[],
    postconditions=["electrode_rinsed"],
))


# ---------------------------------------------------------------------------
# Built-in workflows
# ---------------------------------------------------------------------------

register_cleaning_workflow(CleaningWorkflow(
    id="pre_deposition_clean",
    name="Pre-Deposition Clean",
    purpose="pre_deposition",
    skill_sequence=["water_flush_reactor", "ultrasonic_water_clean"],
    description="Flush reactor with water and ultrasonic clean before deposition.",
))

register_cleaning_workflow(CleaningWorkflow(
    id="post_deposition_clean",
    name="Post-Deposition Clean",
    purpose="post_deposition",
    skill_sequence=[
        "acid_rinse_reactor",
        "ultrasonic_acid_clean",
        "water_flush_reactor",
        "ultrasonic_water_clean",
    ],
    description="Full acid + water cleaning cycle after deposition.",
))

register_cleaning_workflow(CleaningWorkflow(
    id="tool_clean_after_deposition",
    name="Tool Clean After Deposition",
    purpose="tool_clean",
    skill_sequence=["electrode_acid_clean", "electrode_water_rinse"],
    description="Clean deposition tool with acid then water rinse.",
))

register_cleaning_workflow(CleaningWorkflow(
    id="full_cycle_clean",
    name="Full Cycle Clean",
    purpose="full_cycle",
    skill_sequence=[
        "water_flush_reactor",
        "ultrasonic_water_clean",
        "acid_rinse_reactor",
        "ultrasonic_acid_clean",
        "water_flush_reactor",
        "ultrasonic_water_clean",
        "electrode_acid_clean",
        "electrode_water_rinse",
    ],
    description="Complete cleaning cycle: reactor flush, acid, rinse, plus electrode cleaning.",
))
