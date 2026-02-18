"""Cleaning Agent — expand cleaning workflows into compiler-ready protocol steps.

Wraps the ``cleaning_skills`` service with the BaseAgent interface.
Selects a workflow by purpose or accepts explicit skill IDs, validates
safety constraints, and returns expanded protocol steps.

Layer: L1 (compilation helper)
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O models
# ---------------------------------------------------------------------------


class CleaningInput(BaseModel):
    """Input for the Cleaning agent."""

    purpose: str = Field(
        default="",
        description=(
            "Cleaning purpose: pre_deposition | post_deposition | tool_clean | "
            "full_cycle | custom"
        ),
    )
    workflow_id: str = Field(
        default="",
        description="Explicit workflow ID (overrides purpose-based lookup)",
    )
    skill_ids: list[str] = Field(
        default_factory=list,
        description="Explicit skill ID sequence (overrides workflow_id and purpose)",
    )
    param_overrides: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-skill parameter overrides keyed by skill_id",
    )
    step_prefix: str = Field(
        default="",
        description="Prefix for generated step keys (e.g. 'round_3_pre_')",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (deck_plan, available hardware, etc.)",
    )


class CleaningOutput(BaseModel):
    """Output from the Cleaning agent."""

    status: str = Field(
        ...,
        description="Status: success | validation_warning | error",
    )
    protocol_steps: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Compiler-ready protocol step dicts",
    )
    estimated_duration_s: int = 0
    skills_used: list[str] = Field(default_factory=list)
    workflow_id: str = ""
    validation_issues: list[str] = Field(default_factory=list)
    chat_message: str = ""


# ---------------------------------------------------------------------------
# Purpose → workflow mapping
# ---------------------------------------------------------------------------

_PURPOSE_WORKFLOW_MAP: dict[str, str] = {
    "pre_deposition": "pre_deposition_clean",
    "post_deposition": "post_deposition_clean",
    "tool_clean": "tool_clean_after_deposition",
    "full_cycle": "full_cycle_clean",
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CleaningAgent(BaseAgent[CleaningInput, CleaningOutput]):
    """Expand cleaning workflows into protocol steps.

    Resolution order:
    1. ``skill_ids`` (explicit skill sequence)
    2. ``workflow_id`` (registered workflow)
    3. ``purpose`` (lookup in purpose→workflow map)
    """

    name = "cleaning"
    description = "Cleaning workflow → compiler-ready protocol steps"
    layer = "L1"

    def validate_input(self, input_data: CleaningInput) -> list[str]:
        errors: list[str] = []
        if not input_data.skill_ids and not input_data.workflow_id and not input_data.purpose:
            errors.append(
                "At least one of skill_ids, workflow_id, or purpose must be provided"
            )
        return errors

    async def process(self, input_data: CleaningInput) -> CleaningOutput:
        from app.services.cleaning_skills import (
            compose_workflow,
            expand_workflow,
            get_cleaning_workflow,
            validate_skill_composition,
        )

        # Resolve skill IDs
        skill_ids: list[str] = []
        workflow_id = ""

        if input_data.skill_ids:
            skill_ids = input_data.skill_ids
            workflow_id = "custom"
        elif input_data.workflow_id:
            workflow_id = input_data.workflow_id
            wf = get_cleaning_workflow(workflow_id)
            if wf is None:
                return CleaningOutput(
                    status="error",
                    chat_message=f"Cleaning workflow not found: {workflow_id}",
                )
            skill_ids = wf.skill_sequence
        elif input_data.purpose:
            workflow_id = _PURPOSE_WORKFLOW_MAP.get(input_data.purpose, "")
            if not workflow_id:
                return CleaningOutput(
                    status="error",
                    chat_message=(
                        f"No workflow mapped for purpose '{input_data.purpose}'. "
                        f"Available: {list(_PURPOSE_WORKFLOW_MAP.keys())}"
                    ),
                )
            wf = get_cleaning_workflow(workflow_id)
            if wf is None:
                return CleaningOutput(
                    status="error",
                    chat_message=f"Cleaning workflow '{workflow_id}' not found in registry",
                )
            skill_ids = wf.skill_sequence

        # Validate safety
        issues = validate_skill_composition(skill_ids)

        # Expand into protocol steps
        try:
            steps = compose_workflow(
                skill_ids=skill_ids,
                params_override=input_data.param_overrides or None,
                step_prefix=input_data.step_prefix,
            )
        except ValueError as exc:
            return CleaningOutput(
                status="error",
                chat_message=str(exc),
                validation_issues=issues,
            )

        # Estimate duration
        from app.services.cleaning_skills import get_cleaning_skill

        total_duration = 0
        for sid in skill_ids:
            skill = get_cleaning_skill(sid)
            if skill:
                total_duration += skill.estimated_duration_s

        status = "validation_warning" if issues else "success"
        chat_msg = (
            f"Cleaning workflow '{workflow_id}' expanded: "
            f"{len(steps)} steps, ~{total_duration}s estimated."
        )
        if issues:
            chat_msg += f"\nWarnings: {'; '.join(issues)}"

        return CleaningOutput(
            status=status,
            protocol_steps=steps,
            estimated_duration_s=total_duration,
            skills_used=skill_ids,
            workflow_id=workflow_id,
            validation_issues=issues,
            chat_message=chat_msg,
        )
