"""Code Writer Agent -- L1 compilation layer.

Wraps the ot2-nlp-agent's Planner and Compiler to convert
natural language experiment descriptions into executable OT-2 protocol code.

The ot2-nlp-agent lives in a sibling directory and is imported via sys.path.
If it is not available, the agent degrades gracefully with a clear error.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent

# ---------------------------------------------------------------------------
# Lazy import helpers for ot2-nlp-agent
# ---------------------------------------------------------------------------

_OT2_AGENT_DIR = str(Path(__file__).resolve().parents[2] / "ot2-nlp-agent")
_ot2_available: bool | None = None


def _ensure_ot2_agent_importable() -> bool:
    """Add ot2-nlp-agent to sys.path if needed. Return True if importable."""
    global _ot2_available
    if _ot2_available is not None:
        return _ot2_available

    if _OT2_AGENT_DIR not in sys.path:
        sys.path.insert(0, _OT2_AGENT_DIR)
    try:
        import ot2_agent  # noqa: F401
        _ot2_available = True
    except ImportError:
        _ot2_available = False
    return _ot2_available


# ---------------------------------------------------------------------------
# Input / Output models
# ---------------------------------------------------------------------------


class CodeWriterInput(BaseModel):
    """Input for the CodeWriter agent."""

    intent: str = Field(
        ...,
        description="Natural language experiment description (EN or ZH)",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Planning context: devices, materials, constraints, etc.",
    )
    filled_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Pre-filled parameters. If provided, skips interactive step.",
    )
    candidate_idx: int = Field(
        default=0,
        description="Which candidate workflow to compile (0-based index).",
    )
    plan_only: bool = Field(
        default=False,
        description="If True, return plan candidates without compiling.",
    )


class CodeWriterOutput(BaseModel):
    """Output from the CodeWriter agent."""

    plan_candidates: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Serialized WorkflowDraft list from the Planner.",
    )
    selected_candidate_idx: int = Field(
        default=0,
        description="Index of the candidate that was compiled.",
    )
    missing_parameters: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Parameters the user still needs to fill in.",
    )
    python_code: str = Field(
        default="",
        description="Generated OT-2 protocol Python code.",
    )
    workflow_json: str = Field(
        default="",
        description="Workflow JSON representation.",
    )
    validation_errors: list[str] = Field(
        default_factory=list,
        description="Validation errors from the compiler.",
    )
    validation_warnings: list[str] = Field(
        default_factory=list,
        description="Validation warnings from the compiler.",
    )
    device_actions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Compiled device actions.",
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CodeWriterAgent(BaseAgent[CodeWriterInput, CodeWriterOutput]):
    """Convert natural-language experiment intent into OT-2 protocol code.

    Delegates to the ot2-nlp-agent's Planner (intent -> candidate workflows)
    and Compiler (confirmed workflow -> executable code).

    Layer: L1 (compilation).
    """

    name = "code_writer_agent"
    description = "NL intent -> OT-2 protocol code via ot2-nlp-agent Planner+Compiler"
    layer = "L1"

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate_input(self, input_data: CodeWriterInput) -> list[str]:
        errors: list[str] = []
        if not input_data.intent or not input_data.intent.strip():
            errors.append("intent must be a non-empty string")
        if input_data.candidate_idx < 0:
            errors.append("candidate_idx must be >= 0")
        if not _ensure_ot2_agent_importable():
            errors.append(
                "ot2-nlp-agent is not available. "
                f"Expected at: {_OT2_AGENT_DIR}"
            )
        return errors

    # ------------------------------------------------------------------ #
    # Processing
    # ------------------------------------------------------------------ #

    async def process(self, input_data: CodeWriterInput) -> CodeWriterOutput:
        # Import lazily so the module can be loaded even when ot2-nlp-agent
        # is absent (validate_input will catch that first).
        from ot2_agent.planner import Planner, ConfirmedWorkflow
        from ot2_agent.compiler import Compiler

        # --- Step 1: Plan ---
        planner = Planner()
        planner_output = planner.plan(
            user_input=input_data.intent,
            context=input_data.context or None,
        )

        candidates = planner_output.candidates
        plan_dicts = [c.to_dict() for c in candidates]

        # Clamp candidate_idx
        idx = min(input_data.candidate_idx, max(len(candidates) - 1, 0))

        # Collect missing parameters from the selected candidate
        missing_params: list[dict[str, Any]] = []
        if candidates:
            selected = candidates[idx]
            missing_params = [mi.to_dict() for mi in selected.missing_info]

        # --- plan_only: return early ---
        if input_data.plan_only or not candidates:
            return CodeWriterOutput(
                plan_candidates=plan_dicts,
                selected_candidate_idx=idx,
                missing_parameters=missing_params,
            )

        # --- Step 2: Compile ---
        selected_draft = candidates[idx]
        confirmed = ConfirmedWorkflow(
            draft=selected_draft,
            filled_parameters=input_data.filled_parameters,
        )

        compiler = Compiler()
        compiler_output = compiler.compile(confirmed)

        # Extract validation errors/warnings from EnhancedValidationResult
        validation_errors: list[str] = []
        validation_warnings: list[str] = []
        vr = compiler_output.validation_result
        vr_dict = vr.to_dict()

        for issue in vr_dict.get("issues", []):
            if issue.get("severity") == "ERROR":
                validation_errors.append(issue.get("message", ""))
            elif issue.get("severity") == "WARNING":
                validation_warnings.append(issue.get("message", ""))

        for conflict in vr_dict.get("resource_conflicts", []):
            msg = conflict.get("message", "")
            if conflict.get("severity") == "error":
                validation_errors.append(msg)
            else:
                validation_warnings.append(msg)

        for topo in vr_dict.get("topology_issues", []):
            msg = topo.get("message", "")
            if topo.get("severity") == "error":
                validation_errors.append(msg)
            else:
                validation_warnings.append(msg)

        # Serialize device_actions
        device_action_dicts = [a.to_dict() for a in compiler_output.device_actions]

        return CodeWriterOutput(
            plan_candidates=plan_dicts,
            selected_candidate_idx=idx,
            missing_parameters=missing_params,
            python_code=compiler_output.python_code,
            workflow_json=compiler_output.workflow_json,
            validation_errors=validation_errors,
            validation_warnings=validation_warnings,
            device_actions=device_action_dicts,
        )
