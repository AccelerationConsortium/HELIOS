"""NLP Code Agent — bridge between ot2-nlp-agent and the main orchestrator.

Wraps the CodeWriterAgent with a user confirmation flow:
1. Generate code via CodeWriterAgent
2. If auto_approve=False, create a confirmation request and pause
3. After user approval, convert workflow_json into compiler-ready protocol steps

Layer: L1 (compilation)
"""
from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O models
# ---------------------------------------------------------------------------


class NLPCodeInput(BaseModel):
    """Input for the NLPCode agent."""

    phase: str = Field(
        default="generate",
        description="Phase: generate | confirm_response",
    )

    # --- generate phase ---
    intent: str = Field(
        default="",
        description="Natural language experiment description",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Planning context (devices, materials, constraints)",
    )
    filled_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Pre-filled parameters",
    )
    auto_approve: bool = Field(
        default=False,
        description="If True, skip user confirmation",
    )
    campaign_id: str = ""

    # --- confirm_response phase ---
    confirmation_request_id: str = ""
    approved: bool = False
    modified_code: str | None = None
    modified_steps: list[dict[str, Any]] | None = None

    # --- carry-over ---
    previous_result: dict[str, Any] | None = None


class NLPCodeOutput(BaseModel):
    """Output from the NLPCode agent."""

    status: str = Field(
        ...,
        description=(
            "Status: needs_confirmation | confirmed | rejected | "
            "auto_approved | error"
        ),
    )

    # Code generation results
    confirmation_request_id: str = ""
    python_code: str = ""
    workflow_json: str = ""
    protocol_steps: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Compiler-ready protocol steps (available after confirmation)",
    )
    plan_candidates: list[dict[str, Any]] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)

    # Chat message
    chat_message: str = ""

    # Serialised state
    serialised_result: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class NLPCodeAgent(BaseAgent[NLPCodeInput, NLPCodeOutput]):
    """Bridge NLP code generation with user confirmation.

    Wraps CodeWriterAgent + CodeConfirmation flow to safely inject
    generated code into the orchestrator pipeline.
    """

    name = "nlp_code"
    description = "NL intent → confirmed protocol code"
    layer = "L1"

    def validate_input(self, input_data: NLPCodeInput) -> list[str]:
        errors: list[str] = []
        if input_data.phase == "generate":
            if not input_data.intent.strip():
                errors.append("intent is required for generate phase")
        elif input_data.phase == "confirm_response":
            if not input_data.confirmation_request_id:
                errors.append("confirmation_request_id is required")
            if not input_data.previous_result:
                errors.append("previous_result is required for confirm_response phase")
        else:
            errors.append(f"Unknown phase: {input_data.phase}")
        return errors

    async def process(self, input_data: NLPCodeInput) -> NLPCodeOutput:
        if input_data.phase == "generate":
            return await self._handle_generate(input_data)
        elif input_data.phase == "confirm_response":
            return self._handle_confirm_response(input_data)
        else:
            raise ValueError(f"Unknown phase: {input_data.phase}")

    # ------------------------------------------------------------------ #
    # Phase handlers
    # ------------------------------------------------------------------ #

    async def _handle_generate(self, input_data: NLPCodeInput) -> NLPCodeOutput:
        from app.agents.code_writer_agent import CodeWriterAgent, CodeWriterInput
        from app.services.code_confirmation import (
            CodeConfirmationRequest,
            request_code_confirmation,
        )

        # Step 1: Generate code via CodeWriterAgent
        writer = CodeWriterAgent()
        writer_input = CodeWriterInput(
            intent=input_data.intent,
            context=input_data.context,
            filled_parameters=input_data.filled_parameters,
        )

        writer_result = await writer.run(writer_input)

        if not writer_result.success or writer_result.output is None:
            return NLPCodeOutput(
                status="error",
                chat_message=(
                    "Code generation failed: "
                    + "; ".join(writer_result.errors)
                ),
                validation_errors=writer_result.errors,
            )

        output = writer_result.output

        # Convert workflow_json to protocol steps
        protocol_steps = self._convert_workflow_to_steps(
            output.workflow_json, output.device_actions
        )

        # Step 2: Create confirmation request
        confirmation_req = CodeConfirmationRequest(
            python_code=output.python_code,
            workflow_json=output.workflow_json,
            protocol_steps=protocol_steps,
            validation_errors=output.validation_errors,
            validation_warnings=output.validation_warnings,
            plan_candidates=output.plan_candidates,
            selected_candidate_idx=output.selected_candidate_idx,
            auto_approve=input_data.auto_approve,
            campaign_id=input_data.campaign_id,
        )

        request_id = request_code_confirmation(confirmation_req)

        serialised = {
            "request_id": request_id,
            "python_code": output.python_code,
            "workflow_json": output.workflow_json,
            "protocol_steps": protocol_steps,
            "plan_candidates": output.plan_candidates,
        }

        if input_data.auto_approve:
            return NLPCodeOutput(
                status="auto_approved",
                confirmation_request_id=request_id,
                python_code=output.python_code,
                workflow_json=output.workflow_json,
                protocol_steps=protocol_steps,
                plan_candidates=output.plan_candidates,
                validation_errors=output.validation_errors,
                validation_warnings=output.validation_warnings,
                chat_message="Code auto-approved and ready for injection.",
                serialised_result=serialised,
            )

        # Build code preview message
        code_preview = output.python_code[:500]
        if len(output.python_code) > 500:
            code_preview += "\n... (truncated)"

        chat_msg = (
            "## Generated Protocol Code\n\n"
            f"```python\n{code_preview}\n```\n\n"
            f"**{len(protocol_steps)} protocol steps** generated.\n"
        )
        if output.validation_errors:
            chat_msg += f"\nValidation errors: {output.validation_errors}\n"
        if output.validation_warnings:
            chat_msg += f"\nWarnings: {output.validation_warnings}\n"
        chat_msg += (
            f"\nConfirmation ID: `{request_id}`\n"
            "Please review and approve/reject this code."
        )

        return NLPCodeOutput(
            status="needs_confirmation",
            confirmation_request_id=request_id,
            python_code=output.python_code,
            workflow_json=output.workflow_json,
            protocol_steps=protocol_steps,
            plan_candidates=output.plan_candidates,
            validation_errors=output.validation_errors,
            validation_warnings=output.validation_warnings,
            chat_message=chat_msg,
            serialised_result=serialised,
        )

    def _handle_confirm_response(self, input_data: NLPCodeInput) -> NLPCodeOutput:
        from app.services.code_confirmation import (
            CodeConfirmationResponse,
            get_confirmed_code,
            respond_to_confirmation,
        )

        response = CodeConfirmationResponse(
            request_id=input_data.confirmation_request_id,
            approved=input_data.approved,
            modified_code=input_data.modified_code,
            modified_steps=input_data.modified_steps,
        )

        status = respond_to_confirmation(response)

        if not input_data.approved:
            return NLPCodeOutput(
                status="rejected",
                confirmation_request_id=input_data.confirmation_request_id,
                chat_message="Code generation rejected by user.",
                serialised_result=input_data.previous_result or {},
            )

        # Get the (possibly modified) confirmed code
        confirmed = get_confirmed_code(input_data.confirmation_request_id)
        if confirmed is None:
            return NLPCodeOutput(
                status="error",
                chat_message="Could not retrieve confirmed code.",
            )

        python_code, workflow_json, protocol_steps = confirmed

        return NLPCodeOutput(
            status="confirmed",
            confirmation_request_id=input_data.confirmation_request_id,
            python_code=python_code,
            workflow_json=workflow_json,
            protocol_steps=protocol_steps,
            chat_message="Code approved and ready for pipeline injection.",
            serialised_result=input_data.previous_result or {},
        )

    # ------------------------------------------------------------------ #
    # Conversion helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _convert_workflow_to_steps(
        workflow_json: str,
        device_actions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert ot2-nlp-agent workflow JSON and device actions
        into NeverOT compiler-ready protocol steps.

        Each device action maps to a protocol step with:
        - step_key: unique identifier
        - primitive: device.action format
        - params: action parameters
        """
        steps: list[dict[str, Any]] = []

        for i, action in enumerate(device_actions):
            device = action.get("device", "unknown")
            action_type = action.get("action", "unknown")
            params = action.get("params", {})

            step = {
                "step_key": f"nlp_generated_step_{i}",
                "primitive": f"{device}.{action_type}",
                "params": params,
            }

            # Preserve metadata from the action
            if "description" in action:
                step["description"] = action["description"]
            if "preconditions" in action:
                step["preconditions"] = action["preconditions"]

            steps.append(step)

        return steps
