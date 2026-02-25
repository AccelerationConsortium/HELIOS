"""Onboarding Agent — integrates new instruments into the OTbot platform.

Wraps the InstrumentOnboardingService with the BaseAgent interface so it
can be driven by the orchestrator or called directly from the API layer.

The agent follows a multi-turn conversation pattern:
  1. User provides an InstrumentSpec (via chat or structured input)
  2. Agent generates integration code and surfaces confirmation prompts
  3. User confirms/adjusts safety classifications, communication, KPI
  4. Agent regenerates code with confirmed values
  5. Agent writes files to disk (with user approval)

The confirmation flow uses ``pending_confirmations`` in the output.  When
the orchestrator sees pending confirmations, it formats them as chat
messages using ``format_confirmations_for_chat()`` and pauses for user
input.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent, AgentResult
from app.services.instrument_onboarding import (
    CommunicationType,
    ConfirmationItem,
    InstrumentOnboardingService,
    InstrumentSpec,
    OnboardingResult,
    ParamInput,
    PrimitiveInput,
)
from app.services.llm_gateway import get_llm_provider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O models
# ---------------------------------------------------------------------------


class PrimitiveSpec(BaseModel):
    """API-friendly representation of a single primitive."""

    name: str
    description: str = ""
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    hazardous: bool = False
    generates_data: bool = False
    timeout_seconds: int = 30
    retries: int = 1
    preconditions: list[str] = Field(default_factory=list)
    effects: list[str] = Field(default_factory=list)


class OnboardingInput(BaseModel):
    """Input for the onboarding agent.

    Two modes:
    - ``phase="generate"``: Provide ``instrument`` spec to generate code.
    - ``phase="confirm"``:  Provide ``confirmations`` dict to approve items.
    """

    phase: str = Field(
        default="generate",
        description="Phase: discover | generate | confirm | write",
    )

    # --- generate phase ---
    instrument_name: str = ""
    manufacturer: str = ""
    model: str = ""
    communication: str = "usb"
    description: str = ""
    sdk_package: str = ""
    docs_url: str = ""  # Vendor documentation URL (used in discover phase)
    primitives: list[PrimitiveSpec] = Field(default_factory=list)

    # --- confirm phase ---
    confirmations: dict[str, Any] = Field(default_factory=dict)

    # --- carry-over state ---
    # The serialised OnboardingResult from a previous turn.
    # In a real multi-turn chat this is stored in the session; for the
    # agent interface it's passed explicitly.
    previous_result: dict[str, Any] | None = None

    # --- write phase ---
    force_write: bool = False


class ConfirmationItemOut(BaseModel):
    """A single confirmation item returned to the chat UI."""

    id: str
    type: str
    primitive_name: str
    question: str
    current_value: Any
    options: list[str] | None = None
    confirmed: bool = False


class GeneratedFileOut(BaseModel):
    """Summary of a generated file."""

    path: str
    is_patch: bool = False
    description: str = ""


class OnboardingOutput(BaseModel):
    """Output from the onboarding agent."""

    status: str = Field(
        ...,
        description="Status: needs_confirmation | ready_to_write | written | error",
    )
    instrument_name: str = ""
    display_name: str = ""

    # Confirmation flow
    pending_confirmations: list[ConfirmationItemOut] = Field(default_factory=list)
    confirmed_count: int = 0
    total_confirmations: int = 0

    # Chat-friendly message for the user
    chat_message: str = ""

    # Generated files (summaries, not full content)
    files: list[GeneratedFileOut] = Field(default_factory=list)
    written_paths: list[str] = Field(default_factory=list)

    # Manual follow-ups
    manual_todo: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # Serialised result for multi-turn state
    serialised_result: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class OnboardingAgent(BaseAgent[OnboardingInput, OnboardingOutput]):
    """Agent that onboards new instruments into OTbot.

    Lifecycle (multi-turn):
    1. generate: InstrumentSpec → code + confirmations
    2. confirm:  user approvals → regenerated code
    3. write:    write files to disk
    """

    name = "onboarding"
    description = "Instrument onboarding agent"
    layer = "L3"  # task-entry layer

    def __init__(self) -> None:
        super().__init__()
        self._service = InstrumentOnboardingService()

    def validate_input(self, input_data: OnboardingInput) -> list[str]:
        errors: list[str] = []
        if input_data.phase == "discover":
            if not input_data.instrument_name:
                errors.append("instrument_name is required for discover phase")
        elif input_data.phase == "generate":
            if not input_data.instrument_name:
                errors.append("instrument_name is required")
            if not input_data.primitives:
                errors.append("At least one primitive is required")
        elif input_data.phase == "confirm":
            if not input_data.confirmations:
                errors.append("confirmations dict is required")
            if not input_data.previous_result:
                errors.append("previous_result is required for confirm phase")
        elif input_data.phase == "write":
            if not input_data.previous_result:
                errors.append("previous_result is required for write phase")
        else:
            errors.append(f"Unknown phase: {input_data.phase}")
        return errors

    async def process(self, input_data: OnboardingInput) -> OnboardingOutput:
        if input_data.phase == "discover":
            return await self._handle_discover(input_data)
        elif input_data.phase == "generate":
            return self._handle_generate(input_data)
        elif input_data.phase == "confirm":
            return self._handle_confirm(input_data)
        elif input_data.phase == "write":
            return self._handle_write(input_data)
        else:
            raise ValueError(f"Unknown phase: {input_data.phase}")

    # ------------------------------------------------------------------ #
    # Phase handlers
    # ------------------------------------------------------------------ #

    def _handle_generate(self, input_data: OnboardingInput) -> OnboardingOutput:
        """Generate integration code from instrument spec."""
        # Build InstrumentSpec from input
        spec = self._build_spec(input_data)

        # Generate
        result = self._service.generate(spec)

        # Build output
        return self._result_to_output(result)

    def _handle_confirm(self, input_data: OnboardingInput) -> OnboardingOutput:
        """Apply user confirmations and regenerate code."""
        result = self._deserialise_result(input_data.previous_result or {})

        # Apply confirmations
        result = self._service.confirm(result, input_data.confirmations)

        return self._result_to_output(result)

    def _handle_write(self, input_data: OnboardingInput) -> OnboardingOutput:
        """Write files to disk."""
        result = self._deserialise_result(input_data.previous_result or {})

        try:
            written = self._service.write_files(
                result, force=input_data.force_write,
            )
        except RuntimeError as exc:
            return OnboardingOutput(
                status="error",
                instrument_name=result.spec.name,
                display_name=result.spec.display_name,
                chat_message=str(exc),
                warnings=[str(exc)],
                serialised_result=self._serialise_result(result),
            )

        return OnboardingOutput(
            status="written",
            instrument_name=result.spec.name,
            display_name=result.spec.display_name,
            chat_message=(
                f"✅ Successfully generated {len(written)} files for "
                f"**{result.spec.display_name}**:\n\n"
                + "\n".join(f"- `{p}`" for p in written)
                + "\n\n**Manual follow-ups:**\n"
                + "\n".join(f"- {t}" for t in result.manual_todo)
            ),
            files=[
                GeneratedFileOut(
                    path=gf.path, is_patch=gf.is_patch, description=gf.description,
                )
                for gf in result.files
            ],
            written_paths=written,
            manual_todo=result.manual_todo,
            warnings=result.warnings,
            serialised_result=self._serialise_result(result),
        )

    async def _handle_discover(self, input_data: OnboardingInput) -> OnboardingOutput:
        """Auto-discover device primitives via LLM inference.

        Calls the LLM with a structured prompt describing the instrument and
        requests a JSON list of hardware control primitives.  Parsing failures
        are surfaced as warnings rather than hard errors so the user can still
        add primitives manually.
        """
        system_prompt = (
            "You are a laboratory automation integration specialist.\n"
            "Given a lab instrument, you propose hardware control primitives for OTbot.\n"
            "Return ONLY valid JSON (no markdown, no prose):\n"
            '{"primitives": [{'
            '"name": "string", '
            '"description": "string", '
            '"params": {"param_name": {"type": "number|integer|boolean|string|array", '
            '"description": "string", "optional": false}}, '
            '"hazardous": false, '
            '"generates_data": false, '
            '"timeout_seconds": 30, '
            '"retries": 1, '
            '"preconditions": [], '
            '"effects": []}]}\n'
            'param types: "number"|"integer"|"boolean"|"string"|"array"\n'
            "Propose 3-10 meaningful primitives."
        )

        name_line = input_data.instrument_name
        if input_data.manufacturer or input_data.model:
            name_line += f" ({input_data.manufacturer} {input_data.model})".rstrip()

        user_message = (
            f"Instrument: {name_line}\n"
            f"SDK package (if known): {input_data.sdk_package or 'not provided'}\n"
            f"Documentation: {input_data.docs_url or 'not provided'}\n\n"
            "List all meaningful hardware control primitives for OTbot integration."
        )

        discovered: list[dict[str, Any]] = []
        warnings: list[str] = []

        try:
            llm = get_llm_provider()
            response = await llm.complete(
                messages=[{"role": "user", "content": user_message}],
                system=system_prompt,
            )
            raw = response.content.strip()
            # Strip markdown code fences if the LLM wraps output anyway
            if raw.startswith("```"):
                raw = "\n".join(
                    line for line in raw.splitlines() if not line.startswith("```")
                ).strip()
            data = json.loads(raw)
            discovered = data.get("primitives", [])
            if not discovered:
                warnings.append(
                    "LLM returned an empty primitives list — "
                    "try adding more detail to instrument_name / model."
                )
        except json.JSONDecodeError as exc:
            logger.warning("discover: LLM returned non-JSON: %s", exc)
            warnings.append(f"Could not parse LLM response as JSON: {exc}")
        except Exception as exc:
            logger.warning("discover: LLM call failed: %s", exc)
            warnings.append(f"LLM discovery failed: {exc}")

        primitive_count = len(discovered)
        display = (
            f"{input_data.manufacturer} {input_data.model}".strip()
            or input_data.instrument_name
        )
        chat_message = f"Discovered **{primitive_count}** primitive(s) for **{display}**.\n\n"
        if discovered:
            chat_message += "Primitives found:\n" + "\n".join(
                f"- `{p.get('name', '?')}` — {p.get('description', '')}"
                for p in discovered
            )
            chat_message += (
                "\n\nReview the list above, then proceed to the **generate** phase "
                "to create full integration code."
            )
        else:
            chat_message += (
                "No primitives were discovered. "
                "Please add primitives manually or provide more instrument details."
            )

        if warnings:
            chat_message += "\n\n" + "\n".join(f"⚠️ {w}" for w in warnings)

        return OnboardingOutput(
            status="discovered",
            instrument_name=input_data.instrument_name,
            display_name=display,
            chat_message=chat_message,
            warnings=warnings,
            serialised_result={"discovered_primitives": discovered},
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_spec(self, input_data: OnboardingInput) -> InstrumentSpec:
        """Convert OnboardingInput → InstrumentSpec."""
        primitives = []
        for p in input_data.primitives:
            params = {}
            for pname, pdef in p.params.items():
                params[pname] = ParamInput(
                    type=pdef.get("type", "number"),
                    description=pdef.get("description", ""),
                    default=pdef.get("default"),
                    optional=pdef.get("optional", False),
                )
            primitives.append(PrimitiveInput(
                name=p.name,
                description=p.description,
                params=params,
                hazardous=p.hazardous,
                generates_data=p.generates_data,
                timeout_seconds=p.timeout_seconds,
                retries=p.retries,
                preconditions=p.preconditions,
                effects=p.effects,
            ))

        # Map communication string to enum
        try:
            comm = CommunicationType(input_data.communication.lower())
        except ValueError:
            comm = CommunicationType.USB

        return InstrumentSpec(
            name=input_data.instrument_name,
            manufacturer=input_data.manufacturer,
            model=input_data.model,
            communication=comm,
            description=input_data.description,
            primitives=primitives,
            sdk_package=input_data.sdk_package,
        )

    def _result_to_output(self, result: OnboardingResult) -> OnboardingOutput:
        """Convert OnboardingResult → OnboardingOutput."""
        pending = [c for c in result.pending_confirmations if not c.confirmed]

        if pending:
            status = "needs_confirmation"
            chat_message = self._service.format_confirmations_for_chat(result)
        elif result.ready_to_write:
            status = "ready_to_write"
            chat_message = (
                f"✅ All confirmations approved for **{result.spec.display_name}**.\n\n"
                f"I've generated {len(result.files)} files. "
                f"Shall I write them to disk?\n\n"
                + "\n".join(
                    f"- `{gf.path}` — {gf.description}"
                    for gf in result.files
                )
            )
        else:
            status = "needs_confirmation"
            chat_message = self._service.format_confirmations_for_chat(result)

        return OnboardingOutput(
            status=status,
            instrument_name=result.spec.name,
            display_name=result.spec.display_name,
            pending_confirmations=[
                ConfirmationItemOut(
                    id=c.id,
                    type=c.type.value,
                    primitive_name=c.primitive_name,
                    question=c.question,
                    current_value=c.current_value,
                    options=c.options,
                    confirmed=c.confirmed,
                )
                for c in result.pending_confirmations
            ],
            confirmed_count=result.confirmed_count,
            total_confirmations=result.total_confirmations,
            chat_message=chat_message,
            files=[
                GeneratedFileOut(
                    path=gf.path, is_patch=gf.is_patch, description=gf.description,
                )
                for gf in result.files
            ],
            manual_todo=result.manual_todo,
            warnings=result.warnings,
            serialised_result=self._serialise_result(result),
        )

    # ------------------------------------------------------------------ #
    # Serialisation (for multi-turn state)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _serialise_result(result: OnboardingResult) -> dict[str, Any]:
        """Serialise OnboardingResult for passing between agent turns."""
        return {
            "spec": {
                "name": result.spec.name,
                "manufacturer": result.spec.manufacturer,
                "model": result.spec.model,
                "communication": result.spec.communication.value,
                "description": result.spec.description,
                "sdk_package": result.spec.sdk_package,
                "resource_id": result.spec.resource_id,
                "primitives": [
                    {
                        "name": p.name,
                        "description": p.description,
                        "params": {
                            pname: {
                                "type": pdef.type,
                                "description": pdef.description,
                                "default": pdef.default,
                                "optional": pdef.optional,
                            }
                            for pname, pdef in p.params.items()
                        },
                        "hazardous": p.hazardous,
                        "generates_data": p.generates_data,
                        "error_class": p.error_class,
                        "safety_class": p.safety_class,
                        "timeout_seconds": p.timeout_seconds,
                        "retries": p.retries,
                        "preconditions": p.preconditions,
                        "effects": p.effects,
                    }
                    for p in result.spec.primitives
                ],
            },
            "confirmations": [
                {
                    "id": c.id,
                    "type": c.type.value,
                    "primitive_name": c.primitive_name,
                    "question": c.question,
                    "current_value": c.current_value,
                    "options": c.options,
                    "confirmed": c.confirmed,
                    "confirmed_value": c.confirmed_value,
                }
                for c in result.pending_confirmations
            ],
            "warnings": result.warnings,
            "manual_todo": result.manual_todo,
            "files": [
                {
                    "path": gf.path,
                    "content": gf.content,
                    "is_patch": gf.is_patch,
                    "patch_marker": gf.patch_marker,
                    "description": gf.description,
                }
                for gf in result.files
            ],
        }

    @staticmethod
    def _deserialise_result(data: dict[str, Any]) -> OnboardingResult:
        """Reconstruct OnboardingResult from serialised dict."""
        from app.services.instrument_onboarding import (
            ConfirmationType,
            GeneratedFile,
        )

        spec_data = data.get("spec", {})
        primitives = []
        for p in spec_data.get("primitives", []):
            params = {}
            for pname, pdef in p.get("params", {}).items():
                params[pname] = ParamInput(
                    type=pdef.get("type", "number"),
                    description=pdef.get("description", ""),
                    default=pdef.get("default"),
                    optional=pdef.get("optional", False),
                )
            primitives.append(PrimitiveInput(
                name=p["name"],
                description=p.get("description", ""),
                params=params,
                hazardous=p.get("hazardous", False),
                generates_data=p.get("generates_data", False),
                error_class=p.get("error_class", ""),
                safety_class=p.get("safety_class", ""),
                timeout_seconds=p.get("timeout_seconds", 30),
                retries=p.get("retries", 1),
                preconditions=p.get("preconditions", []),
                effects=p.get("effects", []),
            ))

        try:
            comm = CommunicationType(spec_data.get("communication", "usb"))
        except ValueError:
            comm = CommunicationType.USB

        spec = InstrumentSpec(
            name=spec_data.get("name", "unknown"),
            manufacturer=spec_data.get("manufacturer", ""),
            model=spec_data.get("model", ""),
            communication=comm,
            description=spec_data.get("description", ""),
            primitives=primitives,
            sdk_package=spec_data.get("sdk_package", ""),
            resource_id=spec_data.get("resource_id", ""),
        )

        confirmations = []
        for c in data.get("confirmations", []):
            confirmations.append(ConfirmationItem(
                id=c["id"],
                type=ConfirmationType(c["type"]),
                primitive_name=c["primitive_name"],
                question=c["question"],
                current_value=c["current_value"],
                options=c.get("options"),
                confirmed=c.get("confirmed", False),
                confirmed_value=c.get("confirmed_value"),
            ))

        files = []
        for f in data.get("files", []):
            files.append(GeneratedFile(
                path=f["path"],
                content=f["content"],
                is_patch=f.get("is_patch", False),
                patch_marker=f.get("patch_marker", ""),
                description=f.get("description", ""),
            ))

        return OnboardingResult(
            spec=spec,
            files=files,
            pending_confirmations=confirmations,
            warnings=data.get("warnings", []),
            manual_todo=data.get("manual_todo", []),
        )
