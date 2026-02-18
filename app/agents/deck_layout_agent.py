"""Deck Layout Agent — auto-parse NL deck descriptions into structured DeckPlan.

Wraps the ``deck_parser`` service with the BaseAgent interface, supporting
multi-turn confirmation for custom labware slots.

Supports both OT-2 (numeric slots) and Flex (alphanumeric slots).

Lifecycle (multi-turn):
1. **parse**: NL text → DeckParserResult (with possible confirmations)
2. **confirm**: User approves custom labware assignments
3. **finalize**: Produce final DeckPlan dict for the compiler

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


class DeckLayoutInput(BaseModel):
    """Input for the DeckLayout agent."""

    phase: str = Field(
        default="parse",
        description="Phase: parse | confirm | finalize",
    )

    # --- robot type ---
    robot_type: str = Field(
        default="ot2",
        description="Target robot: 'ot2' or 'flex'",
    )

    # --- parse phase ---
    deck_text: str = Field(default="", description="NL deck description text")
    protocol_steps: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Protocol steps for context (used to infer roles if needed)",
    )

    # --- confirm phase ---
    confirmations: dict[str, Any] = Field(
        default_factory=dict,
        description="User confirmations keyed by slot_id (str) → labware_definition",
    )

    # --- carry-over state ---
    previous_result: dict[str, Any] | None = None


class DeckLayoutOutput(BaseModel):
    """Output from the DeckLayout agent."""

    status: str = Field(
        ...,
        description="Status: parsed | needs_confirmation | finalized | error",
    )

    # Parsed result summary
    slot_count: int = 0
    custom_count: int = 0

    # For the orchestrator to build the DeckPlan
    deck_plan: dict[str, Any] = Field(
        default_factory=dict,
        description="Final DeckPlan dict ready for plan_deck_layout()",
    )
    custom_assignments: dict[int | str, dict[str, str]] = Field(
        default_factory=dict,
        description="Slot → assignment mapping",
    )
    custom_labware_definitions: list[dict[str, Any]] = Field(
        default_factory=list,
    )

    # Confirmation flow
    pending_confirmations: list[dict[str, Any]] = Field(default_factory=list)

    # Chat message for user (includes ASCII deck diagram)
    chat_message: str = ""

    # Rich HTML deck visualization (for frontend rendering)
    deck_visualization_html: str = ""

    # Warnings
    warnings: list[str] = Field(default_factory=list)

    # Serialised result for multi-turn state
    serialised_result: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class DeckLayoutAgent(BaseAgent[DeckLayoutInput, DeckLayoutOutput]):
    """Auto-parse NL deck descriptions and produce DeckPlan configs.

    Supports multi-turn confirmation for custom labware.
    Supports both OT-2 and Flex robot types.
    """

    name = "deck_layout"
    description = "NL deck description → structured DeckPlan"
    layer = "L1"

    def validate_input(self, input_data: DeckLayoutInput) -> list[str]:
        errors: list[str] = []
        if input_data.phase == "parse":
            if not input_data.deck_text.strip():
                errors.append("deck_text is required for parse phase")
        elif input_data.phase == "confirm":
            if not input_data.previous_result:
                errors.append("previous_result is required for confirm phase")
        elif input_data.phase == "finalize":
            if not input_data.previous_result:
                errors.append("previous_result is required for finalize phase")
        else:
            errors.append(f"Unknown phase: {input_data.phase}")
        if input_data.robot_type not in ("ot2", "flex"):
            errors.append(f"Unknown robot_type: {input_data.robot_type}")
        return errors

    async def process(self, input_data: DeckLayoutInput) -> DeckLayoutOutput:
        if input_data.phase == "parse":
            return self._handle_parse(input_data)
        elif input_data.phase == "confirm":
            return self._handle_confirm(input_data)
        elif input_data.phase == "finalize":
            return self._handle_finalize(input_data)
        else:
            raise ValueError(f"Unknown phase: {input_data.phase}")

    # ------------------------------------------------------------------ #
    # Phase handlers
    # ------------------------------------------------------------------ #

    def _handle_parse(self, input_data: DeckLayoutInput) -> DeckLayoutOutput:
        from app.services.deck_parser import (
            parse_deck_description,
            build_deck_plan_from_description,
        )

        rt = input_data.robot_type
        desc = parse_deck_description(input_data.deck_text, robot_type=rt)
        result = build_deck_plan_from_description(desc)

        serialised = {
            "robot_type": rt,
            "custom_assignments": {
                str(k): v for k, v in result.custom_assignments.items()
            },
            "custom_labware_definitions": result.custom_labware_definitions,
            "warnings": result.warnings,
            "needs_confirmation": result.needs_confirmation,
            "deck_description": result.deck_description.model_dump(mode="json"),
        }

        # Generate deck visualization (ASCII + HTML)
        from app.services.deck_visualizer import render_deck_ascii, render_deck_html
        deck_diagram = render_deck_ascii(result.custom_assignments, robot_type=rt)
        deck_html = render_deck_html(result.custom_assignments, compact=True, robot_type=rt)

        if result.needs_confirmation:
            # Build chat message with visualization + confirmation requests
            lines = ["I've parsed your deck layout:\n"]
            lines.append("```")
            lines.append(deck_diagram)
            lines.append("```")
            lines.append("\nThe following custom items need confirmation:\n")
            for conf in result.needs_confirmation:
                lines.append(f"- **Slot {conf['slot_number']}**: {conf['question']}")
            lines.append("\nPlease confirm or provide corrected labware definitions.")
            chat_msg = "\n".join(lines)

            return DeckLayoutOutput(
                status="needs_confirmation",
                slot_count=len(desc.slots),
                custom_count=len(result.custom_labware_definitions),
                custom_assignments=result.custom_assignments,
                custom_labware_definitions=result.custom_labware_definitions,
                pending_confirmations=result.needs_confirmation,
                chat_message=chat_msg,
                deck_visualization_html=deck_html,
                warnings=result.warnings,
                serialised_result=serialised,
            )

        # No confirmations needed — auto-finalize
        return self._build_finalized_output(result, serialised, rt)

    def _handle_confirm(self, input_data: DeckLayoutInput) -> DeckLayoutOutput:
        prev = input_data.previous_result or {}
        rt = prev.get("robot_type", input_data.robot_type)
        custom_assignments: dict[int | str, dict[str, str]] = {}

        # Restore assignments — preserve original key types
        for k, v in prev.get("custom_assignments", {}).items():
            if rt == "flex":
                custom_assignments[k] = v  # str keys
            else:
                custom_assignments[int(k)] = v  # int keys

        # Apply user confirmations
        for slot_str, labware_def in input_data.confirmations.items():
            slot_key: int | str = slot_str if rt == "flex" else int(slot_str)
            if slot_key in custom_assignments:
                custom_assignments[slot_key]["labware_definition"] = labware_def

        # Remove confirmed items from pending
        remaining = []
        confirmed_slots = set(input_data.confirmations.keys())
        if rt != "flex":
            confirmed_slots_typed: set[int | str] = {int(k) for k in confirmed_slots}
        else:
            confirmed_slots_typed = set(confirmed_slots)

        for conf in prev.get("needs_confirmation", []):
            if conf["slot_number"] not in confirmed_slots_typed:
                remaining.append(conf)

        serialised = {
            **prev,
            "custom_assignments": {str(k): v for k, v in custom_assignments.items()},
            "needs_confirmation": remaining,
        }

        if remaining:
            return DeckLayoutOutput(
                status="needs_confirmation",
                slot_count=len(prev.get("deck_description", {}).get("slots", [])),
                custom_count=len(prev.get("custom_labware_definitions", [])),
                custom_assignments=custom_assignments,
                pending_confirmations=remaining,
                chat_message="Some slots still need confirmation.",
                warnings=prev.get("warnings", []),
                serialised_result=serialised,
            )

        # All confirmed
        return self._build_finalized_output_from_dict(serialised, rt)

    def _handle_finalize(self, input_data: DeckLayoutInput) -> DeckLayoutOutput:
        prev = input_data.previous_result or {}
        rt = prev.get("robot_type", input_data.robot_type)
        return self._build_finalized_output_from_dict(prev, rt)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_finalized_output(
        self, result: Any, serialised: dict, robot_type: str = "ot2",
    ) -> DeckLayoutOutput:
        from app.services.deck_visualizer import render_deck_ascii, render_deck_html

        deck_plan = self._assignments_to_deck_plan(result.custom_assignments)
        diagram = render_deck_ascii(
            result.custom_assignments, show_custom_markers=False, robot_type=robot_type,
        )
        deck_html = render_deck_html(
            result.custom_assignments, compact=True, show_custom_markers=False, robot_type=robot_type,
        )
        chat_msg = f"Deck layout finalized successfully.\n\n```\n{diagram}\n```"

        return DeckLayoutOutput(
            status="finalized",
            slot_count=len(result.deck_description.slots),
            custom_count=len(result.custom_labware_definitions),
            deck_plan=deck_plan,
            custom_assignments=result.custom_assignments,
            custom_labware_definitions=result.custom_labware_definitions,
            chat_message=chat_msg,
            deck_visualization_html=deck_html,
            warnings=result.warnings,
            serialised_result=serialised,
        )

    def _build_finalized_output_from_dict(
        self, data: dict, robot_type: str = "ot2",
    ) -> DeckLayoutOutput:
        from app.services.deck_visualizer import render_deck_ascii, render_deck_html

        custom_assignments: dict[int | str, dict[str, str]] = {}
        for k, v in data.get("custom_assignments", {}).items():
            if robot_type == "flex":
                custom_assignments[k] = v
            else:
                custom_assignments[int(k)] = v

        deck_plan = self._assignments_to_deck_plan(custom_assignments)
        diagram = render_deck_ascii(
            custom_assignments, show_custom_markers=False, robot_type=robot_type,
        )
        deck_html = render_deck_html(
            custom_assignments, compact=True, show_custom_markers=False, robot_type=robot_type,
        )
        chat_msg = f"Deck layout finalized with confirmed assignments.\n\n```\n{diagram}\n```"

        return DeckLayoutOutput(
            status="finalized",
            slot_count=len(data.get("deck_description", {}).get("slots", [])),
            custom_count=len(data.get("custom_labware_definitions", [])),
            deck_plan=deck_plan,
            custom_assignments=custom_assignments,
            custom_labware_definitions=data.get("custom_labware_definitions", []),
            chat_message=chat_msg,
            deck_visualization_html=deck_html,
            warnings=data.get("warnings", []),
            serialised_result=data,
        )

    @staticmethod
    def _assignments_to_deck_plan(
        assignments: dict[int | str, dict[str, str]],
    ) -> dict[str, Any]:
        """Convert custom assignments into a DeckPlan-style dict."""
        plan: dict[str, Any] = {"slots": {}}
        for slot_id, assignment in assignments.items():
            plan["slots"][str(slot_id)] = {
                "role": assignment.get("role", "custom"),
                "labware_name": assignment.get("labware_name", ""),
                "labware_definition": assignment.get("labware_definition", ""),
            }
        return plan
