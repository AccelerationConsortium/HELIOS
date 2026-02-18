"""Tool Holder Dialog Agent — conversational multi-turn position collection.

Guides the user through defining electrode/tool holder positions via chat.
Follows the OnboardingAgent multi-turn pattern (phase-based state machine).

Phases:
1. **start**: Collect holder name, slot number, number of positions
2. **collect_positions**: Collect each position one by one
3. **confirm**: Present summary for user review
4. **calibrate**: Provide physical calibration guidance

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


class ToolHolderDialogInput(BaseModel):
    """Input for the ToolHolderDialog agent."""

    phase: str = Field(
        default="start",
        description="Phase: start | collect_positions | confirm | calibrate",
    )

    # --- start phase ---
    holder_name: str = ""
    slot_number: int = 0
    position_count: int = 0
    labware_name: str = "custom_tool_holder_4pos"

    # --- collect_positions phase ---
    position_data: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Data for one position: {name, quadrant, tool_type, "
            "well_name (optional), offset_x, offset_y, offset_z (optional)}"
        ),
    )

    # --- confirm phase ---
    confirmed: bool = False
    edits: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Edits keyed by position name → field overrides",
    )

    # --- carry-over state ---
    previous_result: dict[str, Any] | None = None


class ToolHolderDialogOutput(BaseModel):
    """Output from the ToolHolderDialog agent."""

    status: str = Field(
        ...,
        description=(
            "Status: collecting | needs_position | awaiting_confirmation | "
            "finalized | calibration_guide | error"
        ),
    )

    # Progress tracking
    collected_count: int = 0
    total_positions: int = 0

    # Resulting config (available after confirm)
    config_path: str = ""

    # Chat message
    chat_message: str = ""

    # Next question for the user
    next_question: str = ""

    # Serialised state
    serialised_result: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Quadrant → well heuristic
# ---------------------------------------------------------------------------

_QUADRANT_WELL_MAP = {
    "upper-left": "A1",
    "upper-right": "A2",
    "lower-left": "B1",
    "lower-right": "B2",
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ToolHolderDialogAgent(BaseAgent[ToolHolderDialogInput, ToolHolderDialogOutput]):
    """Conversational agent for collecting tool holder positions.

    Multi-turn state machine following the OnboardingAgent pattern.
    """

    name = "tool_holder_dialog"
    description = "Conversational tool holder position collection"
    layer = "L1"

    def validate_input(self, input_data: ToolHolderDialogInput) -> list[str]:
        errors: list[str] = []
        if input_data.phase == "start":
            if not input_data.holder_name:
                errors.append("holder_name is required for start phase")
            from app.services.deck_layout import valid_slot, RobotType
            rt = RobotType(getattr(input_data, 'robot_type', 'ot2') or 'ot2')
            if not valid_slot(input_data.slot_number, rt):
                errors.append(f"slot_number {input_data.slot_number} is invalid for {rt.value}")
            if input_data.position_count < 1:
                errors.append("position_count must be >= 1")
        elif input_data.phase in ("collect_positions", "confirm", "calibrate"):
            if not input_data.previous_result:
                errors.append(f"previous_result is required for {input_data.phase} phase")
        else:
            errors.append(f"Unknown phase: {input_data.phase}")
        return errors

    async def process(self, input_data: ToolHolderDialogInput) -> ToolHolderDialogOutput:
        if input_data.phase == "start":
            return self._handle_start(input_data)
        elif input_data.phase == "collect_positions":
            return self._handle_collect(input_data)
        elif input_data.phase == "confirm":
            return self._handle_confirm(input_data)
        elif input_data.phase == "calibrate":
            return self._handle_calibrate(input_data)
        else:
            raise ValueError(f"Unknown phase: {input_data.phase}")

    # ------------------------------------------------------------------ #
    # Phase handlers
    # ------------------------------------------------------------------ #

    def _handle_start(self, input_data: ToolHolderDialogInput) -> ToolHolderDialogOutput:
        state = {
            "holder_name": input_data.holder_name,
            "slot_number": input_data.slot_number,
            "labware_name": input_data.labware_name,
            "position_count": input_data.position_count,
            "positions": [],
        }

        return ToolHolderDialogOutput(
            status="needs_position",
            collected_count=0,
            total_positions=input_data.position_count,
            chat_message=(
                f"Setting up tool holder **{input_data.holder_name}** "
                f"on Slot {input_data.slot_number} with {input_data.position_count} positions.\n\n"
                f"Let's define each position."
            ),
            next_question=self._position_question(1, input_data.position_count),
            serialised_result=state,
        )

    def _handle_collect(self, input_data: ToolHolderDialogInput) -> ToolHolderDialogOutput:
        state = dict(input_data.previous_result or {})
        positions = list(state.get("positions", []))
        total = state.get("position_count", 0)

        # Add the new position
        pos_data = input_data.position_data
        if pos_data:
            # Infer well from quadrant if not explicitly given
            if not pos_data.get("well_name") and pos_data.get("quadrant"):
                pos_data["well_name"] = _QUADRANT_WELL_MAP.get(
                    pos_data["quadrant"], "A1"
                )

            positions.append(pos_data)
            state["positions"] = positions

        collected = len(positions)

        if collected < total:
            return ToolHolderDialogOutput(
                status="needs_position",
                collected_count=collected,
                total_positions=total,
                chat_message=f"Position {collected} recorded.",
                next_question=self._position_question(collected + 1, total),
                serialised_result=state,
            )

        # All positions collected — show summary
        summary_lines = [
            f"All {total} positions collected for **{state['holder_name']}**:\n"
        ]
        for i, p in enumerate(positions, 1):
            name = p.get("name", f"pos_{i}")
            quadrant = p.get("quadrant", "?")
            tool_type = p.get("tool_type", "unspecified")
            well = p.get("well_name", "?")
            summary_lines.append(
                f"  {i}. **{name}** — {tool_type} at well {well} ({quadrant})"
            )
        summary_lines.append("\nDoes this look correct? (confirm / edit)")

        return ToolHolderDialogOutput(
            status="awaiting_confirmation",
            collected_count=collected,
            total_positions=total,
            chat_message="\n".join(summary_lines),
            serialised_result=state,
        )

    def _handle_confirm(self, input_data: ToolHolderDialogInput) -> ToolHolderDialogOutput:
        state = dict(input_data.previous_result or {})
        positions = list(state.get("positions", []))

        # Apply edits if any
        if input_data.edits:
            for pos_name, overrides in input_data.edits.items():
                for p in positions:
                    if p.get("name") == pos_name:
                        p.update(overrides)
            state["positions"] = positions

        if not input_data.confirmed:
            return ToolHolderDialogOutput(
                status="awaiting_confirmation",
                collected_count=len(positions),
                total_positions=state.get("position_count", 0),
                chat_message="Edits applied. Please review and confirm.",
                serialised_result=state,
            )

        # Save the config
        from app.services.tool_holder_config import (
            ToolHolderConfig,
            ToolPosition,
            save_tool_holder_config,
        )

        tool_positions = []
        for p in positions:
            tool_positions.append(ToolPosition(
                name=p.get("name", ""),
                well_name=p.get("well_name", "A1"),
                offset_x=float(p.get("offset_x", 0)),
                offset_y=float(p.get("offset_y", 0)),
                offset_z=float(p.get("offset_z", 0)),
                tool_type=p.get("tool_type", ""),
                quadrant=p.get("quadrant", ""),
                description=p.get("description", ""),
            ))

        config = ToolHolderConfig(
            holder_name=state["holder_name"],
            slot_number=state["slot_number"],
            labware_name=state.get("labware_name", "custom_tool_holder_4pos"),
            positions=tool_positions,
            created_by="dialog",
        )

        saved_path = save_tool_holder_config(config)

        return ToolHolderDialogOutput(
            status="finalized",
            collected_count=len(positions),
            total_positions=state.get("position_count", 0),
            config_path=saved_path,
            chat_message=(
                f"Tool holder config saved to `{saved_path}`.\n\n"
                f"For best accuracy, run a physical calibration to fine-tune offsets."
            ),
            next_question="Would you like calibration guidance?",
            serialised_result=state,
        )

    def _handle_calibrate(self, input_data: ToolHolderDialogInput) -> ToolHolderDialogOutput:
        state = dict(input_data.previous_result or {})
        slot = state.get("slot_number", "?")
        positions = state.get("positions", [])

        guide_lines = [
            f"## Calibration Guide for Slot {slot}\n",
            "1. Home the OT-2 robot (`robot.home()`)",
            f"2. Attach your tool holder to Slot {slot}",
            "3. For each position, jog the pipette to the exact center:",
        ]
        for p in positions:
            name = p.get("name", "?")
            well = p.get("well_name", "?")
            guide_lines.append(
                f"   - **{name}** (well {well}): Jog and record final X/Y/Z offsets"
            )
        guide_lines.extend([
            "4. Update the config file with measured offsets",
            "5. Run a test move to each position to verify",
            "\nTip: Use `opentrons_execute` or the OT App's jog controls.",
        ])

        return ToolHolderDialogOutput(
            status="calibration_guide",
            collected_count=len(positions),
            total_positions=state.get("position_count", 0),
            chat_message="\n".join(guide_lines),
            serialised_result=state,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _position_question(pos_num: int, total: int) -> str:
        return (
            f"**Position {pos_num}/{total}**: Please provide:\n"
            f"- **name**: e.g., 'counter_electrode_2e'\n"
            f"- **quadrant**: upper-left | upper-right | lower-left | lower-right\n"
            f"- **tool_type**: counter_electrode | reference_electrode | flush_nozzle | custom\n"
            f"- **offset_x/y/z** (optional, in mm, defaults to 0)"
        )
