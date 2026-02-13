"""Deck Parser — natural language deck description to structured DeckPlan.

Parses user-provided deck layout descriptions (as found in case.md-style
requirements) into structured ``DeckPlan`` configurations that the
``plan_deck_layout()`` function can consume.

Uses pure Python regex/keyword matching — no LLM in the critical path.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "DeckSlotDescription",
    "DeckDescription",
    "DeckParserResult",
    "parse_deck_description",
    "build_deck_plan_from_description",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class DeckSlotDescription(BaseModel):
    """Parsed description of a single slot from NL input."""

    slot_number: int | str  # int for OT-2 (1-11), str for Flex ("A1"-"D3")
    description: str = ""
    role: str = ""  # inferred: "source" | "destination" | "tips" | "waste" | "reagent" | "wash" | "custom"
    labware_type: str = ""  # inferred type key
    labware_name: str = ""  # user-given or auto-generated name
    labware_definition: str = ""  # Opentrons definition name (if resolved)
    contents: dict[str, Any] = Field(default_factory=dict)
    is_custom: bool = False
    custom_params: dict[str, Any] = Field(default_factory=dict)


class DeckDescription(BaseModel):
    """Full parsed deck from NL input."""

    slots: list[DeckSlotDescription] = Field(default_factory=list)
    pipettes: dict[str, str] = Field(default_factory=dict)  # mount -> model
    notes: list[str] = Field(default_factory=list)


class DeckParserResult(BaseModel):
    """Output of the deck parser middleware."""

    deck_description: DeckDescription
    custom_assignments: dict[int | str, dict[str, str]] = Field(default_factory=dict)
    custom_labware_definitions: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    needs_confirmation: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Labware alias mapping
# ---------------------------------------------------------------------------

# Maps common NL phrases to (role, labware_definition, is_custom)
_LABWARE_ALIASES: dict[str, tuple[str, str, bool]] = {
    # Tube racks
    "centrifuge tube": ("source", "opentrons_15_tuberack_falcon_15ml_conical", False),
    "15 ml tube": ("source", "opentrons_15_tuberack_falcon_15ml_conical", False),
    "tube rack": ("source", "opentrons_15_tuberack_falcon_15ml_conical", False),
    "tuberack": ("source", "opentrons_15_tuberack_falcon_15ml_conical", False),
    "stock solution": ("source", "opentrons_15_tuberack_falcon_15ml_conical", False),
    # Tip racks
    "tip rack": ("tips", "opentrons_96_tiprack_300ul", False),
    "tiprack": ("tips", "opentrons_96_tiprack_300ul", False),
    "tip box": ("tips", "opentrons_96_tiprack_300ul", False),
    "p20 tip": ("tips", "opentrons_96_tiprack_20ul", False),
    "p300 tip": ("tips", "opentrons_96_tiprack_300ul", False),
    # Well plates / reactors
    "24-well": ("destination", "corning_24_wellplate_3.4ml_flat", False),
    "24 well": ("destination", "corning_24_wellplate_3.4ml_flat", False),
    "reactor": ("destination", "corning_24_wellplate_3.4ml_flat", False),
    "96-well": ("destination", "nest_96_wellplate_200ul_flat", False),
    "96 well": ("destination", "nest_96_wellplate_200ul_flat", False),
    "wellplate": ("destination", "nest_96_wellplate_200ul_flat", False),
    # Reservoirs
    "reservoir": ("reagent", "nest_12_reservoir_15ml", False),
    "trough": ("reagent", "nest_12_reservoir_15ml", False),
    # Trash
    "trash": ("waste", "opentrons_1_trash_1100ml_fixed", False),
    "waste": ("waste", "opentrons_1_trash_1100ml_fixed", False),
    # Custom / non-standard
    "ultrasonic bath": ("wash", "custom_ultrasonic_bath_2chamber", True),
    "ultrasonic": ("wash", "custom_ultrasonic_bath_2chamber", True),
    "cleaning station": ("wash", "custom_ultrasonic_bath_2chamber", True),
    "electrode holder": ("custom", "custom_electrode_holder_2x2", True),
    "tool holder": ("custom", "custom_tool_holder_4pos", True),
    "tool rack": ("custom", "custom_tool_holder_4pos", True),
    "electrode/tool holder": ("custom", "custom_tool_holder_4pos", True),
    # pH measurement labware
    "ph unit": ("custom", "phunit", True),
    "ph sensor": ("custom", "phunit", True),
    "ph strip": ("custom", "phunit", True),
    "ph meter": ("custom", "phunit", True),
    "phunit": ("custom", "phunit", True),
    # Flex-specific labware
    "flex tip rack": ("tips", "opentrons_flex_96_tiprack_1000ul", False),
    "flex tiprack": ("tips", "opentrons_flex_96_tiprack_1000ul", False),
    "flex filter tip": ("tips", "opentrons_flex_96_filtertiprack_1000ul", False),
    "1000ul filter tip": ("tips", "opentrons_flex_96_filtertiprack_1000ul", False),
    "50ul filter tip": ("tips", "opentrons_flex_96_filtertiprack_50ul", False),
    "6 well vial": ("source", "20mlvial_6_wellplate", True),
    "6-vial rack": ("source", "20mlvial_6_wellplate", True),
    "20ml vial": ("source", "20mlvial_6_wellplate", True),
    "24 well vial": ("destination", "al24wellplate_24_wellplate_15000ul", True),
    "24-vial plate": ("destination", "al24wellplate_24_wellplate_15000ul", True),
    "corning 96": ("destination", "corning_96_wellplate_360ul_flat", False),
    "96 flat plate": ("destination", "corning_96_wellplate_360ul_flat", False),
}

# Regex for slot extraction:
#   OT-2:  "Slot 5:" or "• Slot 5:"
#   Flex:  "Slot A2:" or "A2:" or "• A2:"
_SLOT_PATTERN = re.compile(
    r"(?:^|\n)\s*[•\-\*]?\s*(?:Slot\s+)?([A-D][1-3]|\d{1,2})\s*:\s*(.*?)(?=\n\s*[•\-\*]?\s*(?:Slot\s+)?(?:[A-D][1-3]|\d)|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# Valid Flex slot names for validation
_FLEX_VALID_SLOTS = {
    "A1", "A2", "A3", "B1", "B2", "B3",
    "C1", "C2", "C3", "D1", "D2", "D3",
}

# Content extraction patterns
_CONTENT_COUNT_PATTERN = re.compile(
    r"(\d+)\s+(?:stock\s+)?solutions?", re.IGNORECASE
)
_CONTENT_ITEMS_PATTERN = re.compile(
    r"(?:holding|with|contains?)\s+(.*?)(?:\.|$)", re.IGNORECASE
)
_CHAMBER_PATTERN = re.compile(
    r"(left|right|upper|lower)\s+chamber\s*=\s*(.*?)(?:;|\.|$)", re.IGNORECASE
)
_POSITION_PATTERN = re.compile(
    r"\((upper-left|lower-left|upper-right|lower-right)\)\s+(.*?)(?:\n|$)", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------------


def _match_labware(text: str) -> tuple[str, str, bool] | None:
    """Match text against labware aliases. Returns (role, definition, is_custom) or None."""
    text_lower = text.lower()
    best_match: tuple[str, str, bool] | None = None
    best_len = 0

    for alias, info in _LABWARE_ALIASES.items():
        if alias in text_lower and len(alias) > best_len:
            best_match = info
            best_len = len(alias)

    return best_match


def _extract_contents(text: str) -> dict[str, Any]:
    """Extract content descriptions from slot text."""
    contents: dict[str, Any] = {}

    # Check for chamber descriptions (ultrasonic bath)
    chambers = _CHAMBER_PATTERN.findall(text)
    if chambers:
        for side, desc in chambers:
            contents[f"{side.lower()}_chamber"] = desc.strip()
        return contents

    # Check for position descriptions (electrode holder)
    positions = _POSITION_PATTERN.findall(text)
    if positions:
        for quadrant, desc in positions:
            contents[quadrant.lower()] = desc.strip()
        return contents

    # Check for count-based contents
    count_match = _CONTENT_COUNT_PATTERN.search(text)
    if count_match:
        contents["count"] = int(count_match.group(1))

    # Check for item descriptions
    items_match = _CONTENT_ITEMS_PATTERN.search(text)
    if items_match:
        items_text = items_match.group(1)
        items = [item.strip() for item in re.split(r"[,+&]|and", items_text) if item.strip()]
        if items:
            contents["items"] = items

    return contents


def parse_deck_description(
    text: str,
    robot_type: str = "ot2",
) -> DeckDescription:
    """Parse a natural language deck description into structured form.

    Handles text like::

        OT-2: Slot 1: rack holding twelve 15 mL centrifuge tubes
        Flex: A1: ultrasonic bath; left chamber = water, right chamber = acid

    Parameters
    ----------
    text:
        Natural language deck description.
    robot_type:
        "ot2" or "flex". Controls slot ID format and validation range.

    Returns
    -------
    DeckDescription
        Parsed slot descriptions, pipette info, and notes.
    """
    slots: list[DeckSlotDescription] = []
    notes: list[str] = []

    # Extract slot descriptions
    matches = _SLOT_PATTERN.findall(text)

    if not matches:
        fmt = "'A1: description'" if robot_type == "flex" else "'Slot N: description'"
        notes.append(f"No slot descriptions found in text. Expected format: {fmt}")
        return DeckDescription(slots=slots, notes=notes)

    for slot_id_str, desc_text in matches:
        desc_text = desc_text.strip()

        # Parse and validate slot ID based on robot type
        if robot_type == "flex":
            slot_id: int | str = slot_id_str.upper()
            if slot_id not in _FLEX_VALID_SLOTS:
                notes.append(f"Slot {slot_id} is outside Flex range (A1-D3), skipping")
                continue
        else:
            try:
                slot_id = int(slot_id_str)
            except ValueError:
                notes.append(f"Slot '{slot_id_str}' is not a valid OT-2 slot number, skipping")
                continue
            if slot_id < 1 or slot_id > 11:
                notes.append(f"Slot {slot_id} is outside OT-2 range (1-11), skipping")
                continue

        # Match labware type
        labware_match = _match_labware(desc_text)
        if labware_match is None:
            notes.append(f"Slot {slot_id}: could not identify labware type from '{desc_text[:50]}...'")
            role = "custom"
            labware_def = ""
            is_custom = True
        else:
            role, labware_def, is_custom = labware_match

        # Extract contents
        contents = _extract_contents(desc_text)

        # Generate labware name
        labware_name = f"slot_{slot_id}_{role}"

        # Extract custom params for non-standard labware
        custom_params: dict[str, Any] = {}
        if is_custom:
            if "positions" in str(contents):
                custom_params["n_positions"] = len(contents)
            # Check for "N positions" or "N-well"
            well_match = re.search(r"(\d+)[- ](?:well|position)", desc_text, re.IGNORECASE)
            if well_match:
                custom_params["n_wells"] = int(well_match.group(1))

        slots.append(DeckSlotDescription(
            slot_number=slot_id,
            description=desc_text,
            role=role,
            labware_type=labware_def.split("_")[0] if labware_def else "custom",
            labware_name=labware_name,
            labware_definition=labware_def,
            contents=contents,
            is_custom=is_custom,
            custom_params=custom_params,
        ))

    return DeckDescription(slots=slots, notes=notes)


def build_deck_plan_from_description(
    desc: DeckDescription,
) -> DeckParserResult:
    """Convert a parsed DeckDescription into a DeckParserResult.

    Produces ``custom_assignments`` suitable for ``plan_deck_layout()``,
    custom labware definitions for non-standard items, and a list of
    items needing user confirmation.

    Parameters
    ----------
    desc:
        Parsed deck description.

    Returns
    -------
    DeckParserResult
    """
    custom_assignments: dict[int | str, dict[str, str]] = {}
    custom_labware_defs: list[dict[str, Any]] = []
    warnings: list[str] = list(desc.notes)
    needs_confirmation: list[dict[str, Any]] = []

    for slot in desc.slots:
        assignment: dict[str, str] = {
            "role": slot.role,
            "labware_name": slot.labware_name,
        }

        if slot.labware_definition:
            assignment["labware_definition"] = slot.labware_definition

        if slot.is_custom:
            # Create a custom labware definition placeholder
            custom_def = {
                "load_name": slot.labware_definition or f"custom_slot_{slot.slot_number}",
                "display_name": slot.labware_name,
                "slot_number": slot.slot_number,
                "contents": slot.contents,
                "custom_params": slot.custom_params,
                "source_description": slot.description,
            }
            custom_labware_defs.append(custom_def)

            # Flag for confirmation
            needs_confirmation.append({
                "slot_number": slot.slot_number,
                "question": (
                    f"Slot {slot.slot_number} uses custom labware "
                    f"({slot.description[:60]}). Please confirm or provide "
                    f"the labware JSON definition."
                ),
                "current_value": slot.labware_definition,
                "field": "labware_definition",
            })

        custom_assignments[slot.slot_number] = assignment

    return DeckParserResult(
        deck_description=desc,
        custom_assignments=custom_assignments,
        custom_labware_definitions=custom_labware_defs,
        warnings=warnings,
        needs_confirmation=needs_confirmation,
    )
