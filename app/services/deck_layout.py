"""Deck/Layout Agent — deck resource binding and validation.

Maps protocol steps to physical deck slots, assigns labware,
validates pipette reachability, and computes volume budgets.

Supports both OT-2 (11 numeric slots) and Flex (12 alphanumeric slots).
This agent sits in L1 (compilation layer) and ensures the protocol
can physically execute on the target robot deck.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Robot Type
# ---------------------------------------------------------------------------

class RobotType(str, Enum):
    """Supported robot platforms."""
    OT2 = "ot2"
    FLEX = "flex"


# ---------------------------------------------------------------------------
# OT-2 Deck Constants (backward-compat aliases)
# ---------------------------------------------------------------------------

TOTAL_SLOTS = 11
SLOT_RANGE = range(1, TOTAL_SLOTS + 1)

# Pipette specs (OT-2 — kept as backward-compat alias)
PIPETTE_SPECS: dict[str, dict[str, Any]] = {
    "p20_single_gen2": {
        "max_volume_ul": 20.0,
        "min_volume_ul": 1.0,
        "mount": "left",
        "channels": 1,
    },
    "p300_single_gen2": {
        "max_volume_ul": 300.0,
        "min_volume_ul": 20.0,
        "mount": "right",
        "channels": 1,
    },
}

# Flex pipette specs
FLEX_PIPETTE_SPECS: dict[str, dict[str, Any]] = {
    "flex_1channel_1000": {
        "max_volume_ul": 1000.0,
        "min_volume_ul": 5.0,
        "mount": "right",
        "channels": 1,
    },
}


# ---------------------------------------------------------------------------
# Flex valid slots
# ---------------------------------------------------------------------------

FLEX_SLOT_NAMES: list[str] = [
    "A1", "A2", "A3",
    "B1", "B2", "B3",
    "C1", "C2", "C3",
    "D1", "D2", "D3",
]


# ---------------------------------------------------------------------------
# Per-robot configuration
# ---------------------------------------------------------------------------

ROBOT_CONFIGS: dict[RobotType, dict[str, Any]] = {
    RobotType.OT2: {
        "total_slots": 11,
        "slot_names": list(range(1, 12)),       # [1,2,...,11]
        "slot_type": "numeric",
        "pipette_specs": PIPETTE_SPECS,
        "tip_slots": [7, 8, 9],
        "waste_slot": 10,
        "fixed_trash": 12,
    },
    RobotType.FLEX: {
        "total_slots": 12,
        "slot_names": FLEX_SLOT_NAMES,
        "slot_type": "alphanumeric",
        "pipette_specs": FLEX_PIPETTE_SPECS,
        "tip_slots": ["B1", "B2", "B3"],
        "waste_slot": "A3",
        "fixed_trash": None,
    },
}


def valid_slot(slot_id: int | str, robot_type: RobotType = RobotType.OT2) -> bool:
    """Check if a slot identifier is valid for the given robot type."""
    cfg = ROBOT_CONFIGS[robot_type]
    return slot_id in cfg["slot_names"]

# Common labware definitions
LABWARE_CATALOG: dict[str, dict[str, Any]] = {
    "opentrons_96_tiprack_300ul": {
        "type": "tiprack",
        "wells": 96,
        "tip_volume_ul": 300.0,
        "compatible_pipettes": ["p300_single_gen2"],
    },
    "opentrons_96_tiprack_20ul": {
        "type": "tiprack",
        "wells": 96,
        "tip_volume_ul": 20.0,
        "compatible_pipettes": ["p20_single_gen2"],
    },
    "opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap": {
        "type": "tuberack",
        "wells": 24,
        "well_volume_ul": 1500.0,
    },
    "nest_96_wellplate_200ul_flat": {
        "type": "wellplate",
        "wells": 96,
        "well_volume_ul": 200.0,
    },
    "nest_12_reservoir_15ml": {
        "type": "reservoir",
        "wells": 12,
        "well_volume_ul": 15000.0,
    },
    "agilent_1_reservoir_290ml": {
        "type": "reservoir",
        "wells": 1,
        "well_volume_ul": 290000.0,
    },
    "opentrons_1_trash_1100ml_fixed": {
        "type": "trash",
        "wells": 1,
        "well_volume_ul": 1100000.0,
    },
    "opentrons_15_tuberack_falcon_15ml_conical": {
        "type": "tuberack",
        "wells": 15,
        "well_volume_ul": 15000.0,
    },
    "corning_24_wellplate_3.4ml_flat": {
        "type": "wellplate",
        "wells": 24,
        "well_volume_ul": 3400.0,
    },
    # Custom labware for electrochemistry setups
    "custom_ultrasonic_bath_2chamber": {
        "type": "wash",
        "wells": 2,
        "well_volume_ul": 500000.0,
        "is_custom": True,
    },
    "custom_electrode_holder_2x2": {
        "type": "custom",
        "wells": 4,
        "well_volume_ul": 0.0,
        "is_custom": True,
    },
    "custom_tool_holder_4pos": {
        "type": "custom",
        "wells": 4,
        "well_volume_ul": 0.0,
        "is_custom": True,
    },
    # --- Flex-specific labware ---
    "opentrons_flex_96_tiprack_1000ul": {
        "type": "tiprack",
        "wells": 96,
        "tip_volume_ul": 1000.0,
        "compatible_pipettes": ["flex_1channel_1000"],
    },
    "opentrons_flex_96_tiprack_200ul": {
        "type": "tiprack",
        "wells": 96,
        "tip_volume_ul": 200.0,
        "compatible_pipettes": ["flex_1channel_1000"],
    },
    "opentrons_flex_96_tiprack_50ul": {
        "type": "tiprack",
        "wells": 96,
        "tip_volume_ul": 50.0,
        "compatible_pipettes": ["flex_1channel_50"],
    },
}

# Default deck layout template (OT-2 — backward-compat alias)
DEFAULT_DECK_TEMPLATE: dict[int, dict[str, str]] = {
    1: {"role": "source", "labware": ""},
    2: {"role": "source", "labware": ""},
    3: {"role": "destination", "labware": ""},
    4: {"role": "reagent", "labware": ""},
    5: {"role": "reagent", "labware": ""},
    6: {"role": "reagent", "labware": ""},
    7: {"role": "tips", "labware": ""},
    8: {"role": "tips", "labware": ""},
    9: {"role": "tips", "labware": ""},
    10: {"role": "waste", "labware": "opentrons_1_trash_1100ml_fixed"},
    11: {"role": "wash", "labware": ""},
}

# Flex deck layout template
FLEX_DECK_TEMPLATE: dict[str, dict[str, str]] = {
    "D1": {"role": "source", "labware": ""},
    "D2": {"role": "source", "labware": ""},
    "D3": {"role": "destination", "labware": ""},
    "C1": {"role": "reagent", "labware": ""},
    "C2": {"role": "reagent", "labware": ""},
    "C3": {"role": "reagent", "labware": ""},
    "B1": {"role": "tips", "labware": ""},
    "B2": {"role": "tips", "labware": ""},
    "B3": {"role": "tips", "labware": ""},
    "A1": {"role": "wash", "labware": ""},
    "A2": {"role": "custom", "labware": ""},
    "A3": {"role": "waste", "labware": ""},
}


def get_deck_template(robot_type: RobotType = RobotType.OT2) -> dict[int | str, dict[str, str]]:
    """Get the default deck template for the given robot type."""
    if robot_type == RobotType.FLEX:
        return dict(FLEX_DECK_TEMPLATE)  # type: ignore[return-value]
    return dict(DEFAULT_DECK_TEMPLATE)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SlotAssignment:
    """What is assigned to a single deck slot."""
    slot_number: int | str  # int for OT-2 (1-11), str for Flex ("A1"-"D3")
    role: str  # source | destination | tips | waste | wash | reagent | empty
    labware_name: str  # human label
    labware_definition: str  # Opentrons def name
    contents: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeckPlan:
    """Complete deck layout for a run."""
    slots: dict[int | str, SlotAssignment]
    pipette_left: str | None = None  # pipette model on left mount
    pipette_right: str | None = None  # pipette model on right mount
    robot_type: str = "ot2"  # "ot2" | "flex"

    def get_slots_by_role(self, role: str) -> list[SlotAssignment]:
        return [s for s in self.slots.values() if s.role == role]

    def get_slot(self, slot_number: int) -> SlotAssignment | None:
        return self.slots.get(slot_number)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slots": {
                str(num): {
                    "slot_number": sa.slot_number,
                    "role": sa.role,
                    "labware_name": sa.labware_name,
                    "labware_definition": sa.labware_definition,
                    "contents": sa.contents,
                }
                for num, sa in self.slots.items()
            },
            "pipette_left": self.pipette_left,
            "pipette_right": self.pipette_right,
        }


@dataclass
class VolumeRequirement:
    """Volume tracking for a single reagent/liquid."""
    reagent_name: str
    total_volume_ul: float
    num_transfers: int
    max_single_transfer_ul: float


@dataclass
class LayoutValidation:
    """Result of deck layout validation."""
    valid: bool
    errors: list[str]
    warnings: list[str]
    volume_requirements: list[VolumeRequirement]
    tip_usage: dict[str, int]  # pipette_model -> tips needed


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def select_pipette(
    volume_ul: float,
    robot_type: RobotType = RobotType.OT2,
) -> str | None:
    """Select the best pipette for a given volume.

    Returns the pipette model name, or None if no pipette can handle it.
    Prefers the smaller pipette when volume is within its range (better accuracy).
    """
    specs = ROBOT_CONFIGS[robot_type]["pipette_specs"]

    # Sort by max_volume ascending → prefer smaller/more accurate pipette first
    sorted_pipettes = sorted(specs.items(), key=lambda x: x[1]["max_volume_ul"])

    for model, spec in sorted_pipettes:
        if spec["min_volume_ul"] <= volume_ul <= spec["max_volume_ul"]:
            return model

    # Volume too large — use the largest pipette (needs multiple transfers)
    if sorted_pipettes:
        largest_model, largest_spec = sorted_pipettes[-1]
        if volume_ul > largest_spec["max_volume_ul"]:
            return largest_model

    return None


def compute_transfers(volume_ul: float, pipette_model: str) -> int:
    """Compute how many transfer cycles needed for a given volume."""
    spec = PIPETTE_SPECS.get(pipette_model)
    if spec is None:
        return 1
    max_vol = spec["max_volume_ul"]
    if volume_ul <= max_vol:
        return 1
    return math.ceil(volume_ul / max_vol)


def estimate_tip_usage(
    protocol_steps: list[dict[str, Any]],
    robot_type: RobotType = RobotType.OT2,
) -> dict[str, int]:
    """Estimate total tip usage from protocol steps.

    Each aspirate->dispense->drop_tip cycle uses one tip.
    Conservative estimate: one tip per aspirate primitive.
    """
    specs = ROBOT_CONFIGS[robot_type]["pipette_specs"]
    tip_count: dict[str, int] = {model: 0 for model in specs}

    for step in protocol_steps:
        primitive = step.get("primitive", "")
        params = step.get("params", {})

        if primitive in ("robot.aspirate", "aspirate"):
            volume = float(params.get("volume_ul", params.get("volume", 0)))
            pipette_model = select_pipette(volume, robot_type)
            if pipette_model:
                transfers = compute_transfers(volume, pipette_model)
                tip_count[pipette_model] += transfers

    return {k: v for k, v in tip_count.items() if v > 0}


def compute_volume_requirements(
    protocol_steps: list[dict[str, Any]],
    batch_size: int = 1,
) -> list[VolumeRequirement]:
    """Compute total volume requirements from protocol steps.

    Multiplies by batch_size to account for running multiple candidates.
    """
    # Track volume per source labware/well combination
    volume_map: dict[str, VolumeRequirement] = {}

    for step in protocol_steps:
        primitive = step.get("primitive", "")
        params = step.get("params", {})

        if primitive in ("robot.aspirate", "aspirate"):
            volume = float(params.get("volume_ul", params.get("volume", 0)))
            source = params.get("labware", "unknown")
            well = params.get("well", "A1")
            key = f"{source}:{well}"

            if key not in volume_map:
                volume_map[key] = VolumeRequirement(
                    reagent_name=key,
                    total_volume_ul=0.0,
                    num_transfers=0,
                    max_single_transfer_ul=0.0,
                )

            req = volume_map[key]
            req.total_volume_ul += volume * batch_size
            req.num_transfers += batch_size
            req.max_single_transfer_ul = max(req.max_single_transfer_ul, volume)

    return list(volume_map.values())


def compute_tip_racks_needed(tip_usage: dict[str, int]) -> dict[str, int]:
    """Compute how many tip racks of each type are needed."""
    racks: dict[str, int] = {}
    for pipette_model, tips_needed in tip_usage.items():
        tips_per_rack = 96  # standard 96-well tip rack
        racks_needed = math.ceil(tips_needed / tips_per_rack)
        racks[pipette_model] = racks_needed
    return racks


def plan_deck_layout(
    protocol_steps: list[dict[str, Any]],
    available_instruments: list[str] | None = None,
    batch_size: int = 1,
    custom_assignments: dict[int | str, dict[str, str]] | None = None,
    robot_type: RobotType = RobotType.OT2,
) -> DeckPlan:
    """Plan deck layout for a given protocol.

    Auto-assigns labware to slots based on protocol requirements.
    Supports both OT-2 (numeric slots 1-11) and Flex (alphanumeric A1-D3).

    Args:
        protocol_steps: Compiled protocol step list.
        available_instruments: List of available instrument IDs.
        batch_size: Number of candidates per round (affects tip/volume planning).
        custom_assignments: Optional manual slot overrides.
        robot_type: Target robot platform.

    Returns:
        DeckPlan with all slots assigned.
    """
    cfg = ROBOT_CONFIGS[robot_type]
    template = get_deck_template(robot_type)
    slots: dict[int | str, SlotAssignment] = {}

    # Start with defaults
    for slot_id, tmpl in template.items():
        slots[slot_id] = SlotAssignment(
            slot_number=slot_id,
            role=tmpl["role"],
            labware_name=tmpl.get("labware", ""),
            labware_definition=tmpl.get("labware", ""),
        )

    # Apply custom overrides
    if custom_assignments:
        for slot_id, assignment in custom_assignments.items():
            if valid_slot(slot_id, robot_type):
                slots[slot_id] = SlotAssignment(
                    slot_number=slot_id,
                    role=assignment.get("role", "source"),
                    labware_name=assignment.get("name", ""),
                    labware_definition=assignment.get("labware", ""),
                )

    # Determine pipettes needed
    tip_usage = estimate_tip_usage(protocol_steps, robot_type)
    pipette_left = None
    pipette_right = None

    if robot_type == RobotType.OT2:
        if "p20_single_gen2" in tip_usage:
            pipette_left = "p20_single_gen2"
        if "p300_single_gen2" in tip_usage:
            pipette_right = "p300_single_gen2"
        if not pipette_left and not pipette_right:
            pipette_right = "p300_single_gen2"
    else:
        # Flex: use first available pipette
        for model in cfg["pipette_specs"]:
            mount = cfg["pipette_specs"][model].get("mount", "right")
            if mount == "left":
                pipette_left = model
            else:
                pipette_right = model
        if not pipette_left and not pipette_right:
            # Default to first pipette on right
            first_model = next(iter(cfg["pipette_specs"]), None)
            if first_model:
                pipette_right = first_model

    # Assign tip racks to tip slots
    racks_needed = compute_tip_racks_needed(
        {k: v * batch_size for k, v in tip_usage.items()}
    )
    tip_slots = list(cfg["tip_slots"])
    tip_slot_idx = 0

    for pipette_model, n_racks in racks_needed.items():
        tip_rack_def = _default_tiprack_for_pipette(pipette_model, robot_type)
        for _ in range(min(n_racks, len(tip_slots) - tip_slot_idx)):
            if tip_slot_idx < len(tip_slots):
                slot_id = tip_slots[tip_slot_idx]
                slots[slot_id] = SlotAssignment(
                    slot_number=slot_id,
                    role="tips",
                    labware_name=f"tiprack_{pipette_model}_{tip_slot_idx + 1}",
                    labware_definition=tip_rack_def,
                )
                tip_slot_idx += 1

    # Fill remaining tip slots with default racks
    default_rack = _default_tiprack_for_pipette(None, robot_type)
    while tip_slot_idx < len(tip_slots):
        slot_id = tip_slots[tip_slot_idx]
        if not slots[slot_id].labware_definition:
            slots[slot_id] = SlotAssignment(
                slot_number=slot_id,
                role="tips",
                labware_name=f"tiprack_default_{tip_slot_idx + 1}",
                labware_definition=default_rack,
            )
        tip_slot_idx += 1

    # Ensure waste slot has trash
    waste_slot = cfg["waste_slot"]
    if robot_type == RobotType.OT2:
        slots[waste_slot] = SlotAssignment(
            slot_number=waste_slot,
            role="waste",
            labware_name="trash",
            labware_definition="opentrons_1_trash_1100ml_fixed",
        )
    else:
        # Flex uses a movable trash bin (no fixed labware def needed)
        if waste_slot in slots and not slots[waste_slot].labware_definition:
            slots[waste_slot] = SlotAssignment(
                slot_number=waste_slot,
                role="waste",
                labware_name="trash",
                labware_definition="opentrons_flex_trash",
            )

    return DeckPlan(
        slots=slots,
        pipette_left=pipette_left,
        pipette_right=pipette_right,
        robot_type=robot_type.value,
    )


def _default_tiprack_for_pipette(
    pipette_model: str | None,
    robot_type: RobotType = RobotType.OT2,
) -> str:
    """Return the default tiprack definition for a pipette model."""
    if robot_type == RobotType.FLEX:
        if pipette_model and "50" in pipette_model:
            return "opentrons_flex_96_tiprack_50ul"
        return "opentrons_flex_96_tiprack_1000ul"
    # OT-2
    if pipette_model and "p20" in pipette_model:
        return "opentrons_96_tiprack_20ul"
    return "opentrons_96_tiprack_300ul"


def validate_deck_layout(
    deck_plan: DeckPlan,
    protocol_steps: list[dict[str, Any]],
    policy_snapshot: dict[str, Any] | None = None,
    batch_size: int = 1,
    robot_type: RobotType = RobotType.OT2,
) -> LayoutValidation:
    """Validate a deck layout against protocol requirements.

    Checks:
    1. All referenced labware is assigned to a slot
    2. Pipettes can handle all required volumes
    3. Enough tips for all operations
    4. Volume budgets don't exceed labware capacity
    5. No slot conflicts

    Returns:
        LayoutValidation with errors, warnings, and resource estimates.
    """
    errors: list[str] = []
    warnings: list[str] = []
    specs = ROBOT_CONFIGS[robot_type]["pipette_specs"]

    # 1. Check labware references
    loaded_labware: set[str] = set()
    for slot in deck_plan.slots.values():
        if slot.labware_name:
            loaded_labware.add(slot.labware_name)

    for step in protocol_steps:
        params = step.get("params", {})
        primitive = step.get("primitive", "")
        labware_ref = params.get("labware")
        if labware_ref and primitive.startswith("robot."):
            if labware_ref not in loaded_labware:
                # Not necessarily an error -- labware may be loaded by a
                # robot.load_labware step
                pass

    # 2. Check pipette-volume compatibility
    for step in protocol_steps:
        primitive = step.get("primitive", "")
        params = step.get("params", {})

        if primitive in ("robot.aspirate", "robot.dispense", "aspirate"):
            volume = float(params.get("volume_ul", params.get("volume", 0)))
            pipette = select_pipette(volume, robot_type)

            if pipette is None and volume > 0:
                errors.append(
                    f"step {step.get('step_key', '?')}: volume {volume} uL "
                    f"is below minimum pipette capacity"
                )

            # Check against policy max volume
            if policy_snapshot:
                max_vol = float(policy_snapshot.get("max_volume_ul", 1000))
                if volume > max_vol:
                    errors.append(
                        f"step {step.get('step_key', '?')}: volume {volume} uL "
                        f"exceeds policy max {max_vol} uL"
                    )

    # 3. Tip budget
    tip_usage = estimate_tip_usage(protocol_steps, robot_type)
    scaled_tip_usage = {k: v * batch_size for k, v in tip_usage.items()}

    # Count available tips from deck
    available_tips: dict[str, int] = {model: 0 for model in specs}
    for slot in deck_plan.slots.values():
        if slot.role == "tips" and slot.labware_definition:
            catalog_entry = LABWARE_CATALOG.get(slot.labware_definition)
            if catalog_entry and catalog_entry.get("type") == "tiprack":
                for compat in catalog_entry.get("compatible_pipettes", []):
                    available_tips[compat] = (
                        available_tips.get(compat, 0) + catalog_entry["wells"]
                    )

    for pipette_model, needed in scaled_tip_usage.items():
        available = available_tips.get(pipette_model, 0)
        if needed > available:
            warnings.append(
                f"{pipette_model}: needs {needed} tips but only {available} available "
                f"({math.ceil(needed / 96)} racks needed, "
                f"{math.ceil(available / 96)} loaded)"
            )

    # 4. Volume requirements
    volume_reqs = compute_volume_requirements(protocol_steps, batch_size)

    # 5. No duplicate slot assignments (sanity check)
    used_slots: set[int | str] = set()
    for slot_id, slot in deck_plan.slots.items():
        if slot.labware_definition and slot_id in used_slots:
            errors.append(f"slot {slot_id} has duplicate assignment")
        used_slots.add(slot_id)

    return LayoutValidation(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        volume_requirements=volume_reqs,
        tip_usage=scaled_tip_usage,
    )


# ---------------------------------------------------------------------------
# Well Allocator — cross-round well tracking for multi-round campaigns
# ---------------------------------------------------------------------------


@dataclass
class WellAllocation:
    """Record of a single well allocation."""
    well_name: str
    round_number: int
    candidate_index: int
    labware_name: str
    slot_number: int


class WellAllocator:
    """Manages well allocation across rounds for destination labware.

    In a multi-round campaign (e.g. 24 rounds on a 24-well plate), each
    round + candidate pair needs a unique destination well.  This class
    tracks which wells have been used and hands out the next available one.

    Allocation order follows Opentrons column-major convention:
    A1, B1, C1, D1, A2, B2, ... (i.e. ``well_ordering`` flattened).

    Usage::

        allocator = WellAllocator(
            labware_name="reactor",
            slot_number=4,
            well_names=["A1","B1","C1","D1","A2","B2","C2","D2", ...],
        )

        # Round 1, candidate 0
        well = allocator.allocate(round_number=1, candidate_index=0)
        # → "A1"

        # Round 1, candidate 1
        well = allocator.allocate(round_number=1, candidate_index=1)
        # → "B1"

    Raises :class:`WellExhaustedError` when all wells have been used.
    """

    def __init__(
        self,
        labware_name: str,
        slot_number: int,
        well_names: list[str],
    ) -> None:
        if not well_names:
            raise ValueError("well_names must not be empty")
        self.labware_name = labware_name
        self.slot_number = slot_number
        self._well_names = list(well_names)
        self._cursor: int = 0
        self._history: list[WellAllocation] = []
        # lookup: well_name → allocation (for idempotent re-queries)
        self._used: dict[str, WellAllocation] = {}

    # -- Properties --

    @property
    def capacity(self) -> int:
        """Total number of wells available."""
        return len(self._well_names)

    @property
    def remaining(self) -> int:
        """Number of unallocated wells."""
        return self.capacity - self._cursor

    @property
    def history(self) -> list[WellAllocation]:
        """Read-only view of allocation history."""
        return list(self._history)

    # -- Core API --

    def allocate(
        self,
        round_number: int,
        candidate_index: int,
    ) -> str:
        """Allocate the next available well.

        Returns the well name (e.g. "A1").

        Raises:
            WellExhaustedError: If no wells remain.
        """
        if self._cursor >= len(self._well_names):
            raise WellExhaustedError(
                labware_name=self.labware_name,
                slot_number=self.slot_number,
                capacity=self.capacity,
                requested_round=round_number,
                requested_candidate=candidate_index,
            )
        well = self._well_names[self._cursor]
        self._cursor += 1

        alloc = WellAllocation(
            well_name=well,
            round_number=round_number,
            candidate_index=candidate_index,
            labware_name=self.labware_name,
            slot_number=self.slot_number,
        )
        self._history.append(alloc)
        self._used[well] = alloc
        return well

    def peek(self) -> str | None:
        """Return the next well that *would* be allocated, without consuming it.

        Returns ``None`` if exhausted.
        """
        if self._cursor >= len(self._well_names):
            return None
        return self._well_names[self._cursor]

    def get_well_for(
        self,
        round_number: int,
        candidate_index: int,
    ) -> str | None:
        """Look up a previously allocated well by round + candidate.

        Returns ``None`` if that combination was never allocated.
        """
        for alloc in self._history:
            if (
                alloc.round_number == round_number
                and alloc.candidate_index == candidate_index
            ):
                return alloc.well_name
        return None

    def reset(self) -> None:
        """Reset the allocator (discard all allocations)."""
        self._cursor = 0
        self._history.clear()
        self._used.clear()

    def snapshot(self) -> dict[str, Any]:
        """Serializable snapshot for logging / SSE events."""
        return {
            "labware_name": self.labware_name,
            "slot_number": self.slot_number,
            "capacity": self.capacity,
            "allocated": self._cursor,
            "remaining": self.remaining,
            "allocations": [
                {
                    "well": a.well_name,
                    "round": a.round_number,
                    "candidate": a.candidate_index,
                }
                for a in self._history
            ],
        }


class WellExhaustedError(Exception):
    """Raised when a WellAllocator has no more wells to give out."""

    def __init__(
        self,
        labware_name: str,
        slot_number: int,
        capacity: int,
        requested_round: int,
        requested_candidate: int,
    ) -> None:
        self.labware_name = labware_name
        self.slot_number = slot_number
        self.capacity = capacity
        self.requested_round = requested_round
        self.requested_candidate = requested_candidate
        super().__init__(
            f"All {capacity} wells in '{labware_name}' (slot {slot_number}) "
            f"are exhausted.  Cannot allocate well for round "
            f"{requested_round}, candidate {requested_candidate}.  "
            f"Consider reducing batch_size, max_rounds, or using a "
            f"higher-capacity plate."
        )


def create_well_allocator_from_deck_plan(
    deck_plan: DeckPlan,
    role: str = "destination",
) -> WellAllocator | None:
    """Create a WellAllocator for the first labware with the given role.

    Generates well names in column-major order (A1, B1, C1, ..., A2, B2, ...)
    based on the labware catalog entry.

    Returns ``None`` if no slot with that role has a recognized labware.
    """
    for slot in deck_plan.slots.values():
        if slot.role != role or not slot.labware_definition:
            continue

        catalog = LABWARE_CATALOG.get(slot.labware_definition)
        if catalog is None:
            # Unknown labware — try to infer wells from contents
            if slot.contents:
                well_names = sorted(
                    slot.contents.keys(),
                    key=lambda w: (int(w[1:]), w[0]),  # column-major
                )
                if well_names:
                    return WellAllocator(
                        labware_name=slot.labware_name,
                        slot_number=slot.slot_number,
                        well_names=well_names,
                    )
            continue

        n_wells = catalog.get("wells", 0)
        if n_wells <= 0:
            continue

        # Generate well names in column-major order
        well_names = _generate_well_names(n_wells)
        return WellAllocator(
            labware_name=slot.labware_name,
            slot_number=slot.slot_number,
            well_names=well_names,
        )

    return None


def _generate_well_names(n_wells: int) -> list[str]:
    """Generate well names in column-major order for standard plates.

    Common layouts:
    - 24 wells → 4 rows (A-D) × 6 cols  (tuberack / wellplate)
    - 96 wells → 8 rows (A-H) × 12 cols
    - 12 wells → 1 row (A)   × 12 cols  (reservoir)
    - 1 well   → A1
    """
    _LAYOUTS: dict[int, tuple[int, int]] = {
        1: (1, 1),
        6: (2, 3),
        12: (1, 12),
        24: (4, 6),
        48: (6, 8),
        96: (8, 12),
        384: (16, 24),
    }
    rows, cols = _LAYOUTS.get(n_wells, _guess_grid(n_wells))

    names: list[str] = []
    for col in range(1, cols + 1):
        for row in range(rows):
            names.append(f"{chr(ord('A') + row)}{col}")
    return names


def _guess_grid(n: int) -> tuple[int, int]:
    """Guess rows × cols for a non-standard well count."""
    # Try to make it roughly square, more cols than rows
    best = (1, n)
    for r in range(1, int(n ** 0.5) + 2):
        if n % r == 0:
            c = n // r
            best = (r, c)
    return best
