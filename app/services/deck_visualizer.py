"""Deck Visualizer — ASCII and HTML renderings of robot deck layouts.

Generates user-friendly visual representations of parsed deck layouts
for confirmation in chat messages and SSE events.

Supports both OT-2 and Flex deck formats.

OT-2 deck physical layout (view from front):

    ┌─────────┬─────────┬─────────┐
    │  Slot 10 │ Slot 11 │ (trash) │
    ├─────────┼─────────┼─────────┤
    │  Slot 7  │  Slot 8 │  Slot 9 │
    ├─────────┼─────────┼─────────┤
    │  Slot 4  │  Slot 5 │  Slot 6 │
    ├─────────┼─────────┼─────────┤
    │  Slot 1  │  Slot 2 │  Slot 3 │
    └─────────┴─────────┴─────────┘

Flex deck physical layout (view from front):

    ┌─────────┬─────────┬─────────┐
    │   A1    │   A2    │   A3    │   ← back
    ├─────────┼─────────┼─────────┤
    │   B1    │   B2    │   B3    │
    ├─────────┼─────────┼─────────┤
    │   C1    │   C2    │   C3    │
    ├─────────┼─────────┼─────────┤
    │   D1    │   D2    │   D3    │   ← front
    └─────────┴─────────┴─────────┘

Slot 12 is the OT-2 fixed trash (not shown in grid).
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Role → display config
# ---------------------------------------------------------------------------

_ROLE_EMOJI: dict[str, str] = {
    "source": "🧪",
    "destination": "🎯",
    "tips": "📌",
    "waste": "🗑️",
    "wash": "🫧",
    "reagent": "💧",
    "custom": "⚙️",
}

_ROLE_SYMBOL: dict[str, str] = {
    "source": "SRC",
    "destination": "DST",
    "tips": "TIP",
    "waste": "WST",
    "wash": "WSH",
    "reagent": "RGT",
    "custom": "CUS",
}

_ROLE_COLOR: dict[str, str] = {
    "source": "#4A90D9",
    "destination": "#E74C3C",
    "tips": "#95A5A6",
    "waste": "#7F8C8D",
    "wash": "#3498DB",
    "reagent": "#2ECC71",
    "custom": "#F39C12",
}

# OT-2 deck grid rows (bottom to top, 3 columns each)
# Last row has only 2 slots (10, 11) + fixed trash at 12
_GRID_ROWS: list[list[int]] = [
    [10, 11, -1],  # -1 = fixed trash placeholder
    [7, 8, 9],
    [4, 5, 6],
    [1, 2, 3],
]

# Flex deck grid rows (back to front, 3 columns each)
_FLEX_GRID_ROWS: list[list[str]] = [
    ["A1", "A2", "A3"],   # back row
    ["B1", "B2", "B3"],
    ["C1", "C2", "C3"],
    ["D1", "D2", "D3"],   # front row
]


# ---------------------------------------------------------------------------
# Slot info extraction
# ---------------------------------------------------------------------------

def _slot_info(
    slot_id: int | str,
    assignments: dict[int | str, dict[str, str]],
) -> dict[str, Any]:
    """Extract display info for a slot."""
    if slot_id == -1:
        return {"role": "waste", "label": "Fixed Trash", "labware": "trash", "is_custom": False}

    entry = assignments.get(slot_id)
    if entry is None:
        return {"role": "", "label": "(empty)", "labware": "", "is_custom": False}

    role = entry.get("role", "")
    labware_name = entry.get("labware_name", "")
    labware_def = entry.get("labware_definition", "")

    # Build a short human-readable label
    label = _short_labware_label(labware_name, labware_def, role)
    is_custom = "custom" in labware_def.lower() if labware_def else False

    return {"role": role, "label": label, "labware": labware_def, "is_custom": is_custom}


def _short_labware_label(name: str, definition: str, role: str) -> str:
    """Generate a short display label from labware info."""
    if not definition and not name:
        return role.capitalize() if role else "(empty)"

    # Use the human-readable name if it's informative
    if name and not name.startswith("slot_"):
        return _truncate(name, 18)

    # Parse definition into readable form
    if definition:
        return _def_to_label(definition)

    return role.capitalize()


def _def_to_label(definition: str) -> str:
    """Convert an Opentrons definition name to a short label."""
    mapping = {
        # OT-2 labware
        "opentrons_96_tiprack_300ul": "96 Tips 300µL",
        "opentrons_96_tiprack_20ul": "96 Tips 20µL",
        "opentrons_15_tuberack_falcon_15ml_conical": "15mL Tubes",
        "opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap": "1.5mL Tubes",
        "nest_96_wellplate_200ul_flat": "96-Well Plate",
        "nest_12_reservoir_15ml": "12-Reservoir",
        "agilent_1_reservoir_290ml": "290mL Reservoir",
        "corning_24_wellplate_3.4ml_flat": "24-Well Plate",
        "opentrons_1_trash_1100ml_fixed": "Trash",
        "custom_ultrasonic_bath_2chamber": "Ultrasonic Bath",
        "custom_electrode_holder_2x2": "Electrode Holder",
        "custom_tool_holder_4pos": "Tool Holder",
        # Flex labware
        "opentrons_flex_96_tiprack_1000ul": "96 Tips 1000µL",
        "opentrons_flex_96_tiprack_200ul": "96 Tips 200µL",
        "opentrons_flex_96_tiprack_50ul": "96 Tips 50µL",
        "opentrons_flex_trash": "Trash Bin",
    }
    if definition in mapping:
        return mapping[definition]
    # Fallback: clean up the definition name
    return _truncate(definition.replace("_", " ").title(), 18)


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _get_grid_rows(robot_type: str = "ot2") -> list[list[int | str]]:
    """Return the grid rows for the given robot type."""
    if robot_type == "flex":
        return _FLEX_GRID_ROWS  # type: ignore[return-value]
    return _GRID_ROWS  # type: ignore[return-value]


def _slot_label_str(slot_id: int | str) -> str:
    """Return a display label for a slot ID."""
    if slot_id == -1:
        return "Slot 12 (Trash)"
    if isinstance(slot_id, int):
        return f"Slot {slot_id}"
    return str(slot_id)  # Flex: "A1", "B2", etc.


# ---------------------------------------------------------------------------
# ASCII rendering
# ---------------------------------------------------------------------------

_CELL_WIDTH = 22
_SEPARATOR = "+" + (("-" * _CELL_WIDTH + "+") * 3)


def render_deck_ascii(
    assignments: dict[int | str, dict[str, str]],
    *,
    show_custom_markers: bool = True,
    robot_type: str = "ot2",
) -> str:
    """Render deck as ASCII art.

    Parameters
    ----------
    assignments:
        Slot ID → assignment dict with keys: role, labware_name,
        labware_definition. Same format as DeckLayoutOutput.custom_assignments.
    show_custom_markers:
        If True, mark custom labware slots with [*].
    robot_type:
        "ot2" or "flex".

    Returns
    -------
    str
        Multi-line ASCII deck diagram.
    """
    grid_rows = _get_grid_rows(robot_type)
    title = "Flex Deck Layout" if robot_type == "flex" else "OT-2 Deck Layout"

    lines: list[str] = []
    lines.append("")
    lines.append(f"  {title}")
    lines.append("  " + "=" * (_CELL_WIDTH * 3 + 4))
    lines.append("  " + _SEPARATOR)

    for row in grid_rows:
        # Line 1: slot labels
        slot_line = "  |"
        for slot_id in row:
            header = _slot_label_str(slot_id)
            cell = f" {header:<{_CELL_WIDTH - 2}} "
            slot_line += cell[:_CELL_WIDTH] + "|"
        lines.append(slot_line)

        # Line 2: role + emoji
        role_line = "  |"
        for slot_id in row:
            info = _slot_info(slot_id, assignments)
            role = info["role"]
            symbol = _ROLE_SYMBOL.get(role, "   ")
            emoji = _ROLE_EMOJI.get(role, "  ")
            marker = " *" if (show_custom_markers and info.get("is_custom")) else "  "
            cell = f" {emoji} {symbol}{marker}           "
            role_line += cell[:_CELL_WIDTH] + "|"
        lines.append(role_line)

        # Line 3: labware label
        label_line = "  |"
        for slot_id in row:
            info = _slot_info(slot_id, assignments)
            label = _truncate(info["label"], _CELL_WIDTH - 2)
            cell = f" {label:<{_CELL_WIDTH - 2}} "
            label_line += cell[:_CELL_WIDTH] + "|"
        lines.append(label_line)

        lines.append("  " + _SEPARATOR)

    if show_custom_markers:
        lines.append("")
        lines.append("  * = custom labware (needs JSON definition)")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def render_deck_html(
    assignments: dict[int | str, dict[str, str]],
    *,
    show_custom_markers: bool = True,
    compact: bool = False,
    robot_type: str = "ot2",
) -> str:
    """Render deck as an HTML table.

    Parameters
    ----------
    assignments:
        Same format as render_deck_ascii.
    show_custom_markers:
        If True, highlight custom labware cells.
    compact:
        If True, use smaller cell size for chat embedding.
    robot_type:
        "ot2" or "flex".

    Returns
    -------
    str
        Self-contained HTML string with inline CSS.
    """
    grid_rows = _get_grid_rows(robot_type)
    cell_w = "120px" if compact else "160px"
    cell_h = "70px" if compact else "90px"
    font_size = "11px" if compact else "13px"

    html_parts: list[str] = []
    html_parts.append(
        f'<div style="font-family:monospace;font-size:{font_size};">'
        f'<table style="border-collapse:collapse;margin:8px 0;">'
    )

    for row in grid_rows:
        html_parts.append("<tr>")
        for slot_id in row:
            info = _slot_info(slot_id, assignments)
            role = info["role"]
            color = _ROLE_COLOR.get(role, "#BDC3C7")
            emoji = _ROLE_EMOJI.get(role, "")
            label = info["label"]
            is_custom = info.get("is_custom", False)

            border_style = f"3px dashed {color}" if is_custom else f"2px solid {color}"
            bg = f"{color}18"  # very light tint
            slot_label = _slot_label_str(slot_id)

            custom_badge = (
                ' <span style="color:#E67E22;font-weight:bold;" title="Custom labware">*</span>'
                if is_custom and show_custom_markers else ""
            )

            html_parts.append(
                f'<td style="'
                f"width:{cell_w};height:{cell_h};"
                f"border:{border_style};"
                f"background:{bg};"
                f"padding:4px 6px;vertical-align:top;"
                f"border-radius:4px;"
                f'">'
                f'<div style="font-size:10px;color:#666;">{slot_label}{custom_badge}</div>'
                f'<div style="font-size:16px;margin:2px 0;">{emoji}</div>'
                f'<div style="font-size:{font_size};font-weight:bold;color:{color};">'
                f"{_truncate(label, 20)}</div>"
                f"</td>"
            )
        html_parts.append("</tr>")

    html_parts.append("</table>")

    if show_custom_markers:
        html_parts.append(
            '<div style="font-size:10px;color:#888;margin-top:4px;">'
            "* = custom labware (needs JSON definition)"
            "</div>"
        )

    html_parts.append("</div>")
    return "".join(html_parts)


# ---------------------------------------------------------------------------
# Combined rendering (for chat_message)
# ---------------------------------------------------------------------------


def render_deck_for_chat(
    assignments: dict[int | str, dict[str, str]],
    *,
    format: str = "ascii",
    robot_type: str = "ot2",
) -> str:
    """Render deck visualization for embedding in chat messages.

    Parameters
    ----------
    assignments:
        Slot ID → assignment dict.
    format:
        "ascii" for text/markdown, "html" for rich rendering.
    robot_type:
        "ot2" or "flex".

    Returns
    -------
    str
        Formatted deck visualization string.
    """
    if format == "html":
        return render_deck_html(assignments, compact=True, robot_type=robot_type)
    return render_deck_ascii(assignments, robot_type=robot_type)
