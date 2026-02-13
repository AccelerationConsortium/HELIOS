"""Tool Holder Configuration — structured position definitions for electrode/tool holders.

Shared data model used by both the conversational dialog agent and
the PDF blueprint reader agent.  Produces offset maps consumed by
the ActionDispatcher for ``robot.move_to_well`` calls.

Persistence: JSON files in ``data/tool_holders/``.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "ToolPosition",
    "ToolHolderConfig",
    "save_tool_holder_config",
    "load_tool_holder_config",
    "list_tool_holder_configs",
]

_DEFAULT_CONFIG_DIR = "data/tool_holders"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ToolPosition(BaseModel):
    """A single named position on a tool holder."""

    name: str = Field(..., description="Human-readable position name, e.g. 'counter_electrode_2e'")
    description: str = ""
    well_name: str = Field(
        default="A1",
        description="OT-2 well reference on the labware, e.g. 'A1'",
    )
    offset_x: float = Field(default=0.0, description="X offset from well center in mm")
    offset_y: float = Field(default=0.0, description="Y offset from well center in mm")
    offset_z: float = Field(default=0.0, description="Z offset from well top in mm")
    tool_type: str = Field(
        default="",
        description="Tool category: counter_electrode | reference_electrode | flush_nozzle | custom",
    )
    quadrant: str = Field(
        default="",
        description="Spatial hint: upper-left | upper-right | lower-left | lower-right",
    )


class ToolHolderConfig(BaseModel):
    """Complete tool holder configuration for one deck slot."""

    holder_name: str = Field(..., description="Unique name for this holder config")
    slot_number: int | str = Field(..., description="Deck slot identifier (OT-2: 1-11, Flex: A1-D3)")
    labware_name: str = Field(
        default="custom_tool_holder",
        description="OT-2 labware load name used for referencing in protocols",
    )
    positions: list[ToolPosition] = Field(default_factory=list)
    holder_dimensions: dict[str, float] = Field(
        default_factory=dict,
        description="Physical dimensions in mm: x_total, y_total, z_height",
    )
    created_by: str = Field(
        default="manual",
        description="Creation method: dialog | pdf_reader | manual",
    )

    def get_position(self, name: str) -> ToolPosition | None:
        """Look up a position by its name."""
        for p in self.positions:
            if p.name == name:
                return p
        return None

    def get_position_by_type(self, tool_type: str) -> list[ToolPosition]:
        """Find all positions matching a tool type."""
        return [p for p in self.positions if p.tool_type == tool_type]

    def to_offset_map(self) -> dict[str, dict[str, Any]]:
        """Convert positions to an offset map for the dispatcher.

        Returns a dict keyed by position name with values containing
        the well reference and x/y/z offsets.
        """
        return {
            p.name: {
                "well": p.well_name,
                "offset_x": p.offset_x,
                "offset_y": p.offset_y,
                "offset_z": p.offset_z,
            }
            for p in self.positions
        }

    def position_names(self) -> list[str]:
        """Return all position names."""
        return [p.name for p in self.positions]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _ensure_config_dir(config_dir: str | None = None) -> Path:
    """Ensure the config directory exists and return its Path."""
    d = Path(config_dir or _DEFAULT_CONFIG_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_tool_holder_config(
    config: ToolHolderConfig,
    path: str | None = None,
    config_dir: str | None = None,
) -> str:
    """Save a ToolHolderConfig to a JSON file.

    Parameters
    ----------
    config:
        The configuration to save.
    path:
        Explicit file path.  If not provided, uses
        ``{config_dir}/{holder_name}.json``.
    config_dir:
        Directory for auto-generated paths.

    Returns
    -------
    str
        The path where the file was written.
    """
    if path is None:
        d = _ensure_config_dir(config_dir)
        safe_name = config.holder_name.replace(" ", "_").lower()
        path = str(d / f"{safe_name}.json")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    logger.info("Saved tool holder config '%s' to %s", config.holder_name, path)
    return path


def load_tool_holder_config(path: str) -> ToolHolderConfig:
    """Load a ToolHolderConfig from a JSON file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ToolHolderConfig(**data)


def list_tool_holder_configs(config_dir: str | None = None) -> list[dict[str, str]]:
    """List all saved tool holder configs.

    Returns a list of dicts with ``name`` and ``path`` keys.
    """
    d = _ensure_config_dir(config_dir)
    results: list[dict[str, str]] = []
    for fp in sorted(d.glob("*.json")):
        try:
            cfg = load_tool_holder_config(str(fp))
            results.append({"name": cfg.holder_name, "path": str(fp)})
        except Exception:
            logger.debug("Skipping invalid config file: %s", fp, exc_info=True)
    return results
