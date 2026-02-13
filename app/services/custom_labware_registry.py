"""Custom Labware Registry — register and persist non-standard labware definitions.

Bridges the NeverOT main app with the Opentrons custom labware system.
Stores JSON labware definitions to ``data/custom_labware/`` and provides
lookup by load name.

Custom labware (ultrasonic baths, electrode holders, etc.) that don't
exist in the standard Opentrons catalog are registered here and referenced
during deck layout planning and protocol execution.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "register_custom_labware",
    "get_custom_labware_path",
    "get_custom_labware_definition",
    "list_custom_labware",
    "BUILTIN_CUSTOM_LABWARE",
]

_DEFAULT_DIR = "data/custom_labware"

# In-memory index: load_name -> file path
_REGISTRY: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Built-in custom labware templates
# ---------------------------------------------------------------------------

# Minimal labware definitions for common non-standard items.
# These follow the Opentrons custom labware JSON schema (simplified).

BUILTIN_CUSTOM_LABWARE: dict[str, dict[str, Any]] = {
    "custom_ultrasonic_bath_2chamber": {
        "ordering": [["A1", "A2"]],
        "brand": {"brand": "Custom", "brandId": []},
        "metadata": {
            "displayName": "Ultrasonic Bath 2-Chamber",
            "displayCategory": "wellPlate",
            "displayVolumeUnits": "mL",
            "tags": ["ultrasonic", "cleaning"],
        },
        "dimensions": {
            "xDimension": 127.76,
            "yDimension": 85.48,
            "zDimension": 50.0,
        },
        "wells": {
            "A1": {
                "depth": 40.0,
                "totalLiquidVolume": 50000,
                "shape": "rectangular",
                "xDimension": 55.0,
                "yDimension": 75.0,
                "x": 30.0,
                "y": 42.74,
                "z": 10.0,
            },
            "A2": {
                "depth": 40.0,
                "totalLiquidVolume": 50000,
                "shape": "rectangular",
                "xDimension": 55.0,
                "yDimension": 75.0,
                "x": 97.76,
                "y": 42.74,
                "z": 10.0,
            },
        },
        "groups": [
            {
                "metadata": {"wellBottomShape": "flat"},
                "wells": ["A1", "A2"],
            }
        ],
        "parameters": {
            "format": "irregular",
            "quirks": [],
            "isTiprack": False,
            "isMagneticModuleCompatible": False,
            "loadName": "custom_ultrasonic_bath_2chamber",
        },
        "namespace": "custom_neverot",
        "version": 1,
        "schemaVersion": 2,
        "cornerOffsetFromSlot": {"x": 0, "y": 0, "z": 0},
    },
    "custom_electrode_holder_2x2": {
        "ordering": [["A1", "B1"], ["A2", "B2"]],
        "brand": {"brand": "Custom", "brandId": []},
        "metadata": {
            "displayName": "Electrode/Tool Holder 2x2",
            "displayCategory": "wellPlate",
            "displayVolumeUnits": "mL",
            "tags": ["electrode", "tool_holder"],
        },
        "dimensions": {
            "xDimension": 127.76,
            "yDimension": 85.48,
            "zDimension": 40.0,
        },
        "wells": {
            "A1": {
                "depth": 30.0,
                "totalLiquidVolume": 0,
                "shape": "circular",
                "diameter": 15.0,
                "x": 32.0,
                "y": 60.0,
                "z": 10.0,
            },
            "B1": {
                "depth": 30.0,
                "totalLiquidVolume": 0,
                "shape": "circular",
                "diameter": 15.0,
                "x": 32.0,
                "y": 25.0,
                "z": 10.0,
            },
            "A2": {
                "depth": 30.0,
                "totalLiquidVolume": 0,
                "shape": "circular",
                "diameter": 15.0,
                "x": 96.0,
                "y": 60.0,
                "z": 10.0,
            },
            "B2": {
                "depth": 30.0,
                "totalLiquidVolume": 0,
                "shape": "circular",
                "diameter": 15.0,
                "x": 96.0,
                "y": 25.0,
                "z": 10.0,
            },
        },
        "groups": [
            {
                "metadata": {"wellBottomShape": "flat"},
                "wells": ["A1", "B1", "A2", "B2"],
            }
        ],
        "parameters": {
            "format": "irregular",
            "quirks": [],
            "isTiprack": False,
            "isMagneticModuleCompatible": False,
            "loadName": "custom_electrode_holder_2x2",
        },
        "namespace": "custom_neverot",
        "version": 1,
        "schemaVersion": 2,
        "cornerOffsetFromSlot": {"x": 0, "y": 0, "z": 0},
    },
    "custom_tool_holder_4pos": {
        "ordering": [["A1", "B1"], ["A2", "B2"]],
        "brand": {"brand": "Custom", "brandId": []},
        "metadata": {
            "displayName": "Tool Holder 4-Position",
            "displayCategory": "wellPlate",
            "displayVolumeUnits": "mL",
            "tags": ["tool_holder", "electrode"],
        },
        "dimensions": {
            "xDimension": 127.76,
            "yDimension": 85.48,
            "zDimension": 60.0,
        },
        "wells": {
            "A1": {
                "depth": 40.0,
                "totalLiquidVolume": 0,
                "shape": "circular",
                "diameter": 20.0,
                "x": 32.0,
                "y": 60.0,
                "z": 20.0,
            },
            "B1": {
                "depth": 40.0,
                "totalLiquidVolume": 0,
                "shape": "circular",
                "diameter": 20.0,
                "x": 32.0,
                "y": 25.0,
                "z": 20.0,
            },
            "A2": {
                "depth": 40.0,
                "totalLiquidVolume": 0,
                "shape": "circular",
                "diameter": 20.0,
                "x": 96.0,
                "y": 60.0,
                "z": 20.0,
            },
            "B2": {
                "depth": 40.0,
                "totalLiquidVolume": 0,
                "shape": "circular",
                "diameter": 20.0,
                "x": 96.0,
                "y": 25.0,
                "z": 20.0,
            },
        },
        "groups": [
            {
                "metadata": {"wellBottomShape": "flat"},
                "wells": ["A1", "B1", "A2", "B2"],
            }
        ],
        "parameters": {
            "format": "irregular",
            "quirks": [],
            "isTiprack": False,
            "isMagneticModuleCompatible": False,
            "loadName": "custom_tool_holder_4pos",
        },
        "namespace": "custom_neverot",
        "version": 1,
        "schemaVersion": 2,
        "cornerOffsetFromSlot": {"x": 0, "y": 0, "z": 0},
    },
}


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def _ensure_dir(config_dir: str | None = None) -> Path:
    d = Path(config_dir or _DEFAULT_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def register_custom_labware(
    definition: dict[str, Any],
    save_path: str | None = None,
    config_dir: str | None = None,
) -> str:
    """Register a custom labware definition.

    Saves the JSON to disk and adds it to the in-memory index.

    Parameters
    ----------
    definition:
        Opentrons-compatible labware definition dict.
    save_path:
        Explicit file path. If not provided, auto-generates from load name.
    config_dir:
        Directory for auto-generated paths.

    Returns
    -------
    str
        The load_name of the registered labware.
    """
    params = definition.get("parameters", {})
    load_name = params.get("loadName", "")
    if not load_name:
        raise ValueError("Labware definition missing 'parameters.loadName'")

    if save_path is None:
        d = _ensure_dir(config_dir)
        save_path = str(d / f"{load_name}.json")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(definition, f, indent=2, ensure_ascii=False)

    _REGISTRY[load_name] = save_path
    logger.info("Registered custom labware '%s' at %s", load_name, save_path)
    return load_name


def get_custom_labware_path(load_name: str) -> str | None:
    """Get the filesystem path for a registered custom labware JSON.

    Returns None if not registered.
    """
    return _REGISTRY.get(load_name)


def get_custom_labware_definition(load_name: str) -> dict[str, Any] | None:
    """Load and return a custom labware definition by load name.

    Checks in-memory builtins first, then disk registry.
    """
    # Check builtins
    if load_name in BUILTIN_CUSTOM_LABWARE:
        return BUILTIN_CUSTOM_LABWARE[load_name]

    # Check disk
    path = _REGISTRY.get(load_name)
    if path is None:
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.debug("Failed to load custom labware '%s' from %s", load_name, path, exc_info=True)
        return None


def list_custom_labware(config_dir: str | None = None) -> list[dict[str, str]]:
    """List all registered custom labware.

    Returns dicts with ``load_name``, ``path``, and ``source`` keys.
    """
    results: list[dict[str, str]] = []

    # Builtins
    for name in BUILTIN_CUSTOM_LABWARE:
        results.append({
            "load_name": name,
            "path": "(builtin)",
            "source": "builtin",
        })

    # Disk
    d = _ensure_dir(config_dir)
    for fp in sorted(d.glob("*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            load_name = data.get("parameters", {}).get("loadName", fp.stem)
            results.append({
                "load_name": load_name,
                "path": str(fp),
                "source": "registered",
            })
        except Exception:
            pass

    return results


def ensure_builtins_registered(config_dir: str | None = None) -> None:
    """Write built-in custom labware to disk if not already present."""
    d = _ensure_dir(config_dir)
    for load_name, definition in BUILTIN_CUSTOM_LABWARE.items():
        fp = d / f"{load_name}.json"
        if not fp.exists():
            register_custom_labware(definition, save_path=str(fp), config_dir=config_dir)
