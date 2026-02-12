"""
Custom Labware Support for OT-2 NLP Agent.

Supports:
1. Loading custom labware from JSON files (Opentrons Labware Creator format)
2. Creating custom labware definitions programmatically
3. Common 3D printed labware templates
4. Validation of user-provided physical dimensions

Opentrons Custom Labware Format:
https://labware.opentrons.com/create
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# OT-2 physical constraints (mm)
# ---------------------------------------------------------------------------

# SBS-format deck slot footprint
OT2_SLOT_X_MM = 127.76
OT2_SLOT_Y_MM = 85.48
OT2_MAX_Z_MM = 120.0  # max height before pipette collision

# Minimum practical dimensions
MIN_WELL_DEPTH_MM = 1.0
MIN_WELL_DIAMETER_MM = 1.0
MIN_SPACING_MM = 1.0   # wells must not overlap


class LabwareValidationError(Exception):
    """Raised when a custom labware definition fails physical validation.

    Carries the full :class:`LabwareValidationResult` so callers can
    inspect individual errors, warnings, and missing fields.
    """
    def __init__(self, result: "LabwareValidationResult") -> None:
        self.result = result
        parts = []
        if result.missing_fields:
            parts.append(
                f"Missing fields: {', '.join(result.missing_fields)}"
            )
        if result.errors:
            parts.append(f"Errors: {'; '.join(result.errors)}")
        super().__init__(
            "Custom labware validation failed. " + " | ".join(parts)
        )


@dataclass
class LabwareValidationResult:
    """Result of labware definition validation."""
    valid: bool
    errors: list[str]          # hard errors — will cause robot collision / failure
    warnings: list[str]        # suspicious but potentially valid
    missing_fields: list[str]  # fields the user still needs to provide

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "missing_fields": self.missing_fields,
        }


@dataclass
class WellDefinition:
    """Definition of a single well."""
    depth: float  # mm
    diameter: Optional[float] = None  # mm, for circular wells
    x_dimension: Optional[float] = None  # mm, for rectangular wells
    y_dimension: Optional[float] = None  # mm, for rectangular wells
    total_liquid_volume: float = 0  # µL
    shape: str = "circular"  # "circular" or "rectangular"


@dataclass
class CustomLabwareDefinition:
    """
    Custom labware definition compatible with Opentrons format.

    This can be exported to JSON for use with the OT-2.
    """
    name: str
    display_name: str
    description: str = ""

    # Labware dimensions
    format: str = "irregular"  # "96Standard", "384Standard", "trough", "irregular"
    brand: str = "custom"
    brand_id: Optional[str] = None

    # Dimensions in mm
    x_dimension: float = 127.76  # Standard SBS footprint
    y_dimension: float = 85.48
    z_dimension: float = 15.0

    # Well configuration
    wells: Dict[str, WellDefinition] = field(default_factory=dict)
    well_ordering: List[List[str]] = field(default_factory=list)

    # Grid parameters (for regular grids)
    rows: int = 0
    columns: int = 0
    row_spacing: float = 9.0  # mm
    column_spacing: float = 9.0  # mm

    # First well offset from corner
    x_offset: float = 14.38  # mm from left edge to A1 center
    y_offset: float = 11.24  # mm from front edge to A1 center

    # Metadata
    namespace: str = "custom_beta"
    version: int = 1
    schema_version: int = 2

    def to_opentrons_json(self) -> Dict[str, Any]:
        """Convert to Opentrons labware JSON format."""
        # Build wells dict
        wells_dict = {}
        for well_name, well_def in self.wells.items():
            well_data = {
                "depth": well_def.depth,
                "totalLiquidVolume": well_def.total_liquid_volume,
                "shape": well_def.shape,
            }
            if well_def.shape == "circular":
                well_data["diameter"] = well_def.diameter
            else:
                well_data["xDimension"] = well_def.x_dimension
                well_data["yDimension"] = well_def.y_dimension

            # Calculate position
            row = ord(well_name[0]) - ord('A')
            col = int(well_name[1:]) - 1
            well_data["x"] = self.x_offset + col * self.column_spacing
            well_data["y"] = self.y_dimension - self.y_offset - row * self.row_spacing
            well_data["z"] = self.z_dimension - well_def.depth

            wells_dict[well_name] = well_data

        return {
            "schemaVersion": self.schema_version,
            "version": self.version,
            "namespace": self.namespace,
            "metadata": {
                "displayName": self.display_name,
                "displayCategory": "wellPlate" if self.rows > 0 else "other",
                "displayVolumeUnits": "µL",
                "tags": ["custom", "3d-printed"]
            },
            "brand": {
                "brand": self.brand,
                "brandId": [self.brand_id] if self.brand_id else []
            },
            "parameters": {
                "format": self.format,
                "quirks": [],
                "isTiprack": False,
                "isMagneticModuleCompatible": False,
                "loadName": self.name
            },
            "ordering": self.well_ordering,
            "cornerOffsetFromSlot": {
                "x": 0,
                "y": 0,
                "z": 0
            },
            "dimensions": {
                "xDimension": self.x_dimension,
                "yDimension": self.y_dimension,
                "zDimension": self.z_dimension
            },
            "wells": wells_dict,
            "groups": [{
                "metadata": {
                    "wellBottomShape": "flat"
                },
                "wells": list(wells_dict.keys())
            }]
        }

    def save_json(self, filepath: str):
        """Save labware definition to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_opentrons_json(), f, indent=2)


def validate_labware_definition(
    labware: CustomLabwareDefinition,
) -> LabwareValidationResult:
    """Validate a custom labware definition for physical feasibility.

    Checks that user-supplied dimensions will not cause robot collisions,
    well overlaps, or unreachable positions.  Returns structured feedback
    telling the user *exactly* which parameters are missing or wrong.

    This MUST be called before any custom labware is sent to the OT-2.
    """
    errors: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []

    # --- Required grid parameters ---
    if labware.rows <= 0:
        missing.append("rows (number of rows, e.g. 2)")
    if labware.columns <= 0:
        missing.append("columns (number of columns, e.g. 6)")

    # --- Labware footprint vs. OT-2 slot ---
    if labware.x_dimension > OT2_SLOT_X_MM:
        errors.append(
            f"x_dimension={labware.x_dimension}mm exceeds OT-2 slot width "
            f"({OT2_SLOT_X_MM}mm).  The labware will not fit on the deck."
        )
    if labware.y_dimension > OT2_SLOT_Y_MM:
        errors.append(
            f"y_dimension={labware.y_dimension}mm exceeds OT-2 slot depth "
            f"({OT2_SLOT_Y_MM}mm).  The labware will not fit on the deck."
        )
    if labware.z_dimension > OT2_MAX_Z_MM:
        errors.append(
            f"z_dimension={labware.z_dimension}mm exceeds OT-2 max height "
            f"({OT2_MAX_Z_MM}mm).  The pipette will collide with the labware."
        )

    # --- Well definitions ---
    if not labware.wells:
        missing.append("wells (well definitions with depth, diameter, volume)")
    else:
        sample_well = next(iter(labware.wells.values()))
        if sample_well.depth < MIN_WELL_DEPTH_MM:
            errors.append(
                f"well depth={sample_well.depth}mm is below minimum "
                f"({MIN_WELL_DEPTH_MM}mm).  Pipette cannot reach liquid."
            )
        if sample_well.shape == "circular":
            if sample_well.diameter is None:
                missing.append(
                    "well_diameter (inner diameter in mm — measure at the "
                    "top of the well/tube)"
                )
            elif sample_well.diameter < MIN_WELL_DIAMETER_MM:
                errors.append(
                    f"well diameter={sample_well.diameter}mm is below minimum "
                    f"({MIN_WELL_DIAMETER_MM}mm)."
                )
        else:
            if sample_well.x_dimension is None:
                missing.append("well_x (well X dimension in mm)")
            if sample_well.y_dimension is None:
                missing.append("well_y (well Y dimension in mm)")

        if sample_well.total_liquid_volume <= 0:
            missing.append(
                "well_volume (maximum liquid volume per well in µL)"
            )

    # --- Spacing: wells must not overlap ---
    if labware.rows > 0 and labware.columns > 0 and labware.wells:
        sample_well = next(iter(labware.wells.values()))
        effective_diameter = (
            sample_well.diameter
            if sample_well.shape == "circular" and sample_well.diameter
            else max(sample_well.x_dimension or 0, sample_well.y_dimension or 0)
        )
        if effective_diameter and effective_diameter > 0:
            if labware.row_spacing < effective_diameter:
                errors.append(
                    f"row_spacing={labware.row_spacing}mm is less than "
                    f"well diameter={effective_diameter}mm — wells will "
                    f"overlap physically."
                )
            if labware.column_spacing < effective_diameter:
                errors.append(
                    f"column_spacing={labware.column_spacing}mm is less than "
                    f"well diameter={effective_diameter}mm — wells will "
                    f"overlap physically."
                )

    # --- Grid fits within footprint ---
    if labware.rows > 0 and labware.columns > 0:
        rightmost_x = labware.x_offset + (labware.columns - 1) * labware.column_spacing
        if rightmost_x > labware.x_dimension:
            errors.append(
                f"Rightmost well at x={rightmost_x:.1f}mm exceeds labware "
                f"x_dimension={labware.x_dimension}mm.  Adjust column_spacing "
                f"or x_offset."
            )
        topmost_y = labware.y_offset + (labware.rows - 1) * labware.row_spacing
        if topmost_y > labware.y_dimension:
            errors.append(
                f"Topmost well at y={topmost_y:.1f}mm exceeds labware "
                f"y_dimension={labware.y_dimension}mm.  Adjust row_spacing "
                f"or y_offset."
            )

    # --- Offset sanity ---
    if labware.x_offset < 0:
        errors.append(f"x_offset={labware.x_offset}mm must be ≥ 0")
    if labware.y_offset < 0:
        errors.append(f"y_offset={labware.y_offset}mm must be ≥ 0")

    # --- Soft warnings ---
    if labware.row_spacing < MIN_SPACING_MM and labware.rows > 1:
        warnings.append(
            f"row_spacing={labware.row_spacing}mm is very small; "
            "pipette may have trouble accessing individual wells."
        )
    if labware.column_spacing < MIN_SPACING_MM and labware.columns > 1:
        warnings.append(
            f"column_spacing={labware.column_spacing}mm is very small; "
            "pipette may have trouble accessing individual wells."
        )

    # Default offset warning
    if (
        labware.x_offset == 14.38
        and labware.y_offset == 11.24
        and labware.rows > 0
    ):
        warnings.append(
            "Using default SBS x/y offsets (14.38, 11.24 mm).  For 3D-printed "
            "custom labware, please measure the actual offset from the slot "
            "corner to the center of well A1."
        )

    return LabwareValidationResult(
        valid=len(errors) == 0 and len(missing) == 0,
        errors=errors,
        warnings=warnings,
        missing_fields=missing,
    )


def format_missing_fields_prompt(result: LabwareValidationResult) -> str:
    """Generate a user-friendly prompt listing missing physical parameters.

    Returns an empty string when there are no issues.
    """
    parts: list[str] = []

    if result.missing_fields:
        parts.append(
            "⚠️  The following physical parameters are required before "
            "the labware can be used on the OT-2.  Please measure your "
            "3D-printed labware and provide:"
        )
        for i, field_desc in enumerate(result.missing_fields, 1):
            parts.append(f"  {i}. {field_desc}")

    if result.errors:
        parts.append("")
        parts.append("❌  The following values will cause physical problems:")
        for err in result.errors:
            parts.append(f"  • {err}")

    if result.warnings:
        parts.append("")
        parts.append("💡  Suggestions:")
        for w in result.warnings:
            parts.append(f"  • {w}")

    return "\n".join(parts)


class CustomLabwareManager:
    """
    Manages custom labware definitions.

    Usage:
        manager = CustomLabwareManager()

        # Load from JSON
        manager.load_from_file("my_custom_plate.json")

        # Create programmatically
        labware = manager.create_grid_labware(
            name="battery_holder_4x3",
            rows=3,
            columns=4,
            well_depth=20,
            well_diameter=18,
            well_volume=5000
        )

        # Use in agent
        agent.add_custom_labware(protocol, labware, slot=1)
    """

    def __init__(self, labware_dir: str = None):
        """
        Initialize the custom labware manager.

        Args:
            labware_dir: Directory to search for custom labware JSON files
        """
        self.labware_dir = labware_dir or os.path.expanduser("~/.ot2-labware")
        self.custom_labware: Dict[str, CustomLabwareDefinition] = {}
        self._load_builtin_templates()

    def _load_builtin_templates(self):
        """Load built-in custom labware templates."""
        # Common 3D printed labware templates
        self.templates = {
            # Battery cell holder templates
            "battery_holder_1x4": self._create_battery_holder(1, 4),
            "battery_holder_2x4": self._create_battery_holder(2, 4),
            "battery_holder_3x4": self._create_battery_holder(3, 4),
            "battery_holder_4x6": self._create_battery_holder(4, 6),

            # Vial/tube holders
            "vial_holder_3x4": self._create_vial_holder(3, 4, 15),
            "vial_holder_4x6": self._create_vial_holder(4, 6, 15),

            # Custom reservoir
            "custom_reservoir_4": self._create_reservoir(4),
            "custom_reservoir_8": self._create_reservoir(8),

            # Electrode holder
            "electrode_holder_1x8": self._create_electrode_holder(1, 8),
        }

    def _create_battery_holder(self, rows: int, cols: int) -> CustomLabwareDefinition:
        """Create a battery cell holder labware."""
        labware = CustomLabwareDefinition(
            name=f"custom_battery_holder_{rows}x{cols}",
            display_name=f"Custom Battery Holder {rows}x{cols}",
            description=f"3D printed holder for {rows*cols} battery cells",
            rows=rows,
            columns=cols,
            z_dimension=25.0,
            row_spacing=20.0,  # 20mm between cells
            column_spacing=20.0,
            x_offset=20.0,
            y_offset=20.0,
        )

        # Generate wells
        well_def = WellDefinition(
            depth=20.0,
            diameter=18.0,  # 18650 battery diameter
            total_liquid_volume=5000,
            shape="circular"
        )

        labware.wells = {}
        labware.well_ordering = []

        for col in range(cols):
            col_wells = []
            for row in range(rows):
                well_name = f"{chr(ord('A') + row)}{col + 1}"
                labware.wells[well_name] = well_def
                col_wells.append(well_name)
            labware.well_ordering.append(col_wells)

        return labware

    def _create_vial_holder(self, rows: int, cols: int, vial_diameter: float) -> CustomLabwareDefinition:
        """Create a vial holder labware."""
        labware = CustomLabwareDefinition(
            name=f"custom_vial_holder_{rows}x{cols}",
            display_name=f"Custom Vial Holder {rows}x{cols}",
            description=f"3D printed holder for {rows*cols} vials",
            rows=rows,
            columns=cols,
            z_dimension=50.0,
            row_spacing=vial_diameter + 2,
            column_spacing=vial_diameter + 2,
        )

        well_def = WellDefinition(
            depth=45.0,
            diameter=vial_diameter,
            total_liquid_volume=15000,  # 15mL
            shape="circular"
        )

        labware.wells = {}
        labware.well_ordering = []

        for col in range(cols):
            col_wells = []
            for row in range(rows):
                well_name = f"{chr(ord('A') + row)}{col + 1}"
                labware.wells[well_name] = well_def
                col_wells.append(well_name)
            labware.well_ordering.append(col_wells)

        return labware

    def _create_reservoir(self, channels: int) -> CustomLabwareDefinition:
        """Create a custom reservoir labware."""
        labware = CustomLabwareDefinition(
            name=f"custom_reservoir_{channels}",
            display_name=f"Custom {channels}-Channel Reservoir",
            description=f"3D printed {channels}-channel reservoir",
            format="trough",
            rows=1,
            columns=channels,
            z_dimension=40.0,
            column_spacing=127.76 / channels,
        )

        well_def = WellDefinition(
            depth=35.0,
            x_dimension=8.0,
            y_dimension=70.0,
            total_liquid_volume=20000,  # 20mL per channel
            shape="rectangular"
        )

        labware.wells = {}
        labware.well_ordering = []

        for col in range(channels):
            well_name = f"A{col + 1}"
            labware.wells[well_name] = well_def
            labware.well_ordering.append([well_name])

        return labware

    def _create_electrode_holder(self, rows: int, cols: int) -> CustomLabwareDefinition:
        """Create an electrode holder labware."""
        labware = CustomLabwareDefinition(
            name=f"custom_electrode_holder_{rows}x{cols}",
            display_name=f"Electrode Holder {rows}x{cols}",
            description=f"3D printed holder for {rows*cols} electrodes",
            rows=rows,
            columns=cols,
            z_dimension=30.0,
            column_spacing=15.0,
        )

        well_def = WellDefinition(
            depth=25.0,
            diameter=10.0,
            total_liquid_volume=2000,
            shape="circular"
        )

        labware.wells = {}
        labware.well_ordering = []

        for col in range(cols):
            col_wells = []
            for row in range(rows):
                well_name = f"{chr(ord('A') + row)}{col + 1}"
                labware.wells[well_name] = well_def
                col_wells.append(well_name)
            labware.well_ordering.append(col_wells)

        return labware

    def get_template(self, name: str) -> Optional[CustomLabwareDefinition]:
        """Get a built-in template by name."""
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        """List available built-in templates."""
        return list(self.templates.keys())

    def load_from_file(self, filepath: str) -> CustomLabwareDefinition:
        """
        Load custom labware from an Opentrons JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            CustomLabwareDefinition object
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        name = data['parameters']['loadName']
        labware = CustomLabwareDefinition(
            name=name,
            display_name=data['metadata']['displayName'],
            x_dimension=data['dimensions']['xDimension'],
            y_dimension=data['dimensions']['yDimension'],
            z_dimension=data['dimensions']['zDimension'],
        )

        # Parse wells
        for well_name, well_data in data['wells'].items():
            well_def = WellDefinition(
                depth=well_data['depth'],
                total_liquid_volume=well_data['totalLiquidVolume'],
                shape=well_data['shape'],
            )
            if well_data['shape'] == 'circular':
                well_def.diameter = well_data.get('diameter')
            else:
                well_def.x_dimension = well_data.get('xDimension')
                well_def.y_dimension = well_data.get('yDimension')

            labware.wells[well_name] = well_def

        labware.well_ordering = data.get('ordering', [])

        self.custom_labware[name] = labware
        return labware

    def create_grid_labware(
        self,
        name: str,
        rows: int,
        columns: int,
        well_depth: float,
        well_diameter: float = None,
        well_x: float = None,
        well_y: float = None,
        well_volume: float = 1000,
        row_spacing: float = 9.0,
        column_spacing: float = 9.0,
        x_offset: float = 14.38,
        y_offset: float = 11.24,
        display_name: str = None,
        description: str = "",
        skip_validation: bool = False,
    ) -> CustomLabwareDefinition:
        """
        Create a custom grid labware programmatically.

        Args:
            name: Unique labware name
            rows: Number of rows (A, B, C...)
            columns: Number of columns (1, 2, 3...)
            well_depth: Well depth in mm
            well_diameter: Well diameter in mm (for circular wells)
            well_x: Well X dimension in mm (for rectangular wells)
            well_y: Well Y dimension in mm (for rectangular wells)
            well_volume: Well volume in µL
            row_spacing: Spacing between rows in mm
            column_spacing: Spacing between columns in mm
            x_offset: mm from left edge of labware to center of A1
            y_offset: mm from front edge of labware to center of A1
            display_name: Human-readable name
            description: Description
            skip_validation: If True, skip physical validation (for tests only)

        Returns:
            CustomLabwareDefinition object

        Raises:
            LabwareValidationError: If physical parameters are invalid or missing.
        """
        shape = "circular" if well_diameter else "rectangular"

        labware = CustomLabwareDefinition(
            name=name,
            display_name=display_name or name,
            description=description,
            rows=rows,
            columns=columns,
            row_spacing=row_spacing,
            column_spacing=column_spacing,
            x_offset=x_offset,
            y_offset=y_offset,
            z_dimension=well_depth + 5,  # Add some clearance
        )

        well_def = WellDefinition(
            depth=well_depth,
            diameter=well_diameter,
            x_dimension=well_x,
            y_dimension=well_y,
            total_liquid_volume=well_volume,
            shape=shape
        )

        labware.wells = {}
        labware.well_ordering = []

        for col in range(columns):
            col_wells = []
            for row in range(rows):
                well_name = f"{chr(ord('A') + row)}{col + 1}"
                labware.wells[well_name] = well_def
                col_wells.append(well_name)
            labware.well_ordering.append(col_wells)

        # Validate before accepting
        if not skip_validation:
            result = validate_labware_definition(labware)
            if not result.valid:
                raise LabwareValidationError(result)

        self.custom_labware[name] = labware
        return labware

    def get_labware(self, name: str) -> Optional[CustomLabwareDefinition]:
        """Get a loaded custom labware by name."""
        return self.custom_labware.get(name) or self.templates.get(name)


# Singleton instance
_labware_manager: Optional[CustomLabwareManager] = None


def get_labware_manager() -> CustomLabwareManager:
    """Get or create the global CustomLabwareManager instance."""
    global _labware_manager
    if _labware_manager is None:
        _labware_manager = CustomLabwareManager()
    return _labware_manager
