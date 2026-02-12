"""Tests for custom labware validation and parameter prompting."""
import sys
import os

# Ensure ot2-nlp-agent is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ot2-nlp-agent"))

import pytest
from ot2_agent.custom_labware import (
    CustomLabwareDefinition,
    CustomLabwareManager,
    LabwareValidationError,
    LabwareValidationResult,
    WellDefinition,
    format_missing_fields_prompt,
    validate_labware_definition,
)


class TestValidateLabwareDefinition:
    """Test physical validation of custom labware."""

    def _make_valid_labware(self) -> CustomLabwareDefinition:
        """Build a valid 2×6 tube rack definition."""
        labware = CustomLabwareDefinition(
            name="test_tuberack_2x6",
            display_name="Test Tube Rack 2x6",
            rows=2,
            columns=6,
            x_dimension=127.0,
            y_dimension=85.0,
            z_dimension=50.0,
            row_spacing=20.0,
            column_spacing=20.0,
            x_offset=14.0,
            y_offset=12.0,
        )
        well = WellDefinition(
            depth=45.0,
            diameter=15.0,
            total_liquid_volume=1500,
            shape="circular",
        )
        labware.wells = {}
        labware.well_ordering = []
        for col in range(6):
            col_wells = []
            for row in range(2):
                name = f"{chr(ord('A') + row)}{col + 1}"
                labware.wells[name] = well
                col_wells.append(name)
            labware.well_ordering.append(col_wells)
        return labware

    def test_valid_labware_passes(self):
        labware = self._make_valid_labware()
        result = validate_labware_definition(labware)
        assert result.valid
        assert len(result.errors) == 0
        assert len(result.missing_fields) == 0

    def test_oversized_x_dimension(self):
        labware = self._make_valid_labware()
        labware.x_dimension = 200.0  # exceeds OT2_SLOT_X_MM (127.76)
        result = validate_labware_definition(labware)
        assert not result.valid
        assert any("x_dimension" in e for e in result.errors)

    def test_oversized_y_dimension(self):
        labware = self._make_valid_labware()
        labware.y_dimension = 100.0  # exceeds OT2_SLOT_Y_MM (85.48)
        result = validate_labware_definition(labware)
        assert not result.valid
        assert any("y_dimension" in e for e in result.errors)

    def test_oversized_z_dimension(self):
        labware = self._make_valid_labware()
        labware.z_dimension = 150.0  # exceeds OT2_MAX_Z_MM (120)
        result = validate_labware_definition(labware)
        assert not result.valid
        assert any("z_dimension" in e for e in result.errors)

    def test_missing_rows(self):
        labware = self._make_valid_labware()
        labware.rows = 0
        result = validate_labware_definition(labware)
        assert not result.valid
        assert any("rows" in f for f in result.missing_fields)

    def test_missing_columns(self):
        labware = self._make_valid_labware()
        labware.columns = 0
        result = validate_labware_definition(labware)
        assert not result.valid
        assert any("columns" in f for f in result.missing_fields)

    def test_missing_wells(self):
        labware = self._make_valid_labware()
        labware.wells = {}
        result = validate_labware_definition(labware)
        assert not result.valid
        assert any("wells" in f for f in result.missing_fields)

    def test_missing_well_diameter(self):
        labware = self._make_valid_labware()
        # Set diameter to None
        for w in labware.wells.values():
            w.diameter = None
        result = validate_labware_definition(labware)
        assert not result.valid
        assert any("diameter" in f for f in result.missing_fields)

    def test_missing_well_volume(self):
        labware = self._make_valid_labware()
        for w in labware.wells.values():
            w.total_liquid_volume = 0
        result = validate_labware_definition(labware)
        assert not result.valid
        assert any("volume" in f.lower() for f in result.missing_fields)

    def test_well_depth_too_small(self):
        labware = self._make_valid_labware()
        for w in labware.wells.values():
            w.depth = 0.5  # below MIN_WELL_DEPTH_MM (1.0)
        result = validate_labware_definition(labware)
        assert not result.valid
        assert any("depth" in e for e in result.errors)

    def test_wells_overlap_row_spacing(self):
        labware = self._make_valid_labware()
        labware.row_spacing = 10.0  # less than diameter=15mm
        result = validate_labware_definition(labware)
        assert not result.valid
        assert any("overlap" in e for e in result.errors)

    def test_wells_overlap_column_spacing(self):
        labware = self._make_valid_labware()
        labware.column_spacing = 10.0
        result = validate_labware_definition(labware)
        assert not result.valid
        assert any("overlap" in e for e in result.errors)

    def test_wells_exceed_footprint_x(self):
        labware = self._make_valid_labware()
        labware.column_spacing = 25.0
        # 14 + 5*25 = 139 > 127
        result = validate_labware_definition(labware)
        assert not result.valid
        assert any("Rightmost" in e for e in result.errors)

    def test_wells_exceed_footprint_y(self):
        labware = self._make_valid_labware()
        labware.row_spacing = 80.0
        # 12 + 1*80 = 92 > 85
        result = validate_labware_definition(labware)
        assert not result.valid
        assert any("Topmost" in e for e in result.errors)

    def test_negative_offset(self):
        labware = self._make_valid_labware()
        labware.x_offset = -5.0
        result = validate_labware_definition(labware)
        assert not result.valid
        assert any("x_offset" in e for e in result.errors)

    def test_default_offset_warning(self):
        """Using default SBS offsets should produce a warning."""
        labware = self._make_valid_labware()
        labware.x_offset = 14.38
        labware.y_offset = 11.24
        result = validate_labware_definition(labware)
        # Still valid, but with a warning
        assert result.valid
        assert any("default" in w.lower() for w in result.warnings)

    def test_rectangular_well_missing_dimensions(self):
        labware = self._make_valid_labware()
        for w in labware.wells.values():
            w.shape = "rectangular"
            w.diameter = None
            w.x_dimension = None
            w.y_dimension = None
        result = validate_labware_definition(labware)
        assert not result.valid
        assert any("well_x" in f for f in result.missing_fields)
        assert any("well_y" in f for f in result.missing_fields)


class TestFormatMissingFieldsPrompt:
    def test_no_issues_empty_string(self):
        result = LabwareValidationResult(
            valid=True, errors=[], warnings=[], missing_fields=[],
        )
        assert format_missing_fields_prompt(result) == ""

    def test_missing_fields_listed(self):
        result = LabwareValidationResult(
            valid=False,
            errors=[],
            warnings=[],
            missing_fields=["rows (number of rows)", "columns (number of columns)"],
        )
        prompt = format_missing_fields_prompt(result)
        assert "1. rows" in prompt
        assert "2. columns" in prompt
        assert "⚠️" in prompt

    def test_errors_listed(self):
        result = LabwareValidationResult(
            valid=False,
            errors=["x_dimension exceeds slot width"],
            warnings=[],
            missing_fields=[],
        )
        prompt = format_missing_fields_prompt(result)
        assert "❌" in prompt
        assert "x_dimension" in prompt

    def test_combined_output(self):
        result = LabwareValidationResult(
            valid=False,
            errors=["z too tall"],
            warnings=["default offsets"],
            missing_fields=["diameter"],
        )
        prompt = format_missing_fields_prompt(result)
        assert "⚠️" in prompt
        assert "❌" in prompt
        assert "💡" in prompt


class TestLabwareValidationError:
    def test_exception_carries_result(self):
        result = LabwareValidationResult(
            valid=False,
            errors=["too tall"],
            warnings=[],
            missing_fields=["diameter"],
        )
        err = LabwareValidationError(result)
        assert err.result is result
        assert "diameter" in str(err)
        assert "too tall" in str(err)


class TestCreateGridLabwareValidation:
    def test_valid_creation_succeeds(self):
        mgr = CustomLabwareManager()
        labware = mgr.create_grid_labware(
            name="test_12tube",
            rows=2,
            columns=6,
            well_depth=45.0,
            well_diameter=15.0,
            well_volume=1500,
            row_spacing=20.0,
            column_spacing=20.0,
        )
        assert len(labware.wells) == 12

    def test_invalid_creation_raises(self):
        mgr = CustomLabwareManager()
        with pytest.raises(LabwareValidationError) as exc_info:
            mgr.create_grid_labware(
                name="bad_labware",
                rows=2,
                columns=6,
                well_depth=45.0,
                well_diameter=15.0,
                well_volume=1500,
                row_spacing=10.0,  # < diameter=15 → overlap
                column_spacing=20.0,
            )
        assert "overlap" in str(exc_info.value).lower()

    def test_skip_validation_bypasses_check(self):
        mgr = CustomLabwareManager()
        # Same invalid params, but skip_validation=True
        labware = mgr.create_grid_labware(
            name="bad_but_forced",
            rows=2,
            columns=6,
            well_depth=45.0,
            well_diameter=15.0,
            well_volume=1500,
            row_spacing=10.0,  # overlap, but skipped
            column_spacing=20.0,
            skip_validation=True,
        )
        assert len(labware.wells) == 12

    def test_missing_diameter_raises(self):
        mgr = CustomLabwareManager()
        with pytest.raises(LabwareValidationError) as exc_info:
            mgr.create_grid_labware(
                name="no_diameter",
                rows=2,
                columns=6,
                well_depth=45.0,
                # well_diameter not provided → None → missing
                well_volume=1500,
                row_spacing=20.0,
                column_spacing=20.0,
            )
        assert "missing" in str(exc_info.value).lower() or "well_x" in str(exc_info.value).lower()
