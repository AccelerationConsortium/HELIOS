"""
UO Expander - Expand Unit Operations to Primitives.

This module converts high-level UOs into sequences of
device-agnostic primitive actions.
"""

from typing import Dict, List, Any

from ..ir import UnitOperation, UOType, Primitive, ActionType


class UOExpander:
    """
    Expands Unit Operations into Primitives.

    Each UO type has an expansion strategy that converts it
    into a sequence of primitive actions.
    """

    def __init__(self):
        """Initialize the expander."""
        # Map UO types to expansion methods
        self._expanders = {
            UOType.ELECTRODE_PREPARATION: self._expand_electrode_prep,
            UOType.ELECTROLYTE_PREPARATION: self._expand_electrolyte_prep,
            UOType.CELL_ASSEMBLY: self._expand_cell_assembly,
            UOType.CALIBRATION: self._expand_calibration,
            UOType.MEASUREMENT: self._expand_measurement,
            UOType.DATA_ANALYSIS: self._expand_data_analysis,
            UOType.DATA_LOGGING: self._expand_data_logging,
            UOType.STABILITY_TEST: self._expand_stability_test,
            UOType.CLEANUP: self._expand_cleanup,
            UOType.USER_CHECKPOINT: self._expand_checkpoint,
            UOType.WAIT: self._expand_wait,
        }

    def expand(self, unit_operations: List[UnitOperation]) -> List[Primitive]:
        """
        Expand a list of UOs into primitives.

        Args:
            unit_operations: List of UOs to expand

        Returns:
            List of Primitive actions
        """
        primitives = []

        for uo in unit_operations:
            # Get expander for this UO type
            expander = self._expanders.get(uo.uo_type, self._expand_generic)

            # Expand and add to list
            uo_primitives = expander(uo)
            primitives.extend(uo_primitives)

        return primitives

    def _expand_electrode_prep(self, uo: UnitOperation) -> List[Primitive]:
        """Expand electrode preparation UO."""
        primitives = []
        params = uo.parameters

        # If this is just info collection, no primitives needed
        if uo.name == "ElectrodeInfo":
            # Just a data logging primitive
            primitives.append(Primitive(
                name=f"{uo.name}_record",
                action_type=ActionType.DATA_LOGGING,
                params={
                    "data_type": "electrode_info",
                    "material": params.get("electrode_material"),
                    "area_cm2": params.get("electrode_area_cm2"),
                    "loading": params.get("catalyst_loading_mg_cm2"),
                },
                device_type="data_system",
                description="Record electrode information",
                source_uo=uo.name,
            ))
        else:
            # Full electrode preparation
            if "ink_volume_ul" in params:
                primitives.append(Primitive(
                    name=f"{uo.name}_dispense_ink",
                    action_type=ActionType.DISPENSE,
                    params={
                        "volume_ul": params.get("ink_volume_ul", 10),
                        "destination": "electrode_surface",
                    },
                    device_type="liquid_handler",
                    description="Dispense catalyst ink onto electrode",
                    source_uo=uo.name,
                ))

            # Drying step
            if params.get("drying_time_min"):
                if params.get("drying_temperature_C", 25) > 30:
                    primitives.append(Primitive(
                        name=f"{uo.name}_heat",
                        action_type=ActionType.HEAT,
                        params={
                            "temperature_C": params.get("drying_temperature_C", 60),
                            "duration_s": params.get("drying_time_min", 30) * 60,
                        },
                        device_type="temperature_module",
                        description="Heat electrode for drying",
                        source_uo=uo.name,
                    ))
                else:
                    primitives.append(Primitive(
                        name=f"{uo.name}_wait_dry",
                        action_type=ActionType.WAIT,
                        params={
                            "duration_s": params.get("drying_time_min", 30) * 60,
                        },
                        device_type="none",
                        description="Wait for electrode to dry",
                        source_uo=uo.name,
                    ))

        return primitives

    def _expand_electrolyte_prep(self, uo: UnitOperation) -> List[Primitive]:
        """Expand electrolyte preparation UO."""
        primitives = []
        params = uo.parameters

        # Record electrolyte info
        primitives.append(Primitive(
            name=f"{uo.name}_record",
            action_type=ActionType.DATA_LOGGING,
            params={
                "data_type": "electrolyte_info",
                "electrolyte_type": params.get("electrolyte_type", "1M KOH"),
                "volume_ml": params.get("electrolyte_volume_ml", 50),
            },
            device_type="data_system",
            description="Record electrolyte information",
            source_uo=uo.name,
        ))

        # Gas purge if specified
        if params.get("purge_gas") and params.get("purge_gas") != "None":
            primitives.append(Primitive(
                name=f"{uo.name}_purge",
                action_type=ActionType.WAIT,
                params={
                    "duration_s": params.get("purge_time_min", 30) * 60,
                    "condition": f"gas_purge_{params.get('purge_gas')}",
                },
                device_type="gas_system",
                description=f"Purge electrolyte with {params.get('purge_gas')}",
                source_uo=uo.name,
            ))

        # Temperature control if specified
        if params.get("temperature_C") and params.get("temperature_C") != 25:
            primitives.append(Primitive(
                name=f"{uo.name}_temp_control",
                action_type=ActionType.HEAT,
                params={
                    "temperature_C": params.get("temperature_C"),
                    "hold": True,
                },
                device_type="temperature_module",
                description=f"Set electrolyte temperature to {params.get('temperature_C')}°C",
                source_uo=uo.name,
            ))

        return primitives

    def _expand_cell_assembly(self, uo: UnitOperation) -> List[Primitive]:
        """Expand cell assembly UO."""
        primitives = []
        params = uo.parameters

        # User checkpoint for manual assembly
        primitives.append(Primitive(
            name=f"{uo.name}_manual",
            action_type=ActionType.USER_CHECKPOINT,
            params={
                "message": "Please assemble the electrochemical cell",
                "message_zh": "请组装电化学电池",
                "checklist": [
                    f"Working electrode: {params.get('working_electrode_area_cm2', 'N/A')} cm²",
                    f"Reference electrode: {params.get('reference_electrode', 'Ag/AgCl')}",
                    f"Counter electrode: {params.get('counter_electrode', 'Pt wire')}",
                    "Connect all electrodes to potentiostat",
                ],
            },
            device_type="user",
            description="Manual cell assembly checkpoint",
            source_uo=uo.name,
            estimated_duration_s=300,
        ))

        # Record cell configuration
        primitives.append(Primitive(
            name=f"{uo.name}_record",
            action_type=ActionType.DATA_LOGGING,
            params={
                "data_type": "cell_config",
                "cell_type": params.get("cell_type", "three_electrode"),
                "reference": params.get("reference_electrode"),
                "counter": params.get("counter_electrode"),
            },
            device_type="data_system",
            description="Record cell configuration",
            source_uo=uo.name,
        ))

        return primitives

    def _expand_calibration(self, uo: UnitOperation) -> List[Primitive]:
        """Expand calibration UO."""
        primitives = []
        params = uo.parameters

        # Calculate RHE offset
        # E(RHE) = E(ref) + E°(ref vs SHE) + 0.059 * pH
        ref_potential = params.get("reference_potential_V", 0.197)
        ph = params.get("ph_value", 14)
        rhe_offset = ref_potential + 0.059 * ph

        primitives.append(Primitive(
            name=f"{uo.name}_calculate",
            action_type=ActionType.DATA_LOGGING,
            params={
                "data_type": "reference_calibration",
                "reference_potential_V": ref_potential,
                "ph_value": ph,
                "rhe_offset_V": rhe_offset,
            },
            device_type="data_system",
            description=f"RHE offset calculated: {rhe_offset:.3f} V",
            source_uo=uo.name,
        ))

        # If iR compensation requested
        if params.get("ir_compensation"):
            # EIS measurement for solution resistance
            primitives.append(Primitive(
                name=f"{uo.name}_measure_Rs",
                action_type=ActionType.IMPEDANCE_SCAN,
                params={
                    "frequency_Hz": 100000,
                    "amplitude_mV": 10,
                    "points": 1,
                    "purpose": "solution_resistance",
                },
                device_type="potentiostat",
                description="Measure solution resistance for iR compensation",
                source_uo=uo.name,
            ))

        return primitives

    def _expand_measurement(self, uo: UnitOperation) -> List[Primitive]:
        """Expand measurement UO (LSV, CV, etc.)."""
        primitives = []
        params = uo.parameters
        method = params.get("method", "LSV")

        if method == "LSV":
            primitives.append(Primitive(
                name=f"{uo.name}_lsv",
                action_type=ActionType.POTENTIOSTAT_METHOD,
                params={
                    "method": "LSV",
                    "start_potential_V": params.get("start_potential_V_vs_RHE", 1.0),
                    "end_potential_V": params.get("end_potential_V_vs_RHE", 1.8),
                    "scan_rate_V_s": params.get("scan_rate_mV_s", 5) / 1000,
                    "vs_reference": "RHE",
                },
                device_type="potentiostat",
                description=f"LSV scan from {params.get('start_potential_V_vs_RHE', 1.0)} to {params.get('end_potential_V_vs_RHE', 1.8)} V vs RHE",
                source_uo=uo.name,
                estimated_duration_s=600,
            ))

        elif method == "EIS":
            primitives.append(Primitive(
                name=f"{uo.name}_eis",
                action_type=ActionType.IMPEDANCE_SCAN,
                params={
                    "dc_potential_V": params.get("dc_potential_V_vs_RHE", 1.55),
                    "ac_amplitude_mV": params.get("ac_amplitude_mV", 10),
                    "frequency_start_Hz": params.get("frequency_start_Hz", 100000),
                    "frequency_end_Hz": params.get("frequency_end_Hz", 0.1),
                    "points_per_decade": params.get("points_per_decade", 10),
                    "vs_reference": "RHE",
                },
                device_type="potentiostat",
                description=f"EIS at {params.get('dc_potential_V_vs_RHE', 1.55)} V vs RHE",
                source_uo=uo.name,
                estimated_duration_s=900,
            ))

        return primitives

    def _expand_data_analysis(self, uo: UnitOperation) -> List[Primitive]:
        """Expand data analysis UO."""
        primitives = []
        params = uo.parameters

        # Data analysis runs on data system
        primitives.append(Primitive(
            name=f"{uo.name}_analyze",
            action_type=ActionType.DATA_LOGGING,
            params={
                "action": "analyze",
                "analysis_type": uo.name,
                "params": params,
            },
            device_type="data_system",
            description=f"Run {uo.name} analysis",
            source_uo=uo.name,
        ))

        return primitives

    def _expand_data_logging(self, uo: UnitOperation) -> List[Primitive]:
        """Expand data logging/save UO."""
        primitives = []
        params = uo.parameters

        primitives.append(Primitive(
            name=f"{uo.name}_save",
            action_type=ActionType.DATA_LOGGING,
            params={
                "action": "save",
                "sample_name": params.get("sample_name", "sample"),
                "format": params.get("output_format", "csv"),
                "include_plots": params.get("save_plots", True),
                "notes": params.get("notes", ""),
            },
            device_type="data_system",
            description="Save experiment data",
            source_uo=uo.name,
        ))

        return primitives

    def _expand_stability_test(self, uo: UnitOperation) -> List[Primitive]:
        """Expand stability test UO."""
        primitives = []
        params = uo.parameters

        primitives.append(Primitive(
            name=f"{uo.name}_chronopotentiometry",
            action_type=ActionType.POTENTIOSTAT_METHOD,
            params={
                "method": "chronopotentiometry",
                "current_density_mA_cm2": params.get("current_density_mA_cm2", 10),
                "duration_s": params.get("duration_hours", 10) * 3600,
                "sampling_interval_s": params.get("sampling_interval_s", 10),
            },
            device_type="potentiostat",
            description=f"Stability test at {params.get('current_density_mA_cm2', 10)} mA/cm²",
            source_uo=uo.name,
            estimated_duration_s=params.get("duration_hours", 10) * 3600,
        ))

        return primitives

    def _expand_cleanup(self, uo: UnitOperation) -> List[Primitive]:
        """Expand cleanup UO."""
        primitives = []
        params = uo.parameters

        primitives.append(Primitive(
            name=f"{uo.name}_manual",
            action_type=ActionType.USER_CHECKPOINT,
            params={
                "message": "Please clean up the experiment",
                "message_zh": "请清理实验",
                "checklist": [
                    f"Rinse cycles: {params.get('rinse_cycles', 3)}",
                    f"Dispose electrolyte: {params.get('dispose_electrolyte', True)}",
                    f"Dry electrodes: {params.get('dry_electrodes', True)}",
                ],
            },
            device_type="user",
            description="Manual cleanup checkpoint",
            source_uo=uo.name,
        ))

        return primitives

    def _expand_checkpoint(self, uo: UnitOperation) -> List[Primitive]:
        """Expand user checkpoint UO."""
        return [Primitive(
            name=f"{uo.name}_checkpoint",
            action_type=ActionType.USER_CHECKPOINT,
            params=uo.parameters,
            device_type="user",
            description=uo.description,
            source_uo=uo.name,
        )]

    def _expand_wait(self, uo: UnitOperation) -> List[Primitive]:
        """Expand wait UO."""
        return [Primitive(
            name=f"{uo.name}_wait",
            action_type=ActionType.WAIT,
            params=uo.parameters,
            device_type="none",
            description=uo.description,
            source_uo=uo.name,
        )]

    def _expand_generic(self, uo: UnitOperation) -> List[Primitive]:
        """Generic expansion for unhandled UO types."""
        return [Primitive(
            name=f"{uo.name}_generic",
            action_type=ActionType.DATA_LOGGING,
            params={
                "uo_type": uo.uo_type.value,
                "uo_params": uo.parameters,
            },
            device_type="any",
            description=uo.description,
            source_uo=uo.name,
        )]
