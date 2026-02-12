"""
Potentiostat Operations

Defines all electrochemistry operations supported by potentiostats.
"""

from enum import Enum
from typing import Dict
from ...core.plugin_base import OperationDef


class ElectrochemOperation(Enum):
    """Electrochemistry operation types."""
    OCV = "ocv"  # Open Circuit Voltage
    EIS = "eis"  # Electrochemical Impedance Spectroscopy
    CV = "cv"    # Cyclic Voltammetry
    CP = "cp"    # Chronopotentiometry (constant current)
    CA = "ca"    # Chronoamperometry (constant voltage)
    LSV = "lsv"  # Linear Sweep Voltammetry
    DPV = "dpv"  # Differential Pulse Voltammetry
    SWV = "swv"  # Square Wave Voltammetry
    RESET_PLOT = "reset_plot"
    SAVE_DATA = "save_data"
    SAVE_SNAPSHOT = "save_snapshot"


# Operation definitions with multilingual keywords
ELECTROCHEM_OPERATIONS: Dict[ElectrochemOperation, OperationDef] = {
    ElectrochemOperation.OCV: OperationDef(
        name="ocv",
        action="potentiostat.ocv",
        keywords={
            "en": ["ocv", "open circuit", "open circuit voltage", "rest", "equilibrate"],
            "zh": ["开路电压", "开路", "静置", "平衡"],
        },
        params_schema={
            "duration_s": {"type": "float", "required": True, "default": 30},
            "sampling_interval_s": {"type": "float", "required": False, "default": 0.5},
        },
        description="Measure open circuit voltage over time",
    ),

    ElectrochemOperation.EIS: OperationDef(
        name="eis",
        action="potentiostat.run_eis",
        keywords={
            "en": ["eis", "impedance", "impedance spectroscopy", "nyquist", "peis", "geis"],
            "zh": ["阻抗", "交流阻抗", "阻抗谱", "奈奎斯特"],
        },
        params_schema={
            "freq_start_hz": {"type": "float", "required": True, "default": 10000},
            "freq_stop_hz": {"type": "float", "required": True, "default": 0.1},
            "points_per_decade": {"type": "float", "required": False, "default": 5},
            "amplitude_v": {"type": "float", "required": False, "default": 0.01},
            "bias_voltage": {"type": "float", "required": False, "default": 0},
            "bias_vs_ocp": {"type": "bool", "required": False, "default": True},
        },
        description="Run electrochemical impedance spectroscopy",
    ),

    ElectrochemOperation.CV: OperationDef(
        name="cv",
        action="potentiostat.run_cv",
        keywords={
            "en": ["cv", "cyclic voltammetry", "cyclic", "voltammogram"],
            "zh": ["循环伏安", "cv", "伏安"],
        },
        params_schema={
            "start_v": {"type": "float", "required": True},
            "vertex1_v": {"type": "float", "required": True},
            "vertex2_v": {"type": "float", "required": True},
            "end_v": {"type": "float", "required": True},
            "scan_rate_v_s": {"type": "float", "required": True, "default": 0.05},
            "cycles": {"type": "int", "required": False, "default": 3},
        },
        description="Run cyclic voltammetry scan",
    ),

    ElectrochemOperation.CP: OperationDef(
        name="cp",
        action="potentiostat.run_cp",
        keywords={
            "en": ["cp", "chronopotentiometry", "constant current", "galvanostatic", "deposition", "dissolution"],
            "zh": ["恒电流", "计时电位", "沉积", "溶解"],
        },
        params_schema={
            "current_a": {"type": "float", "required": True},
            "duration_s": {"type": "float", "required": True},
            "sampling_interval_s": {"type": "float", "required": False, "default": 0.5},
            "max_voltage": {"type": "float", "required": False},
            "min_voltage": {"type": "float", "required": False},
        },
        description="Apply constant current and measure voltage",
    ),

    ElectrochemOperation.CA: OperationDef(
        name="ca",
        action="potentiostat.run_ca",
        keywords={
            "en": ["ca", "chronoamperometry", "constant voltage", "potentiostatic", "step"],
            "zh": ["恒电位", "计时电流", "电位阶跃"],
        },
        params_schema={
            "voltage_v": {"type": "float", "required": True},
            "duration_s": {"type": "float", "required": True},
            "sampling_interval_s": {"type": "float", "required": False, "default": 0.5},
        },
        description="Apply constant voltage and measure current",
    ),

    ElectrochemOperation.LSV: OperationDef(
        name="lsv",
        action="potentiostat.run_lsv",
        keywords={
            "en": ["lsv", "linear sweep", "linear sweep voltammetry", "polarization"],
            "zh": ["线性扫描", "极化曲线"],
        },
        params_schema={
            "start_v": {"type": "float", "required": True},
            "end_v": {"type": "float", "required": True},
            "scan_rate_v_s": {"type": "float", "required": True, "default": 0.01},
        },
        description="Run linear sweep voltammetry",
    ),

    ElectrochemOperation.DPV: OperationDef(
        name="dpv",
        action="potentiostat.run_dpv",
        keywords={
            "en": ["dpv", "differential pulse", "differential pulse voltammetry"],
            "zh": ["差分脉冲", "差分脉冲伏安"],
        },
        params_schema={
            "start_v": {"type": "float", "required": True},
            "end_v": {"type": "float", "required": True},
            "step_v": {"type": "float", "required": False, "default": 0.005},
            "pulse_height_v": {"type": "float", "required": False, "default": 0.05},
            "pulse_width_s": {"type": "float", "required": False, "default": 0.05},
        },
        description="Run differential pulse voltammetry",
    ),

    ElectrochemOperation.SWV: OperationDef(
        name="swv",
        action="potentiostat.run_swv",
        keywords={
            "en": ["swv", "square wave", "square wave voltammetry"],
            "zh": ["方波", "方波伏安"],
        },
        params_schema={
            "start_v": {"type": "float", "required": True},
            "end_v": {"type": "float", "required": True},
            "step_v": {"type": "float", "required": False, "default": 0.005},
            "amplitude_v": {"type": "float", "required": False, "default": 0.025},
            "frequency_hz": {"type": "float", "required": False, "default": 25},
        },
        description="Run square wave voltammetry",
    ),

    ElectrochemOperation.RESET_PLOT: OperationDef(
        name="reset_plot",
        action="potentiostat.reset_plot",
        keywords={
            "en": ["reset plot", "clear plot", "new plot"],
            "zh": ["重置图表", "清除图表", "新建图表"],
        },
        params_schema={
            "title": {"type": "string", "required": False},
        },
        description="Reset the live plot display",
    ),

    ElectrochemOperation.SAVE_DATA: OperationDef(
        name="save_data",
        action="potentiostat.save_data",
        keywords={
            "en": ["save data", "export data", "save results"],
            "zh": ["保存数据", "导出数据"],
        },
        params_schema={
            "filename": {"type": "string", "required": False},
            "format": {"type": "string", "required": False, "default": "csv"},
        },
        description="Save experiment data to file",
    ),

    ElectrochemOperation.SAVE_SNAPSHOT: OperationDef(
        name="save_snapshot",
        action="potentiostat.save_snapshot",
        keywords={
            "en": ["save snapshot", "screenshot", "capture plot"],
            "zh": ["保存截图", "截图"],
        },
        params_schema={
            "phase": {"type": "string", "required": False},
            "cycle": {"type": "int", "required": False},
        },
        description="Save a snapshot of the current plot",
    ),
}


def get_operation(op_type: ElectrochemOperation) -> OperationDef:
    """Get operation definition by type."""
    return ELECTROCHEM_OPERATIONS[op_type]


def get_all_operations() -> Dict[ElectrochemOperation, OperationDef]:
    """Get all operation definitions."""
    return ELECTROCHEM_OPERATIONS.copy()
