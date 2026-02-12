"""
Pump Controller Plugin

Main plugin class for pump systems (PLC, peristaltic, syringe, microfluidic).
"""

import re
from enum import Enum
from typing import Any, Dict, List, Optional
from ...core.plugin_base import PluginBase, ParserBase, OperationDef


class PumpOperation(Enum):
    """Pump operation types."""
    DISPENSE = "dispense"
    ASPIRATE = "aspirate"
    PUMP_ON = "pump_on"
    PUMP_OFF = "pump_off"
    SET_FLOW_RATE = "set_flow_rate"
    PRIME = "prime"
    PURGE = "purge"
    FILL = "fill"
    DRAIN = "drain"
    SET_TIMER = "set_timer"


# Operation definitions
PUMP_OPERATIONS: Dict[PumpOperation, OperationDef] = {
    PumpOperation.DISPENSE: OperationDef(
        name="dispense",
        action="pump_controller.dispense",
        keywords={
            "en": ["pump", "dispense", "deliver", "add", "inject"],
            "zh": ["泵送", "分配", "添加", "注入"],
        },
        params_schema={
            "pump": {"type": "int", "required": True},
            "volume_ml": {"type": "float", "required": True},
        },
        description="Dispense a volume using the pump",
    ),

    PumpOperation.ASPIRATE: OperationDef(
        name="aspirate",
        action="pump_controller.aspirate",
        keywords={
            "en": ["withdraw", "draw", "remove", "extract", "pull"],
            "zh": ["抽取", "提取", "移除"],
        },
        params_schema={
            "pump": {"type": "int", "required": True},
            "volume_ml": {"type": "float", "required": True},
        },
        description="Aspirate/withdraw using the pump",
    ),

    PumpOperation.PUMP_ON: OperationDef(
        name="pump_on",
        action="pump_controller.pump_on",
        keywords={
            "en": ["start pump", "pump on", "turn on pump", "run pump"],
            "zh": ["启动泵", "开泵", "打开泵"],
        },
        params_schema={
            "pump": {"type": "int", "required": True},
        },
        description="Turn pump on",
    ),

    PumpOperation.PUMP_OFF: OperationDef(
        name="pump_off",
        action="pump_controller.pump_off",
        keywords={
            "en": ["stop pump", "pump off", "turn off pump"],
            "zh": ["停止泵", "关泵", "关闭泵"],
        },
        params_schema={
            "pump": {"type": "int", "required": True},
        },
        description="Turn pump off",
    ),

    PumpOperation.SET_FLOW_RATE: OperationDef(
        name="set_flow_rate",
        action="pump_controller.set_flow_rate",
        keywords={
            "en": ["flow rate", "set rate", "rate"],
            "zh": ["流速", "流量"],
        },
        params_schema={
            "pump": {"type": "int", "required": True},
            "rate_ml_min": {"type": "float", "required": True},
        },
        description="Set pump flow rate",
    ),

    PumpOperation.PRIME: OperationDef(
        name="prime",
        action="pump_controller.prime",
        keywords={
            "en": ["prime", "prime pump", "fill line"],
            "zh": ["灌注", "预充"],
        },
        params_schema={
            "pump": {"type": "int", "required": True},
            "volume_ml": {"type": "float", "required": False},
        },
        description="Prime the pump line",
    ),

    PumpOperation.FILL: OperationDef(
        name="fill",
        action="pump_controller.fill",
        keywords={
            "en": ["fill", "fill up", "refill"],
            "zh": ["填充", "加满", "灌满"],
        },
        params_schema={
            "pump": {"type": "int", "required": True},
            "volume_ml": {"type": "float", "required": False},
        },
        description="Fill target container",
    ),

    PumpOperation.DRAIN: OperationDef(
        name="drain",
        action="pump_controller.drain",
        keywords={
            "en": ["drain", "empty", "remove all"],
            "zh": ["排空", "排出", "清空"],
        },
        params_schema={
            "pump": {"type": "int", "required": True},
        },
        description="Drain target container",
    ),

    PumpOperation.SET_TIMER: OperationDef(
        name="set_timer",
        action="pump_controller.set_timer",
        keywords={
            "en": ["pump timer", "run for", "pump for"],
            "zh": ["定时泵送", "泵送时间"],
        },
        params_schema={
            "pump": {"type": "int", "required": True},
            "duration_ms": {"type": "int", "required": True},
        },
        description="Run pump for specified duration",
    ),
}


class PumpControllerParser(ParserBase):
    """Parser for pump controller instructions."""

    def __init__(self):
        self._operations = PUMP_OPERATIONS

    def parse(self, instruction: str) -> Dict[str, Any]:
        """Parse pump instruction."""
        language = self.detect_language(instruction)

        # Find matching operation
        best_match = None
        best_confidence = 0.0

        for op_type, op_def in self._operations.items():
            matches, confidence = op_def.matches(instruction, language)
            if matches and confidence > best_confidence:
                best_match = op_type
                best_confidence = confidence

        if not best_match:
            return {
                "operation": None,
                "action": None,
                "params": {},
                "confidence": 0.0,
                "language": language,
                "description": instruction,
            }

        # Extract parameters
        params = self._extract_params(instruction, best_match)

        op_def = self._operations[best_match]
        return {
            "operation": best_match.value,
            "action": op_def.action,
            "params": params,
            "confidence": best_confidence,
            "language": language,
            "description": instruction,
        }

    def _extract_params(self, instruction: str, op_type: PumpOperation) -> Dict[str, Any]:
        """Extract pump parameters."""
        params = {}

        # Pump number
        pump_match = re.search(r'pump\s*(\d+)', instruction, re.IGNORECASE)
        if pump_match:
            params['pump'] = int(pump_match.group(1))

        # Volume
        vol_match = re.search(r'(\d+(?:\.\d+)?)\s*ml', instruction, re.IGNORECASE)
        if vol_match:
            params['volume_ml'] = float(vol_match.group(1))

        # Duration (ms or s)
        dur_match = re.search(r'(\d+)\s*ms', instruction, re.IGNORECASE)
        if dur_match:
            params['duration_ms'] = int(dur_match.group(1))
        else:
            dur_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:s|sec)', instruction, re.IGNORECASE)
            if dur_match:
                params['duration_ms'] = int(float(dur_match.group(1)) * 1000)

        # Flow rate
        rate_match = re.search(r'(\d+(?:\.\d+)?)\s*ml/min', instruction, re.IGNORECASE)
        if rate_match:
            params['rate_ml_min'] = float(rate_match.group(1))

        return params


class PumpControllerPlugin(PluginBase):
    """
    Plugin for pump control systems.

    Supports:
    - PLC-controlled pumps
    - Peristaltic pumps
    - Syringe pumps
    - Microfluidic pumps
    """

    name = "pump_controller"
    device_type = "pump_controller"
    version = "1.0.0"
    description = "Pump systems (PLC, peristaltic, syringe, microfluidic)"

    def _register_operations(self):
        """Register pump operations."""
        for op_type, op_def in PUMP_OPERATIONS.items():
            self.register_operation(op_def)

    def _create_parser(self) -> ParserBase:
        """Create pump parser."""
        return PumpControllerParser()

    def get_supported_operations(self) -> List[str]:
        """Get list of supported operation names."""
        return [op.value for op in PumpOperation]
