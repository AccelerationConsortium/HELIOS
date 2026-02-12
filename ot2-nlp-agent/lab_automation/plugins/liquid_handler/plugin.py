"""
Liquid Handler Plugin

Main plugin class for liquid handling robots.
"""

from typing import Any, Dict, List
from ...core.plugin_base import PluginBase, ParserBase, OperationDef
from .operations import LIQUID_OPERATIONS, LiquidOperation
from .parser import LiquidHandlerParser


class LiquidHandlerPlugin(PluginBase):
    """
    Plugin for liquid handling robots.

    Supports various liquid handling platforms through adapters:
    - OT-2 (Opentrons)
    - Hamilton
    - Tecan
    - Custom gantry systems

    Example:
        plugin = LiquidHandlerPlugin()

        # Parse instruction
        result = plugin.parse("transfer 100ul from A1 to B1")
        # Returns: {
        #     "operation": "transfer",
        #     "action": "liquid_handler.transfer",
        #     "params": {"volume": 100, "source": "A1", "destination": "B1"},
        #     "confidence": 0.8,
        #     ...
        # }

        # Register hardware adapter
        plugin.register_adapter("ot2", OT2Adapter())
    """

    name = "liquid_handler"
    device_type = "liquid_handler"
    version = "1.0.0"
    description = "Liquid handling robots (OT-2, Hamilton, Tecan, etc.)"

    def _register_operations(self):
        """Register all liquid handling operations."""
        for op_type, op_def in LIQUID_OPERATIONS.items():
            self.register_operation(op_def)

    def _create_parser(self) -> ParserBase:
        """Create the liquid handler parser."""
        return LiquidHandlerParser()

    def get_supported_operations(self) -> List[str]:
        """Get list of supported operation names."""
        return [op.value for op in LiquidOperation]

    def get_keywords(self, language: str = "en") -> Dict[str, List[str]]:
        """Get all keywords organized by operation."""
        result = {}
        for op_type, op_def in LIQUID_OPERATIONS.items():
            result[op_type.value] = op_def.keywords.get(language, [])
        return result


# Adapter base class
class LiquidHandlerAdapter:
    """
    Base class for liquid handler hardware adapters.

    Each adapter translates generic liquid handling operations
    to specific hardware commands.
    """

    name: str = "generic"
    manufacturer: str = "Generic"
    model: str = "Generic Liquid Handler"

    def translate_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate generic action to hardware-specific command.

        Args:
            action: Generic action string (e.g., "liquid_handler.transfer")
            params: Action parameters

        Returns:
            Hardware-specific command dictionary
        """
        # Default: pass through unchanged
        return {
            "action": action,
            "params": params,
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """Get adapter capabilities."""
        return {
            "name": self.name,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "operations": list(LIQUID_OPERATIONS.keys()),
        }
