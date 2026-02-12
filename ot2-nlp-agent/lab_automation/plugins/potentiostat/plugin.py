"""
Potentiostat Plugin

Main plugin class for electrochemistry instruments.
"""

from typing import Dict, List
from ...core.plugin_base import PluginBase, ParserBase
from .operations import ELECTROCHEM_OPERATIONS, ElectrochemOperation
from .parser import PotentiostatParser


class PotentiostatPlugin(PluginBase):
    """
    Plugin for potentiostat/electrochemistry instruments.

    Supports various electrochemistry platforms through adapters:
    - SquidStat (Admiral Instruments)
    - Gamry
    - BioLogic
    - Autolab
    - CH Instruments

    Example:
        plugin = PotentiostatPlugin()

        # Parse instruction
        result = plugin.parse("run EIS from 10kHz to 0.1Hz")
        # Returns: {
        #     "operation": "eis",
        #     "action": "potentiostat.run_eis",
        #     "params": {"freq_start_hz": 10000, "freq_stop_hz": 0.1},
        #     ...
        # }
    """

    name = "potentiostat"
    device_type = "potentiostat"
    version = "1.0.0"
    description = "Electrochemistry instruments (SquidStat, Gamry, BioLogic, etc.)"

    def _register_operations(self):
        """Register all electrochemistry operations."""
        for op_type, op_def in ELECTROCHEM_OPERATIONS.items():
            self.register_operation(op_def)

    def _create_parser(self) -> ParserBase:
        """Create the potentiostat parser."""
        return PotentiostatParser()

    def get_supported_operations(self) -> List[str]:
        """Get list of supported operation names."""
        return [op.value for op in ElectrochemOperation]

    def get_keywords(self, language: str = "en") -> Dict[str, List[str]]:
        """Get all keywords organized by operation."""
        result = {}
        for op_type, op_def in ELECTROCHEM_OPERATIONS.items():
            result[op_type.value] = op_def.keywords.get(language, [])
        return result
