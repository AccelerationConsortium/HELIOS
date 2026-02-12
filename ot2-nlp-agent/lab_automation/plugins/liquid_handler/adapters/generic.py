"""
Generic Gantry Adapter

A generic adapter for custom gantry-based liquid handling systems.
"""

from typing import Any, Dict


class GenericGantryAdapter:
    """
    Adapter for generic/custom gantry systems.

    This adapter provides a pass-through implementation that
    can be customized for specific hardware.
    """

    name = "generic_gantry"
    manufacturer = "Custom"
    model = "Generic Gantry System"

    def translate_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate generic action to gantry command.

        Default implementation passes through unchanged.
        Override in subclass for specific hardware.
        """
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
            "description": "Generic adapter for custom gantry systems",
        }
