"""
Lab Automation Plugins

Each plugin provides natural language parsing and operation definitions
for a category of lab instruments.
"""

from .camera import CameraPlugin
from .liquid_handler import LiquidHandlerPlugin
from .potentiostat import PotentiostatPlugin
from .pump_controller import PumpControllerPlugin

__all__ = [
    'LiquidHandlerPlugin',
    'PotentiostatPlugin',
    'PumpControllerPlugin',
    'CameraPlugin',
]
