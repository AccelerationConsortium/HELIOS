"""
Real laboratory hardware device implementations.

This package provides concrete implementations for actual laboratory equipment,
enabling the Experiment Agent to interface with real hardware devices.
"""

from .communication import (
    CommunicationInterface,
    NetworkCommunication,
    RealDevice,
    SerialCommunication,
)
from .config import (
    ConfigManager,
    DeviceConfig,
    LabConfig,
    create_network_heater_config,
    create_serial_heater_config,
)
from .heater import (
    IKAHeater,
    NetworkHeater,
    RealHeater,
    SerialHeater,
    create_ika_heater,
    create_network_heater,
    create_serial_heater,
)

__all__ = [
    # Communication interfaces
    "CommunicationInterface",
    "SerialCommunication",
    "NetworkCommunication",
    "RealDevice",
    # Device implementations
    "RealHeater",
    "SerialHeater",
    "NetworkHeater",
    "IKAHeater",
    # Factory functions
    "create_serial_heater",
    "create_network_heater",
    "create_ika_heater",
    # Configuration
    "DeviceConfig",
    "LabConfig",
    "ConfigManager",
    "create_serial_heater_config",
    "create_network_heater_config",
]
