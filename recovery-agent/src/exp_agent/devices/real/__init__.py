"""
Real laboratory hardware device implementations.

This package provides concrete implementations for actual laboratory equipment,
enabling the Experiment Agent to interface with real hardware devices.
"""

from .communication import (
    CommunicationInterface,
    SerialCommunication,
    NetworkCommunication,
    RealDevice,
)
from .heater import (
    RealHeater,
    SerialHeater,
    NetworkHeater,
    IKAHeater,
    create_serial_heater,
    create_network_heater,
    create_ika_heater,
)
from .config import (
    DeviceConfig,
    LabConfig,
    ConfigManager,
    create_serial_heater_config,
    create_network_heater_config,
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
