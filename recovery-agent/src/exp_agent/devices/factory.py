"""
Device factory for creating device instances from configuration.
"""

from typing import Dict, Any, Type
from ..base import Device
from ..real import RealHeater, SerialHeater, NetworkHeater, IKAHeater, DeviceConfig
from ..simulated.heater import SimHeater


class DeviceFactory:
    """Factory for creating device instances."""

    # Registry of device types and their constructors
    DEVICE_TYPES: Dict[str, Dict[str, Type[Device]]] = {
        "heater": {
            "simulated": SimHeater,
            "serial": SerialHeater,
            "network": NetworkHeater,
            "ika": IKAHeater,
        }
    }

    @classmethod
    def create_device(
        cls, config: DeviceConfig, simulation_mode: bool = False
    ) -> Device:
        """Create a device instance from configuration."""

        if simulation_mode:
            # Use simulated devices
            device_class = cls.DEVICE_TYPES.get(config.type, {}).get("simulated")
            if not device_class:
                raise ValueError(
                    f"No simulated device available for type: {config.type}"
                )

            # Create with fault mode for simulation
            if config.type == "heater":
                return device_class(name=config.name, fault_mode="none")
            else:
                return device_class(name=config.name)
        else:
            # Use real devices
            device_class = cls.DEVICE_TYPES.get(config.type, {}).get(
                config.connection_type
            )
            if not device_class:
                raise ValueError(
                    f"No device class found for type '{config.type}' and connection '{config.connection_type}'"
                )

            if config.type == "heater":
                # Create communication interface
                comm_interface = config.create_communication_interface()
                return device_class(name=config.name, comm_interface=comm_interface)
            else:
                raise ValueError(f"Unsupported device type: {config.type}")

    @classmethod
    def create_simulated_device(cls, device_type: str, name: str, **kwargs) -> Device:
        """Create a simulated device for testing."""
        device_class = cls.DEVICE_TYPES.get(device_type, {}).get("simulated")
        if not device_class:
            raise ValueError(f"No simulated device available for type: {device_type}")

        return device_class(name=name, **kwargs)

    @classmethod
    def get_supported_types(cls) -> Dict[str, list]:
        """Get all supported device types and their connection methods."""
        return {
            device_type: list(methods.keys())
            for device_type, methods in cls.DEVICE_TYPES.items()
        }
