"""
Configuration management for real devices.

This module provides configuration classes and utilities for managing
device connections, protocols, and settings.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path

from .communication import (
    CommunicationInterface,
    SerialCommunication,
    NetworkCommunication,
)


@dataclass
class DeviceConfig:
    """Configuration for a single device."""

    name: str
    type: str  # e.g., "heater", "pump", "sensor"
    brand: Optional[str] = None  # e.g., "IKA", "ThermoFisher"
    model: Optional[str] = None

    # Connection settings
    connection_type: str  # "serial", "network", "usb"
    connection_params: Dict[str, Any] = field(default_factory=dict)

    # Device-specific settings
    safety_limits: Dict[str, Any] = field(default_factory=dict)
    calibration: Dict[str, Any] = field(default_factory=dict)

    # Agent settings
    enabled: bool = True
    monitoring_interval: float = 1.0  # seconds
    timeout: float = 5.0  # seconds

    def create_communication_interface(self) -> CommunicationInterface:
        """Create the appropriate communication interface."""
        if self.connection_type == "serial":
            return SerialCommunication(
                port=self.connection_params.get("port", "/dev/ttyUSB0"),
                baudrate=self.connection_params.get("baudrate", 9600),
                timeout=self.timeout,
            )
        elif self.connection_type == "network":
            return NetworkCommunication(
                host=self.connection_params.get("host", "192.168.1.100"),
                port=self.connection_params.get("port", 5025),
                timeout=self.timeout,
            )
        else:
            raise ValueError(f"Unsupported connection type: {self.connection_type}")


@dataclass
class LabConfig:
    """Configuration for the entire lab setup."""

    devices: List[DeviceConfig] = field(default_factory=list)
    global_settings: Dict[str, Any] = field(default_factory=dict)

    # Recovery settings
    recovery_enabled: bool = True
    max_retry_attempts: int = 3
    emergency_stop_on_critical: bool = True

    # Monitoring settings
    health_check_interval: int = 30  # seconds
    log_level: str = "INFO"

    # Safety settings
    global_safety_limits: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Manages loading and saving device configurations."""

    def __init__(self, config_dir: str = "~/.exp_agent"):
        self.config_dir = Path(config_dir).expanduser()
        self.config_dir.mkdir(exist_ok=True)
        self.current_config: Optional[LabConfig] = None

    def load_config(self, config_file: str = "lab_config.json") -> LabConfig:
        """Load configuration from file."""
        config_path = self.config_dir / config_file

        if not config_path.exists():
            # Create default config
            self.current_config = self._create_default_config()
            self.save_config(self.current_config, config_file)
        else:
            with open(config_path, "r") as f:
                data = json.load(f)
                self.current_config = self._parse_config(data)

        return self.current_config

    def save_config(self, config: LabConfig, config_file: str = "lab_config.json"):
        """Save configuration to file."""
        config_path = self.config_dir / config_file

        with open(config_path, "w") as f:
            json.dump(self._serialize_config(config), f, indent=2)

    def _create_default_config(self) -> LabConfig:
        """Create a default configuration."""
        return LabConfig(
            devices=[
                DeviceConfig(
                    name="heater_1",
                    type="heater",
                    brand="Generic",
                    connection_type="serial",
                    connection_params={"port": "/dev/ttyUSB0", "baudrate": 9600},
                    safety_limits={"max_temperature": 200.0, "min_temperature": -20.0},
                )
            ],
            global_settings={"timezone": "UTC", "units": "celsius"},
        )

    def _parse_config(self, data: Dict[str, Any]) -> LabConfig:
        """Parse configuration from dictionary."""
        devices = []
        for device_data in data.get("devices", []):
            devices.append(DeviceConfig(**device_data))

        return LabConfig(
            devices=devices,
            global_settings=data.get("global_settings", {}),
            recovery_enabled=data.get("recovery_enabled", True),
            max_retry_attempts=data.get("max_retry_attempts", 3),
            emergency_stop_on_critical=data.get("emergency_stop_on_critical", True),
            health_check_interval=data.get("health_check_interval", 30),
            log_level=data.get("log_level", "INFO"),
            global_safety_limits=data.get("global_safety_limits", {}),
        )

    def _serialize_config(self, config: LabConfig) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "devices": [self._device_to_dict(device) for device in config.devices],
            "global_settings": config.global_settings,
            "recovery_enabled": config.recovery_enabled,
            "max_retry_attempts": config.max_retry_attempts,
            "emergency_stop_on_critical": config.emergency_stop_on_critical,
            "health_check_interval": config.health_check_interval,
            "log_level": config.log_level,
            "global_safety_limits": config.global_safety_limits,
        }

    def _device_to_dict(self, device: DeviceConfig) -> Dict[str, Any]:
        """Convert DeviceConfig to dictionary."""
        return {
            "name": device.name,
            "type": device.type,
            "brand": device.brand,
            "model": device.model,
            "connection_type": device.connection_type,
            "connection_params": device.connection_params,
            "safety_limits": device.safety_limits,
            "calibration": device.calibration,
            "enabled": device.enabled,
            "monitoring_interval": device.monitoring_interval,
            "timeout": device.timeout,
        }


# Example configurations for common devices
def create_serial_heater_config(
    name: str, port: str, brand: str = "Generic"
) -> DeviceConfig:
    """Create configuration for a serial-connected heater."""
    return DeviceConfig(
        name=name,
        type="heater",
        brand=brand,
        connection_type="serial",
        connection_params={"port": port, "baudrate": 9600},
        safety_limits={
            "max_temperature": 200.0 if brand == "IKA" else 150.0,
            "min_temperature": -20.0,
        },
    )


def create_network_heater_config(
    name: str, host: str, port: int = 5025
) -> DeviceConfig:
    """Create configuration for a network-connected heater."""
    return DeviceConfig(
        name=name,
        type="heater",
        brand="Network",
        connection_type="network",
        connection_params={"host": host, "port": port},
        safety_limits={"max_temperature": 150.0, "min_temperature": -20.0},
    )
