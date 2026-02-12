"""
Sensor Drivers - Protocol adapters for different sensor sources.

Each driver is responsible for:
1. Connecting to the sensor source (serial, MQTT, Modbus, HTTP, etc.)
2. Parsing raw readings
3. Normalizing units and ranges
4. Producing SensorEvents

Drivers are pluggable - the SensorHub doesn't care where data comes from.
"""

from exp_agent.sensing.drivers.base import SensorDriver, DriverConfig
from exp_agent.sensing.drivers.mock_driver import (
    MockSensorDriver,
    MockSensorConfig,
    SensorProfile,
    TemperatureProfile,
    AirflowProfile,
    PressureProfile,
    create_lab_sensor_set,
)

__all__ = [
    "SensorDriver",
    "DriverConfig",
    "MockSensorDriver",
    "MockSensorConfig",
    "SensorProfile",
    "TemperatureProfile",
    "AirflowProfile",
    "PressureProfile",
    "create_lab_sensor_set",
]
