"""
MockSensorDriver - Simulated sensor driver for testing and development.

Features:
- Configurable sensor profiles (temperature, airflow, pressure, etc.)
- Realistic value generation with noise
- Support for fault injection (via FaultInjector)
- Replay mode for reproducing incidents from logs
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional
import random
import math

from exp_agent.sensing.drivers.base import SensorDriver, DriverConfig
from exp_agent.sensing.protocol.sensor_event import (
    SensorEvent,
    SensorType,
    SensorQuality,
    SensorMeta,
    QualityStatus,
)


@dataclass
class SensorProfile:
    """Configuration for a simulated sensor."""

    sensor_id: str
    sensor_type: SensorType = SensorType.GENERIC
    location: str = ""

    # Value generation
    base_value: float = 25.0              # Baseline value
    noise_std: float = 0.5                # Gaussian noise standard deviation
    drift_rate: float = 0.0               # Value drift per second
    unit: str = ""

    # Valid range (for OUT_OF_RANGE detection)
    valid_min: Optional[float] = None
    valid_max: Optional[float] = None

    # Update behavior
    update_interval_ms: float = 1000.0    # How often this sensor updates

    # Fault injection hooks (set by FaultInjector)
    fault_override: Optional[Callable[[], Optional[float]]] = None
    quality_override: Optional[Callable[[], Optional[QualityStatus]]] = None


@dataclass
class TemperatureProfile(SensorProfile):
    """Profile for temperature sensors."""

    sensor_type: SensorType = field(default=SensorType.TEMPERATURE)
    base_value: float = 25.0
    noise_std: float = 0.5
    unit: str = "C"
    valid_min: float = -40.0
    valid_max: float = 200.0


@dataclass
class AirflowProfile(SensorProfile):
    """Profile for fume hood airflow sensors."""

    sensor_type: SensorType = field(default=SensorType.AIRFLOW)
    base_value: float = 0.5                # 0.5 m/s is typical safe airflow
    noise_std: float = 0.02
    unit: str = "m/s"
    valid_min: float = 0.0
    valid_max: float = 2.0


@dataclass
class PressureProfile(SensorProfile):
    """Profile for pressure sensors."""

    sensor_type: SensorType = field(default=SensorType.PRESSURE)
    base_value: float = 101.3              # Atmospheric pressure in kPa
    noise_std: float = 0.1
    unit: str = "kPa"
    valid_min: float = 0.0
    valid_max: float = 500.0


@dataclass
class MockSensorConfig(DriverConfig):
    """Configuration for MockSensorDriver."""

    sensors: list[SensorProfile] = field(default_factory=list)

    # Simulation behavior
    time_acceleration: float = 1.0         # Speed up time for testing
    start_time: Optional[datetime] = None  # Fixed start time (for replay)


class MockSensorDriver(SensorDriver):
    """
    Simulated sensor driver for testing.

    Generates realistic sensor readings with configurable noise,
    drift, and fault injection. Perfect for end-to-end testing
    without real hardware.
    """

    def __init__(self, config: MockSensorConfig):
        super().__init__(config)
        self.mock_config = config
        self._sensors: dict[str, SensorProfile] = {}
        self._sequence_nums: dict[str, int] = {}
        self._start_time = config.start_time or datetime.now(timezone.utc)
        self._current_values: dict[str, float] = {}

        # Initialize sensors
        for profile in config.sensors:
            self._sensors[profile.sensor_id] = profile
            self._sequence_nums[profile.sensor_id] = 0
            self._current_values[profile.sensor_id] = profile.base_value

    async def connect(self) -> bool:
        """Mock connection always succeeds."""
        self._connected = True
        return True

    async def disconnect(self) -> None:
        """Mock disconnection."""
        self._connected = False

    async def read(self) -> list[SensorEvent]:
        """Generate readings for all configured sensors."""
        events = []
        now = datetime.now(timezone.utc)

        for sensor_id, profile in self._sensors.items():
            event = self._generate_reading(sensor_id, profile, now)
            events.append(event)

        return events

    def _generate_reading(
        self,
        sensor_id: str,
        profile: SensorProfile,
        timestamp: datetime,
    ) -> SensorEvent:
        """Generate a single sensor reading."""
        # Calculate time-based drift
        elapsed_seconds = (timestamp - self._start_time).total_seconds()
        elapsed_seconds *= self.mock_config.time_acceleration
        drift = profile.drift_rate * elapsed_seconds

        # Get base value with drift
        base = profile.base_value + drift

        # Add noise
        noise = random.gauss(0, profile.noise_std) if profile.noise_std > 0 else 0

        # Check for fault override
        if profile.fault_override:
            override_value = profile.fault_override()
            if override_value is not None:
                value = override_value
            else:
                value = base + noise
        else:
            value = base + noise

        # Store current value for stuck detection
        self._current_values[sensor_id] = value

        # Determine quality
        quality_status = QualityStatus.OK
        confidence = 1.0

        if profile.quality_override:
            override_quality = profile.quality_override()
            if override_quality is not None:
                quality_status = override_quality
                confidence = 0.5 if quality_status != QualityStatus.OK else 1.0

        # Check range
        if profile.valid_min is not None and value < profile.valid_min:
            quality_status = QualityStatus.OUT_OF_RANGE
            confidence = 0.3
        elif profile.valid_max is not None and value > profile.valid_max:
            quality_status = QualityStatus.OUT_OF_RANGE
            confidence = 0.3

        # Increment sequence number
        self._sequence_nums[sensor_id] += 1

        return SensorEvent(
            ts=timestamp,
            sensor_id=sensor_id,
            sensor_type=profile.sensor_type,
            value=value,
            unit=profile.unit,
            quality=SensorQuality(status=quality_status, confidence=confidence),
            meta=SensorMeta(
                location=profile.location,
                source="mock",
                driver_id=self.driver_id,
                sequence_num=self._sequence_nums[sensor_id],
            ),
        )

    def add_sensor(self, profile: SensorProfile) -> None:
        """Add a sensor to this driver."""
        self._sensors[profile.sensor_id] = profile
        self._sequence_nums[profile.sensor_id] = 0
        self._current_values[profile.sensor_id] = profile.base_value

    def remove_sensor(self, sensor_id: str) -> None:
        """Remove a sensor from this driver."""
        self._sensors.pop(sensor_id, None)
        self._sequence_nums.pop(sensor_id, None)
        self._current_values.pop(sensor_id, None)

    def get_sensor_ids(self) -> list[str]:
        """Get list of sensor IDs managed by this driver."""
        return list(self._sensors.keys())

    def set_value(self, sensor_id: str, value: float) -> None:
        """Directly set a sensor's current value (for testing)."""
        if sensor_id in self._sensors:
            self._sensors[sensor_id].base_value = value

    def set_drift(self, sensor_id: str, drift_rate: float) -> None:
        """Set drift rate for a sensor (value change per second)."""
        if sensor_id in self._sensors:
            self._sensors[sensor_id].drift_rate = drift_rate

    def inject_fault(
        self,
        sensor_id: str,
        value_override: Optional[Callable[[], Optional[float]]] = None,
        quality_override: Optional[Callable[[], Optional[QualityStatus]]] = None,
    ) -> None:
        """Inject a fault into a sensor."""
        if sensor_id in self._sensors:
            self._sensors[sensor_id].fault_override = value_override
            self._sensors[sensor_id].quality_override = quality_override

    def clear_fault(self, sensor_id: str) -> None:
        """Clear any injected fault."""
        if sensor_id in self._sensors:
            self._sensors[sensor_id].fault_override = None
            self._sensors[sensor_id].quality_override = None


def create_lab_sensor_set(location_prefix: str = "SDL1") -> MockSensorConfig:
    """
    Create a standard set of lab sensors for testing.

    Returns a MockSensorConfig with P0 sensors:
    - Temperature (reactor, hotplate)
    - Pressure (reactor)
    - Airflow (fume hood)
    - E-stop status
    """
    sensors = [
        # Temperature sensors
        TemperatureProfile(
            sensor_id=f"{location_prefix}_reactor_temp",
            location=f"{location_prefix}_reactor",
            base_value=25.0,
            noise_std=0.3,
        ),
        TemperatureProfile(
            sensor_id=f"{location_prefix}_hotplate_temp",
            location=f"{location_prefix}_hotplate",
            base_value=25.0,
            noise_std=0.5,
        ),
        TemperatureProfile(
            sensor_id=f"{location_prefix}_chamber_temp",
            location=f"{location_prefix}_chamber",
            base_value=22.0,
            noise_std=0.2,
        ),
        # Pressure sensor
        PressureProfile(
            sensor_id=f"{location_prefix}_reactor_pressure",
            location=f"{location_prefix}_reactor",
            base_value=101.3,
            noise_std=0.1,
        ),
        # Airflow sensors (fume hood)
        AirflowProfile(
            sensor_id=f"{location_prefix}_hood_airflow",
            location=f"{location_prefix}_hood_A",
            base_value=0.5,
            noise_std=0.02,
        ),
        # E-stop (binary, no noise)
        SensorProfile(
            sensor_id=f"{location_prefix}_estop",
            sensor_type=SensorType.ESTOP,
            location=f"{location_prefix}_main",
            base_value=0.0,  # 0 = not triggered
            noise_std=0.0,
            unit="bool",
        ),
        # Power status
        SensorProfile(
            sensor_id=f"{location_prefix}_power",
            sensor_type=SensorType.POWER,
            location=f"{location_prefix}_main",
            base_value=1.0,  # 1 = power on
            noise_std=0.0,
            unit="bool",
        ),
    ]

    return MockSensorConfig(
        driver_id=f"{location_prefix}_mock_driver",
        sensors=sensors,
        poll_interval_ms=1000.0,
    )
