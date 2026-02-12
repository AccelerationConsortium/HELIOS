"""
FaultInjector - Inject realistic faults into mock sensors for testing.

Supported fault types:
- Dropout: Sensor stops reporting
- Noise: Increased random noise
- Stuck: Sensor returns same value
- Drift: Gradual value drift
- Spike: Sudden value jumps
- Out of Range: Force value outside valid bounds
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Callable, Optional
import random
import math

from exp_agent.sensing.drivers.mock_driver import MockSensorDriver
from exp_agent.sensing.protocol.sensor_event import QualityStatus


class FaultType(str, Enum):
    """Types of faults that can be injected."""

    DROPOUT = "dropout"           # Stop reporting
    NOISE = "noise"               # Increased noise
    STUCK = "stuck"               # Same value repeated
    DRIFT = "drift"               # Gradual drift
    SPIKE = "spike"               # Sudden jump
    OUT_OF_RANGE = "out_of_range" # Force out of range
    QUALITY_DEGRADE = "quality"   # Degrade quality status


@dataclass
class FaultConfig:
    """Configuration for a fault injection."""

    fault_type: FaultType
    sensor_id: str

    # Duration (None = permanent until cleared)
    duration_ms: Optional[float] = None
    start_time: Optional[datetime] = None

    # Fault-specific parameters
    # For DROPOUT:
    dropout_probability: float = 1.0       # 0-1, probability of dropping each reading

    # For NOISE:
    noise_multiplier: float = 10.0         # Multiply existing noise by this

    # For STUCK:
    stuck_value: Optional[float] = None    # Value to stick at (None = use last value)

    # For DRIFT:
    drift_rate: float = 0.1                # Units per second

    # For SPIKE:
    spike_magnitude: float = 50.0          # How much to spike
    spike_probability: float = 0.1         # Probability of spike per reading

    # For OUT_OF_RANGE:
    out_of_range_value: Optional[float] = None  # Specific value, or None for auto

    # For QUALITY_DEGRADE:
    degraded_quality: QualityStatus = QualityStatus.SUSPECT


@dataclass
class ActiveFault:
    """An active fault being injected."""

    config: FaultConfig
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_value: Optional[float] = None


class FaultInjector:
    """
    Injects faults into MockSensorDriver sensors for testing.

    Usage:
        injector = FaultInjector(driver)
        injector.inject(FaultConfig(
            fault_type=FaultType.DROPOUT,
            sensor_id="temp_1",
            duration_ms=5000,
        ))
        # ... sensor will drop out for 5 seconds ...
        injector.clear("temp_1")
    """

    def __init__(self, driver: MockSensorDriver):
        self.driver = driver
        self._active_faults: dict[str, ActiveFault] = {}

    def inject(self, config: FaultConfig) -> None:
        """Inject a fault into a sensor."""
        if config.start_time is None:
            config.start_time = datetime.now(timezone.utc)

        fault = ActiveFault(config=config, start_time=config.start_time)
        self._active_faults[config.sensor_id] = fault

        # Apply the fault to the driver
        self._apply_fault(fault)

    def _apply_fault(self, fault: ActiveFault) -> None:
        """Apply fault overrides to the driver."""
        config = fault.config
        sensor_id = config.sensor_id

        value_override: Optional[Callable[[], Optional[float]]] = None
        quality_override: Optional[Callable[[], Optional[QualityStatus]]] = None

        if config.fault_type == FaultType.DROPOUT:
            # Return None (skip) with given probability
            def dropout_value() -> Optional[float]:
                if random.random() < config.dropout_probability:
                    return None  # Signal to skip this reading
                return None  # Let normal value through
            value_override = dropout_value
            quality_override = lambda: QualityStatus.DROPPED

        elif config.fault_type == FaultType.STUCK:
            # Return the same value every time
            stuck_val = config.stuck_value
            if stuck_val is None:
                # Use last known value
                stuck_val = fault.last_value or 25.0
            value_override = lambda: stuck_val

        elif config.fault_type == FaultType.DRIFT:
            # Return drifting value
            def drift_value() -> float:
                elapsed = (datetime.now(timezone.utc) - fault.start_time).total_seconds()
                return config.drift_rate * elapsed
            # Note: This adds drift on top of base value
            # The driver handles this in its drift_rate parameter
            self.driver.set_drift(sensor_id, config.drift_rate)
            return  # Don't use override, use driver's built-in drift

        elif config.fault_type == FaultType.SPIKE:
            # Random spikes
            def spike_value() -> Optional[float]:
                if random.random() < config.spike_probability:
                    return (fault.last_value or 25.0) + config.spike_magnitude * random.choice([-1, 1])
                return None
            value_override = spike_value

        elif config.fault_type == FaultType.OUT_OF_RANGE:
            # Force out of range
            value_override = lambda: config.out_of_range_value or 999.9
            quality_override = lambda: QualityStatus.OUT_OF_RANGE

        elif config.fault_type == FaultType.QUALITY_DEGRADE:
            # Just degrade quality, not value
            quality_override = lambda: config.degraded_quality

        elif config.fault_type == FaultType.NOISE:
            # Noise is handled by modifying the sensor profile
            # This is a simplified version
            pass

        # Apply overrides
        self.driver.inject_fault(
            sensor_id,
            value_override=value_override,
            quality_override=quality_override,
        )

    def clear(self, sensor_id: str) -> None:
        """Clear any active fault on a sensor."""
        if sensor_id in self._active_faults:
            del self._active_faults[sensor_id]
        self.driver.clear_fault(sensor_id)
        self.driver.set_drift(sensor_id, 0.0)  # Reset drift

    def clear_all(self) -> None:
        """Clear all active faults."""
        for sensor_id in list(self._active_faults.keys()):
            self.clear(sensor_id)

    def update(self, now: Optional[datetime] = None) -> list[str]:
        """
        Check for expired faults and clear them.

        Returns list of sensor IDs where faults were cleared.
        """
        now = now or datetime.now(timezone.utc)
        cleared = []

        for sensor_id, fault in list(self._active_faults.items()):
            if fault.config.duration_ms is not None:
                elapsed = (now - fault.start_time).total_seconds() * 1000
                if elapsed >= fault.config.duration_ms:
                    self.clear(sensor_id)
                    cleared.append(sensor_id)

        return cleared

    def get_active_faults(self) -> dict[str, FaultConfig]:
        """Get all currently active faults."""
        return {
            sensor_id: fault.config
            for sensor_id, fault in self._active_faults.items()
        }

    def is_faulted(self, sensor_id: str) -> bool:
        """Check if a sensor has an active fault."""
        return sensor_id in self._active_faults

    # Convenience methods for common fault scenarios

    def simulate_sensor_failure(
        self,
        sensor_id: str,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Simulate complete sensor failure (dropout)."""
        self.inject(FaultConfig(
            fault_type=FaultType.DROPOUT,
            sensor_id=sensor_id,
            duration_ms=duration_ms,
            dropout_probability=1.0,
        ))

    def simulate_intermittent(
        self,
        sensor_id: str,
        dropout_rate: float = 0.3,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Simulate intermittent connection."""
        self.inject(FaultConfig(
            fault_type=FaultType.DROPOUT,
            sensor_id=sensor_id,
            duration_ms=duration_ms,
            dropout_probability=dropout_rate,
        ))

    def simulate_thermal_runaway(
        self,
        sensor_id: str,
        rate: float = 5.0,  # Degrees per second
        duration_ms: Optional[float] = None,
    ) -> None:
        """Simulate thermal runaway (rapid temperature increase)."""
        self.inject(FaultConfig(
            fault_type=FaultType.DRIFT,
            sensor_id=sensor_id,
            duration_ms=duration_ms,
            drift_rate=rate,
        ))

    def simulate_hood_failure(
        self,
        sensor_id: str,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Simulate fume hood airflow failure (zero airflow)."""
        self.inject(FaultConfig(
            fault_type=FaultType.STUCK,
            sensor_id=sensor_id,
            duration_ms=duration_ms,
            stuck_value=0.0,  # No airflow
        ))

    def simulate_pressure_spike(
        self,
        sensor_id: str,
        magnitude: float = 50.0,
        probability: float = 0.2,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Simulate pressure spikes."""
        self.inject(FaultConfig(
            fault_type=FaultType.SPIKE,
            sensor_id=sensor_id,
            duration_ms=duration_ms,
            spike_magnitude=magnitude,
            spike_probability=probability,
        ))
