"""
StuckDetector - Detect sensors reporting the same value for too long.

A sensor is considered stuck if:
- The value hasn't changed for > stuck_threshold_ms
- And the value is not a valid constant (like a binary sensor in steady state)
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import math


@dataclass
class StuckConfig:
    """Configuration for stuck detection."""

    # How long same value = stuck (ms)
    threshold_ms: float = 30000.0

    # Minimum change to count as "different"
    epsilon: float = 0.001

    # Sensors that are allowed to be constant (e.g., binary sensors)
    constant_sensors: set[str] = None

    def __post_init__(self):
        if self.constant_sensors is None:
            self.constant_sensors = set()


@dataclass
class StuckState:
    """Tracked state for stuck detection."""

    sensor_id: str
    last_value: Optional[float] = None
    value_since: Optional[datetime] = None
    is_stuck: bool = False
    stuck_since: Optional[datetime] = None
    stuck_count: int = 0


class StuckDetector:
    """
    Detects stuck sensors (sensors reporting the same value for too long).

    This can indicate a frozen sensor, broken communication, or firmware bug.

    Usage:
        detector = StuckDetector()
        detector.update("temp_1", 25.0)
        # ... later ...
        is_stuck = detector.check("temp_1")
    """

    def __init__(self, config: Optional[StuckConfig] = None):
        self.config = config or StuckConfig()
        self._sensors: dict[str, StuckState] = {}

    def mark_as_constant(self, sensor_id: str) -> None:
        """Mark a sensor as allowed to be constant (e.g., binary sensor)."""
        self.config.constant_sensors.add(sensor_id)

    def update(
        self,
        sensor_id: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Update with a new value and return whether value changed.

        Returns True if the value is different from the previous value.
        """
        if sensor_id not in self._sensors:
            self._sensors[sensor_id] = StuckState(sensor_id=sensor_id)

        state = self._sensors[sensor_id]
        now = timestamp or datetime.now(timezone.utc)

        # Check if value changed
        value_changed = False
        if state.last_value is None:
            value_changed = True
        elif not math.isclose(value, state.last_value, abs_tol=self.config.epsilon):
            value_changed = True

        if value_changed:
            state.last_value = value
            state.value_since = now
            # Recover from stuck
            if state.is_stuck:
                state.is_stuck = False
                state.stuck_since = None

        return value_changed

    def check(
        self,
        sensor_id: str,
        now: Optional[datetime] = None,
    ) -> bool:
        """
        Check if a sensor is stuck.

        Returns True if the sensor value hasn't changed in too long.
        """
        # Constant sensors are never "stuck"
        if sensor_id in self.config.constant_sensors:
            return False

        if sensor_id not in self._sensors:
            return False  # Unknown sensor = not stuck (no data)

        state = self._sensors[sensor_id]
        now = now or datetime.now(timezone.utc)

        if state.value_since is None:
            return False  # No data yet

        # Calculate how long value has been the same
        duration_ms = (now - state.value_since).total_seconds() * 1000

        is_stuck = duration_ms > self.config.threshold_ms

        # Update state
        if is_stuck and not state.is_stuck:
            state.is_stuck = True
            state.stuck_since = now
            state.stuck_count += 1
        elif not is_stuck and state.is_stuck:
            state.is_stuck = False
            state.stuck_since = None

        return is_stuck

    def check_all(self, now: Optional[datetime] = None) -> dict[str, bool]:
        """Check stuck status for all sensors."""
        now = now or datetime.now(timezone.utc)
        return {
            sensor_id: self.check(sensor_id, now)
            for sensor_id in self._sensors
        }

    def get_stuck_sensors(self, now: Optional[datetime] = None) -> list[str]:
        """Get list of stuck sensor IDs."""
        statuses = self.check_all(now)
        return [sensor_id for sensor_id, is_stuck in statuses.items() if is_stuck]

    def get_state(self, sensor_id: str) -> Optional[StuckState]:
        """Get the stuck state for a sensor."""
        return self._sensors.get(sensor_id)

    def get_stuck_duration_ms(
        self,
        sensor_id: str,
        now: Optional[datetime] = None,
    ) -> Optional[float]:
        """Get how long the sensor has been at the same value."""
        state = self._sensors.get(sensor_id)
        if state is None or state.value_since is None:
            return None
        now = now or datetime.now(timezone.utc)
        return (now - state.value_since).total_seconds() * 1000
