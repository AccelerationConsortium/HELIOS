"""
StaleDetector - Detect sensors that haven't reported in too long.

A sensor is considered stale if:
- No reading for > 2x expected_period
- Or no reading for > absolute_threshold (e.g., 30 seconds)
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class StaleConfig:
    """Configuration for stale detection."""

    # Multiplier of expected period to consider stale
    period_multiplier: float = 2.0

    # Absolute maximum time without reading (ms)
    absolute_threshold_ms: float = 30000.0

    # Grace period after startup (ms)
    startup_grace_ms: float = 5000.0


@dataclass
class StalenessState:
    """Tracked state for staleness detection."""

    sensor_id: str
    expected_period_ms: float = 1000.0
    last_seen: Optional[datetime] = None
    first_seen: Optional[datetime] = None
    is_stale: bool = False
    stale_since: Optional[datetime] = None
    stale_count: int = 0  # How many times this sensor went stale


class StaleDetector:
    """
    Detects stale sensors (sensors that haven't reported recently).

    Usage:
        detector = StaleDetector()
        detector.register_sensor("temp_1", expected_period_ms=1000)
        detector.update("temp_1", now)
        is_stale = detector.check("temp_1")
    """

    def __init__(self, config: Optional[StaleConfig] = None):
        self.config = config or StaleConfig()
        self._sensors: dict[str, StalenessState] = {}

    def register_sensor(
        self,
        sensor_id: str,
        expected_period_ms: float = 1000.0,
    ) -> None:
        """Register a sensor for stale monitoring."""
        self._sensors[sensor_id] = StalenessState(
            sensor_id=sensor_id,
            expected_period_ms=expected_period_ms,
        )

    def update(self, sensor_id: str, timestamp: Optional[datetime] = None) -> None:
        """Record that a sensor reported at the given time."""
        if sensor_id not in self._sensors:
            self.register_sensor(sensor_id)

        state = self._sensors[sensor_id]
        now = timestamp or datetime.now(timezone.utc)

        if state.first_seen is None:
            state.first_seen = now

        # Check if we were stale and are now recovered
        if state.is_stale:
            state.is_stale = False
            state.stale_since = None

        state.last_seen = now

    def check(self, sensor_id: str, now: Optional[datetime] = None) -> bool:
        """
        Check if a sensor is stale.

        Returns True if the sensor hasn't reported in too long.
        """
        if sensor_id not in self._sensors:
            return True  # Unknown sensor = stale

        state = self._sensors[sensor_id]
        now = now or datetime.now(timezone.utc)

        # Not seen yet
        if state.last_seen is None:
            # Check if still in startup grace period
            if state.first_seen is None:
                return True
            elapsed = (now - state.first_seen).total_seconds() * 1000
            return elapsed > self.config.startup_grace_ms

        # Calculate age
        age_ms = (now - state.last_seen).total_seconds() * 1000

        # Check against thresholds
        period_threshold = state.expected_period_ms * self.config.period_multiplier
        is_stale = age_ms > period_threshold or age_ms > self.config.absolute_threshold_ms

        # Update state
        if is_stale and not state.is_stale:
            state.is_stale = True
            state.stale_since = now
            state.stale_count += 1
        elif not is_stale and state.is_stale:
            state.is_stale = False
            state.stale_since = None

        return is_stale

    def check_all(self, now: Optional[datetime] = None) -> dict[str, bool]:
        """Check staleness for all registered sensors."""
        now = now or datetime.now(timezone.utc)
        return {
            sensor_id: self.check(sensor_id, now)
            for sensor_id in self._sensors
        }

    def get_stale_sensors(self, now: Optional[datetime] = None) -> list[str]:
        """Get list of stale sensor IDs."""
        statuses = self.check_all(now)
        return [sensor_id for sensor_id, is_stale in statuses.items() if is_stale]

    def get_state(self, sensor_id: str) -> Optional[StalenessState]:
        """Get the staleness state for a sensor."""
        return self._sensors.get(sensor_id)

    def get_age_ms(self, sensor_id: str, now: Optional[datetime] = None) -> Optional[float]:
        """Get how long since the sensor last reported."""
        state = self._sensors.get(sensor_id)
        if state is None or state.last_seen is None:
            return None
        now = now or datetime.now(timezone.utc)
        return (now - state.last_seen).total_seconds() * 1000
