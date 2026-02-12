"""
OutOfRangeDetector - Detect sensor values outside valid bounds.

A reading is out of range if:
- value < valid_min OR value > valid_max
- This could indicate sensor malfunction or dangerous condition
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class RangeBounds:
    """Valid range bounds for a sensor."""

    sensor_id: str
    valid_min: Optional[float] = None
    valid_max: Optional[float] = None

    # Soft limits for warnings (before hard limits)
    warn_min: Optional[float] = None
    warn_max: Optional[float] = None


@dataclass
class RangeState:
    """Tracked state for range detection."""

    sensor_id: str
    last_value: Optional[float] = None
    last_check: Optional[datetime] = None
    is_out_of_range: bool = False
    is_in_warning: bool = False
    out_of_range_since: Optional[datetime] = None
    out_of_range_count: int = 0


class OutOfRangeDetector:
    """
    Detects sensor values outside their valid range.

    Usage:
        detector = OutOfRangeDetector()
        detector.set_bounds("temp_1", valid_min=-40, valid_max=200)
        result = detector.check("temp_1", 250.0)  # True - out of range
    """

    def __init__(self):
        self._bounds: dict[str, RangeBounds] = {}
        self._states: dict[str, RangeState] = {}

    def set_bounds(
        self,
        sensor_id: str,
        valid_min: Optional[float] = None,
        valid_max: Optional[float] = None,
        warn_min: Optional[float] = None,
        warn_max: Optional[float] = None,
    ) -> None:
        """Set valid range bounds for a sensor."""
        self._bounds[sensor_id] = RangeBounds(
            sensor_id=sensor_id,
            valid_min=valid_min,
            valid_max=valid_max,
            warn_min=warn_min,
            warn_max=warn_max,
        )
        if sensor_id not in self._states:
            self._states[sensor_id] = RangeState(sensor_id=sensor_id)

    def check(
        self,
        sensor_id: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Check if a value is out of range.

        Returns True if value is outside valid bounds.
        """
        if sensor_id not in self._bounds:
            return False  # No bounds set = in range

        bounds = self._bounds[sensor_id]
        now = timestamp or datetime.now(timezone.utc)

        # Ensure state exists
        if sensor_id not in self._states:
            self._states[sensor_id] = RangeState(sensor_id=sensor_id)
        state = self._states[sensor_id]
        state.last_value = value
        state.last_check = now

        # Check hard limits
        out_of_range = False
        if bounds.valid_min is not None and value < bounds.valid_min:
            out_of_range = True
        if bounds.valid_max is not None and value > bounds.valid_max:
            out_of_range = True

        # Check soft limits (warnings)
        in_warning = False
        if not out_of_range:
            if bounds.warn_min is not None and value < bounds.warn_min:
                in_warning = True
            if bounds.warn_max is not None and value > bounds.warn_max:
                in_warning = True

        # Update state
        if out_of_range and not state.is_out_of_range:
            state.is_out_of_range = True
            state.out_of_range_since = now
            state.out_of_range_count += 1
        elif not out_of_range and state.is_out_of_range:
            state.is_out_of_range = False
            state.out_of_range_since = None

        state.is_in_warning = in_warning

        return out_of_range

    def is_in_warning(
        self,
        sensor_id: str,
        value: float,
    ) -> bool:
        """Check if value is in warning range (but not out of range)."""
        if sensor_id not in self._bounds:
            return False

        bounds = self._bounds[sensor_id]

        # First check if out of range (not a warning)
        if bounds.valid_min is not None and value < bounds.valid_min:
            return False
        if bounds.valid_max is not None and value > bounds.valid_max:
            return False

        # Check warning limits
        if bounds.warn_min is not None and value < bounds.warn_min:
            return True
        if bounds.warn_max is not None and value > bounds.warn_max:
            return True

        return False

    def get_out_of_range_sensors(self) -> list[str]:
        """Get list of sensors currently out of range."""
        return [
            sensor_id
            for sensor_id, state in self._states.items()
            if state.is_out_of_range
        ]

    def get_warning_sensors(self) -> list[str]:
        """Get list of sensors currently in warning range."""
        return [
            sensor_id
            for sensor_id, state in self._states.items()
            if state.is_in_warning and not state.is_out_of_range
        ]

    def get_state(self, sensor_id: str) -> Optional[RangeState]:
        """Get the range state for a sensor."""
        return self._states.get(sensor_id)

    def get_bounds(self, sensor_id: str) -> Optional[RangeBounds]:
        """Get the bounds for a sensor."""
        return self._bounds.get(sensor_id)

    def get_violation_details(
        self,
        sensor_id: str,
        value: float,
    ) -> dict:
        """Get details about a range violation."""
        bounds = self._bounds.get(sensor_id)
        if bounds is None:
            return {"in_range": True}

        result = {
            "in_range": True,
            "value": value,
            "valid_min": bounds.valid_min,
            "valid_max": bounds.valid_max,
        }

        if bounds.valid_min is not None and value < bounds.valid_min:
            result["in_range"] = False
            result["violation"] = "below_min"
            result["violation_amount"] = bounds.valid_min - value
        elif bounds.valid_max is not None and value > bounds.valid_max:
            result["in_range"] = False
            result["violation"] = "above_max"
            result["violation_amount"] = value - bounds.valid_max

        return result
