"""
HealthMonitor - Central health monitoring for all sensors.

Combines multiple detectors to produce a unified health status
for each sensor. Emits SensorHealthEvents when status changes.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional
from enum import Enum

from exp_agent.sensing.protocol.sensor_event import SensorEvent, SensorType
from exp_agent.sensing.protocol.health_event import (
    SensorHealthEvent,
    HealthStatus,
    HealthMetrics,
)
from exp_agent.sensing.health.detectors.stale import StaleDetector, StaleConfig
from exp_agent.sensing.health.detectors.stuck import StuckDetector, StuckConfig
from exp_agent.sensing.health.detectors.out_of_range import OutOfRangeDetector


@dataclass
class SensorHealthConfig:
    """Per-sensor health monitoring configuration."""

    sensor_id: str
    sensor_type: SensorType = SensorType.GENERIC

    # Expected update interval
    expected_period_ms: float = 1000.0

    # Valid range
    valid_min: Optional[float] = None
    valid_max: Optional[float] = None
    warn_min: Optional[float] = None
    warn_max: Optional[float] = None

    # Stuck detection
    stuck_threshold_ms: float = 30000.0
    is_constant_sensor: bool = False  # Binary sensors, etc.


@dataclass
class HealthMonitorConfig:
    """Global health monitor configuration."""

    # Stale detection
    stale_period_multiplier: float = 2.0
    stale_absolute_threshold_ms: float = 30000.0

    # Stuck detection
    default_stuck_threshold_ms: float = 30000.0

    # Check interval (for background monitoring)
    check_interval_ms: float = 1000.0


class HealthMonitor:
    """
    Central health monitor that tracks all sensor health.

    Integrates multiple detectors:
    - Stale: No readings for too long
    - Stuck: Same value for too long
    - Out of Range: Value outside valid bounds

    Produces SensorHealthEvents when status changes.
    """

    def __init__(self, config: Optional[HealthMonitorConfig] = None):
        self.config = config or HealthMonitorConfig()

        # Detectors
        self._stale_detector = StaleDetector(StaleConfig(
            period_multiplier=self.config.stale_period_multiplier,
            absolute_threshold_ms=self.config.stale_absolute_threshold_ms,
        ))
        self._stuck_detector = StuckDetector(StuckConfig(
            threshold_ms=self.config.default_stuck_threshold_ms,
        ))
        self._range_detector = OutOfRangeDetector()

        # Per-sensor config
        self._sensor_configs: dict[str, SensorHealthConfig] = {}

        # Current health status per sensor
        self._health_status: dict[str, HealthStatus] = {}
        self._health_metrics: dict[str, HealthMetrics] = {}

        # Callbacks
        self._health_callbacks: list[Callable[[SensorHealthEvent], None]] = []

    def register_sensor(self, config: SensorHealthConfig) -> None:
        """Register a sensor for health monitoring."""
        self._sensor_configs[config.sensor_id] = config

        # Configure stale detector
        self._stale_detector.register_sensor(
            config.sensor_id,
            expected_period_ms=config.expected_period_ms,
        )

        # Configure stuck detector
        if config.is_constant_sensor:
            self._stuck_detector.mark_as_constant(config.sensor_id)

        # Configure range detector
        if config.valid_min is not None or config.valid_max is not None:
            self._range_detector.set_bounds(
                config.sensor_id,
                valid_min=config.valid_min,
                valid_max=config.valid_max,
                warn_min=config.warn_min,
                warn_max=config.warn_max,
            )

        # Initialize health status
        self._health_status[config.sensor_id] = HealthStatus.UNKNOWN
        self._health_metrics[config.sensor_id] = HealthMetrics(
            expected_period_ms=config.expected_period_ms,
            valid_min=config.valid_min,
            valid_max=config.valid_max,
            stuck_threshold_ms=config.stuck_threshold_ms,
        )

    def set_health_callback(
        self,
        callback: Callable[[SensorHealthEvent], None],
    ) -> None:
        """Add a callback for health status changes."""
        self._health_callbacks.append(callback)

    def process_event(self, event: SensorEvent) -> Optional[SensorHealthEvent]:
        """
        Process a sensor event and update health status.

        Returns a SensorHealthEvent if status changed, None otherwise.
        """
        sensor_id = event.sensor_id

        # Auto-register if not known
        if sensor_id not in self._sensor_configs:
            self.register_sensor(SensorHealthConfig(
                sensor_id=sensor_id,
                sensor_type=event.sensor_type,
            ))

        now = event.ts
        metrics = self._health_metrics[sensor_id]

        # Update stale detector
        self._stale_detector.update(sensor_id, now)

        # Update stuck detector
        self._stuck_detector.update(sensor_id, event.value, now)

        # Check range
        is_out_of_range = self._range_detector.check(sensor_id, event.value, now)

        # Update metrics
        metrics.last_seen = now
        metrics.last_value = event.value
        metrics.total_readings += 1

        # Calculate health status
        previous_status = self._health_status.get(sensor_id, HealthStatus.UNKNOWN)
        new_status = self._calculate_status(sensor_id, now)

        # Update stored status
        self._health_status[sensor_id] = new_status

        # Emit event if status changed
        if new_status != previous_status:
            reason = self._get_status_reason(sensor_id, new_status, now)
            health_event = SensorHealthEvent(
                ts=now,
                sensor_id=sensor_id,
                status=new_status,
                previous_status=previous_status,
                reason=reason,
                metrics=metrics,
            )

            # Notify callbacks
            for callback in self._health_callbacks:
                try:
                    callback(health_event)
                except Exception as e:
                    print(f"Error in health callback: {e}")

            return health_event

        return None

    def _calculate_status(
        self,
        sensor_id: str,
        now: datetime,
    ) -> HealthStatus:
        """Calculate the overall health status for a sensor."""
        is_stale = self._stale_detector.check(sensor_id, now)
        is_stuck = self._stuck_detector.check(sensor_id, now)

        state = self._range_detector.get_state(sensor_id)
        is_out_of_range = state.is_out_of_range if state else False
        is_in_warning = state.is_in_warning if state else False

        # Priority: OFFLINE > UNHEALTHY > DEGRADED > HEALTHY
        if is_stale:
            return HealthStatus.OFFLINE

        if is_out_of_range or is_stuck:
            return HealthStatus.UNHEALTHY

        if is_in_warning:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def _get_status_reason(
        self,
        sensor_id: str,
        status: HealthStatus,
        now: datetime,
    ) -> str:
        """Get a human-readable reason for the status."""
        reasons = []

        if self._stale_detector.check(sensor_id, now):
            age_ms = self._stale_detector.get_age_ms(sensor_id, now)
            reasons.append(f"stale ({age_ms:.0f}ms since last reading)")

        if self._stuck_detector.check(sensor_id, now):
            duration_ms = self._stuck_detector.get_stuck_duration_ms(sensor_id, now)
            reasons.append(f"stuck at same value for {duration_ms:.0f}ms")

        state = self._range_detector.get_state(sensor_id)
        if state and state.is_out_of_range:
            bounds = self._range_detector.get_bounds(sensor_id)
            if bounds:
                reasons.append(f"value {state.last_value} outside range [{bounds.valid_min}, {bounds.valid_max}]")

        if not reasons:
            if status == HealthStatus.HEALTHY:
                return "all checks passed"
            return "unknown"

        return "; ".join(reasons)

    def check_health(
        self,
        sensor_id: str,
        now: Optional[datetime] = None,
    ) -> HealthStatus:
        """Get current health status for a sensor."""
        if sensor_id not in self._health_status:
            return HealthStatus.UNKNOWN
        now = now or datetime.now(timezone.utc)
        return self._calculate_status(sensor_id, now)

    def check_all_health(
        self,
        now: Optional[datetime] = None,
    ) -> dict[str, HealthStatus]:
        """Get health status for all sensors."""
        now = now or datetime.now(timezone.utc)
        return {
            sensor_id: self._calculate_status(sensor_id, now)
            for sensor_id in self._sensor_configs
        }

    def get_unhealthy_sensors(
        self,
        now: Optional[datetime] = None,
    ) -> list[str]:
        """Get list of unhealthy sensor IDs."""
        statuses = self.check_all_health(now)
        return [
            sensor_id
            for sensor_id, status in statuses.items()
            if status in (HealthStatus.UNHEALTHY, HealthStatus.OFFLINE)
        ]

    def get_metrics(self, sensor_id: str) -> Optional[HealthMetrics]:
        """Get health metrics for a sensor."""
        return self._health_metrics.get(sensor_id)

    def get_all_metrics(self) -> dict[str, HealthMetrics]:
        """Get health metrics for all sensors."""
        return dict(self._health_metrics)

    def get_summary(self, now: Optional[datetime] = None) -> dict:
        """Get a summary of all sensor health."""
        now = now or datetime.now(timezone.utc)
        statuses = self.check_all_health(now)

        return {
            "total_sensors": len(statuses),
            "healthy": sum(1 for s in statuses.values() if s == HealthStatus.HEALTHY),
            "degraded": sum(1 for s in statuses.values() if s == HealthStatus.DEGRADED),
            "unhealthy": sum(1 for s in statuses.values() if s == HealthStatus.UNHEALTHY),
            "offline": sum(1 for s in statuses.values() if s == HealthStatus.OFFLINE),
            "unknown": sum(1 for s in statuses.values() if s == HealthStatus.UNKNOWN),
            "sensors": {
                sensor_id: {
                    "status": status.value,
                    "metrics": self._health_metrics[sensor_id].to_dict()
                    if sensor_id in self._health_metrics else None,
                }
                for sensor_id, status in statuses.items()
            },
        }
