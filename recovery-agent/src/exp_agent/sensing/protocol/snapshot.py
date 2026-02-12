"""
Snapshot - Point-in-time view of sensor state for decision layers.

The Snapshot API provides a "current state panel" that decision layers
(SafetyChecker, RecoveryAgent, LLMAdvisor) can pull to understand the
current state of the system.

Key features:
- Latest value + health for each sensor
- Recent window for trend analysis
- Aggregated system-level health
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from enum import Enum

from exp_agent.sensing.protocol.sensor_event import SensorEvent, SensorType, QualityStatus
from exp_agent.sensing.protocol.health_event import HealthStatus, HealthMetrics


class SystemHealthStatus(str, Enum):
    """Overall system health based on sensor states."""

    NOMINAL = "NOMINAL"             # All sensors healthy
    DEGRADED = "DEGRADED"           # Some sensors degraded but operational
    CRITICAL = "CRITICAL"           # Critical sensors unhealthy
    EMERGENCY = "EMERGENCY"         # Safety-critical failure detected
    UNKNOWN = "UNKNOWN"             # Insufficient sensor data


@dataclass
class SensorSnapshot:
    """
    Point-in-time snapshot of a single sensor's state.

    Provides both the latest reading and health status, plus a
    recent history window for trend analysis.
    """

    sensor_id: str
    sensor_type: SensorType

    # Latest reading
    latest_event: Optional[SensorEvent] = None
    latest_value: Optional[float] = None
    latest_unit: str = ""
    latest_quality: QualityStatus = QualityStatus.UNKNOWN

    # Health status
    health_status: HealthStatus = HealthStatus.UNKNOWN
    health_metrics: HealthMetrics = field(default_factory=HealthMetrics)

    # Recent history (last N seconds, for trend)
    recent_values: list[float] = field(default_factory=list)
    recent_timestamps: list[datetime] = field(default_factory=list)
    window_seconds: float = 60.0  # How much history to keep

    # Derived metrics
    trend_slope: Optional[float] = None       # Rate of change (units/sec)
    value_min: Optional[float] = None         # Min in window
    value_max: Optional[float] = None         # Max in window
    value_mean: Optional[float] = None        # Mean in window

    @property
    def is_healthy(self) -> bool:
        """Check if sensor is in healthy state."""
        return self.health_status == HealthStatus.HEALTHY

    @property
    def is_trustworthy(self) -> bool:
        """Check if sensor can be trusted for safety decisions."""
        return (
            self.health_status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
            and self.latest_quality in (QualityStatus.OK, QualityStatus.CALIBRATION_DUE)
        )

    @property
    def age_seconds(self) -> Optional[float]:
        """How old is the latest reading?"""
        if self.latest_event is None:
            return None
        return (datetime.now(timezone.utc) - self.latest_event.ts).total_seconds()

    def add_reading(self, event: SensorEvent, window_seconds: float = 60.0) -> None:
        """Add a new reading and update derived metrics."""
        self.latest_event = event
        self.latest_value = event.value
        self.latest_unit = event.unit
        self.latest_quality = event.quality.status

        # Infer health status from quality if not explicitly set
        if event.is_ok:
            self.health_status = HealthStatus.HEALTHY
        elif event.quality.status in (QualityStatus.STALE, QualityStatus.SUSPECT):
            self.health_status = HealthStatus.DEGRADED
        elif event.quality.status in (QualityStatus.DROPPED, QualityStatus.OUT_OF_RANGE):
            self.health_status = HealthStatus.UNHEALTHY
        else:
            self.health_status = HealthStatus.UNKNOWN

        # Add to history
        self.recent_values.append(event.value)
        self.recent_timestamps.append(event.ts)

        # Trim old readings outside window
        cutoff = datetime.now(timezone.utc).timestamp() - window_seconds
        while (
            self.recent_timestamps
            and self.recent_timestamps[0].timestamp() < cutoff
        ):
            self.recent_timestamps.pop(0)
            self.recent_values.pop(0)

        # Update derived metrics
        if self.recent_values:
            self.value_min = min(self.recent_values)
            self.value_max = max(self.recent_values)
            self.value_mean = sum(self.recent_values) / len(self.recent_values)

            # Calculate trend (simple linear regression)
            if len(self.recent_values) >= 2:
                n = len(self.recent_values)
                t0 = self.recent_timestamps[0].timestamp()
                times = [(ts.timestamp() - t0) for ts in self.recent_timestamps]
                mean_t = sum(times) / n
                mean_v = self.value_mean

                numerator = sum((times[i] - mean_t) * (self.recent_values[i] - mean_v) for i in range(n))
                denominator = sum((times[i] - mean_t) ** 2 for i in range(n))

                self.trend_slope = numerator / denominator if denominator != 0 else 0.0
            else:
                self.trend_slope = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "latest_value": self.latest_value,
            "latest_unit": self.latest_unit,
            "latest_quality": self.latest_quality.value,
            "latest_timestamp": self.latest_event.ts.isoformat() if self.latest_event else None,
            "age_seconds": self.age_seconds,
            "health_status": self.health_status.value,
            "health_metrics": self.health_metrics.to_dict(),
            "trend_slope": self.trend_slope,
            "value_min": self.value_min,
            "value_max": self.value_max,
            "value_mean": self.value_mean,
            "is_healthy": self.is_healthy,
            "is_trustworthy": self.is_trustworthy,
        }


@dataclass
class SystemSnapshot:
    """
    System-wide snapshot of all sensor states.

    This is what the decision layers consume to understand
    "what's happening in the lab right now".
    """

    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # All sensor snapshots indexed by sensor_id
    sensors: dict[str, SensorSnapshot] = field(default_factory=dict)

    # Aggregated status
    system_status: SystemHealthStatus = SystemHealthStatus.UNKNOWN

    # Counts by status
    healthy_count: int = 0
    degraded_count: int = 0
    unhealthy_count: int = 0
    offline_count: int = 0

    # Critical sensor status (P0 sensors)
    critical_sensors_ok: bool = True
    critical_sensor_issues: list[str] = field(default_factory=list)

    def add_sensor(self, snapshot: SensorSnapshot) -> None:
        """Add or update a sensor snapshot."""
        self.sensors[snapshot.sensor_id] = snapshot
        self._update_aggregates()

    def get_sensor(self, sensor_id: str) -> Optional[SensorSnapshot]:
        """Get snapshot for a specific sensor."""
        return self.sensors.get(sensor_id)

    def get_sensors_by_type(self, sensor_type: SensorType) -> list[SensorSnapshot]:
        """Get all sensors of a specific type."""
        return [s for s in self.sensors.values() if s.sensor_type == sensor_type]

    def get_value(self, sensor_id: str) -> Optional[float]:
        """Get latest value for a sensor."""
        snapshot = self.sensors.get(sensor_id)
        return snapshot.latest_value if snapshot else None

    def _update_aggregates(self) -> None:
        """Update aggregate counts and system status."""
        self.healthy_count = 0
        self.degraded_count = 0
        self.unhealthy_count = 0
        self.offline_count = 0
        self.critical_sensor_issues = []

        critical_types = {
            SensorType.TEMPERATURE,
            SensorType.PRESSURE,
            SensorType.AIRFLOW,
            SensorType.ESTOP,
            SensorType.POWER,
        }

        for sensor_id, snapshot in self.sensors.items():
            if snapshot.health_status == HealthStatus.HEALTHY:
                self.healthy_count += 1
            elif snapshot.health_status == HealthStatus.DEGRADED:
                self.degraded_count += 1
            elif snapshot.health_status == HealthStatus.UNHEALTHY:
                self.unhealthy_count += 1
            elif snapshot.health_status == HealthStatus.OFFLINE:
                self.offline_count += 1

            # Check critical sensors
            if snapshot.sensor_type in critical_types:
                if snapshot.health_status not in (HealthStatus.HEALTHY, HealthStatus.DEGRADED):
                    self.critical_sensor_issues.append(sensor_id)

        self.critical_sensors_ok = len(self.critical_sensor_issues) == 0

        # Determine system status
        if self.unhealthy_count > 0 or self.offline_count > 0:
            if not self.critical_sensors_ok:
                self.system_status = SystemHealthStatus.CRITICAL
            else:
                self.system_status = SystemHealthStatus.DEGRADED
        elif self.degraded_count > 0:
            self.system_status = SystemHealthStatus.DEGRADED
        elif self.healthy_count > 0:
            self.system_status = SystemHealthStatus.NOMINAL
        else:
            self.system_status = SystemHealthStatus.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "ts": self.ts.isoformat(),
            "system_status": self.system_status.value,
            "healthy_count": self.healthy_count,
            "degraded_count": self.degraded_count,
            "unhealthy_count": self.unhealthy_count,
            "offline_count": self.offline_count,
            "critical_sensors_ok": self.critical_sensors_ok,
            "critical_sensor_issues": self.critical_sensor_issues,
            "sensors": {k: v.to_dict() for k, v in self.sensors.items()},
        }

    @classmethod
    def from_sensor_events(cls, events: list[SensorEvent]) -> "SystemSnapshot":
        """Create a snapshot from a list of sensor events."""
        snapshot = cls()
        for event in events:
            if event.sensor_id not in snapshot.sensors:
                snapshot.sensors[event.sensor_id] = SensorSnapshot(
                    sensor_id=event.sensor_id,
                    sensor_type=event.sensor_type,
                )
            snapshot.sensors[event.sensor_id].add_reading(event)
            snapshot.sensors[event.sensor_id].health_status = (
                HealthStatus.HEALTHY if event.is_ok else HealthStatus.DEGRADED
            )
        snapshot._update_aggregates()
        return snapshot
