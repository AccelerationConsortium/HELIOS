"""
SensorHealthEvent - Health status tracking for individual sensors.

The HealthMonitor component maintains these metrics for each sensor_id
and emits health events when status changes. SafetyChecker uses these
to make degraded-mode decisions.

Health checks:
- Stale: No reading for > 2x expected_period
- Stuck: Same value for too long (sensor frozen)
- Out of Range: Value outside valid bounds
- Drift: Gradual deviation from baseline
- Dropout: Communication failures
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid


class HealthStatus(str, Enum):
    """Overall health status of a sensor."""

    HEALTHY = "HEALTHY"             # Normal operation
    DEGRADED = "DEGRADED"           # Reduced confidence but usable
    UNHEALTHY = "UNHEALTHY"         # Should not be trusted
    OFFLINE = "OFFLINE"             # No communication
    UNKNOWN = "UNKNOWN"             # Just started, insufficient data


@dataclass
class HealthMetrics:
    """Detailed health metrics for a sensor."""

    # Timing metrics
    last_seen: Optional[datetime] = None
    expected_period_ms: float = 1000.0      # Expected update interval
    actual_period_ms: Optional[float] = None  # Measured update interval

    # Reliability metrics
    dropout_rate: float = 0.0               # 0.0 - 1.0, fraction of missed readings
    dropout_count: int = 0                  # Total missed readings
    total_readings: int = 0                 # Total readings received

    # Value metrics
    stuck_duration_ms: float = 0.0          # How long value hasn't changed
    stuck_threshold_ms: float = 30000.0     # When to consider "stuck"
    last_value: Optional[float] = None

    # Statistical metrics (optional, for drift detection)
    baseline_value: Optional[float] = None
    current_mean: Optional[float] = None
    current_variance: Optional[float] = None
    drift_from_baseline: Optional[float] = None

    # Range metrics
    valid_min: Optional[float] = None
    valid_max: Optional[float] = None
    out_of_range_count: int = 0

    @property
    def is_stale(self) -> bool:
        """Check if sensor is stale (no recent readings)."""
        if self.last_seen is None:
            return True
        age_ms = (datetime.now(timezone.utc) - self.last_seen).total_seconds() * 1000
        return age_ms > (self.expected_period_ms * 2)

    @property
    def is_stuck(self) -> bool:
        """Check if sensor value is stuck."""
        return self.stuck_duration_ms > self.stuck_threshold_ms

    @property
    def is_out_of_range(self) -> bool:
        """Check if last value was out of valid range."""
        if self.last_value is None:
            return False
        if self.valid_min is not None and self.last_value < self.valid_min:
            return True
        if self.valid_max is not None and self.last_value > self.valid_max:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "expected_period_ms": self.expected_period_ms,
            "actual_period_ms": self.actual_period_ms,
            "dropout_rate": self.dropout_rate,
            "dropout_count": self.dropout_count,
            "total_readings": self.total_readings,
            "stuck_duration_ms": self.stuck_duration_ms,
            "stuck_threshold_ms": self.stuck_threshold_ms,
            "last_value": self.last_value,
            "baseline_value": self.baseline_value,
            "current_mean": self.current_mean,
            "current_variance": self.current_variance,
            "drift_from_baseline": self.drift_from_baseline,
            "valid_min": self.valid_min,
            "valid_max": self.valid_max,
            "out_of_range_count": self.out_of_range_count,
            "is_stale": self.is_stale,
            "is_stuck": self.is_stuck,
            "is_out_of_range": self.is_out_of_range,
        }


@dataclass(frozen=True)
class SensorHealthEvent:
    """
    Health status event for a sensor, emitted by HealthMonitor.

    These events are consumed by SafetyChecker to make degraded-mode
    decisions. For example, if a critical sensor is OFFLINE, the
    system may need to block high-risk operations.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    sensor_id: str = ""
    status: HealthStatus = HealthStatus.UNKNOWN
    previous_status: Optional[HealthStatus] = None

    # Why did status change?
    reason: str = ""

    # Detailed metrics at time of event
    metrics: HealthMetrics = field(default_factory=HealthMetrics)

    def __post_init__(self):
        """Ensure ts is timezone-aware."""
        if self.ts.tzinfo is None:
            object.__setattr__(self, 'ts', self.ts.replace(tzinfo=timezone.utc))

    @property
    def is_status_change(self) -> bool:
        """Check if this event represents a status change."""
        return self.previous_status is not None and self.previous_status != self.status

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "event_id": self.event_id,
            "ts": self.ts.isoformat(),
            "sensor_id": self.sensor_id,
            "status": self.status.value,
            "previous_status": self.previous_status.value if self.previous_status else None,
            "reason": self.reason,
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SensorHealthEvent":
        """Create from dictionary."""
        metrics_data = data.get("metrics", {})
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            ts=datetime.fromisoformat(data["ts"]) if isinstance(data.get("ts"), str) else datetime.now(timezone.utc),
            sensor_id=data.get("sensor_id", ""),
            status=HealthStatus(data.get("status", "UNKNOWN")),
            previous_status=HealthStatus(data["previous_status"]) if data.get("previous_status") else None,
            reason=data.get("reason", ""),
            metrics=HealthMetrics(
                last_seen=datetime.fromisoformat(metrics_data["last_seen"]) if metrics_data.get("last_seen") else None,
                expected_period_ms=metrics_data.get("expected_period_ms", 1000.0),
                dropout_rate=metrics_data.get("dropout_rate", 0.0),
                total_readings=metrics_data.get("total_readings", 0),
            ),
        )
