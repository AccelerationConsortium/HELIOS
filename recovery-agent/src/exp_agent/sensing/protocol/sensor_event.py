"""
SensorEvent - The core data structure for all sensor readings.

All inputs, regardless of source (serial, MQTT, Modbus, HTTP), are normalized
to this format. Upper layers (SafetyChecker, RecoveryAgent, LLMAdvisor) only
see this standardized format.

Design principles:
- Timestamp is always UTC ISO-8601
- Quality status enables degraded-mode decisions
- Meta preserves provenance for audit/replay
- Immutable after creation (frozen dataclass)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid


class QualityStatus(str, Enum):
    """Quality status of a sensor reading."""

    OK = "OK"                       # Normal reading
    STALE = "STALE"                 # Reading older than expected period
    DROPPED = "DROPPED"             # Communication failure, using last known
    OUT_OF_RANGE = "OUT_OF_RANGE"   # Value outside valid bounds
    CALIBRATION_DUE = "CALIBRATION_DUE"  # Sensor needs calibration
    SUSPECT = "SUSPECT"             # Anomaly detected, may be valid
    UNKNOWN = "UNKNOWN"             # Cannot determine quality


class SensorType(str, Enum):
    """Types of sensors supported by the sensing layer (P0/P1/P2 priority)."""

    # P0 - Critical safety sensors (must have)
    TEMPERATURE = "temperature"           # Reactor/hotplate/chamber temp
    PRESSURE = "pressure"                 # Closed system pressure
    AIRFLOW = "airflow"                   # Fume hood airflow (m/s)
    ESTOP = "estop"                       # Emergency stop status
    POWER = "power"                       # Device power/mains status

    # P1 - Important safety sensors
    GAS = "gas"                           # VOC/flammable/toxic gas
    LIQUID_LEVEL = "liquid_level"         # Tray/reservoir level
    LEAK = "leak"                         # Floor/tray leak detection
    DOOR = "door"                         # Door/access status

    # P2 - Enhanced monitoring
    CAMERA = "camera"                     # Visual detection events
    OCCUPANCY = "occupancy"               # Room/zone occupancy
    HUMIDITY = "humidity"                 # Environmental humidity
    VIBRATION = "vibration"               # Mechanical vibration

    # Generic
    GENERIC = "generic"                   # Unclassified sensor


@dataclass(frozen=True)
class SensorQuality:
    """Quality metadata for a sensor reading."""

    status: QualityStatus = QualityStatus.OK
    confidence: float = 1.0  # 0.0 - 1.0, how much to trust this reading

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            object.__setattr__(self, 'confidence', max(0.0, min(1.0, self.confidence)))


@dataclass(frozen=True)
class SensorMeta:
    """Provenance metadata for a sensor reading."""

    location: str                              # Physical location (e.g., "SDL1_hood_A")
    source: str                                # Protocol source: "modbus|mqtt|serial|http|mock"
    raw: Optional[str] = None                  # Raw value for debugging/audit
    driver_id: Optional[str] = None            # Which driver produced this
    sequence_num: Optional[int] = None         # For ordering/dedup

    @classmethod
    def mock(cls, location: str = "mock_location") -> "SensorMeta":
        """Create metadata for mock/test sensors."""
        return cls(location=location, source="mock")


@dataclass(frozen=True)
class SensorEvent:
    """
    Standard sensor event format consumed by all upper layers.

    This is the "core asset" of the sensing layer - all drivers must
    produce events in this format, and all consumers only depend on this format.

    Example:
        {
          "ts": "2026-02-05T14:20:11.123Z",
          "sensor_id": "hood_01_airflow",
          "type": "airflow",
          "value": 0.42,
          "unit": "m/s",
          "quality": {"status": "OK", "confidence": 1.0},
          "meta": {"location": "SDL1_hood_A", "source": "modbus"}
        }
    """

    # Unique identifier for this event
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Timestamp in UTC ISO-8601 format
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Sensor identification
    sensor_id: str = ""                        # Unique sensor identifier
    sensor_type: SensorType = SensorType.GENERIC

    # Reading value
    value: float = 0.0                         # Normalized numeric value
    unit: str = ""                             # SI unit or standard unit

    # Quality and provenance
    quality: SensorQuality = field(default_factory=SensorQuality)
    meta: SensorMeta = field(default_factory=lambda: SensorMeta.mock())

    def __post_init__(self):
        """Ensure ts is timezone-aware."""
        if self.ts.tzinfo is None:
            object.__setattr__(self, 'ts', self.ts.replace(tzinfo=timezone.utc))

    @property
    def is_ok(self) -> bool:
        """Check if the reading is normal quality."""
        return self.quality.status == QualityStatus.OK

    @property
    def is_stale(self) -> bool:
        """Check if the reading is stale."""
        return self.quality.status == QualityStatus.STALE

    @property
    def is_trustworthy(self) -> bool:
        """Check if the reading can be trusted for safety decisions."""
        return (
            self.quality.status in (QualityStatus.OK, QualityStatus.CALIBRATION_DUE)
            and self.quality.confidence >= 0.7
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "event_id": self.event_id,
            "ts": self.ts.isoformat(),
            "sensor_id": self.sensor_id,
            "type": self.sensor_type.value,
            "value": self.value,
            "unit": self.unit,
            "quality": {
                "status": self.quality.status.value,
                "confidence": self.quality.confidence,
            },
            "meta": {
                "location": self.meta.location,
                "source": self.meta.source,
                "raw": self.meta.raw,
                "driver_id": self.meta.driver_id,
                "sequence_num": self.meta.sequence_num,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SensorEvent":
        """Create from dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            ts=datetime.fromisoformat(data["ts"]) if isinstance(data.get("ts"), str) else data.get("ts", datetime.now(timezone.utc)),
            sensor_id=data.get("sensor_id", ""),
            sensor_type=SensorType(data.get("type", "generic")),
            value=float(data.get("value", 0.0)),
            unit=data.get("unit", ""),
            quality=SensorQuality(
                status=QualityStatus(data.get("quality", {}).get("status", "OK")),
                confidence=float(data.get("quality", {}).get("confidence", 1.0)),
            ),
            meta=SensorMeta(
                location=data.get("meta", {}).get("location", ""),
                source=data.get("meta", {}).get("source", "unknown"),
                raw=data.get("meta", {}).get("raw"),
                driver_id=data.get("meta", {}).get("driver_id"),
                sequence_num=data.get("meta", {}).get("sequence_num"),
            ),
        )


# Convenience factory functions
def temperature_event(
    sensor_id: str,
    value: float,
    unit: str = "C",
    location: str = "",
    source: str = "mock",
) -> SensorEvent:
    """Create a temperature sensor event."""
    return SensorEvent(
        sensor_id=sensor_id,
        sensor_type=SensorType.TEMPERATURE,
        value=value,
        unit=unit,
        meta=SensorMeta(location=location, source=source),
    )


def airflow_event(
    sensor_id: str,
    value: float,
    unit: str = "m/s",
    location: str = "",
    source: str = "mock",
) -> SensorEvent:
    """Create an airflow sensor event (fume hood)."""
    return SensorEvent(
        sensor_id=sensor_id,
        sensor_type=SensorType.AIRFLOW,
        value=value,
        unit=unit,
        meta=SensorMeta(location=location, source=source),
    )


def pressure_event(
    sensor_id: str,
    value: float,
    unit: str = "kPa",
    location: str = "",
    source: str = "mock",
) -> SensorEvent:
    """Create a pressure sensor event."""
    return SensorEvent(
        sensor_id=sensor_id,
        sensor_type=SensorType.PRESSURE,
        value=value,
        unit=unit,
        meta=SensorMeta(location=location, source=source),
    )


def estop_event(
    sensor_id: str,
    triggered: bool,
    location: str = "",
    source: str = "mock",
) -> SensorEvent:
    """Create an emergency stop status event."""
    return SensorEvent(
        sensor_id=sensor_id,
        sensor_type=SensorType.ESTOP,
        value=1.0 if triggered else 0.0,
        unit="bool",
        meta=SensorMeta(location=location, source=source),
    )
