"""
Sensing Layer - Unified sensor data pipeline for safety-critical lab automation.

This module provides:
- Standardized SensorEvent format for all sensor inputs
- Pluggable drivers for different protocols (mock, modbus, mqtt, serial, http)
- SensorHub for aggregation, deduplication, and buffering
- HealthMonitor for sensor health tracking (stale, stuck, out-of-range)
- FaultInjector for testing and simulation
- Snapshot API for decision layer consumption
- Integration with SafetyChecker/RecoveryAgent
"""

from exp_agent.sensing.protocol.sensor_event import (
    SensorEvent,
    QualityStatus,
    SensorQuality,
    SensorMeta,
    SensorType,
)
from exp_agent.sensing.protocol.health_event import (
    SensorHealthEvent,
    HealthStatus,
    HealthMetrics,
)
from exp_agent.sensing.protocol.snapshot import (
    SensorSnapshot,
    SystemSnapshot,
)
from exp_agent.sensing.integration import (
    SensingIntegration,
    SensingInterlockConfig,
    SafetyVeto,
    SafetyVetoReason,
)

__all__ = [
    # Core event types
    "SensorEvent",
    "QualityStatus",
    "SensorQuality",
    "SensorMeta",
    "SensorType",
    # Health types
    "SensorHealthEvent",
    "HealthStatus",
    "HealthMetrics",
    # Snapshot types
    "SensorSnapshot",
    "SystemSnapshot",
    # Integration
    "SensingIntegration",
    "SensingInterlockConfig",
    "SafetyVeto",
    "SafetyVetoReason",
]
