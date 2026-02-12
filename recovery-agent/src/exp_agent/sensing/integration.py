"""
Integration layer between Sensing and Safety/Recovery systems.

This module provides the bridge between the real-time sensing layer
and the safety-critical decision making components.

Key Integration Points (from plan.md 0205):
1. Hood airflow < threshold → veto: block volatile steps, safe shutdown
2. Temperature slope > threshold → veto: stop heating, alarm
3. Pressure > threshold → veto: stop heating, execute pressure relief
4. Sensor stale (> 2x expected period) → degrade: block high-risk steps
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Callable, Any
from enum import Enum

from exp_agent.sensing.hub.sensor_hub import SensorHub, HubConfig
from exp_agent.sensing.health.health_monitor import (
    HealthMonitor,
    HealthMonitorConfig,
    SensorHealthConfig,
)
from exp_agent.sensing.api.sensing_api import SensingAPI
from exp_agent.sensing.drivers.mock_driver import MockSensorDriver, create_lab_sensor_set
from exp_agent.sensing.protocol.sensor_event import SensorEvent, SensorType
from exp_agent.sensing.protocol.health_event import HealthStatus
from exp_agent.sensing.protocol.snapshot import SystemSnapshot


class SafetyVetoReason(str, Enum):
    """Reasons for safety veto based on sensor data."""

    HOOD_AIRFLOW_LOW = "hood_airflow_low"
    TEMPERATURE_SLOPE_HIGH = "temperature_slope_high"
    PRESSURE_HIGH = "pressure_high"
    SENSOR_STALE = "sensor_stale"
    SENSOR_UNHEALTHY = "sensor_unhealthy"
    ESTOP_TRIGGERED = "estop_triggered"
    POWER_FAILURE = "power_failure"


@dataclass
class SafetyVeto:
    """A safety veto from the sensing layer."""

    reason: SafetyVetoReason
    sensor_id: str
    current_value: Optional[float]
    threshold: Optional[float]
    message: str
    severity: str = "critical"  # critical, warning

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value,
            "sensor_id": self.sensor_id,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class SensingInterlockConfig:
    """Configuration for safety interlocks based on sensing data."""

    # Hood airflow interlock
    hood_airflow_min: float = 0.3  # m/s - minimum safe airflow
    hood_sensor_id: str = ""       # Empty = auto-detect

    # Temperature interlocks
    temperature_max: float = 130.0         # °C - maximum safe temperature
    temperature_slope_max: float = 3.0     # °C/min - maximum rate of change
    temperature_sensor_id: str = ""        # Empty = auto-detect

    # Pressure interlocks
    pressure_max: float = 200.0    # kPa - maximum safe pressure
    pressure_sensor_id: str = ""   # Empty = auto-detect

    # General settings
    stale_threshold_multiplier: float = 2.0  # Consider stale if > 2x expected period


class SensingIntegration:
    """
    Integrates the sensing layer with safety decision making.

    This class provides:
    1. Real-time safety interlocks based on sensor data
    2. Telemetry data for SafetyChecker/RecoveryAgent
    3. Sensor health status for degraded-mode decisions
    4. Event callbacks for safety-critical changes

    Usage:
        integration = SensingIntegration()
        await integration.start()

        # Check safety before operation
        vetoes = integration.check_interlocks()
        if vetoes:
            # Block operation
            ...

        # Get telemetry for SafetyChecker
        telemetry = integration.get_telemetry()

        await integration.stop()
    """

    def __init__(
        self,
        config: Optional[SensingInterlockConfig] = None,
        hub: Optional[SensorHub] = None,
        monitor: Optional[HealthMonitor] = None,
    ):
        self.config = config or SensingInterlockConfig()
        self._hub = hub
        self._monitor = monitor
        self._api: Optional[SensingAPI] = None
        self._drivers: list = []
        self._running = False

        # Veto callbacks
        self._veto_callbacks: list[Callable[[SafetyVeto], None]] = []

        # Detected sensor IDs
        self._hood_sensors: list[str] = []
        self._temperature_sensors: list[str] = []
        self._pressure_sensors: list[str] = []
        self._estop_sensors: list[str] = []

    async def start(self) -> None:
        """Start the sensing integration."""
        if self._hub is None:
            self._hub = SensorHub()

        if self._monitor is None:
            self._monitor = HealthMonitor()

        # Connect monitor to hub
        self._hub.set_event_callback(self._on_sensor_event)

        self._api = SensingAPI(hub=self._hub, health_monitor=self._monitor)
        self._running = True

        await self._hub.start()

    async def stop(self) -> None:
        """Stop the sensing integration."""
        self._running = False
        if self._hub:
            await self._hub.stop()

    def add_driver(self, driver) -> None:
        """Add a sensor driver."""
        self._drivers.append(driver)
        if self._hub:
            self._hub.add_driver(driver)

    def add_mock_lab_sensors(self, location_prefix: str = "SDL1") -> MockSensorDriver:
        """Add a set of mock lab sensors for testing."""
        config = create_lab_sensor_set(location_prefix)
        driver = MockSensorDriver(config)
        self.add_driver(driver)

        # Register sensors with health monitor
        for sensor_id in driver.get_sensor_ids():
            sensor_type = SensorType.GENERIC
            if "temp" in sensor_id:
                sensor_type = SensorType.TEMPERATURE
                self._temperature_sensors.append(sensor_id)
            elif "airflow" in sensor_id or "hood" in sensor_id:
                sensor_type = SensorType.AIRFLOW
                self._hood_sensors.append(sensor_id)
            elif "pressure" in sensor_id:
                sensor_type = SensorType.PRESSURE
                self._pressure_sensors.append(sensor_id)
            elif "estop" in sensor_id:
                sensor_type = SensorType.ESTOP
                self._estop_sensors.append(sensor_id)

            if self._monitor:
                self._monitor.register_sensor(SensorHealthConfig(
                    sensor_id=sensor_id,
                    sensor_type=sensor_type,
                ))

        return driver

    def set_veto_callback(self, callback: Callable[[SafetyVeto], None]) -> None:
        """Set callback for safety vetoes."""
        self._veto_callbacks.append(callback)

    def _on_sensor_event(self, event: SensorEvent) -> None:
        """Handle incoming sensor event."""
        if self._monitor:
            self._monitor.process_event(event)

        # Check for immediate vetoes
        vetoes = self._check_event_for_vetoes(event)
        for veto in vetoes:
            for callback in self._veto_callbacks:
                try:
                    callback(veto)
                except Exception as e:
                    print(f"Error in veto callback: {e}")

    def _check_event_for_vetoes(self, event: SensorEvent) -> list[SafetyVeto]:
        """Check if an event triggers any vetoes."""
        vetoes = []

        # E-stop check
        if event.sensor_type == SensorType.ESTOP and event.value > 0.5:
            vetoes.append(SafetyVeto(
                reason=SafetyVetoReason.ESTOP_TRIGGERED,
                sensor_id=event.sensor_id,
                current_value=event.value,
                threshold=0.5,
                message="Emergency stop triggered",
                severity="critical",
            ))

        # Hood airflow check
        if event.sensor_type == SensorType.AIRFLOW:
            if event.value < self.config.hood_airflow_min:
                vetoes.append(SafetyVeto(
                    reason=SafetyVetoReason.HOOD_AIRFLOW_LOW,
                    sensor_id=event.sensor_id,
                    current_value=event.value,
                    threshold=self.config.hood_airflow_min,
                    message=f"Hood airflow {event.value:.2f} m/s below minimum {self.config.hood_airflow_min} m/s",
                    severity="critical",
                ))

        # Temperature check
        if event.sensor_type == SensorType.TEMPERATURE:
            if event.value > self.config.temperature_max:
                vetoes.append(SafetyVeto(
                    reason=SafetyVetoReason.TEMPERATURE_SLOPE_HIGH,
                    sensor_id=event.sensor_id,
                    current_value=event.value,
                    threshold=self.config.temperature_max,
                    message=f"Temperature {event.value:.1f}°C exceeds maximum {self.config.temperature_max}°C",
                    severity="critical",
                ))

        # Pressure check
        if event.sensor_type == SensorType.PRESSURE:
            if event.value > self.config.pressure_max:
                vetoes.append(SafetyVeto(
                    reason=SafetyVetoReason.PRESSURE_HIGH,
                    sensor_id=event.sensor_id,
                    current_value=event.value,
                    threshold=self.config.pressure_max,
                    message=f"Pressure {event.value:.1f} kPa exceeds maximum {self.config.pressure_max} kPa",
                    severity="critical",
                ))

        return vetoes

    def check_interlocks(self) -> list[SafetyVeto]:
        """
        Check all safety interlocks based on current sensor state.

        Returns list of vetoes. Empty list means all clear.

        This is the main entry point for SafetyChecker integration.
        """
        vetoes = []

        if self._api is None:
            return vetoes

        snapshot = self._api.get_snapshot()
        sensors = snapshot.get("sensors", {})

        # Check hood airflow
        for sensor_id in self._hood_sensors:
            if sensor_id in sensors:
                sensor = sensors[sensor_id]
                value = sensor.get("latest_value")
                if value is not None and value < self.config.hood_airflow_min:
                    vetoes.append(SafetyVeto(
                        reason=SafetyVetoReason.HOOD_AIRFLOW_LOW,
                        sensor_id=sensor_id,
                        current_value=value,
                        threshold=self.config.hood_airflow_min,
                        message=f"Hood airflow {value:.2f} m/s below minimum {self.config.hood_airflow_min} m/s",
                    ))

        # Check temperature
        for sensor_id in self._temperature_sensors:
            if sensor_id in sensors:
                sensor = sensors[sensor_id]
                value = sensor.get("latest_value")
                if value is not None and value > self.config.temperature_max:
                    vetoes.append(SafetyVeto(
                        reason=SafetyVetoReason.TEMPERATURE_SLOPE_HIGH,
                        sensor_id=sensor_id,
                        current_value=value,
                        threshold=self.config.temperature_max,
                        message=f"Temperature {value:.1f}°C exceeds maximum {self.config.temperature_max}°C",
                    ))

                # Check temperature slope
                slope = sensor.get("trend_slope")
                if slope is not None:
                    slope_per_min = slope * 60  # Convert /sec to /min
                    if slope_per_min > self.config.temperature_slope_max:
                        vetoes.append(SafetyVeto(
                            reason=SafetyVetoReason.TEMPERATURE_SLOPE_HIGH,
                            sensor_id=sensor_id,
                            current_value=slope_per_min,
                            threshold=self.config.temperature_slope_max,
                            message=f"Temperature rising at {slope_per_min:.1f}°C/min exceeds maximum {self.config.temperature_slope_max}°C/min",
                        ))

        # Check pressure
        for sensor_id in self._pressure_sensors:
            if sensor_id in sensors:
                sensor = sensors[sensor_id]
                value = sensor.get("latest_value")
                if value is not None and value > self.config.pressure_max:
                    vetoes.append(SafetyVeto(
                        reason=SafetyVetoReason.PRESSURE_HIGH,
                        sensor_id=sensor_id,
                        current_value=value,
                        threshold=self.config.pressure_max,
                        message=f"Pressure {value:.1f} kPa exceeds maximum {self.config.pressure_max} kPa",
                    ))

        # Check E-stop
        for sensor_id in self._estop_sensors:
            if sensor_id in sensors:
                sensor = sensors[sensor_id]
                value = sensor.get("latest_value")
                if value is not None and value > 0.5:
                    vetoes.append(SafetyVeto(
                        reason=SafetyVetoReason.ESTOP_TRIGGERED,
                        sensor_id=sensor_id,
                        current_value=value,
                        threshold=0.5,
                        message="Emergency stop triggered",
                    ))

        # Check sensor health
        unhealthy = self._api.get_unhealthy_sensors()
        for sensor_id in unhealthy:
            vetoes.append(SafetyVeto(
                reason=SafetyVetoReason.SENSOR_UNHEALTHY,
                sensor_id=sensor_id,
                current_value=None,
                threshold=None,
                message=f"Sensor {sensor_id} is unhealthy",
                severity="warning",
            ))

        return vetoes

    def get_telemetry(self) -> dict[str, Any]:
        """
        Get current telemetry dict for SafetyChecker.

        Returns a dictionary that can be used as state.telemetry
        in check_action_safety().
        """
        if self._api is None:
            return {}

        snapshot = self._api.get_snapshot()
        sensors = snapshot.get("sensors", {})

        telemetry = {}
        for sensor_id, sensor in sensors.items():
            # Add value with sensor_id as key
            if sensor.get("latest_value") is not None:
                telemetry[sensor_id] = sensor["latest_value"]

            # Add common aliases
            sensor_type = sensor.get("sensor_type", "")
            if sensor_type == "temperature":
                if "reactor" in sensor_id:
                    telemetry["reactor_temperature"] = sensor["latest_value"]
                elif "hotplate" in sensor_id:
                    telemetry["hotplate_temperature"] = sensor["latest_value"]
                telemetry["current_temp"] = sensor["latest_value"]
            elif sensor_type == "pressure":
                telemetry["current_pressure"] = sensor["latest_value"]
            elif sensor_type == "airflow":
                telemetry["hood_airflow"] = sensor["latest_value"]

        return telemetry

    def get_snapshot(self) -> Optional[SystemSnapshot]:
        """Get the current system snapshot."""
        if self._hub:
            return self._hub.get_snapshot()
        return None

    def is_safe_to_proceed(self) -> tuple[bool, list[str]]:
        """
        Check if it's safe to proceed with operations.

        Returns:
            (is_safe, list of issue messages)
        """
        vetoes = self.check_interlocks()
        critical_vetoes = [v for v in vetoes if v.severity == "critical"]

        if critical_vetoes:
            return (False, [v.message for v in critical_vetoes])

        return (True, [v.message for v in vetoes])  # Warning vetoes only
