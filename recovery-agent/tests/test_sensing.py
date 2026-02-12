"""
Tests for the sensing layer.

Tests cover:
- SensorEvent creation and serialization
- MockSensorDriver functionality
- SensorHub aggregation
- HealthMonitor detection
- FaultInjector scenarios
"""

import pytest
import asyncio

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio(loop_scope="function")
from datetime import datetime, timezone, timedelta

from exp_agent.sensing.protocol.sensor_event import (
    SensorEvent,
    SensorType,
    SensorQuality,
    SensorMeta,
    QualityStatus,
    temperature_event,
    airflow_event,
    pressure_event,
)
from exp_agent.sensing.protocol.health_event import (
    SensorHealthEvent,
    HealthStatus,
    HealthMetrics,
)
from exp_agent.sensing.protocol.snapshot import (
    SensorSnapshot,
    SystemSnapshot,
    SystemHealthStatus,
)
from exp_agent.sensing.drivers.mock_driver import (
    MockSensorDriver,
    MockSensorConfig,
    SensorProfile,
    TemperatureProfile,
    AirflowProfile,
    create_lab_sensor_set,
)
from exp_agent.sensing.hub.sensor_hub import SensorHub, HubConfig
from exp_agent.sensing.hub.ring_buffer import RingBuffer
from exp_agent.sensing.health.health_monitor import (
    HealthMonitor,
    HealthMonitorConfig,
    SensorHealthConfig,
)
from exp_agent.sensing.health.detectors.stale import StaleDetector
from exp_agent.sensing.health.detectors.stuck import StuckDetector
from exp_agent.sensing.health.detectors.out_of_range import OutOfRangeDetector
from exp_agent.sensing.simulator.fault_injector import (
    FaultInjector,
    FaultType,
    FaultConfig,
)
from exp_agent.sensing.api.sensing_api import SensingAPI


class TestSensorEvent:
    """Tests for SensorEvent data structure."""

    def test_create_sensor_event(self):
        """Test basic event creation."""
        event = SensorEvent(
            sensor_id="temp_1",
            sensor_type=SensorType.TEMPERATURE,
            value=25.5,
            unit="C",
        )
        assert event.sensor_id == "temp_1"
        assert event.value == 25.5
        assert event.is_ok
        assert event.is_trustworthy

    def test_temperature_event_factory(self):
        """Test temperature event factory function."""
        event = temperature_event("reactor_temp", 80.0, location="SDL1")
        assert event.sensor_type == SensorType.TEMPERATURE
        assert event.value == 80.0
        assert event.unit == "C"

    def test_airflow_event_factory(self):
        """Test airflow event factory function."""
        event = airflow_event("hood_1", 0.5, location="SDL1_hood")
        assert event.sensor_type == SensorType.AIRFLOW
        assert event.value == 0.5
        assert event.unit == "m/s"

    def test_event_serialization(self):
        """Test event to/from dict."""
        event = temperature_event("temp_1", 25.0)
        data = event.to_dict()
        assert data["sensor_id"] == "temp_1"
        assert data["value"] == 25.0

        restored = SensorEvent.from_dict(data)
        assert restored.sensor_id == event.sensor_id
        assert restored.value == event.value

    def test_event_quality(self):
        """Test event quality status."""
        ok_event = SensorEvent(
            sensor_id="temp_1",
            value=25.0,
            quality=SensorQuality(status=QualityStatus.OK),
        )
        assert ok_event.is_ok
        assert ok_event.is_trustworthy

        stale_event = SensorEvent(
            sensor_id="temp_1",
            value=25.0,
            quality=SensorQuality(status=QualityStatus.STALE, confidence=0.5),
        )
        assert stale_event.is_stale
        assert not stale_event.is_trustworthy


class TestMockSensorDriver:
    """Tests for MockSensorDriver."""

    @pytest.mark.asyncio
    async def test_driver_read(self):
        """Test reading from mock driver."""
        config = MockSensorConfig(
            driver_id="test_driver",
            sensors=[
                TemperatureProfile(
                    sensor_id="temp_1",
                    location="test_loc",
                    base_value=25.0,
                    noise_std=0.0,  # No noise for predictable testing
                ),
            ],
        )
        driver = MockSensorDriver(config)
        await driver.connect()

        events = await driver.read()
        assert len(events) == 1
        assert events[0].sensor_id == "temp_1"
        assert events[0].value == 25.0

        await driver.disconnect()

    @pytest.mark.asyncio
    async def test_lab_sensor_set(self):
        """Test creating standard lab sensor set."""
        config = create_lab_sensor_set("TEST")
        driver = MockSensorDriver(config)
        await driver.connect()

        events = await driver.read()
        sensor_ids = {e.sensor_id for e in events}

        assert "TEST_reactor_temp" in sensor_ids
        assert "TEST_hood_airflow" in sensor_ids
        assert "TEST_estop" in sensor_ids

        await driver.disconnect()


class TestSensorHub:
    """Tests for SensorHub."""

    @pytest.mark.asyncio
    async def test_hub_collect_events(self):
        """Test hub collecting events from driver."""
        config = create_lab_sensor_set("HUB_TEST")
        driver = MockSensorDriver(config)

        hub = SensorHub()
        hub.add_driver(driver)

        # Do a single read
        events = await hub.read_once()
        assert len(events) > 0

        # Check snapshot
        snapshot = hub.get_snapshot()
        assert len(snapshot.sensors) > 0

    @pytest.mark.asyncio
    async def test_hub_snapshot(self):
        """Test hub snapshot generation."""
        config = MockSensorConfig(
            driver_id="test",
            sensors=[
                TemperatureProfile(sensor_id="temp_1", location="loc1", base_value=25.0),
            ],
        )
        driver = MockSensorDriver(config)

        hub = SensorHub()
        hub.add_driver(driver)

        await hub.read_once()
        snapshot = hub.get_snapshot()

        assert "temp_1" in snapshot.sensors
        sensor = snapshot.sensors["temp_1"]
        assert sensor.latest_value is not None


class TestRingBuffer:
    """Tests for RingBuffer."""

    def test_buffer_append(self):
        """Test adding items to buffer."""
        buffer = RingBuffer[int](max_size=10)
        for i in range(5):
            buffer.append(i)
        assert len(buffer) == 5
        assert buffer.newest == 4

    def test_buffer_overflow(self):
        """Test buffer overflow behavior."""
        buffer = RingBuffer[int](max_size=5)
        for i in range(10):
            buffer.append(i)
        assert len(buffer) == 5
        assert buffer.oldest == 5
        assert buffer.newest == 9


class TestHealthDetectors:
    """Tests for health detectors."""

    def test_stale_detector(self):
        """Test stale detection."""
        detector = StaleDetector()
        detector.register_sensor("temp_1", expected_period_ms=1000)

        now = datetime.now(timezone.utc)
        detector.update("temp_1", now)

        # Should not be stale immediately
        assert not detector.check("temp_1", now)

        # Should be stale after 2.5x expected period
        later = now + timedelta(milliseconds=2500)
        assert detector.check("temp_1", later)

    def test_stuck_detector(self):
        """Test stuck detection."""
        detector = StuckDetector()

        now = datetime.now(timezone.utc)
        detector.update("temp_1", 25.0, now)

        # Not stuck immediately
        assert not detector.check("temp_1", now)

        # Stuck after threshold
        later = now + timedelta(milliseconds=35000)
        assert detector.check("temp_1", later)

        # Recovered after value change
        detector.update("temp_1", 26.0, later)
        assert not detector.check("temp_1", later)

    def test_out_of_range_detector(self):
        """Test out of range detection."""
        detector = OutOfRangeDetector()
        detector.set_bounds("temp_1", valid_min=0, valid_max=100)

        assert not detector.check("temp_1", 50.0)  # In range
        assert detector.check("temp_1", 150.0)     # Out of range
        assert detector.check("temp_1", -10.0)     # Out of range


class TestHealthMonitor:
    """Tests for HealthMonitor."""

    def test_monitor_process_event(self):
        """Test processing events and updating health."""
        monitor = HealthMonitor()
        monitor.register_sensor(SensorHealthConfig(
            sensor_id="temp_1",
            sensor_type=SensorType.TEMPERATURE,
            valid_min=0,
            valid_max=100,
        ))

        event = temperature_event("temp_1", 25.0)
        health_event = monitor.process_event(event)

        # Should transition from UNKNOWN to HEALTHY
        assert health_event is not None
        assert health_event.status == HealthStatus.HEALTHY

    def test_monitor_detect_out_of_range(self):
        """Test detecting out of range values."""
        monitor = HealthMonitor()
        monitor.register_sensor(SensorHealthConfig(
            sensor_id="temp_1",
            valid_max=100,
        ))

        # Normal event
        event1 = temperature_event("temp_1", 50.0)
        monitor.process_event(event1)

        # Out of range event
        event2 = SensorEvent(
            sensor_id="temp_1",
            sensor_type=SensorType.TEMPERATURE,
            value=150.0,
            unit="C",
        )
        health_event = monitor.process_event(event2)

        assert health_event is not None
        assert health_event.status == HealthStatus.UNHEALTHY


class TestFaultInjector:
    """Tests for FaultInjector."""

    @pytest.mark.asyncio
    async def test_inject_stuck_fault(self):
        """Test injecting stuck fault."""
        config = MockSensorConfig(
            driver_id="test",
            sensors=[
                TemperatureProfile(sensor_id="temp_1", location="loc1", base_value=25.0, noise_std=0.5),
            ],
        )
        driver = MockSensorDriver(config)
        injector = FaultInjector(driver)

        await driver.connect()

        # Normal reading (with noise)
        events1 = await driver.read()
        normal_value = events1[0].value

        # Inject stuck fault
        injector.inject(FaultConfig(
            fault_type=FaultType.STUCK,
            sensor_id="temp_1",
            stuck_value=99.0,
        ))

        # Now should return stuck value
        events2 = await driver.read()
        assert events2[0].value == 99.0

        # Clear fault
        injector.clear("temp_1")

        await driver.disconnect()


class TestSensingAPI:
    """Tests for SensingAPI."""

    @pytest.mark.asyncio
    async def test_api_get_snapshot(self):
        """Test API snapshot endpoint."""
        config = create_lab_sensor_set("API_TEST")
        driver = MockSensorDriver(config)
        hub = SensorHub()
        hub.add_driver(driver)

        await hub.read_once()

        api = SensingAPI(hub=hub)
        snapshot = api.get_snapshot()

        assert "ts" in snapshot
        assert "sensors" in snapshot
        assert len(snapshot["sensors"]) > 0

    @pytest.mark.asyncio
    async def test_api_safety_check(self):
        """Test API safety check."""
        config = create_lab_sensor_set("SAFETY_TEST")
        driver = MockSensorDriver(config)
        hub = SensorHub()
        hub.add_driver(driver)

        await hub.read_once()

        api = SensingAPI(hub=hub)
        is_safe, issues = api.is_safe_to_proceed()

        # Should be safe with healthy mock sensors
        assert is_safe
        assert len(issues) == 0


class TestIntegration:
    """Integration tests for the complete sensing pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test complete sensing pipeline: driver -> hub -> monitor -> api."""
        # Create driver with lab sensors
        config = create_lab_sensor_set("INTEG_TEST")
        driver = MockSensorDriver(config)

        # Create hub
        hub = SensorHub()
        hub.add_driver(driver)

        # Create health monitor
        monitor = HealthMonitor()
        for sensor_id in driver.get_sensor_ids():
            monitor.register_sensor(SensorHealthConfig(sensor_id=sensor_id))

        # Connect hub to monitor
        hub.set_event_callback(lambda e: monitor.process_event(e))

        # Create API
        api = SensingAPI(hub=hub, health_monitor=monitor)

        # Read some events
        await hub.read_once()

        # Verify everything works
        snapshot = api.get_snapshot()
        assert len(snapshot["sensors"]) > 0

        health = api.get_all_health()
        assert health["total_sensors"] > 0

        is_safe, _ = api.is_safe_to_proceed()
        assert is_safe

    @pytest.mark.asyncio
    async def test_fault_detection_pipeline(self):
        """Test that faults are detected through the pipeline."""
        config = MockSensorConfig(
            driver_id="fault_test",
            sensors=[
                AirflowProfile(
                    sensor_id="hood_airflow",
                    location="test_hood",
                    base_value=0.5,
                    noise_std=0.0,
                ),
            ],
        )
        driver = MockSensorDriver(config)

        # Disable rate limiting for testing
        hub_config = HubConfig(min_interval_ms=0, dedup_window_ms=0)
        hub = SensorHub(config=hub_config)
        hub.add_driver(driver)

        # Create monitor with airflow threshold
        monitor = HealthMonitor()
        monitor.register_sensor(SensorHealthConfig(
            sensor_id="hood_airflow",
            sensor_type=SensorType.AIRFLOW,
            valid_min=0.3,  # Minimum safe airflow
        ))

        hub.set_event_callback(lambda e: monitor.process_event(e))

        api = SensingAPI(hub=hub, health_monitor=monitor)

        # Normal operation
        await hub.read_once()
        is_safe, _ = api.is_safe_to_proceed()
        assert is_safe

        # Inject hood failure
        injector = FaultInjector(driver)
        injector.simulate_hood_failure("hood_airflow")

        await hub.read_once()

        # Should now detect the issue
        health = api.get_health("hood_airflow")
        assert health["status"] == "UNHEALTHY"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
