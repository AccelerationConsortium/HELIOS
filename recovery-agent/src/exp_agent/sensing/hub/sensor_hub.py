"""
SensorHub - Central aggregation point for all sensor data.

The hub connects to multiple drivers, collects their events, and provides
a unified interface for downstream consumers (SafetyChecker, RecoveryAgent).

Features:
- Multi-driver support
- Event deduplication
- Rate limiting
- Ring buffer for history
- Snapshot generation
- Event callbacks for real-time processing
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Callable, Optional, Any
from collections import defaultdict

from exp_agent.sensing.drivers.base import SensorDriver
from exp_agent.sensing.protocol.sensor_event import SensorEvent
from exp_agent.sensing.protocol.health_event import SensorHealthEvent, HealthStatus
from exp_agent.sensing.protocol.snapshot import SensorSnapshot, SystemSnapshot
from exp_agent.sensing.hub.ring_buffer import RingBuffer


@dataclass
class HubConfig:
    """Configuration for SensorHub."""

    # Buffer settings
    buffer_size: int = 10000
    buffer_max_age_seconds: float = 3600.0  # 1 hour

    # Rate limiting (per sensor)
    min_interval_ms: float = 100.0          # Minimum time between events
    max_events_per_second: float = 10.0     # Max events per sensor per second

    # Deduplication
    dedup_window_ms: float = 50.0           # Ignore duplicate events within window

    # Snapshot settings
    snapshot_window_seconds: float = 60.0   # History window for snapshots


class SensorHub:
    """
    Central hub for collecting and distributing sensor events.

    Usage:
        hub = SensorHub()
        hub.add_driver(mock_driver)
        hub.set_event_callback(process_event)
        await hub.start()
        ...
        snapshot = hub.get_snapshot()
        await hub.stop()
    """

    def __init__(self, config: Optional[HubConfig] = None):
        self.config = config or HubConfig()
        self._drivers: dict[str, SensorDriver] = {}
        self._buffer: RingBuffer[SensorEvent] = RingBuffer(
            max_size=self.config.buffer_size,
            max_age_seconds=self.config.buffer_max_age_seconds,
        )

        # Latest event per sensor for dedup and snapshots
        self._latest_events: dict[str, SensorEvent] = {}
        self._last_event_times: dict[str, datetime] = {}

        # Sensor snapshots for quick access
        self._sensor_snapshots: dict[str, SensorSnapshot] = {}

        # Callbacks
        self._event_callbacks: list[Callable[[SensorEvent], None]] = []
        self._health_callbacks: list[Callable[[SensorHealthEvent], None]] = []

        # State
        self._running = False
        self._tasks: list[asyncio.Task] = []

        # Statistics
        self._stats: dict[str, Any] = {
            "total_events": 0,
            "events_per_sensor": defaultdict(int),
            "dropped_events": 0,
            "start_time": None,
        }

    def add_driver(self, driver: SensorDriver) -> None:
        """Add a sensor driver to the hub."""
        self._drivers[driver.driver_id] = driver
        driver.set_event_callback(self._on_driver_event)

    def remove_driver(self, driver_id: str) -> None:
        """Remove a sensor driver from the hub."""
        self._drivers.pop(driver_id, None)

    def set_event_callback(self, callback: Callable[[SensorEvent], None]) -> None:
        """Add a callback for new events."""
        self._event_callbacks.append(callback)

    def set_health_callback(self, callback: Callable[[SensorHealthEvent], None]) -> None:
        """Add a callback for health events."""
        self._health_callbacks.append(callback)

    async def start(self) -> None:
        """Start the hub and all drivers."""
        if self._running:
            return

        self._running = True
        self._stats["start_time"] = datetime.now(timezone.utc)

        # Start all drivers
        for driver in self._drivers.values():
            await driver.start()

        # Start collection tasks
        for driver_id, driver in self._drivers.items():
            task = asyncio.create_task(self._collect_from_driver(driver))
            self._tasks.append(task)

    async def stop(self) -> None:
        """Stop the hub and all drivers."""
        self._running = False

        # Cancel collection tasks
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

        # Stop all drivers
        for driver in self._drivers.values():
            await driver.stop()

    async def _collect_from_driver(self, driver: SensorDriver) -> None:
        """Collect events from a single driver."""
        try:
            async for event in driver.stream():
                if not self._running:
                    break
                self._process_event(event)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            # Log error but don't crash
            print(f"Error collecting from driver {driver.driver_id}: {e}")

    def _on_driver_event(self, event: SensorEvent) -> None:
        """Callback from driver for immediate processing."""
        self._process_event(event)

    def _process_event(self, event: SensorEvent) -> None:
        """Process an incoming event."""
        sensor_id = event.sensor_id

        # Deduplication check
        if sensor_id in self._last_event_times:
            elapsed_ms = (event.ts - self._last_event_times[sensor_id]).total_seconds() * 1000
            if elapsed_ms < self.config.dedup_window_ms:
                self._stats["dropped_events"] += 1
                return

        # Rate limiting check
        if sensor_id in self._last_event_times:
            elapsed_ms = (event.ts - self._last_event_times[sensor_id]).total_seconds() * 1000
            if elapsed_ms < self.config.min_interval_ms:
                self._stats["dropped_events"] += 1
                return

        # Accept the event
        self._buffer.append(event, event.ts)
        self._latest_events[sensor_id] = event
        self._last_event_times[sensor_id] = event.ts

        # Update snapshot
        if sensor_id not in self._sensor_snapshots:
            self._sensor_snapshots[sensor_id] = SensorSnapshot(
                sensor_id=sensor_id,
                sensor_type=event.sensor_type,
            )
        self._sensor_snapshots[sensor_id].add_reading(
            event,
            window_seconds=self.config.snapshot_window_seconds,
        )

        # Update statistics
        self._stats["total_events"] += 1
        self._stats["events_per_sensor"][sensor_id] += 1

        # Notify callbacks
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Error in event callback: {e}")

    def get_latest(self, sensor_id: str) -> Optional[SensorEvent]:
        """Get the latest event for a sensor."""
        return self._latest_events.get(sensor_id)

    def get_latest_value(self, sensor_id: str) -> Optional[float]:
        """Get the latest value for a sensor."""
        event = self._latest_events.get(sensor_id)
        return event.value if event else None

    def get_all_latest(self) -> dict[str, SensorEvent]:
        """Get latest events for all sensors."""
        return dict(self._latest_events)

    def get_history(
        self,
        sensor_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[SensorEvent]:
        """Get historical events."""
        if since:
            events = self._buffer.get_since(since)
        else:
            events = self._buffer.get_all()

        if sensor_id:
            events = [e for e in events if e.sensor_id == sensor_id]

        if limit:
            events = events[-limit:]

        return events

    def get_sensor_snapshot(self, sensor_id: str) -> Optional[SensorSnapshot]:
        """Get snapshot for a specific sensor."""
        return self._sensor_snapshots.get(sensor_id)

    def get_snapshot(self) -> SystemSnapshot:
        """Get a system-wide snapshot of all sensors."""
        snapshot = SystemSnapshot(ts=datetime.now(timezone.utc))

        for sensor_id, sensor_snapshot in self._sensor_snapshots.items():
            snapshot.add_sensor(sensor_snapshot)

        return snapshot

    def get_stats(self) -> dict[str, Any]:
        """Get hub statistics."""
        return {
            **self._stats,
            "buffer_stats": self._buffer.get_stats(),
            "driver_count": len(self._drivers),
            "sensor_count": len(self._latest_events),
            "running": self._running,
            "events_per_sensor": dict(self._stats["events_per_sensor"]),
        }

    def get_driver_status(self) -> dict[str, dict]:
        """Get status of all drivers."""
        return {
            driver_id: driver.get_status()
            for driver_id, driver in self._drivers.items()
        }

    # Convenience methods for common sensor queries

    def get_temperature(self, sensor_id: str) -> Optional[float]:
        """Get current temperature reading."""
        return self.get_latest_value(sensor_id)

    def get_airflow(self, sensor_id: str) -> Optional[float]:
        """Get current airflow reading."""
        return self.get_latest_value(sensor_id)

    def get_pressure(self, sensor_id: str) -> Optional[float]:
        """Get current pressure reading."""
        return self.get_latest_value(sensor_id)

    def is_estop_triggered(self, sensor_id: str) -> bool:
        """Check if e-stop is triggered."""
        value = self.get_latest_value(sensor_id)
        return value is not None and value > 0.5

    def is_power_on(self, sensor_id: str) -> bool:
        """Check if power is on."""
        value = self.get_latest_value(sensor_id)
        return value is not None and value > 0.5

    async def read_once(self) -> list[SensorEvent]:
        """
        Do a single read from all drivers (useful for testing).

        This bypasses the streaming loop and just reads once.
        """
        events = []
        for driver in self._drivers.values():
            driver_events = await driver.read()
            for event in driver_events:
                self._process_event(event)
                events.append(event)
        return events
