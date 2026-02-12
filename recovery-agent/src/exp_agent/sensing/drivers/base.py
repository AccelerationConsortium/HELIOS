"""
Base class for sensor drivers.

A SensorDriver connects to a sensor source and produces SensorEvents.
All drivers share this common interface so SensorHub can treat them uniformly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional
from datetime import datetime, timezone

from exp_agent.sensing.protocol.sensor_event import SensorEvent


@dataclass
class DriverConfig:
    """Base configuration for sensor drivers."""

    driver_id: str                          # Unique identifier for this driver
    enabled: bool = True                    # Is this driver active?
    poll_interval_ms: float = 1000.0        # How often to poll (if polling-based)
    reconnect_interval_ms: float = 5000.0   # How long to wait before reconnecting
    max_reconnect_attempts: int = 3         # Max reconnection attempts


class SensorDriver(ABC):
    """
    Abstract base class for sensor drivers.

    A driver can operate in two modes:
    1. Polling: Call read() periodically
    2. Streaming: Use stream() to get an async iterator

    Subclasses must implement at least read(). stream() has a default
    polling implementation but can be overridden for push-based sources.
    """

    def __init__(self, config: DriverConfig):
        self.config = config
        self.driver_id = config.driver_id
        self._running = False
        self._connected = False
        self._last_error: Optional[Exception] = None
        self._event_callback: Optional[Callable[[SensorEvent], None]] = None

    @property
    def is_running(self) -> bool:
        """Check if driver is running."""
        return self._running

    @property
    def is_connected(self) -> bool:
        """Check if driver is connected to its source."""
        return self._connected

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the sensor source.

        Returns:
            True if connection successful, False otherwise.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the sensor source."""
        pass

    @abstractmethod
    async def read(self) -> list[SensorEvent]:
        """
        Read current values from all sensors managed by this driver.

        Returns:
            List of SensorEvents, one per sensor.
        """
        pass

    async def start(self) -> None:
        """Start the driver."""
        if not self._connected:
            await self.connect()
        self._running = True

    async def stop(self) -> None:
        """Stop the driver."""
        self._running = False
        await self.disconnect()

    async def stream(self) -> AsyncIterator[SensorEvent]:
        """
        Stream sensor events as they arrive.

        Default implementation polls at poll_interval_ms.
        Override for push-based sources (MQTT, WebSocket, etc.).
        """
        import asyncio

        while self._running:
            try:
                events = await self.read()
                for event in events:
                    yield event
                    if self._event_callback:
                        self._event_callback(event)
            except Exception as e:
                self._last_error = e
                # Log error but continue trying
            await asyncio.sleep(self.config.poll_interval_ms / 1000.0)

    def set_event_callback(self, callback: Callable[[SensorEvent], None]) -> None:
        """Set a callback to be called for each event."""
        self._event_callback = callback

    def get_last_error(self) -> Optional[Exception]:
        """Get the last error that occurred."""
        return self._last_error

    def get_status(self) -> dict:
        """Get driver status summary."""
        return {
            "driver_id": self.driver_id,
            "running": self._running,
            "connected": self._connected,
            "enabled": self.config.enabled,
            "last_error": str(self._last_error) if self._last_error else None,
        }
