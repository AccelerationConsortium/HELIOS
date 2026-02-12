"""
ReplayDriver - Replay historical sensor events for incident reproduction.

Load events from:
- JSON log files
- CSV files
- In-memory event lists

Use for:
- Incident post-mortem analysis
- Regression testing
- Training and validation
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Iterator
import asyncio

from exp_agent.sensing.drivers.base import SensorDriver, DriverConfig
from exp_agent.sensing.protocol.sensor_event import SensorEvent


@dataclass
class ReplayConfig(DriverConfig):
    """Configuration for replay driver."""

    # Source of events
    events: list[SensorEvent] = field(default_factory=list)
    source_file: Optional[str] = None

    # Replay behavior
    time_scale: float = 1.0              # Playback speed (1.0 = real-time)
    loop: bool = False                   # Loop when done
    start_offset_ms: float = 0.0         # Skip first N ms of events
    preserve_timing: bool = True         # Preserve original timing


class ReplayDriver(SensorDriver):
    """
    Replays historical sensor events.

    Events can be loaded from:
    - A list of SensorEvent objects
    - A JSON file containing serialized events
    - A CSV file with event data

    Timing:
    - With preserve_timing=True, events are replayed with original delays
    - With preserve_timing=False, events are emitted as fast as possible
    - time_scale controls playback speed (2.0 = 2x speed)
    """

    def __init__(self, config: ReplayConfig):
        super().__init__(config)
        self.replay_config = config
        self._events: list[SensorEvent] = list(config.events)
        self._current_index: int = 0
        self._replay_start: Optional[datetime] = None
        self._event_start: Optional[datetime] = None

        # Load from file if specified
        if config.source_file:
            self._load_from_file(config.source_file)

    def _load_from_file(self, path: str) -> None:
        """Load events from a file."""
        file_path = Path(path)

        if file_path.suffix == ".json":
            self._load_json(file_path)
        elif file_path.suffix == ".csv":
            self._load_csv(file_path)
        elif file_path.suffix == ".jsonl":
            self._load_jsonl(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _load_json(self, path: Path) -> None:
        """Load events from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            self._events = [SensorEvent.from_dict(e) for e in data]
        elif isinstance(data, dict) and "events" in data:
            self._events = [SensorEvent.from_dict(e) for e in data["events"]]
        else:
            raise ValueError("JSON must be a list of events or {events: [...]}")

    def _load_jsonl(self, path: Path) -> None:
        """Load events from JSON Lines file (one event per line)."""
        self._events = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    event_data = json.loads(line)
                    self._events.append(SensorEvent.from_dict(event_data))

    def _load_csv(self, path: Path) -> None:
        """Load events from CSV file."""
        import csv

        self._events = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert CSV row to event dict format
                event_data = {
                    "ts": row.get("ts") or row.get("timestamp"),
                    "sensor_id": row.get("sensor_id"),
                    "type": row.get("type") or row.get("sensor_type", "generic"),
                    "value": float(row.get("value", 0)),
                    "unit": row.get("unit", ""),
                    "quality": {
                        "status": row.get("quality_status", "OK"),
                        "confidence": float(row.get("confidence", 1.0)),
                    },
                    "meta": {
                        "location": row.get("location", ""),
                        "source": row.get("source", "replay"),
                    },
                }
                self._events.append(SensorEvent.from_dict(event_data))

    async def connect(self) -> bool:
        """Prepare for replay."""
        self._current_index = 0
        self._connected = True

        # Apply start offset
        if self.replay_config.start_offset_ms > 0 and self._events:
            first_ts = self._events[0].ts
            offset = timedelta(milliseconds=self.replay_config.start_offset_ms)
            cutoff = first_ts + offset
            while (
                self._current_index < len(self._events)
                and self._events[self._current_index].ts < cutoff
            ):
                self._current_index += 1

        return True

    async def disconnect(self) -> None:
        """Stop replay."""
        self._connected = False
        self._replay_start = None

    async def read(self) -> list[SensorEvent]:
        """Read the next batch of events."""
        if self._current_index >= len(self._events):
            if self.replay_config.loop:
                self._current_index = 0
                self._replay_start = None
            else:
                return []

        # Return single event (batch of 1)
        event = self._events[self._current_index]
        self._current_index += 1
        return [event]

    async def stream(self) -> Iterator[SensorEvent]:
        """Stream events with timing preservation."""
        self._replay_start = datetime.now(timezone.utc)

        if self._events:
            self._event_start = self._events[0].ts

        while self._running:
            if self._current_index >= len(self._events):
                if self.replay_config.loop:
                    self._current_index = 0
                    self._replay_start = datetime.now(timezone.utc)
                    if self._events:
                        self._event_start = self._events[0].ts
                else:
                    break

            event = self._events[self._current_index]

            if self.replay_config.preserve_timing and self._event_start:
                # Calculate when this event should fire
                event_offset = (event.ts - self._event_start).total_seconds()
                target_time = self._replay_start + timedelta(
                    seconds=event_offset / self.replay_config.time_scale
                )

                # Wait until it's time
                now = datetime.now(timezone.utc)
                if target_time > now:
                    delay = (target_time - now).total_seconds()
                    await asyncio.sleep(delay)

            self._current_index += 1
            yield event

    def add_events(self, events: list[SensorEvent]) -> None:
        """Add more events to the replay queue."""
        self._events.extend(events)

    def get_duration(self) -> Optional[timedelta]:
        """Get the total duration of all events."""
        if not self._events:
            return None
        return self._events[-1].ts - self._events[0].ts

    def get_sensor_ids(self) -> set[str]:
        """Get all unique sensor IDs in the replay data."""
        return {e.sensor_id for e in self._events}

    def get_event_count(self) -> int:
        """Get total number of events."""
        return len(self._events)

    def get_progress(self) -> float:
        """Get replay progress (0.0 - 1.0)."""
        if not self._events:
            return 1.0
        return self._current_index / len(self._events)

    def seek(self, offset_ms: float) -> None:
        """Seek to a position in the replay."""
        if not self._events:
            return

        first_ts = self._events[0].ts
        target_ts = first_ts + timedelta(milliseconds=offset_ms)

        self._current_index = 0
        while (
            self._current_index < len(self._events)
            and self._events[self._current_index].ts < target_ts
        ):
            self._current_index += 1

    def reset(self) -> None:
        """Reset replay to the beginning."""
        self._current_index = 0
        self._replay_start = None


def save_events_to_json(events: list[SensorEvent], path: str) -> None:
    """Save events to a JSON file for later replay."""
    with open(path, "w") as f:
        json.dump([e.to_dict() for e in events], f, indent=2)


def save_events_to_jsonl(events: list[SensorEvent], path: str) -> None:
    """Save events to a JSON Lines file."""
    with open(path, "w") as f:
        for event in events:
            f.write(json.dumps(event.to_dict()) + "\n")
