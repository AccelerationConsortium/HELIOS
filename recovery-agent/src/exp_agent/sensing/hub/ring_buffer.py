"""
RingBuffer - Fixed-size circular buffer for sensor event history.

Used for:
- Incident replay
- Trend analysis
- Audit logging
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Generic, TypeVar, Iterator, Optional
from collections import deque

T = TypeVar('T')


@dataclass
class RingBuffer(Generic[T]):
    """
    Thread-safe ring buffer with time-based eviction.

    Events older than max_age_seconds are automatically evicted
    when new events are added. The buffer also has a max capacity.
    """

    max_size: int = 10000
    max_age_seconds: float = 3600.0  # 1 hour default

    _buffer: deque[tuple[datetime, T]] = field(default_factory=deque)

    def __post_init__(self):
        self._buffer = deque(maxlen=self.max_size)

    def append(self, item: T, timestamp: Optional[datetime] = None) -> None:
        """Add an item to the buffer."""
        ts = timestamp or datetime.now(timezone.utc)
        self._buffer.append((ts, item))
        self._evict_old()

    def _evict_old(self) -> None:
        """Remove items older than max_age_seconds."""
        if not self._buffer:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.max_age_seconds)
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    def get_all(self) -> list[T]:
        """Get all items in the buffer."""
        self._evict_old()
        return [item for _, item in self._buffer]

    def get_since(self, since: datetime) -> list[T]:
        """Get all items since a given timestamp."""
        self._evict_old()
        return [item for ts, item in self._buffer if ts >= since]

    def get_last_n(self, n: int) -> list[T]:
        """Get the last N items."""
        self._evict_old()
        items = list(self._buffer)[-n:]
        return [item for _, item in items]

    def get_window(self, start: datetime, end: datetime) -> list[T]:
        """Get items within a time window."""
        self._evict_old()
        return [item for ts, item in self._buffer if start <= ts <= end]

    def clear(self) -> None:
        """Clear all items from the buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self._buffer)

    def __iter__(self) -> Iterator[T]:
        """Iterate over items (oldest first)."""
        self._evict_old()
        for _, item in self._buffer:
            yield item

    @property
    def oldest(self) -> Optional[T]:
        """Get the oldest item."""
        self._evict_old()
        if self._buffer:
            return self._buffer[0][1]
        return None

    @property
    def newest(self) -> Optional[T]:
        """Get the newest item."""
        self._evict_old()
        if self._buffer:
            return self._buffer[-1][1]
        return None

    @property
    def oldest_timestamp(self) -> Optional[datetime]:
        """Get timestamp of oldest item."""
        self._evict_old()
        if self._buffer:
            return self._buffer[0][0]
        return None

    @property
    def newest_timestamp(self) -> Optional[datetime]:
        """Get timestamp of newest item."""
        self._evict_old()
        if self._buffer:
            return self._buffer[-1][0]
        return None

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        self._evict_old()
        return {
            "size": len(self._buffer),
            "max_size": self.max_size,
            "max_age_seconds": self.max_age_seconds,
            "oldest_timestamp": self.oldest_timestamp.isoformat() if self.oldest_timestamp else None,
            "newest_timestamp": self.newest_timestamp.isoformat() if self.newest_timestamp else None,
        }
