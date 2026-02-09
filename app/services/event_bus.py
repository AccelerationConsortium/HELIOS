"""In-process event bus for SSE real-time streaming.

Thread-safe: sync code (worker threads) calls ``publish()`` which uses
``call_soon_threadsafe`` to dispatch onto the asyncio event loop.
Async subscribers receive events via per-subscriber ``asyncio.Queue`` instances.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventMessage:
    """Immutable event payload that flows through the bus."""

    id: str
    run_id: str | None
    actor: str
    action: str
    details: dict[str, Any]
    created_at: str


# ---------------------------------------------------------------------------
# Subscription handle
# ---------------------------------------------------------------------------


class Subscription:
    """Handle for a single SSE subscriber.

    Iterate asynchronously to receive events.  Iteration ends when
    ``cancel()`` is called (a ``None`` sentinel is placed on the queue).
    """

    def __init__(
        self,
        queue: asyncio.Queue[EventMessage | None],
        run_id: str | None = None,
    ) -> None:
        self.queue = queue
        self.run_id = run_id  # None ⇒ global (all events)

    async def __aiter__(self):  # noqa: ANN204
        while True:
            event = await self.queue.get()
            if event is None:
                break
            yield event

    def cancel(self) -> None:
        """Signal this subscription to stop iterating."""
        try:
            self.queue.put_nowait(None)
        except asyncio.QueueFull:
            # Queue is full — force-drain one item and retry
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self.queue.put_nowait(None)
            except asyncio.QueueFull:
                pass


# ---------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------


class EventBus:
    """In-process fan-out event bus for SSE streaming.

    Lifecycle::

        bus = EventBus()
        await bus.start()      # call from FastAPI lifespan
        ...
        await bus.stop()       # call on shutdown
    """

    def __init__(self, max_queue_size: int = 256) -> None:
        self._subscribers: list[Subscription] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._max_queue_size = max_queue_size

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Capture the running event loop.  Call from FastAPI lifespan."""
        self._loop = asyncio.get_running_loop()

    async def stop(self) -> None:
        """Cancel every active subscription and clear the list."""
        for sub in list(self._subscribers):
            sub.cancel()
        self._subscribers.clear()
        self._loop = None

    # -- publish (thread-safe) -----------------------------------------------

    def publish(self, event: EventMessage) -> None:
        """Publish *event* to all matching subscribers.

        Safe to call from any thread — the actual dispatch is scheduled
        on the event loop via ``call_soon_threadsafe``.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            return  # bus not started or already stopped
        try:
            loop.call_soon_threadsafe(self._dispatch, event)
        except RuntimeError:
            # Event loop already closed — ignore silently.
            pass

    def _dispatch(self, event: EventMessage) -> None:
        """Fan out *event* to every matching subscriber (runs on event loop)."""
        for sub in list(self._subscribers):
            # Filter by run_id when the subscription is scoped.
            if sub.run_id is not None and event.run_id != sub.run_id:
                continue
            try:
                sub.queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "SSE subscriber queue full — dropping event %s (%s)",
                    event.id,
                    event.action,
                )

    # -- subscribe / unsubscribe ---------------------------------------------

    async def subscribe(self, run_id: str | None = None) -> Subscription:
        """Create a new subscription.

        *run_id*=None subscribes to **all** events (global stream).
        """
        queue: asyncio.Queue[EventMessage | None] = asyncio.Queue(
            maxsize=self._max_queue_size,
        )
        sub = Subscription(queue=queue, run_id=run_id)
        self._subscribers.append(sub)
        return sub

    async def unsubscribe(self, sub: Subscription) -> None:
        """Remove *sub* from the subscriber list."""
        try:
            self._subscribers.remove(sub)
        except ValueError:
            pass  # already removed
