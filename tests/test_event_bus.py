"""Tests for the in-process event bus (app/services/event_bus.py)."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.services.event_bus import EventBus, EventMessage, Subscription


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(
    *,
    run_id: str | None = "run-1",
    action: str = "step.state_changed",
    event_id: str = "evt-1",
) -> EventMessage:
    return EventMessage(
        id=event_id,
        run_id=run_id,
        actor="worker",
        action=action,
        details={"status": "running"},
        created_at="2026-02-08T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# EventBus core tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_publish_to_global_subscriber() -> None:
    """Global subscriber (run_id=None) receives all events."""
    bus = EventBus()
    await bus.start()
    sub = await bus.subscribe(run_id=None)

    bus.publish(_msg(run_id="run-1"))
    bus.publish(_msg(run_id="run-2", event_id="evt-2"))

    # Allow event loop to process call_soon_threadsafe callbacks
    await asyncio.sleep(0.05)

    assert sub.queue.qsize() == 2
    e1 = await asyncio.wait_for(sub.queue.get(), timeout=1)
    e2 = await asyncio.wait_for(sub.queue.get(), timeout=1)
    assert e1 is not None and e1.run_id == "run-1"
    assert e2 is not None and e2.run_id == "run-2"

    await bus.stop()


@pytest.mark.anyio
async def test_publish_filtered_by_run_id() -> None:
    """Subscriber scoped to run_id only receives matching events."""
    bus = EventBus()
    await bus.start()
    sub = await bus.subscribe(run_id="run-A")

    bus.publish(_msg(run_id="run-A"))
    bus.publish(_msg(run_id="run-B", event_id="evt-2"))
    bus.publish(_msg(run_id="run-A", event_id="evt-3"))

    await asyncio.sleep(0.05)

    assert sub.queue.qsize() == 2
    e1 = await asyncio.wait_for(sub.queue.get(), timeout=1)
    e2 = await asyncio.wait_for(sub.queue.get(), timeout=1)
    assert e1 is not None and e1.id == "evt-1"
    assert e2 is not None and e2.id == "evt-3"

    await bus.stop()


@pytest.mark.anyio
async def test_fan_out_to_multiple_subscribers() -> None:
    """All matching subscribers receive the same event."""
    bus = EventBus()
    await bus.start()
    sub1 = await bus.subscribe(run_id=None)
    sub2 = await bus.subscribe(run_id=None)
    sub3 = await bus.subscribe(run_id="run-1")

    bus.publish(_msg(run_id="run-1"))
    await asyncio.sleep(0.05)

    assert sub1.queue.qsize() == 1
    assert sub2.queue.qsize() == 1
    assert sub3.queue.qsize() == 1

    await bus.stop()


@pytest.mark.anyio
async def test_publish_before_start_is_noop() -> None:
    """Publishing before start() should not raise."""
    bus = EventBus()
    # Not started — _loop is None
    bus.publish(_msg())
    # No error, no subscribers, nothing happens


@pytest.mark.anyio
async def test_stop_cancels_subscriptions() -> None:
    """Stopping the bus sends None sentinel to all subscribers."""
    bus = EventBus()
    await bus.start()
    sub = await bus.subscribe()

    await bus.stop()

    # The sentinel should be in the queue
    item = await asyncio.wait_for(sub.queue.get(), timeout=1)
    assert item is None


@pytest.mark.anyio
async def test_unsubscribe_removes_subscriber() -> None:
    """After unsubscribe, events are no longer delivered."""
    bus = EventBus()
    await bus.start()
    sub = await bus.subscribe(run_id=None)

    await bus.unsubscribe(sub)

    bus.publish(_msg())
    await asyncio.sleep(0.05)

    assert sub.queue.empty()

    await bus.stop()


@pytest.mark.anyio
async def test_unsubscribe_idempotent() -> None:
    """Unsubscribing twice should not raise."""
    bus = EventBus()
    await bus.start()
    sub = await bus.subscribe()

    await bus.unsubscribe(sub)
    await bus.unsubscribe(sub)  # no error

    await bus.stop()


@pytest.mark.anyio
async def test_full_queue_drops_event() -> None:
    """When a subscriber's queue is full, events are dropped (not crash)."""
    bus = EventBus(max_queue_size=2)
    await bus.start()
    sub = await bus.subscribe(run_id=None)

    # Fill the queue
    bus.publish(_msg(event_id="e1"))
    bus.publish(_msg(event_id="e2"))
    await asyncio.sleep(0.05)
    assert sub.queue.qsize() == 2

    # This should be dropped (queue full)
    bus.publish(_msg(event_id="e3"))
    await asyncio.sleep(0.05)
    assert sub.queue.qsize() == 2  # still 2, not 3

    await bus.stop()


@pytest.mark.anyio
async def test_subscription_async_iteration() -> None:
    """Subscription supports async for ... in sub pattern."""
    bus = EventBus()
    await bus.start()
    sub = await bus.subscribe(run_id=None)

    bus.publish(_msg(event_id="e1"))
    bus.publish(_msg(event_id="e2"))
    await asyncio.sleep(0.05)

    # Cancel after publishing so iteration terminates
    sub.cancel()

    collected: list[str] = []
    async for event in sub:
        collected.append(event.id)

    assert collected == ["e1", "e2"]

    await bus.stop()


@pytest.mark.anyio
async def test_subscription_cancel_terminates_iteration() -> None:
    """Calling cancel() on a blocked subscriber unblocks the iterator."""
    bus = EventBus()
    await bus.start()
    sub = await bus.subscribe()

    collected: list[str] = []

    async def reader() -> None:
        async for event in sub:
            collected.append(event.id)

    task = asyncio.create_task(reader())
    await asyncio.sleep(0.05)

    # Reader should be blocked waiting — cancel it
    sub.cancel()
    await asyncio.wait_for(task, timeout=2)

    assert collected == []

    await bus.stop()


@pytest.mark.anyio
async def test_publish_after_stop_is_noop() -> None:
    """Publishing after stop() should not raise."""
    bus = EventBus()
    await bus.start()
    await bus.stop()

    bus.publish(_msg())
    # No error — _loop is None after stop


# ---------------------------------------------------------------------------
# Integration with audit.record_event()
# ---------------------------------------------------------------------------

import os
import tempfile

_tmpdir = tempfile.mkdtemp(prefix="otbot_bus_test_")
os.environ.setdefault("DATA_DIR", _tmpdir)
os.environ.setdefault("DB_PATH", os.path.join(_tmpdir, "bus_test.db"))
os.environ.setdefault("OBJECT_STORE_DIR", os.path.join(_tmpdir, "obj"))

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import init_db, run_txn  # noqa: E402
from app.services.audit import record_event, set_event_bus  # noqa: E402


@pytest.fixture(autouse=True)
def _init_test_db() -> None:
    get_settings.cache_clear()
    init_db()


@pytest.mark.anyio
async def test_record_event_publishes_to_bus() -> None:
    """record_event() should publish to the event bus when wired."""
    bus = EventBus()
    await bus.start()
    set_event_bus(bus)

    sub = await bus.subscribe(run_id=None)

    try:
        run_txn(
            lambda conn: record_event(
                conn,
                run_id=None,
                actor="test",
                action="test.event",
                details={"key": "value"},
            )
        )

        await asyncio.sleep(0.05)

        assert sub.queue.qsize() == 1
        event = await asyncio.wait_for(sub.queue.get(), timeout=1)
        assert event is not None
        assert event.run_id is None
        assert event.action == "test.event"
        assert event.details == {"key": "value"}
    finally:
        set_event_bus(None)
        await bus.stop()


@pytest.mark.anyio
async def test_record_event_without_bus() -> None:
    """record_event() works normally when no bus is set (backward compat)."""
    set_event_bus(None)

    # Should not raise
    run_txn(
        lambda conn: record_event(
            conn,
            run_id=None,
            actor="test",
            action="test.noop",
            details={},
        )
    )
