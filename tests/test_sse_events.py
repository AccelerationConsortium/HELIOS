"""Regression tests for the campaign SSE event pipeline.

Covers:
- no double-persistence of orchestrator events (events arriving with a
  ``_seq`` already attached must not be written to the DB again)
- seq assignment for events that have not been persisted yet
- the replay→live handover: events published while the DB replay is
  streaming must not be lost, and overlapping events must not be
  delivered twice
- subscriber-queue cleanup (no leaked campaign keys)
- bounded onboarding session store
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.api.v1.endpoints.orchestrate_events import (
    _campaign_event_generator,
    _campaign_queues,
    publish_campaign_event,
)


class _FakeRequest:
    """Minimal stand-in for starlette.Request in generator tests."""

    def __init__(self) -> None:
        self.disconnected = False

    async def is_disconnected(self) -> bool:
        return self.disconnected


@pytest.fixture(autouse=True)
def _clean_queues():
    _campaign_queues.clear()
    yield
    _campaign_queues.clear()


@pytest.fixture()
def fake_log(monkeypatch):
    """Capture log_event calls instead of writing to SQLite."""
    calls: list[tuple[str, str, dict[str, Any]]] = []
    seq = {"n": 0}

    def _log_event(campaign_id: str, event_type: str, payload: dict[str, Any]) -> int:
        calls.append((campaign_id, event_type, payload))
        seq["n"] += 1
        return seq["n"]

    import app.services.campaign_events as ce

    monkeypatch.setattr(ce, "log_event", _log_event)
    return calls


def test_publish_skips_already_persisted_events(fake_log):
    """Orchestrator events carry _seq from _emit; publishing them must not
    insert a second row (previously every event was stored twice)."""
    publish_campaign_event("camp-a", {"type": "round_start", "_seq": 42})
    assert fake_log == []


def test_publish_persists_and_attaches_seq(fake_log):
    event = {"type": "demo_event"}
    publish_campaign_event("camp-a", event)
    assert len(fake_log) == 1
    assert event["_seq"] == 1


def test_publish_delivers_to_subscribers(fake_log):
    q: asyncio.Queue = asyncio.Queue()
    _campaign_queues["camp-b"] = [q]
    publish_campaign_event("camp-b", {"type": "x", "_seq": 7})
    assert q.get_nowait()["_seq"] == 7


async def test_generator_no_gap_and_no_duplicates(monkeypatch, fake_log):
    """Events published during DB replay must be delivered exactly once.

    The replay snapshot contains seqs 1-2. Event 3 is published while the
    replay is being consumed (it only exists in the live queue), and event
    2 is also published live (overlap with replay). The client must see
    1, 2, 3 — no loss, no duplicate.
    """
    import app.services.campaign_events as ce

    def _replay(campaign_id: str, after_seq: int = 0):
        # At this point the live queue must already be registered,
        # otherwise events published "now" would be lost.
        assert _campaign_queues.get("camp-c"), "queue must be subscribed before replay"
        publish_campaign_event("camp-c", {"type": "e2", "_seq": 2})  # overlap
        publish_campaign_event("camp-c", {"type": "e3", "_seq": 3})  # replay gap
        _campaign_queues["camp-c"][0].put_nowait(None)  # sentinel: stop generator
        return [
            {"seq": 1, "event_type": "e1", "payload": {"type": "e1"}},
            {"seq": 2, "event_type": "e2", "payload": {"type": "e2"}},
        ]

    monkeypatch.setattr(ce, "replay_events", _replay)

    chunks = []
    async for chunk in _campaign_event_generator(_FakeRequest(), "camp-c"):
        chunks.append(chunk)

    ids = [line[4:] for c in chunks for line in c.splitlines() if line.startswith("id: ")]
    assert ids == ["1", "2", "3"], f"expected exactly-once delivery, got ids={ids}"


async def test_generator_cleans_up_queue_registry(monkeypatch, fake_log):
    import app.services.campaign_events as ce

    def _replay(campaign_id: str, after_seq: int = 0):
        _campaign_queues["camp-d"][0].put_nowait(None)  # sentinel: stop immediately
        return []

    monkeypatch.setattr(ce, "replay_events", _replay)

    async for _ in _campaign_event_generator(_FakeRequest(), "camp-d"):
        pass

    assert "camp-d" not in _campaign_queues, "campaign key must not leak"


def test_publish_survives_full_subscriber_queue(fake_log):
    q: asyncio.Queue = asyncio.Queue(maxsize=1)
    q.put_nowait({"type": "old"})
    _campaign_queues["camp-e"] = [q]
    # Must not raise even though the queue is full
    publish_campaign_event("camp-e", {"type": "new", "_seq": 9})


def test_onboarding_session_store_is_bounded():
    from app.api.v1.endpoints import onboarding

    onboarding._onboarding_sessions.clear()
    for i in range(onboarding._MAX_ONBOARDING_SESSIONS + 50):
        onboarding._store_session(f"onb-{i}", {"i": i})

    assert len(onboarding._onboarding_sessions) == onboarding._MAX_ONBOARDING_SESSIONS
    # Oldest entries were evicted, newest retained
    assert "onb-0" not in onboarding._onboarding_sessions
    assert f"onb-{onboarding._MAX_ONBOARDING_SESSIONS + 49}" in onboarding._onboarding_sessions
    onboarding._onboarding_sessions.clear()
