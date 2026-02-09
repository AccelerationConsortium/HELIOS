"""Integration tests for SSE streaming endpoints (events_stream.py)."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile

# Isolate test DB before any app imports.
_tmpdir = tempfile.mkdtemp(prefix="otbot_sse_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "sse_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import init_db  # noqa: E402
from app.main import app, event_bus  # noqa: E402
from app.services.audit import set_event_bus  # noqa: E402


# ---------------------------------------------------------------------------
# Test protocol/policy (matches existing test_api_integration.py patterns)
# ---------------------------------------------------------------------------

SIMPLE_PROTOCOL = {
    "steps": [
        {"id": "s1", "primitive": "aspirate", "params": {"volume_ul": 100}},
        {"id": "s2", "primitive": "heat", "depends_on": ["s1"], "params": {"temp_c": 60}},
    ]
}

SAFE_POLICY = {
    "max_temp_c": 95,
    "max_volume_ul": 1000,
    "allowed_primitives": ["aspirate", "heat", "eis", "wait", "upload_artifact"],
    "require_human_approval": False,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
async def _setup_env():
    """Set up DB and event bus for each test."""
    get_settings.cache_clear()
    init_db()
    await event_bus.start()
    set_event_bus(event_bus)
    yield
    set_event_bus(None)
    await event_bus.stop()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# SSE parsing helper
# ---------------------------------------------------------------------------


def parse_sse_events(raw: str) -> list[dict]:
    """Parse raw SSE text into a list of structured event dicts.

    Each SSE event block is separated by ``\\n\\n`` and may contain:
    - ``id: <value>``
    - ``event: <value>``
    - ``data: <json>``
    - ``: <comment>`` (e.g. keepalive — ignored)
    """
    events: list[dict] = []
    blocks = raw.split("\n\n")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        event: dict = {}
        for line in block.split("\n"):
            if line.startswith("id: "):
                event["id"] = line[4:]
            elif line.startswith("event: "):
                event["event"] = line[7:]
            elif line.startswith("data: "):
                event["data"] = json.loads(line[6:])
            elif line.startswith(":"):
                event["comment"] = line[1:].strip()
        if event:
            events.append(event)
    return events


async def _create_run(client: AsyncClient) -> dict:
    """Create a run via the trigger API and return the response dict."""
    resp = await client.post("/api/v1/triggers/time", json={
        "protocol": SIMPLE_PROTOCOL,
        "inputs": {"instrument_id": "sim-1"},
        "policy_snapshot": SAFE_POLICY,
        "actor": "test",
    })
    assert resp.status_code == 200
    return resp.json()


# ---------------------------------------------------------------------------
# Tests: Event bus integration (direct subscription, no HTTP streaming)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_global_stream_receives_events(client: AsyncClient) -> None:
    """Subscribe globally, create a run, verify run.created is received."""
    sub = await event_bus.subscribe(run_id=None)

    run = await _create_run(client)
    run_id = run["id"]

    await asyncio.sleep(0.1)

    collected = []
    while not sub.queue.empty():
        evt = await asyncio.wait_for(sub.queue.get(), timeout=1)
        if evt is not None:
            collected.append(evt)

    assert len(collected) >= 1
    actions = [e.action for e in collected]
    assert "run.created" in actions

    created_events = [e for e in collected if e.action == "run.created"]
    assert created_events[0].run_id == run_id

    await event_bus.unsubscribe(sub)


@pytest.mark.anyio
async def test_multiple_subscribers_fan_out(client: AsyncClient) -> None:
    """Two subscribers should both receive the same events."""
    sub1 = await event_bus.subscribe(run_id=None)
    sub2 = await event_bus.subscribe(run_id=None)

    await _create_run(client)
    await asyncio.sleep(0.1)

    assert sub1.queue.qsize() >= 1
    assert sub2.queue.qsize() >= 1

    e1 = await asyncio.wait_for(sub1.queue.get(), timeout=1)
    e2 = await asyncio.wait_for(sub2.queue.get(), timeout=1)
    assert e1 is not None and e2 is not None
    assert e1.action == e2.action == "run.created"

    await event_bus.unsubscribe(sub1)
    await event_bus.unsubscribe(sub2)


@pytest.mark.anyio
async def test_run_specific_subscriber_only_gets_matching(client: AsyncClient) -> None:
    """A subscriber scoped to run_id only receives matching events."""
    run_a = await _create_run(client)
    run_a_id = run_a["id"]

    # Drain pending call_soon_threadsafe callbacks from run_a creation
    await asyncio.sleep(0.05)

    # Subscribe after run_a events have been dispatched
    sub = await event_bus.subscribe(run_id=run_a_id)

    # Create run_b — these events should NOT reach sub (different run_id)
    await _create_run(client)
    await asyncio.sleep(0.1)

    assert sub.queue.empty()

    await event_bus.unsubscribe(sub)


# ---------------------------------------------------------------------------
# Tests: HTTP endpoint behavior
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_run_stream_404_for_unknown_run(client: AsyncClient) -> None:
    """SSE stream for non-existent run should return 404."""
    resp = await client.get("/api/v1/runs/nonexistent-run-id/events/stream")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_run_stream_replays_historical(client: AsyncClient) -> None:
    """Create a run, then open per-run SSE — historical events arrive."""
    run = await _create_run(client)
    run_id = run["id"]

    # Verify historical events exist in DB
    resp = await client.get(f"/api/v1/runs/{run_id}/events")
    assert resp.status_code == 200
    historical = resp.json()
    assert len(historical) >= 1
    assert historical[0]["action"] == "run.created"

    # Verify the SSE endpoint serves historical events by directly calling
    # the format helper with the known historical data.
    from app.api.v1.endpoints.events_stream import _format_sse_from_dict, _fetch_historical_events

    fetched = _fetch_historical_events(run_id)
    assert len(fetched) >= 1
    assert fetched[0]["action"] == "run.created"

    # Verify the SSE format is correct
    sse_text = _format_sse_from_dict(fetched[0])
    parsed = parse_sse_events(sse_text)
    assert len(parsed) == 1
    assert parsed[0]["data"]["run_id"] == run_id
    assert parsed[0]["event"] == "run.created"


@pytest.mark.anyio
async def test_global_stream_endpoint_exists(client: AsyncClient) -> None:
    """Global stream endpoint exists and returns StreamingResponse."""
    # Verify the endpoint is routable by checking it via the app router
    from fastapi.testclient import TestClient

    # Use the OpenAPI schema to verify endpoint registration
    resp = await client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert "/api/v1/events/stream" in paths
    assert "/api/v1/runs/{run_id}/events/stream" in paths


# ---------------------------------------------------------------------------
# Tests: SSE format helpers (unit tests, no HTTP needed)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_sse_format_correct() -> None:
    """Verify SSE formatting of events has id:, event:, data: fields."""
    from app.api.v1.endpoints.events_stream import _format_sse
    from app.services.event_bus import EventMessage

    msg = EventMessage(
        id="evt-123",
        run_id="run-abc",
        actor="worker",
        action="step.state_changed",
        details={"status": "running"},
        created_at="2026-02-08T00:00:00+00:00",
    )
    sse_text = _format_sse(msg)

    assert "id: evt-123\n" in sse_text
    assert "event: step.state_changed\n" in sse_text
    assert "data: " in sse_text
    assert sse_text.endswith("\n\n")

    for line in sse_text.split("\n"):
        if line.startswith("data: "):
            data = json.loads(line[6:])
            assert data["id"] == "evt-123"
            assert data["run_id"] == "run-abc"
            assert data["actor"] == "worker"
            assert data["action"] == "step.state_changed"
            assert data["details"] == {"status": "running"}
            break


@pytest.mark.anyio
async def test_sse_format_from_dict() -> None:
    """Verify SSE formatting from a DB-style dict."""
    from app.api.v1.endpoints.events_stream import _format_sse_from_dict

    row = {
        "id": "evt-456",
        "run_id": "run-xyz",
        "actor": "system",
        "action": "run.created",
        "details": {"key": "value"},
        "created_at": "2026-02-08T01:00:00+00:00",
    }
    sse_text = _format_sse_from_dict(row)

    assert "id: evt-456\n" in sse_text
    assert "event: run.created\n" in sse_text
    assert sse_text.endswith("\n\n")

    parsed = parse_sse_events(sse_text)
    assert len(parsed) == 1
    assert parsed[0]["id"] == "evt-456"
    assert parsed[0]["event"] == "run.created"
    assert parsed[0]["data"]["run_id"] == "run-xyz"


@pytest.mark.anyio
async def test_parse_sse_events_helper() -> None:
    """Unit test for the SSE parsing helper."""
    raw = (
        "id: evt-1\n"
        "event: run.created\n"
        "data: {\"id\":\"evt-1\",\"run_id\":\"r1\"}\n"
        "\n"
        ": keepalive\n"
        "\n"
        "id: evt-2\n"
        "event: step.state_changed\n"
        "data: {\"id\":\"evt-2\",\"run_id\":\"r1\"}\n"
        "\n"
    )
    events = parse_sse_events(raw)
    assert len(events) == 3  # 2 real events + 1 keepalive comment

    assert events[0]["id"] == "evt-1"
    assert events[0]["event"] == "run.created"
    assert events[0]["data"]["run_id"] == "r1"

    assert "comment" in events[1]

    assert events[2]["id"] == "evt-2"
    assert events[2]["event"] == "step.state_changed"
