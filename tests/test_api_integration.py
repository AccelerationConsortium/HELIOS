"""Integration tests for the FastAPI endpoints — full lifecycle via HTTP."""
from __future__ import annotations

import os
import tempfile

# Isolate test DB before any app imports.
_tmpdir = tempfile.mkdtemp(prefix="otbot_api_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")

import pytest  # noqa: E402
from httpx import AsyncClient, ASGITransport  # noqa: E402

from app.core.config import get_settings  # noqa: E402


# Clear cached settings so test env vars take effect.
get_settings.cache_clear()

from app.core.db import init_db  # noqa: E402
from app.main import app  # noqa: E402


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


@pytest.fixture(autouse=True)
def _setup_db():
    get_settings.cache_clear()
    init_db()
    yield


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ─── health ─────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_health(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


# ─── campaigns ──────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_create_and_list_campaign(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/campaigns", json={
        "name": "nightly-run",
        "cadence_seconds": 3600,
        "protocol": SIMPLE_PROTOCOL,
        "inputs": {"instrument_id": "sim-1"},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "nightly-run"
    campaign_id = data["id"]

    resp = await client.get("/api/v1/campaigns")
    assert resp.status_code == 200
    ids = [c["id"] for c in resp.json()]
    assert campaign_id in ids


# ─── trigger → run creation ─────────────────────────────────────────

@pytest.mark.anyio
async def test_time_trigger_creates_scheduled_run(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/triggers/time", json={
        "protocol": SIMPLE_PROTOCOL,
        "inputs": {"instrument_id": "sim-1"},
        "policy_snapshot": SAFE_POLICY,
        "actor": "test",
    })
    assert resp.status_code == 200
    run = resp.json()
    assert run["status"] == "scheduled"
    assert run["trigger_type"] == "time"
    assert len(run["steps"]) == 2


@pytest.mark.anyio
async def test_event_trigger_creates_run(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/triggers/event", json={
        "protocol": SIMPLE_PROTOCOL,
        "inputs": {"instrument_id": "sim-1"},
        "policy_snapshot": SAFE_POLICY,
        "actor": "sensor",
        "payload": {"sensor": "temp", "value": 42},
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "scheduled"


@pytest.mark.anyio
async def test_external_trigger_creates_run(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/triggers/external", json={
        "protocol": SIMPLE_PROTOCOL,
        "inputs": {"instrument_id": "sim-1"},
        "policy_snapshot": SAFE_POLICY,
        "actor": "bo-optimizer",
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "scheduled"


# ─── safety rejection ───────────────────────────────────────────────

@pytest.mark.anyio
async def test_unsafe_protocol_rejected(client: AsyncClient) -> None:
    dangerous = {
        "steps": [
            {"id": "s1", "primitive": "heat", "params": {"temp_c": 200}},
        ]
    }
    resp = await client.post("/api/v1/triggers/time", json={
        "protocol": dangerous,
        "inputs": {"instrument_id": "sim-1"},
        "policy_snapshot": {"max_temp_c": 95, "allowed_primitives": ["heat"]},
        "actor": "test",
    })
    assert resp.status_code == 200
    run = resp.json()
    assert run["status"] == "rejected"
    assert run["rejection_reason"] is not None


# ─── approval flow ──────────────────────────────────────────────────

@pytest.mark.anyio
async def test_approval_flow(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/triggers/time", json={
        "protocol": SIMPLE_PROTOCOL,
        "inputs": {"instrument_id": "sim-1"},
        "policy_snapshot": {**SAFE_POLICY, "require_human_approval": True},
        "actor": "test",
    })
    assert resp.status_code == 200
    run = resp.json()
    assert run["status"] == "awaiting_approval"
    run_id = run["id"]

    resp = await client.post(f"/api/v1/runs/{run_id}/approve", json={
        "approver": "pi-smith",
        "reason": "looks good",
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "scheduled"


# ─── run detail + events ────────────────────────────────────────────

@pytest.mark.anyio
async def test_get_run_detail_and_events(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/triggers/time", json={
        "protocol": SIMPLE_PROTOCOL,
        "inputs": {"instrument_id": "sim-1"},
        "policy_snapshot": SAFE_POLICY,
        "actor": "test",
    })
    run_id = resp.json()["id"]

    resp = await client.get(f"/api/v1/runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == run_id

    resp = await client.get(f"/api/v1/runs/{run_id}/events")
    assert resp.status_code == 200
    events = resp.json()
    assert len(events) >= 1
    assert events[0]["action"] == "run.created"


# ─── list runs ──────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_list_runs(client: AsyncClient) -> None:
    await client.post("/api/v1/triggers/time", json={
        "protocol": SIMPLE_PROTOCOL,
        "inputs": {"instrument_id": "sim-1"},
        "policy_snapshot": SAFE_POLICY,
        "actor": "test",
    })
    resp = await client.get("/api/v1/runs")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


# ─── locks endpoint ─────────────────────────────────────────────────

@pytest.mark.anyio
async def test_list_locks(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/runs/meta/locks")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ─── campaign-based trigger ─────────────────────────────────────────

@pytest.mark.anyio
async def test_trigger_via_campaign_id(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/campaigns", json={
        "name": "test-campaign",
        "cadence_seconds": 60,
        "protocol": SIMPLE_PROTOCOL,
        "inputs": {"instrument_id": "sim-1"},
        "policy_snapshot": SAFE_POLICY,
    })
    campaign_id = resp.json()["id"]

    resp = await client.post("/api/v1/triggers/event", json={
        "campaign_id": campaign_id,
        "actor": "sensor",
        "payload": {"event": "threshold_exceeded"},
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "scheduled"
    assert resp.json()["campaign_id"] == campaign_id
