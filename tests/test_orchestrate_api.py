"""Tests for the orchestrate API endpoints."""
from __future__ import annotations

import os
import tempfile

# Isolate test DB before any app imports.
_tmpdir = tempfile.mkdtemp(prefix="otbot_orchestrate_api_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "orchestrate_api_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import init_db  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True)
def _setup_db():
    """Initialize a fresh DB for each test."""
    get_settings.cache_clear()
    init_db()
    # Clear module-level campaign tracking dicts between tests
    from app.api.v1.endpoints.orchestrate import (
        _campaign_errors,
        _campaign_results,
        _running_campaigns,
    )
    _running_campaigns.clear()
    _campaign_results.clear()
    _campaign_errors.clear()


# ---------------------------------------------------------------------------
# POST /orchestrate/start
# ---------------------------------------------------------------------------


class TestOrchestrateStart:
    def test_orchestrate_start_dry_run(self):
        """POST to /orchestrate/start with dry_run=True returns campaign_id."""
        resp = client.post(
            "/api/v1/orchestrate/start",
            json={
                "contract_id": "tc-test-001",
                "objective_kpi": "overpotential_mv",
                "direction": "minimize",
                "max_rounds": 3,
                "batch_size": 2,
                "dimensions": [
                    {
                        "param_name": "ratio",
                        "param_type": "number",
                        "min_value": 0.1,
                        "max_value": 10.0,
                    },
                ],
                "protocol_template": {
                    "steps": [
                        {
                            "step_key": "s1",
                            "primitive": "robot.home",
                            "params": {},
                            "depends_on": [],
                            "resources": [],
                        },
                    ],
                },
                "policy_snapshot": {
                    "max_temp_c": 95.0,
                    "max_volume_ul": 1000.0,
                    "allowed_primitives": ["robot.home"],
                },
                "dry_run": True,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "campaign_id" in body
        assert body["campaign_id"].startswith("orch-")
        assert body["status"] == "started"

    def test_orchestrate_start_plan_only(self):
        """POST to /orchestrate/start with plan_only=True returns campaign_id."""
        resp = client.post(
            "/api/v1/orchestrate/start",
            json={
                "contract_id": "tc-test-002",
                "objective_kpi": "overpotential_mv",
                "direction": "minimize",
                "max_rounds": 5,
                "batch_size": 3,
                "dimensions": [
                    {
                        "param_name": "ratio",
                        "param_type": "number",
                        "min_value": 0.1,
                        "max_value": 10.0,
                    },
                ],
                "protocol_template": {"steps": []},
                "plan_only": True,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "campaign_id" in body
        assert body["status"] == "started"


# ---------------------------------------------------------------------------
# GET /orchestrate/{campaign_id}/status
# ---------------------------------------------------------------------------


class TestOrchestrateStatus:
    def test_orchestrate_status_not_found(self):
        """GET /orchestrate/nonexistent/status returns 404."""
        resp = client.get("/api/v1/orchestrate/nonexistent/status")
        assert resp.status_code == 404

    def test_orchestrate_status_after_start(self):
        """Status endpoint returns valid response for a started campaign."""
        # Start a campaign first
        start_resp = client.post(
            "/api/v1/orchestrate/start",
            json={
                "contract_id": "tc-test-003",
                "objective_kpi": "overpotential_mv",
                "direction": "minimize",
                "max_rounds": 1,
                "batch_size": 1,
                "dimensions": [
                    {
                        "param_name": "ratio",
                        "param_type": "number",
                        "min_value": 0.1,
                        "max_value": 10.0,
                    },
                ],
                "protocol_template": {"steps": []},
                "plan_only": True,
            },
        )
        campaign_id = start_resp.json()["campaign_id"]

        # Check status -- could be running or completed depending on timing
        status_resp = client.get(f"/api/v1/orchestrate/{campaign_id}/status")
        assert status_resp.status_code == 200
        body = status_resp.json()
        assert body["campaign_id"] == campaign_id
        assert body["status"] in ("running", "completed", "failed")


# ---------------------------------------------------------------------------
# POST /orchestrate/{campaign_id}/stop
# ---------------------------------------------------------------------------


class TestOrchestrateStop:
    def test_orchestrate_stop_not_found(self):
        """POST /orchestrate/nonexistent/stop returns 404."""
        resp = client.post("/api/v1/orchestrate/nonexistent/stop")
        assert resp.status_code == 404

    def test_orchestrate_stop_after_start(self):
        """Stop endpoint returns valid response for a started campaign."""
        # Start a campaign
        start_resp = client.post(
            "/api/v1/orchestrate/start",
            json={
                "contract_id": "tc-test-004",
                "objective_kpi": "overpotential_mv",
                "direction": "minimize",
                "max_rounds": 100,
                "batch_size": 10,
                "dimensions": [
                    {
                        "param_name": "ratio",
                        "param_type": "number",
                        "min_value": 0.1,
                        "max_value": 10.0,
                    },
                ],
                "protocol_template": {"steps": []},
                "plan_only": True,
            },
        )
        campaign_id = start_resp.json()["campaign_id"]

        # Stop it
        stop_resp = client.post(f"/api/v1/orchestrate/{campaign_id}/stop")
        assert stop_resp.status_code == 200
        body = stop_resp.json()
        assert body["campaign_id"] == campaign_id
        assert body["status"] in ("cancelled", "already_finished")
