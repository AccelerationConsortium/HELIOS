"""Integration tests for the campaign initialization API endpoints."""
from __future__ import annotations

import os
import tempfile

# Isolate test DB before any app imports.
_tmpdir = tempfile.mkdtemp(prefix="otbot_initapi_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "initapi_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import connection, init_db  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True)
def _setup_db():
    get_settings.cache_clear()
    init_db()
    with connection() as conn:
        conn.execute("DELETE FROM conversation_sessions")
        conn.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROUND_1 = {
    "objective_type": "oer_screening",
    "objective_kpi": "overpotential_mv",
    "direction": "minimize",
    "acceptable_range_pct": 10.0,
}

ROUND_2 = {
    "available_instruments": ["ot2", "squidstat"],
    "max_temp_c": 95.0,
    "max_volume_ul": 1000.0,
    "hazardous_reagents": ["KOH"],
    "require_human_approval": False,
}

ROUND_3 = {
    "pattern_id": "oer_screening",
    "optional_steps": [],
}

ROUND_4 = {
    "strategy": "lhs",
    "batch_size": 10,
    "forbidden_combinations": "",
}

ROUND_5 = {
    "max_rounds": 20,
    "plateau_threshold": 0.01,
    "auto_approve_magnitude": 0.3,
    "human_gate_triggers": ["safety_boundary_change"],
}

ALL_ROUNDS = [ROUND_1, ROUND_2, ROUND_3, ROUND_4, ROUND_5]


def _start_session() -> str:
    """Start a session and return the session_id."""
    resp = client.post("/api/v1/init/start?created_by=test_user")
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    return data["session_id"]


def _complete_all_rounds(session_id: str) -> None:
    """Submit all 5 rounds with valid data."""
    for round_data in ALL_ROUNDS:
        resp = client.post(
            f"/api/v1/init/{session_id}/respond",
            json={"responses": round_data},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"], f"Round failed: {body.get('errors')}"


# ---------------------------------------------------------------------------
# /start
# ---------------------------------------------------------------------------


class TestStartEndpoint:
    def test_start_returns_200_with_session_id(self):
        resp = client.post("/api/v1/init/start?created_by=tester")
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["round_number"] == 1
        assert data["round_name"] == "Goal & Success Criteria"

    def test_start_returns_slots(self):
        resp = client.post("/api/v1/init/start")
        data = resp.json()
        assert "slots" in data
        slot_names = [s["name"] for s in data["slots"]]
        assert "objective_type" in slot_names

    def test_start_default_author_is_user(self):
        resp = client.post("/api/v1/init/start")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /respond
# ---------------------------------------------------------------------------


class TestRespondEndpoint:
    def test_valid_round_1_returns_success(self):
        sid = _start_session()
        resp = client.post(
            f"/api/v1/init/{sid}/respond",
            json={"responses": ROUND_1},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"]
        assert body["next_round"]["round_number"] == 2

    def test_invalid_round_1_returns_errors(self):
        sid = _start_session()
        bad = {**ROUND_1, "direction": "wrong"}
        resp = client.post(
            f"/api/v1/init/{sid}/respond",
            json={"responses": bad},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert not body["success"]
        assert "direction" in body["errors"]

    def test_nonexistent_session_returns_400(self):
        resp = client.post(
            "/api/v1/init/fake-session/respond",
            json={"responses": ROUND_1},
        )
        assert resp.status_code == 400

    def test_preview_included_in_response(self):
        sid = _start_session()
        resp = client.post(
            f"/api/v1/init/{sid}/respond",
            json={"responses": ROUND_1},
        )
        body = resp.json()
        assert "injection_pack_preview" in body
        preview = body["injection_pack_preview"]
        assert preview is not None
        assert "goal" in preview


# ---------------------------------------------------------------------------
# /back
# ---------------------------------------------------------------------------


class TestBackEndpoint:
    def test_back_from_round_2_returns_round_1(self):
        sid = _start_session()
        client.post(f"/api/v1/init/{sid}/respond", json={"responses": ROUND_1})
        resp = client.post(f"/api/v1/init/{sid}/back")
        assert resp.status_code == 200
        assert resp.json()["round_number"] == 1

    def test_back_from_round_1_stays(self):
        sid = _start_session()
        resp = client.post(f"/api/v1/init/{sid}/back")
        assert resp.status_code == 200
        assert resp.json()["round_number"] == 1


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------


class TestStatusEndpoint:
    def test_status_after_start(self):
        sid = _start_session()
        resp = client.get(f"/api/v1/init/{sid}/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == sid
        assert body["status"] == "active"
        assert body["current_round"] == 1

    def test_status_nonexistent_returns_404(self):
        resp = client.get("/api/v1/init/fake-id/status")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /round
# ---------------------------------------------------------------------------


class TestRoundEndpoint:
    def test_get_round_returns_current(self):
        sid = _start_session()
        resp = client.get(f"/api/v1/init/{sid}/round")
        assert resp.status_code == 200
        assert resp.json()["round_number"] == 1


# ---------------------------------------------------------------------------
# /confirm — full flow
# ---------------------------------------------------------------------------


class TestConfirmEndpoint:
    def test_confirm_creates_campaign(self):
        sid = _start_session()
        _complete_all_rounds(sid)
        resp = client.post(f"/api/v1/init/{sid}/confirm")
        assert resp.status_code == 200
        body = resp.json()
        assert "campaign_id" in body
        assert body["campaign_id"] is not None
        assert body["status"] == "campaign_created"
        assert "injection_pack" in body
        assert "diff_summary" in body

    def test_confirm_incomplete_returns_400(self):
        sid = _start_session()
        client.post(f"/api/v1/init/{sid}/respond", json={"responses": ROUND_1})
        resp = client.post(f"/api/v1/init/{sid}/confirm")
        assert resp.status_code == 400

    def test_diff_summary_has_entries(self):
        sid = _start_session()
        _complete_all_rounds(sid)
        resp = client.post(f"/api/v1/init/{sid}/confirm")
        body = resp.json()
        diff = body["diff_summary"]
        assert isinstance(diff, list)
        assert len(diff) > 0
        field_names = [d["field"] for d in diff]
        assert "objective_kpi" in field_names
        assert "pattern_id" in field_names

    def test_injection_pack_has_all_sections(self):
        sid = _start_session()
        _complete_all_rounds(sid)
        resp = client.post(f"/api/v1/init/{sid}/confirm")
        pack = resp.json()["injection_pack"]
        assert "goal" in pack
        assert "protocol" in pack
        assert "param_space" in pack
        assert "safety" in pack
        assert "kpi_config" in pack
        assert "human_gate" in pack
        assert "metadata" in pack
        assert pack["metadata"]["checksum"] is not None


# ---------------------------------------------------------------------------
# /patterns & /kpis — reference data
# ---------------------------------------------------------------------------


class TestReferenceDataEndpoints:
    def test_patterns_returns_list(self):
        resp = client.get("/api/v1/init/patterns")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert any(p["id"] == "oer_screening" for p in data)

    def test_kpis_returns_list(self):
        resp = client.get("/api/v1/init/kpis")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0


# ---------------------------------------------------------------------------
# Full E2E flow
# ---------------------------------------------------------------------------


class TestFullFlow:
    def test_complete_conversation_to_campaign(self):
        """Full happy path: start → 5 rounds → confirm → campaign created."""
        # 1. Start
        resp = client.post("/api/v1/init/start?created_by=e2e_test")
        assert resp.status_code == 200
        sid = resp.json()["session_id"]

        # 2. Submit rounds
        for i, data in enumerate(ALL_ROUNDS, 1):
            resp = client.post(
                f"/api/v1/init/{sid}/respond",
                json={"responses": data},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"], f"Round {i} failed: {body.get('errors')}"

        # 3. Check status
        resp = client.get(f"/api/v1/init/{sid}/status")
        assert resp.json()["completed_rounds"] == [1, 2, 3, 4, 5]

        # 4. Confirm
        resp = client.post(f"/api/v1/init/{sid}/confirm")
        assert resp.status_code == 200
        result = resp.json()
        assert result["status"] == "campaign_created"
        assert result["campaign_id"] is not None

        # 5. Session is now completed
        resp = client.get(f"/api/v1/init/{sid}/status")
        assert resp.json()["status"] == "completed"
