"""Tests for campaign state checkpoint & event replay services."""
from __future__ import annotations

import os
import tempfile

# Isolate test DB before any app imports.
_tmpdir = tempfile.mkdtemp(prefix="otbot_campaign_state_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "campaign_state_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import asyncio  # noqa: E402
import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import init_db  # noqa: E402


def _run(coro):
    """Helper to run async code in tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture(autouse=True)
def _setup_db():
    """Initialize a fresh DB for each test."""
    get_settings.cache_clear()
    init_db()


# ---------------------------------------------------------------------------
# campaign_state — CRUD
# ---------------------------------------------------------------------------


class TestCampaignStateCRUD:
    """Basic create / load / update / status transitions."""

    def test_create_and_load(self):
        from app.services.campaign_state import create_campaign, load_campaign

        create_campaign("camp-001", {"objective_kpi": "cv"}, direction="minimize")
        state = load_campaign("camp-001")
        assert state is not None
        assert state["campaign_id"] == "camp-001"
        assert state["status"] == "planning"
        assert state["direction"] == "minimize"
        assert state["input"]["objective_kpi"] == "cv"
        assert state["current_round"] == 0
        assert state["total_rounds"] == 0
        assert state["kpi_history"] == []

    def test_load_missing_returns_none(self):
        from app.services.campaign_state import load_campaign

        assert load_campaign("nonexistent") is None

    def test_update_status(self):
        from app.services.campaign_state import (
            create_campaign,
            load_campaign,
            update_campaign_status,
        )

        create_campaign("camp-002", {}, direction="maximize")
        update_campaign_status("camp-002", "running")
        state = load_campaign("camp-002")
        assert state["status"] == "running"

    def test_update_status_with_kwargs(self):
        from app.services.campaign_state import (
            create_campaign,
            load_campaign,
            update_campaign_status,
        )

        create_campaign("camp-003", {}, direction="minimize")
        update_campaign_status(
            "camp-003", "completed", stop_reason="converged", best_kpi=1.5
        )
        state = load_campaign("camp-003")
        assert state["status"] == "completed"
        assert state["stop_reason"] == "converged"
        assert state["best_kpi"] == 1.5

    def test_save_plan(self):
        from app.services.campaign_state import (
            create_campaign,
            load_campaign,
            save_plan,
        )

        create_campaign("camp-004", {}, direction="minimize")
        save_plan("camp-004", {"plan_id": "p1", "n_rounds": 5}, total_rounds=5)
        state = load_campaign("camp-004")
        assert state["plan"]["plan_id"] == "p1"
        assert state["total_rounds"] == 5

    def test_create_campaign_is_idempotent(self):
        from app.services.campaign_state import create_campaign, load_campaign

        create_campaign("camp-idem", {"a": 1}, direction="minimize")
        # Second call with INSERT OR IGNORE should not fail
        create_campaign("camp-idem", {"a": 2}, direction="maximize")
        state = load_campaign("camp-idem")
        # First insert wins
        assert state["input"]["a"] == 1
        assert state["direction"] == "minimize"


# ---------------------------------------------------------------------------
# campaign_rounds — checkpoint
# ---------------------------------------------------------------------------


class TestRoundCheckpoint:
    def test_start_and_complete_round(self):
        from app.services.campaign_state import (
            complete_round,
            create_campaign,
            load_round_state,
            start_round,
        )

        create_campaign("camp-r01", {}, direction="minimize")
        start_round("camp-r01", 1, "lhs", n_candidates=4)

        rs = load_round_state("camp-r01", 1)
        assert rs is not None
        assert rs["status"] == "running"
        assert rs["strategy"] == "lhs"
        assert rs["n_candidates_total"] == 4

        complete_round("camp-r01", 1, [1.0, 2.0, 3.0], [{"a": 1}, {"a": 2}, {"a": 3}])
        rs = load_round_state("camp-r01", 1)
        assert rs["status"] == "completed"
        assert rs["batch_kpis"] == [1.0, 2.0, 3.0]
        assert len(rs["batch_params"]) == 3

    def test_start_round_updates_current_round(self):
        from app.services.campaign_state import (
            create_campaign,
            load_campaign,
            start_round,
        )

        create_campaign("camp-r02", {}, direction="maximize")
        start_round("camp-r02", 3, "bayesian", n_candidates=2)
        state = load_campaign("camp-r02")
        assert state["current_round"] == 3

    def test_load_missing_round(self):
        from app.services.campaign_state import (
            create_campaign,
            load_round_state,
        )

        create_campaign("camp-r03", {}, direction="minimize")
        assert load_round_state("camp-r03", 99) is None


# ---------------------------------------------------------------------------
# campaign_candidates — checkpoint + idempotent skip
# ---------------------------------------------------------------------------


class TestCandidateCheckpoint:
    def test_start_and_complete_candidate(self):
        from app.services.campaign_state import (
            complete_candidate,
            create_campaign,
            start_candidate,
            start_round,
        )
        from app.core.db import connection

        create_campaign("camp-c01", {}, direction="minimize")
        start_round("camp-c01", 1, "lhs", n_candidates=2)
        start_candidate("camp-c01", 1, 0, {"vol": 10.0})
        complete_candidate("camp-c01", 1, 0, kpi=2.5, qc="good")

        with connection() as conn:
            row = conn.execute(
                "SELECT * FROM campaign_candidates WHERE campaign_id=? AND round_number=? AND candidate_index=?",
                ("camp-c01", 1, 0),
            ).fetchone()
        assert row is not None
        assert row["status"] == "completed"
        assert row["kpi_value"] == 2.5
        assert row["run_id"] is None  # no actual run in test
        assert row["qc_quality"] == "good"

    def test_idempotent_skip_by_pk(self):
        from app.services.campaign_state import (
            complete_candidate,
            create_campaign,
            is_candidate_done,
            start_candidate,
            start_round,
        )

        create_campaign("camp-c02", {}, direction="minimize")
        start_round("camp-c02", 1, "lhs", n_candidates=1)
        start_candidate("camp-c02", 1, 0, {"vol": 5.0})
        complete_candidate("camp-c02", 1, 0, kpi=1.0, status="completed")

        assert is_candidate_done("camp-c02", 1, 0) is True
        assert is_candidate_done("camp-c02", 1, 1) is False

    def test_idempotent_skip_by_graph_hash(self):
        from app.services.campaign_state import (
            complete_candidate,
            create_campaign,
            is_candidate_done,
            start_candidate,
            start_round,
            update_candidate_graph_hash,
        )

        create_campaign("camp-c03", {}, direction="minimize")
        start_round("camp-c03", 1, "lhs", n_candidates=1)
        start_candidate("camp-c03", 1, 0, {"vol": 5.0})
        update_candidate_graph_hash("camp-c03", 1, 0, "hash-abc123")
        complete_candidate("camp-c03", 1, 0, kpi=1.0, status="completed")

        # Different round/index but same hash → done
        assert is_candidate_done("camp-c03", 2, 0, "hash-abc123") is True
        assert is_candidate_done("camp-c03", 2, 0, "hash-different") is False

    def test_failed_candidate_counts_as_done(self):
        from app.services.campaign_state import (
            complete_candidate,
            create_campaign,
            is_candidate_done,
            start_candidate,
            start_round,
        )

        create_campaign("camp-c04", {}, direction="minimize")
        start_round("camp-c04", 1, "lhs", n_candidates=1)
        start_candidate("camp-c04", 1, 0, {"vol": 5.0})
        complete_candidate("camp-c04", 1, 0, status="failed", error="safety_veto")

        assert is_candidate_done("camp-c04", 1, 0) is True


# ---------------------------------------------------------------------------
# checkpoint_kpi — bulk state snapshot
# ---------------------------------------------------------------------------


class TestCheckpointKPI:
    def test_checkpoint_and_restore(self):
        from app.services.campaign_state import (
            checkpoint_kpi,
            create_campaign,
            load_campaign,
        )

        create_campaign("camp-kpi1", {}, direction="minimize")
        checkpoint_kpi(
            "camp-kpi1",
            kpi_history=[1.0, 2.0, 3.0],
            all_kpis=[1.0, 2.0, 3.0],
            all_params=[{"a": 1}, {"a": 2}, {"a": 3}],
            all_rounds=[1, 1, 2],
            best_kpi=1.0,
            total_runs=3,
        )

        state = load_campaign("camp-kpi1")
        assert state["kpi_history"] == [1.0, 2.0, 3.0]
        assert state["all_kpis"] == [1.0, 2.0, 3.0]
        assert state["all_params"] == [{"a": 1}, {"a": 2}, {"a": 3}]
        assert state["all_rounds"] == [1, 1, 2]
        assert state["best_kpi"] == 1.0
        assert state["total_runs"] == 3

    def test_checkpoint_overwrite(self):
        from app.services.campaign_state import (
            checkpoint_kpi,
            create_campaign,
            load_campaign,
        )

        create_campaign("camp-kpi2", {}, direction="maximize")
        checkpoint_kpi("camp-kpi2", [1.0], [1.0], [{"a": 1}], [1], 1.0, 1)
        checkpoint_kpi("camp-kpi2", [1.0, 5.0], [1.0, 5.0], [{"a": 1}, {"a": 2}], [1, 2], 5.0, 2)

        state = load_campaign("camp-kpi2")
        assert state["best_kpi"] == 5.0
        assert state["total_runs"] == 2
        assert len(state["kpi_history"]) == 2


# ---------------------------------------------------------------------------
# campaign_events — log + replay
# ---------------------------------------------------------------------------


class TestCampaignEvents:
    def test_log_event_returns_seq(self):
        from app.services.campaign_events import log_event
        from app.services.campaign_state import create_campaign

        create_campaign("camp-ev1", {}, direction="minimize")
        seq1 = log_event("camp-ev1", "round_start", {"round": 1})
        seq2 = log_event("camp-ev1", "round_end", {"round": 1})
        assert isinstance(seq1, int)
        assert seq2 > seq1

    def test_replay_events(self):
        from app.services.campaign_events import log_event, replay_events
        from app.services.campaign_state import create_campaign

        create_campaign("camp-ev2", {}, direction="minimize")
        s1 = log_event("camp-ev2", "a", {"x": 1})
        s2 = log_event("camp-ev2", "b", {"x": 2})
        s3 = log_event("camp-ev2", "c", {"x": 3})

        # Replay all
        all_events = replay_events("camp-ev2", after_seq=0)
        assert len(all_events) == 3
        assert all_events[0]["seq"] == s1
        assert all_events[2]["seq"] == s3

        # Replay after s1
        partial = replay_events("camp-ev2", after_seq=s1)
        assert len(partial) == 2
        assert partial[0]["seq"] == s2

    def test_replay_empty(self):
        from app.services.campaign_events import replay_events
        from app.services.campaign_state import create_campaign

        create_campaign("camp-ev3", {}, direction="minimize")
        assert replay_events("camp-ev3") == []

    def test_get_latest_seq(self):
        from app.services.campaign_events import get_latest_seq, log_event
        from app.services.campaign_state import create_campaign

        create_campaign("camp-ev4", {}, direction="minimize")
        assert get_latest_seq("camp-ev4") == 0

        log_event("camp-ev4", "test", {"a": 1})
        s = log_event("camp-ev4", "test", {"b": 2})
        assert get_latest_seq("camp-ev4") == s

    def test_events_isolated_by_campaign(self):
        from app.services.campaign_events import log_event, replay_events
        from app.services.campaign_state import create_campaign

        create_campaign("camp-iso-a", {}, direction="minimize")
        create_campaign("camp-iso-b", {}, direction="minimize")
        log_event("camp-iso-a", "x", {"v": 1})
        log_event("camp-iso-b", "y", {"v": 2})
        log_event("camp-iso-a", "z", {"v": 3})

        a_events = replay_events("camp-iso-a")
        b_events = replay_events("camp-iso-b")
        assert len(a_events) == 2
        assert len(b_events) == 1


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


class TestResumeHelpers:
    def test_list_incomplete_campaigns(self):
        from app.services.campaign_state import (
            create_campaign,
            list_incomplete_campaigns,
            update_campaign_status,
        )

        create_campaign("camp-inc-1", {}, direction="minimize")
        create_campaign("camp-inc-2", {}, direction="maximize")
        create_campaign("camp-inc-3", {}, direction="minimize")
        update_campaign_status("camp-inc-1", "running")
        update_campaign_status("camp-inc-3", "completed")

        incomplete = list_incomplete_campaigns()
        ids = [c["campaign_id"] for c in incomplete]
        assert "camp-inc-1" in ids  # running
        assert "camp-inc-2" in ids  # planning
        assert "camp-inc-3" not in ids  # completed

    def test_load_completed_candidates(self):
        from app.services.campaign_state import (
            checkpoint_kpi,
            create_campaign,
            load_completed_candidates,
        )

        create_campaign("camp-lcc", {}, direction="minimize")
        checkpoint_kpi(
            "camp-lcc",
            kpi_history=[10.0, 20.0],
            all_kpis=[10.0, 20.0],
            all_params=[{"x": 1}, {"x": 2}],
            all_rounds=[1, 1],
            best_kpi=10.0,
            total_runs=2,
        )

        restored = load_completed_candidates("camp-lcc")
        assert restored["kpi_history"] == [10.0, 20.0]
        assert restored["best_kpi"] == 10.0
        assert restored["total_runs"] == 2
        assert restored["direction"] == "minimize"

    def test_load_completed_candidates_missing(self):
        from app.services.campaign_state import load_completed_candidates

        restored = load_completed_candidates("nonexistent")
        assert restored["kpi_history"] == []
        assert restored["best_kpi"] is None

    def test_get_completed_rounds(self):
        from app.services.campaign_state import (
            complete_round,
            create_campaign,
            get_completed_rounds,
            start_round,
        )

        create_campaign("camp-gcr", {}, direction="minimize")
        start_round("camp-gcr", 1, "lhs", n_candidates=1)
        complete_round("camp-gcr", 1, [1.0], [{"a": 1}])
        start_round("camp-gcr", 2, "bayesian", n_candidates=1)
        # Round 2 not completed
        start_round("camp-gcr", 3, "lhs", n_candidates=1)
        complete_round("camp-gcr", 3, [2.0], [{"a": 2}])

        completed = get_completed_rounds("camp-gcr")
        assert completed == [1, 3]


# ---------------------------------------------------------------------------
# Full checkpoint → restore workflow
# ---------------------------------------------------------------------------


class TestResumeFromCheckpoint:
    """Simulate a campaign that checkpointed 3 rounds, then reload."""

    def test_full_checkpoint_and_restore(self):
        from app.services.campaign_state import (
            checkpoint_kpi,
            complete_candidate,
            complete_round,
            create_campaign,
            get_completed_rounds,
            load_campaign,
            load_completed_candidates,
            save_plan,
            start_candidate,
            start_round,
            update_campaign_status,
        )

        # Create
        create_campaign("camp-full", {"objective_kpi": "cv"}, direction="minimize")
        save_plan("camp-full", {"plan_id": "p1"}, total_rounds=5)
        update_campaign_status("camp-full", "running")

        # Round 1: 2 candidates
        start_round("camp-full", 1, "lhs", n_candidates=2)
        start_candidate("camp-full", 1, 0, {"vol": 10})
        complete_candidate("camp-full", 1, 0, kpi=5.0, status="completed")
        start_candidate("camp-full", 1, 1, {"vol": 20})
        complete_candidate("camp-full", 1, 1, kpi=3.0, status="completed")
        checkpoint_kpi("camp-full", [5.0, 3.0], [5.0, 3.0],
                       [{"vol": 10}, {"vol": 20}], [1, 1], 3.0, 2)
        complete_round("camp-full", 1, [5.0, 3.0], [{"vol": 10}, {"vol": 20}])

        # Round 2: 1 candidate
        start_round("camp-full", 2, "bayesian", n_candidates=1)
        start_candidate("camp-full", 2, 0, {"vol": 15})
        complete_candidate("camp-full", 2, 0, kpi=2.0, status="completed")
        checkpoint_kpi("camp-full", [5.0, 3.0, 2.0], [5.0, 3.0, 2.0],
                       [{"vol": 10}, {"vol": 20}, {"vol": 15}], [1, 1, 2], 2.0, 3)
        complete_round("camp-full", 2, [2.0], [{"vol": 15}])

        # Round 3: started but crashed
        start_round("camp-full", 3, "bayesian", n_candidates=2)
        start_candidate("camp-full", 3, 0, {"vol": 12})
        # ← crash here: candidate 0 started, not completed

        # --- Verify restored state ---
        state = load_campaign("camp-full")
        assert state["status"] == "running"
        assert state["current_round"] == 3
        assert state["total_rounds"] == 5
        assert state["best_kpi"] == 2.0

        restored = load_completed_candidates("camp-full")
        assert restored["total_runs"] == 3
        assert restored["best_kpi"] == 2.0
        assert len(restored["kpi_history"]) == 3

        completed_rnds = get_completed_rounds("camp-full")
        assert completed_rnds == [1, 2]
        # Round 3 is NOT completed → resume should re-run it


# ---------------------------------------------------------------------------
# Orchestrator checkpoint integration (unit-level)
# ---------------------------------------------------------------------------


class TestOrchestratorCheckpointIntegration:
    """Verify the orchestrator creates DB records during dry_run."""

    def test_dry_run_creates_campaign_state(self):
        from app.agents.orchestrator import OrchestratorAgent, OrchestratorInput
        from app.services.campaign_state import load_campaign

        orch_input = OrchestratorInput(
            contract_id="test-contract",
            objective_kpi="cv",
            direction="minimize",
            max_rounds=1,
            batch_size=2,
            strategy="lhs",
            dimensions=[
                {"param_name": "volume", "param_type": "number", "min_value": 1.0, "max_value": 50.0},
            ],
            protocol_template={"steps": []},
            dry_run=True,
            campaign_id="camp-orch-test",
        )

        agent = OrchestratorAgent()
        result = _run(agent.run(orch_input))

        assert result.success
        assert result.output.status == "completed"

        # Verify DB was populated
        state = load_campaign("camp-orch-test")
        assert state is not None
        assert state["status"] == "completed"
        assert state["best_kpi"] is not None

    def test_plan_only_creates_campaign_state(self):
        from app.agents.orchestrator import OrchestratorAgent, OrchestratorInput
        from app.services.campaign_state import load_campaign

        orch_input = OrchestratorInput(
            contract_id="test-contract-po",
            objective_kpi="cv",
            direction="minimize",
            max_rounds=3,
            batch_size=4,
            strategy="lhs",
            dimensions=[
                {"param_name": "volume", "param_type": "number", "min_value": 1.0, "max_value": 50.0},
            ],
            protocol_template={"steps": []},
            plan_only=True,
            campaign_id="camp-plan-only",
        )

        agent = OrchestratorAgent()
        result = _run(agent.run(orch_input))

        assert result.success
        assert result.output.status == "planned"

        state = load_campaign("camp-plan-only")
        assert state is not None
        assert state["status"] == "completed"  # plan_only → completed
        assert state["plan"] is not None


# ---------------------------------------------------------------------------
# SSE event persistence via _emit
# ---------------------------------------------------------------------------


class TestEmitPersistence:
    """Verify _emit() persists events to campaign_events table."""

    def test_emit_persists_to_db(self):
        from app.agents.orchestrator import OrchestratorAgent
        from app.services.campaign_events import replay_events
        from app.services.campaign_state import create_campaign

        create_campaign("camp-emit", {}, direction="minimize")

        agent = OrchestratorAgent()
        agent._emit("camp-emit", {"type": "test_event", "data": "hello"})
        agent._emit("camp-emit", {"type": "test_event2", "data": "world"})

        events = replay_events("camp-emit")
        assert len(events) == 2
        assert events[0]["event_type"] == "test_event"
        assert events[0]["payload"]["data"] == "hello"
        assert events[1]["event_type"] == "test_event2"

    def test_emit_attaches_seq(self):
        from app.agents.orchestrator import OrchestratorAgent
        from app.services.campaign_state import create_campaign

        create_campaign("camp-emit2", {}, direction="minimize")

        agent = OrchestratorAgent()
        event = {"type": "with_seq", "value": 42}
        agent._emit("camp-emit2", event)
        assert "_seq" in event
        assert isinstance(event["_seq"], int)
