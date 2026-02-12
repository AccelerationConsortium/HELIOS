"""Tests for the intelligent campaign loop orchestrator."""
from __future__ import annotations

import os
import tempfile

_tmpdir = tempfile.mkdtemp(prefix="otbot_campaign_loop_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "campaign_loop_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import connection, init_db, utcnow_iso  # noqa: E402
from app.services.campaign_loop import (  # noqa: E402
    CampaignGoal,
    CampaignResult,
    RoundResult,
    decide_next_action,
    evaluate_round_pure,
    run_campaign_offline,
)
from app.services.candidate_gen import ParameterSpace, SearchDimension  # noqa: E402
from app.services.convergence import ConvergenceConfig, ConvergenceStatus  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _setup_db():
    """Initialize a fresh DB for each test."""
    get_settings.cache_clear()
    init_db()
    with connection() as conn:
        # FK-safe cleanup order
        conn.execute("DELETE FROM evolved_priors")
        conn.execute("DELETE FROM evolution_proposals")
        conn.execute("DELETE FROM protocol_templates")
        conn.execute("DELETE FROM batch_candidates")
        conn.execute("DELETE FROM batch_requests")
        conn.execute("DELETE FROM run_reviews")
        conn.execute("DELETE FROM run_kpis")
        conn.execute("DELETE FROM artifacts")
        conn.execute("DELETE FROM provenance_events")
        conn.execute("DELETE FROM run_steps")
        conn.execute("DELETE FROM snapshot_runs")
        conn.execute("DELETE FROM dataset_snapshots")
        conn.execute("DELETE FROM qc_flags")
        conn.execute("DELETE FROM run_failure_signatures")
        conn.execute("DELETE FROM experiment_index")
        conn.execute("DELETE FROM param_schema")
        conn.execute("DELETE FROM runs")
        conn.execute("DELETE FROM campaigns")
        conn.commit()
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_goal(
    direction: str = "maximize",
    target: float | None = None,
    max_rounds: int = 5,
    batch_size: int = 3,
    strategy: str = "lhs",
    kpi: str = "test_kpi",
) -> CampaignGoal:
    return CampaignGoal(
        objective_kpi=kpi,
        direction=direction,
        target_value=target,
        max_rounds=max_rounds,
        batch_size=batch_size,
        strategy=strategy,
    )


def _make_round(
    round_number: int = 0,
    best_kpi: float = 10.0,
    convergence_status: str = "improving",
) -> RoundResult:
    return RoundResult(
        round_number=round_number,
        run_ids=("run-1",),
        kpi_values=(best_kpi,),
        best_kpi=best_kpi,
        convergence_status=convergence_status,
        timestamp="2025-01-01T00:00:00Z",
    )


def _make_space() -> ParameterSpace:
    return ParameterSpace(
        dimensions=(
            SearchDimension(param_name="x", param_type="number", min_value=0.0, max_value=10.0),
            SearchDimension(param_name="y", param_type="number", min_value=0.0, max_value=5.0),
        ),
        protocol_template={},
    )


def _insert_campaign_row(campaign_id: str) -> None:
    """Insert a campaign row so FK constraints on batch_requests are satisfied."""
    now = utcnow_iso()
    with connection() as conn:
        conn.execute(
            "INSERT INTO campaigns "
            "(id, name, cadence_seconds, protocol_json, inputs_json, "
            "policy_json, next_fire_at, is_active, created_at, updated_at) "
            "VALUES (?, 'test', 60, '{}', '{}', '{}', ?, 1, ?, ?)",
            (campaign_id, now, now, now),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# TestCampaignGoal
# ---------------------------------------------------------------------------


class TestCampaignGoal:
    def test_goal_maximize(self):
        goal = _make_goal(direction="maximize")
        assert goal.maximize is True

    def test_goal_minimize(self):
        goal = _make_goal(direction="minimize")
        assert goal.maximize is False

    def test_goal_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction must be"):
            _make_goal(direction="invalid")

    def test_is_target_reached_maximize(self):
        goal = _make_goal(direction="maximize", target=100.0)
        assert goal.is_target_reached(105.0) is True
        assert goal.is_target_reached(95.0) is False

    def test_is_target_reached_no_target(self):
        goal = _make_goal(direction="maximize", target=None)
        assert goal.is_target_reached(999.0) is False

    def test_goal_to_dict(self):
        goal = _make_goal(direction="minimize", target=42.0, max_rounds=10)
        d = goal.to_dict()
        assert d["objective_kpi"] == "test_kpi"
        assert d["direction"] == "minimize"
        assert d["target_value"] == 42.0
        assert d["max_rounds"] == 10
        assert d["batch_size"] == 3
        assert d["strategy"] == "lhs"


# ---------------------------------------------------------------------------
# TestRoundResult
# ---------------------------------------------------------------------------


class TestRoundResult:
    def test_round_result_frozen(self):
        rr = _make_round()
        with pytest.raises(AttributeError):
            rr.best_kpi = 999.0  # type: ignore[misc]

    def test_round_result_to_dict(self):
        rr = RoundResult(
            round_number=1,
            run_ids=("r1", "r2"),
            kpi_values=(10.0, 20.0),
            best_kpi=20.0,
            convergence_status="improving",
            timestamp="2025-01-01T00:00:00Z",
        )
        d = rr.to_dict()
        assert d["round_number"] == 1
        assert d["run_ids"] == ["r1", "r2"]
        assert d["kpi_values"] == [10.0, 20.0]
        assert d["best_kpi"] == 20.0
        assert d["convergence_status"] == "improving"
        assert d["timestamp"] == "2025-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# TestCampaignResult
# ---------------------------------------------------------------------------


class TestCampaignResult:
    def test_campaign_result_to_dict(self):
        goal = _make_goal(direction="maximize", target=100.0)
        rr = _make_round(round_number=0, best_kpi=50.0)
        cr = CampaignResult(
            goal=goal,
            rounds=(rr,),
            best_kpi=50.0,
            best_round=0,
            total_runs=1,
            converged=False,
            target_reached=False,
            stop_reason="budget_exhausted",
        )
        d = cr.to_dict()
        assert d["goal"]["direction"] == "maximize"
        assert len(d["rounds"]) == 1
        assert d["rounds"][0]["best_kpi"] == 50.0
        assert d["best_kpi"] == 50.0
        assert d["best_round"] == 0
        assert d["total_runs"] == 1
        assert d["converged"] is False
        assert d["target_reached"] is False
        assert d["stop_reason"] == "budget_exhausted"


# ---------------------------------------------------------------------------
# TestEvaluateRoundPure
# ---------------------------------------------------------------------------


class TestEvaluateRoundPure:
    def test_evaluate_maximize(self):
        goal = _make_goal(direction="maximize")
        kpis = [10.0, 20.0, 15.0]
        history = list(kpis)
        best, status = evaluate_round_pure(kpis, history, goal)
        assert best == 20.0

    def test_evaluate_minimize(self):
        goal = _make_goal(direction="minimize")
        kpis = [10.0, 20.0, 15.0]
        history = list(kpis)
        best, status = evaluate_round_pure(kpis, history, goal)
        assert best == 10.0

    def test_evaluate_empty_kpis(self):
        goal_max = _make_goal(direction="maximize")
        best_max, status_max = evaluate_round_pure([], [], goal_max)
        assert best_max == float("-inf")
        assert status_max.status == "insufficient_data"

        goal_min = _make_goal(direction="minimize")
        best_min, status_min = evaluate_round_pure([], [], goal_min)
        assert best_min == float("inf")
        assert status_min.status == "insufficient_data"


# ---------------------------------------------------------------------------
# TestDecideNextAction
# ---------------------------------------------------------------------------


class TestDecideNextAction:
    def test_continue_on_first_round(self):
        goal = _make_goal(direction="maximize")
        conv = ConvergenceStatus(
            status="insufficient_data", confidence=0.0, details={}
        )
        action = decide_next_action(goal, [], conv)
        assert action == "continue"

    def test_stop_target(self):
        goal = _make_goal(direction="maximize", target=100.0)
        rounds = [_make_round(round_number=0, best_kpi=105.0)]
        conv = ConvergenceStatus(status="improving", confidence=0.5, details={})
        action = decide_next_action(goal, rounds, conv)
        assert action == "stop_target"

    def test_stop_budget(self):
        goal = _make_goal(direction="maximize", max_rounds=2, target=1000.0)
        rounds = [
            _make_round(round_number=0, best_kpi=10.0),
            _make_round(round_number=1, best_kpi=20.0),
        ]
        conv = ConvergenceStatus(status="improving", confidence=0.5, details={})
        action = decide_next_action(goal, rounds, conv)
        assert action == "stop_budget"

    def test_stop_converged(self):
        goal = _make_goal(direction="maximize", max_rounds=10, target=1000.0)
        rounds = [_make_round(round_number=0, best_kpi=10.0)]
        conv = ConvergenceStatus(status="plateau", confidence=0.9, details={})
        action = decide_next_action(goal, rounds, conv)
        assert action == "stop_converged"

    def test_stop_diverging(self):
        goal = _make_goal(direction="maximize", max_rounds=10, target=1000.0)
        rounds = [_make_round(round_number=0, best_kpi=10.0)]
        conv = ConvergenceStatus(status="diverging", confidence=0.9, details={})
        action = decide_next_action(goal, rounds, conv)
        assert action == "stop_diverging"


# ---------------------------------------------------------------------------
# TestRunCampaignOffline
# ---------------------------------------------------------------------------


class TestRunCampaignOffline:
    def test_offline_reaches_target(self):
        """sim_fn returns kpi = x * 10; with x in [0,10], target=50 is reachable."""
        goal = _make_goal(
            direction="maximize",
            target=50.0,
            max_rounds=5,
            batch_size=3,
            strategy="lhs",
        )
        space = _make_space()

        def sim_fn(params):
            return {"test_kpi": params.get("x", 0) * 10}

        _insert_campaign_row("test-target")
        result = run_campaign_offline(goal, space, sim_fn, campaign_id="test-target")
        assert result.target_reached is True
        assert result.stop_reason == "stop_target"
        assert result.best_kpi >= 50.0

    def test_offline_budget_exhausted(self):
        """sim_fn returns constant 1.0; unreachable target exhausts budget."""
        goal = _make_goal(
            direction="maximize",
            target=1000.0,
            max_rounds=2,
            batch_size=3,
            strategy="lhs",
        )
        space = _make_space()

        def sim_fn(params):
            return {"test_kpi": 1.0}

        _insert_campaign_row("test-budget")
        result = run_campaign_offline(goal, space, sim_fn, campaign_id="test-budget")
        assert result.stop_reason in ("budget_exhausted", "stop_budget")
        assert result.target_reached is False

    def test_offline_records_all_rounds(self):
        """All rounds are recorded and total_runs > 0."""
        call_count = {"n": 0}

        goal = _make_goal(
            direction="maximize",
            target=None,
            max_rounds=3,
            batch_size=2,
            strategy="lhs",
        )
        space = _make_space()

        def sim_fn(params):
            call_count["n"] += 1
            return {"test_kpi": call_count["n"] * 10.0}

        _insert_campaign_row("test-rounds")
        result = run_campaign_offline(goal, space, sim_fn, campaign_id="test-rounds")
        # May stop early due to convergence, but should have at least 1 round
        assert len(result.rounds) >= 1
        # If it ran all 3 rounds, verify that
        if result.stop_reason == "budget_exhausted":
            assert len(result.rounds) == 3
        assert result.total_runs > 0
        # Verify round numbering is sequential
        for i, rr in enumerate(result.rounds):
            assert rr.round_number == i
