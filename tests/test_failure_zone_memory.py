"""Tests for cross-campaign failure-zone memory (read-only prior)."""
from __future__ import annotations

import pytest


@pytest.fixture
def db_env(monkeypatch, request, tmp_path):
    """Isolated DB for each test, matching the campaign_state test pattern."""
    from app.core.config import get_settings
    from app.core.db import init_db

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "data" / "orchestrator.db"))
    monkeypatch.setenv("OBJECT_STORE_DIR", str(tmp_path / "objects"))
    get_settings.cache_clear()
    request.addfinalizer(get_settings.cache_clear)
    init_db()


def _space():
    from app.services.candidate_gen import ParameterSpace, SearchDimension

    return ParameterSpace(
        dimensions=(SearchDimension("x", "number", 0.0, 10.0),),
        protocol_template={},
    )


def _seed(campaign_id, idx, params, *, status, error=None, kpi=None):
    from app.services.campaign_state import (
        complete_candidate,
        create_campaign,
        start_candidate,
        start_round,
    )

    create_campaign(campaign_id, {"objective": "t"}, direction="maximize")
    start_round(campaign_id, 1, "explore", 9)
    start_candidate(campaign_id, 1, idx, params)
    complete_candidate(campaign_id, 1, idx, kpi=kpi, status=status, error=error)


# --- persistence read helper -------------------------------------------------


def test_load_failed_candidates_filters_status_and_excludes_campaign(db_env):
    from app.services.campaign_state import load_failed_candidates

    _seed("camp-a", 0, {"x": 1.0}, status="failed", error="err_a")
    _seed("camp-a", 1, {"x": 2.0}, status="completed", kpi=5.0)
    _seed("camp-b", 0, {"x": 3.0}, status="failed", error="err_b")

    all_failed = load_failed_candidates()
    assert {r["campaign_id"] for r in all_failed} == {"camp-a", "camp-b"}
    assert all(r["status"] == "failed" for r in all_failed)
    assert {r["params"]["x"] for r in all_failed} == {1.0, 3.0}  # not the completed 2.0

    excl = load_failed_candidates(exclude_campaign_id="camp-a")
    assert [r["campaign_id"] for r in excl] == ["camp-b"]


# --- recall_failure_zones ----------------------------------------------------


def test_recall_failure_zones_cross_campaign_by_similarity(db_env):
    from app.optimization.failure_zone_memory import recall_failure_zones

    _seed("camp-hist", 0, {"x": 1.0}, status="failed", error="low")
    _seed("camp-hist", 1, {"x": 9.0}, status="failed", error="high")

    zones = recall_failure_zones("camp-current", {"x": 8.8}, _space(), k=1)

    assert len(zones) == 1
    assert zones[0].params == {"x": 9.0}
    assert zones[0].campaign_id == "camp-hist"


def test_recall_failure_zones_ignores_successful_candidates(db_env):
    from app.optimization.failure_zone_memory import recall_failure_zones

    _seed("camp-hist", 0, {"x": 2.0}, status="completed", kpi=8.0)
    _seed("camp-hist", 1, {"x": 9.0}, status="failed", error="high")

    # Query sits right on the successful point, but only failures are zones.
    zones = recall_failure_zones("camp-current", {"x": 2.0}, _space(), k=5)

    assert len(zones) == 1
    assert zones[0].params == {"x": 9.0}


def test_recall_failure_zones_excludes_current_campaign_by_default(db_env):
    from app.optimization.failure_zone_memory import recall_failure_zones

    _seed("camp-current", 0, {"x": 5.0}, status="failed", error="own")
    _seed("camp-other", 0, {"x": 9.0}, status="failed", error="other")

    default = recall_failure_zones("camp-current", {"x": 5.0}, _space(), k=5)
    assert [z.params["x"] for z in default] == [9.0]  # own 5.0 excluded

    with_current = recall_failure_zones(
        "camp-current", {"x": 5.0}, _space(), k=5, include_current=True
    )
    assert {z.params["x"] for z in with_current} == {5.0, 9.0}
    assert with_current[0].params == {"x": 5.0}  # nearest first


def test_recall_failure_zones_top_k_ordering_by_distance(db_env):
    from app.optimization.failure_zone_memory import recall_failure_zones

    _seed("camp-hist", 0, {"x": 1.0}, status="failed", error="a")
    _seed("camp-hist", 1, {"x": 5.0}, status="failed", error="b")
    _seed("camp-hist", 2, {"x": 9.0}, status="failed", error="c")

    zones = recall_failure_zones("camp-current", {"x": 0.5}, _space(), k=2)

    assert [z.params["x"] for z in zones] == [1.0, 5.0]
    assert zones[0].distance < zones[1].distance


def test_recall_failure_zones_fail_open_empty_history(db_env):
    from app.optimization.failure_zone_memory import recall_failure_zones
    from app.services.campaign_state import create_campaign

    create_campaign("camp-current", {"objective": "t"}, direction="maximize")

    assert recall_failure_zones("camp-current", {"x": 0.5}, _space(), k=3) == []


def test_recall_failure_zones_preserves_failure_reason(db_env):
    from app.optimization.failure_zone_memory import recall_failure_zones

    _seed("camp-hist", 0, {"x": 9.0}, status="failed", error="gel_formation")

    zones = recall_failure_zones("camp-current", {"x": 8.9}, _space(), k=1)

    assert zones[0].error == "gel_formation"
