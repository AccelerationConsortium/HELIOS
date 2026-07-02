"""Tests for the candidate memory layer (recall over persisted candidates)."""
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


def test_load_all_candidates_returns_persisted_rows(db_env):
    from app.services.campaign_state import (
        complete_candidate,
        create_campaign,
        load_all_candidates,
        start_candidate,
        start_round,
    )

    create_campaign("camp-mem", {"objective": "test"}, direction="maximize")
    start_round("camp-mem", 1, "explore", 2)
    start_candidate("camp-mem", 1, 0, {"x": 0.2, "y": 1.0})
    start_candidate("camp-mem", 1, 1, {"x": 0.9, "y": 0.0})
    complete_candidate("camp-mem", 1, 0, kpi=5.0, status="completed")
    complete_candidate("camp-mem", 1, 1, status="failed", error="qc_abort")

    rows = load_all_candidates("camp-mem")

    assert len(rows) == 2
    by_idx = {r["candidate_index"]: r for r in rows}
    assert by_idx[0]["params"] == {"x": 0.2, "y": 1.0}
    assert by_idx[0]["kpi_value"] == 5.0
    assert by_idx[0]["status"] == "completed"
    assert by_idx[1]["params"] == {"x": 0.9, "y": 0.0}
    assert by_idx[1]["status"] == "failed"
    assert by_idx[1]["error"] == "qc_abort"
    assert by_idx[1]["round_number"] == 1


def _space():
    from app.services.candidate_gen import ParameterSpace, SearchDimension

    return ParameterSpace(
        dimensions=(SearchDimension("x", "number", 0.0, 10.0),),
        protocol_template={},
    )


def _seed_three(campaign_id: str) -> None:
    from app.services.campaign_state import (
        complete_candidate,
        create_campaign,
        start_candidate,
        start_round,
    )

    create_campaign(campaign_id, {"objective": "test"}, direction="maximize")
    start_round(campaign_id, 1, "explore", 3)
    start_candidate(campaign_id, 1, 0, {"x": 0.2})
    start_candidate(campaign_id, 1, 1, {"x": 5.0})
    start_candidate(campaign_id, 1, 2, {"x": 9.0})
    complete_candidate(campaign_id, 1, 0, kpi=1.0, status="completed")
    complete_candidate(campaign_id, 1, 1, kpi=8.0, status="completed")
    complete_candidate(campaign_id, 1, 2, status="failed", error="gel_formation")


def test_recall_ranks_by_similarity_and_returns_top_k(db_env):
    from app.optimization.candidate_memory import recall_similar_candidates

    _seed_three("camp-recall")

    hits = recall_similar_candidates("camp-recall", {"x": 0.3}, _space(), k=2)

    assert len(hits) == 2
    # Nearest to x=0.3 is x=0.2, then x=5.0; x=9.0 is dropped by top-k.
    assert hits[0].params == {"x": 0.2}
    assert hits[1].params == {"x": 5.0}
    assert hits[0].distance < hits[1].distance
    assert hits[0].kpi == 1.0
    assert hits[0].status == "completed"


def test_recall_is_fail_open_when_no_history(db_env):
    from app.optimization.candidate_memory import recall_similar_candidates
    from app.services.campaign_state import create_campaign

    create_campaign("camp-empty", {"objective": "test"}, direction="maximize")

    assert recall_similar_candidates("camp-empty", {"x": 0.5}, _space(), k=3) == []


def test_recall_surfaces_failure_reason_for_avoidance(db_env):
    from app.optimization.candidate_memory import recall_similar_candidates

    _seed_three("camp-fail")

    # Query right next to the failed point x=9.0.
    hits = recall_similar_candidates("camp-fail", {"x": 8.8}, _space(), k=1)

    assert hits[0].params == {"x": 9.0}
    assert hits[0].status == "failed"
    assert hits[0].error == "gel_formation"
