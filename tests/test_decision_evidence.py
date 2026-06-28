"""Tests for Phase ⑤: memory recall attached as decision-trace evidence.

Evidence is *additive*: it enriches the provenance record but never changes
the decision. Recall is fail-open: empty or raising recall must not alter
arbitration.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def db_env(monkeypatch, request, tmp_path):
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


def _request(campaign_id="camp", seed=4):
    from app.optimization.schemas import OptimizationRequest

    return OptimizationRequest(
        campaign_id=campaign_id, space=_space(), n=1, seed=seed, round_index=7
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


class _UnavailableProvider:
    def is_available(self) -> bool:
        return False

    def suggest(self, request):  # pragma: no cover
        raise AssertionError("must not be called")


# --- evidence builder --------------------------------------------------------


def test_build_decision_evidence_includes_similar_candidates(db_env):
    from app.optimization.decision_evidence import build_decision_evidence

    _seed("camp", 0, {"x": 0.2}, status="completed", kpi=5.0)

    ev = build_decision_evidence("camp", {"x": 0.3}, _space())

    assert len(ev.similar_candidates) == 1
    assert ev.similar_candidates[0].params == {"x": 0.2}


def test_build_decision_evidence_includes_cross_campaign_failure_zones(db_env):
    from app.optimization.decision_evidence import build_decision_evidence

    _seed("other", 0, {"x": 9.0}, status="failed", error="gel_formation")

    ev = build_decision_evidence("camp", {"x": 8.8}, _space())

    assert len(ev.failure_zones) == 1
    assert ev.failure_zones[0].error == "gel_formation"
    assert ev.failure_zones[0].campaign_id == "other"


def test_build_decision_evidence_fail_open_on_recall_error(db_env, monkeypatch):
    import app.optimization.decision_evidence as de

    def _boom(*a, **k):
        raise RuntimeError("recall exploded")

    monkeypatch.setattr(de, "recall_similar_candidates", _boom)
    monkeypatch.setattr(de, "recall_failure_zones", _boom)

    ev = de.build_decision_evidence("camp", {"x": 0.3}, _space())

    assert ev.similar_candidates == []
    assert ev.failure_zones == []


def test_evidence_for_decision_is_none_when_no_history(db_env):
    from app.optimization.decision_evidence import evidence_for_decision
    from app.optimization.schemas import DecisionResult

    decision = DecisionResult(accepted=True, final_candidates=({"x": 0.5},))

    assert evidence_for_decision(_request(), decision) is None


def test_evidence_for_decision_is_structured_not_freetext(db_env):
    from app.optimization.decision_evidence import evidence_for_decision
    from app.optimization.schemas import DecisionResult

    _seed("camp", 0, {"x": 0.2}, status="completed", kpi=5.0)
    decision = DecisionResult(accepted=True, final_candidates=({"x": 0.3},))

    ev = evidence_for_decision(_request(), decision)

    assert isinstance(ev, dict)
    assert isinstance(ev["candidates"], list)
    entry = ev["candidates"][0]
    assert entry["params"] == {"x": 0.3}
    assert entry["similar_candidates"][0]["params"] == {"x": 0.2}
    assert "distance" in entry["similar_candidates"][0]


# --- integration through suggest_next ----------------------------------------


def test_suggest_next_attaches_evidence_when_history_exists(db_env):
    from app.optimization.service import suggest_next

    # Same-campaign completed point -> similar; other-campaign failure -> zone.
    _seed("camp", 0, {"x": 4.0}, status="completed", kpi=5.0)
    _seed("other", 0, {"x": 9.0}, status="failed", error="gel")

    outcome = suggest_next(_request(), provider=_UnavailableProvider())

    assert "evidence" in outcome.provenance
    cands = outcome.provenance["evidence"]["candidates"]
    assert cands  # at least the selected candidate has evidence
    assert cands[0]["similar_candidates"] or cands[0]["failure_zones"]


def test_suggest_next_no_evidence_key_when_no_history(db_env):
    from app.optimization.service import suggest_next

    outcome = suggest_next(_request(), provider=_UnavailableProvider())

    assert "evidence" not in outcome.provenance


def test_suggest_next_decision_unchanged_by_evidence(db_env):
    from app.optimization.service import suggest_next

    _seed("camp", 0, {"x": 4.0}, status="completed", kpi=5.0)
    _seed("other", 0, {"x": 9.0}, status="failed", error="gel")

    with_ev = suggest_next(_request(seed=11), provider=_UnavailableProvider())
    without_ev = suggest_next(
        _request(seed=11), provider=_UnavailableProvider(), attach_evidence=False
    )

    assert with_ev.decision.accepted == without_ev.decision.accepted
    assert with_ev.decision.final_candidates == without_ev.decision.final_candidates
    assert with_ev.suggestion.candidates == without_ev.suggestion.candidates


def test_suggest_next_fail_open_when_evidence_raises(db_env, monkeypatch):
    import app.optimization.service as svc

    def _boom(*a, **k):
        raise RuntimeError("evidence exploded")

    monkeypatch.setattr(svc, "evidence_for_decision", _boom)

    outcome = svc.suggest_next(_request(), provider=_UnavailableProvider())

    # Arbitration unbroken: a decision is still produced, no evidence attached.
    assert outcome.decision is not None
    assert "evidence" not in outcome.provenance
