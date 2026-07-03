"""T6: the per-signal verifiable reward is emitted on the event stream so the
inner evaluation loop is visible in /lab (not just an opaque total)."""
from __future__ import annotations

from app.services.decision_outcome import (
    CampaignDecisionOutcome,
    CampaignDecisionRewardCalculator,
)
from app.services.detailed_event_emitter import DetailedEventEmitter


def test_emit_decision_reward_carries_verifier_report():
    events: list[tuple[str, dict]] = []
    emitter = DetailedEventEmitter("camp-1", lambda cid, evt: events.append((cid, evt)))

    reward = CampaignDecisionRewardCalculator().calculate(
        CampaignDecisionOutcome(
            trace_id="t-1",
            campaign_id="camp-1",
            round_index=0,
            execution_success=True,
            objective_delta=0.5,
            proxy_gap_delta=-0.4,
        )
    )
    emitter.emit_decision_reward(reward, decision_id="t-1")

    assert len(events) == 1
    cid, evt = events[0]
    assert cid == "camp-1"
    assert evt["type"] == "decision_reward"
    assert evt["decision_id"] == "t-1"
    assert evt["rubric_version"] == reward.rubric_version
    assert evt["reward"] == reward.reward
    assert evt["process_reward"] == reward.process_reward
    assert evt["outcome_reward"] == reward.outcome_reward
    # every signal's verifier record is present with passed/score/evidence
    names = {v["name"] for v in evt["verifications"]}
    assert {"execution", "objective", "proxy_gap"} <= names
    for v in evt["verifications"]:
        assert set(v) >= {"name", "passed", "score", "evidence", "verifier_type"}


def test_emit_handles_reward_without_verifications():
    events: list[tuple[str, dict]] = []
    emitter = DetailedEventEmitter("camp-1", lambda cid, evt: events.append((cid, evt)))

    class _Bare:
        verifications = []
        rubric_version = "v0.1_static"
        reward = 0.0
        process_reward = 0.0
        outcome_reward = 0.0

    emitter.emit_decision_reward(_Bare())
    assert events[0][1]["verifications"] == []
