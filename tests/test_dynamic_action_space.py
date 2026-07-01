from __future__ import annotations

import json
from datetime import UTC, datetime

from app.services.campaign_mode import CampaignMode, CampaignModeDecision
from app.services.dynamic_action_space import (
    ActionShadowLabel,
    ActionSpec,
    DynamicActionSpaceAssessor,
    build_action_space_snapshot,
)
from app.services.failure_attribution import attribute_failure
from app.services.failure_signatures import classify_failure

_NOW = datetime(2026, 6, 29, 12, 0, 0, tzinfo=UTC)


def _mode(mode: CampaignMode, *, rank: int = 7) -> CampaignModeDecision:
    return CampaignModeDecision(
        campaign_id="camp-1",
        round_index=0,
        mode=mode,
        priority_rank=rank,
        reason=f"test {mode.value}",
    )


def _snapshot(mode_decision, actions, *, available=("robot", "heat", "squidstat"), **kw):
    return build_action_space_snapshot(
        mode_decision=mode_decision,
        actions=actions,
        available_capabilities=list(available),
        now=_NOW,
        **kw,
    )


def _labels(snapshot) -> dict[str, ActionShadowLabel]:
    return {a.name: a.label for a in snapshot.assessments}


def test_bo_optimization_prefers_low_risk_and_flags_high_risk():
    actions = [
        ActionSpec(name="explore", kind="experiment", base_risk=0.2),
        ActionSpec(name="risky_run", kind="experiment", base_risk=0.85),
    ]

    snapshot = _snapshot(_mode(CampaignMode.BO_OPTIMIZATION), actions)
    labels = _labels(snapshot)

    assert labels["explore"] == ActionShadowLabel.PREFERRED
    assert labels["risky_run"] == ActionShadowLabel.RISKY
    assert snapshot.proposed_disabled_actions == []


def test_stop_recommended_proposes_disabling_experiments_but_not_reports():
    actions = [
        ActionSpec(name="run_experiment", kind="experiment"),
        ActionSpec(name="write_report", kind="report"),
    ]

    snapshot = _snapshot(_mode(CampaignMode.STOP_RECOMMENDED, rank=1), actions)
    labels = _labels(snapshot)

    assert labels["run_experiment"] == ActionShadowLabel.PROPOSED_DISABLED
    assert labels["write_report"] != ActionShadowLabel.PROPOSED_DISABLED


def test_calibration_prefers_calibration_and_flags_implicated_instrument():
    attribution = attribute_failure(
        classify_failure(step_key="s1", primitive="heat", error_message="temp overshoot exceeded"),
        now=_NOW,
    )
    actions = [
        ActionSpec(name="recalibrate", kind="calibration", required_capabilities=["heat"]),
        ActionSpec(name="heat_sample", kind="experiment", required_capabilities=["heat"]),
        ActionSpec(name="mix_sample", kind="experiment", required_capabilities=["robot"]),
    ]

    snapshot = _snapshot(
        _mode(CampaignMode.CALIBRATION, rank=3), actions, failure_attribution=attribution
    )
    labels = _labels(snapshot)

    assert labels["recalibrate"] == ActionShadowLabel.PREFERRED
    assert labels["heat_sample"] == ActionShadowLabel.RISKY
    assert labels["mix_sample"] == ActionShadowLabel.NEUTRAL


def test_failure_diagnosis_prefers_diagnostic_and_flags_failing_capability():
    attribution = attribute_failure(
        classify_failure(
            step_key="s1", primitive="robot.pick_up_tip", error_message="no tips available"
        ),
        now=_NOW,
    )
    actions = [
        ActionSpec(name="inspect", kind="diagnostic"),
        ActionSpec(
            name="pipette_again",
            kind="experiment",
            required_capabilities=["robot.pick_up_tip"],
        ),
    ]

    snapshot = _snapshot(
        _mode(CampaignMode.FAILURE_DIAGNOSIS, rank=4),
        actions,
        available=("robot", "robot.pick_up_tip", "heat"),
        failure_attribution=attribution,
    )
    labels = _labels(snapshot)

    assert labels["inspect"] == ActionShadowLabel.PREFERRED
    assert labels["pipette_again"] == ActionShadowLabel.RISKY


def test_context_seeking_prefers_literature_actions():
    actions = [
        ActionSpec(name="query_lit", kind="literature"),
        ActionSpec(name="run_experiment", kind="experiment"),
    ]

    snapshot = _snapshot(_mode(CampaignMode.LITERATURE_CONTEXT_SEEKING, rank=6), actions)
    labels = _labels(snapshot)

    assert labels["query_lit"] == ActionShadowLabel.PREFERRED
    assert labels["run_experiment"] == ActionShadowLabel.NEUTRAL


def test_validation_prefers_validation_actions():
    actions = [
        ActionSpec(name="replicate_best", kind="validation"),
        ActionSpec(name="explore", kind="experiment"),
    ]

    snapshot = _snapshot(_mode(CampaignMode.VALIDATION, rank=5), actions)
    labels = _labels(snapshot)

    assert labels["replicate_best"] == ActionShadowLabel.PREFERRED


def test_missing_capability_proposes_disable_regardless_of_mode():
    actions = [
        ActionSpec(
            name="mass_spec_scan",
            kind="experiment",
            required_capabilities=["mass_spec", "robot"],
        ),
    ]

    snapshot = _snapshot(_mode(CampaignMode.BO_OPTIMIZATION), actions)
    assessment = snapshot.assessments[0]

    assert assessment.label == ActionShadowLabel.PROPOSED_DISABLED
    assert assessment.missing_capabilities == ["mass_spec"]
    assert "mass_spec" in assessment.reason


def test_every_assessment_has_reason_and_evidence():
    actions = [ActionSpec(name="explore", kind="experiment")]

    snapshot = _snapshot(_mode(CampaignMode.BO_OPTIMIZATION), actions)

    for assessment in snapshot.assessments:
        assert assessment.reason
        assert assessment.evidence


def test_snapshot_is_deterministic_and_json_safe():
    actions = [
        ActionSpec(name="explore", kind="experiment", base_risk=0.2, cost=1.0, latency=30.0),
        ActionSpec(name="risky_run", kind="experiment", base_risk=0.85),
    ]
    mode_decision = _mode(CampaignMode.BO_OPTIMIZATION)

    first = build_action_space_snapshot(
        mode_decision=mode_decision,
        actions=actions,
        available_capabilities=["robot", "heat"],
        now=_NOW,
    )
    second = DynamicActionSpaceAssessor().snapshot(
        mode_decision=mode_decision,
        actions=actions,
        available_capabilities=["robot", "heat"],
        now=_NOW,
    )

    assert first.shadow_only is True
    assert first.created_at == _NOW
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    json.dumps(first.model_dump(mode="json"))


def test_stop_recommended_kind_semantics_for_new_kinds():
    actions = [
        ActionSpec(name="clean", kind="cleanup"),
        ActionSpec(name="prep", kind="preparation"),
        ActionSpec(name="flow", kind="workflow"),
    ]

    snapshot = _snapshot(_mode(CampaignMode.STOP_RECOMMENDED, rank=1), actions)
    labels = _labels(snapshot)

    # cleanup may still run during stop; preparation/workflow are proposed-disabled.
    assert labels["clean"] == ActionShadowLabel.NEUTRAL
    assert labels["prep"] == ActionShadowLabel.PROPOSED_DISABLED
    assert labels["flow"] == ActionShadowLabel.PROPOSED_DISABLED


def test_safety_constraint_tightening_labels():
    actions = [
        ActionSpec(name="exp", kind="experiment"),
        ActionSpec(name="prep", kind="preparation"),
        ActionSpec(name="flow", kind="workflow"),
        ActionSpec(name="diag", kind="diagnostic"),
        ActionSpec(name="calib", kind="calibration"),
        ActionSpec(name="rep", kind="report"),
        ActionSpec(name="clean", kind="cleanup"),
    ]

    snapshot = _snapshot(
        _mode(CampaignMode.SAFETY_CONSTRAINT_TIGHTENING, rank=2), actions
    )
    labels = _labels(snapshot)

    assert labels["exp"] == ActionShadowLabel.RISKY
    assert labels["prep"] == ActionShadowLabel.RISKY
    assert labels["flow"] == ActionShadowLabel.RISKY
    assert labels["diag"] == ActionShadowLabel.PREFERRED
    assert labels["calib"] == ActionShadowLabel.PREFERRED
    assert labels["rep"] == ActionShadowLabel.NEUTRAL
    assert labels["clean"] == ActionShadowLabel.NEUTRAL


def test_import_smoke():
    import app.services.dynamic_action_space  # noqa: F401
