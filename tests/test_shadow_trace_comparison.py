from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime

from app.services.adaptive_campaign_substrate import (
    build_adaptive_campaign_substrate_snapshot,
)
from app.services.campaign_mode import CampaignMode
from app.services.decision_models import CampaignDecisionAction, CampaignDecisionPlan
from app.services.dynamic_action_space import ActionSpec
from app.services.failure_attribution import (
    FailureAttributionCategory,
    attribute_failure,
)
from app.services.failure_signatures import classify_failure
from app.services.objective_state import ObjectiveState
from app.services.shadow_trace_comparison import (
    ShadowEquivalenceClass,
    compare_shadow_tracks,
    parse_substrate_log_line,
    summarize_comparisons,
)
from app.services.value_of_information import ActionValueSignals

_NOW = datetime(2026, 6, 29, 12, 0, 0, tzinfo=UTC)


def _attr(error: str, primitive: str):
    return attribute_failure(
        classify_failure(step_key="s1", primitive=primitive, error_message=error),
        now=_NOW,
    )


def _snap(*, failure=None, actions=None, available=None, value_signals=None):
    return build_adaptive_campaign_substrate_snapshot(
        campaign_id="camp-1",
        round_index=0,
        objective_state=ObjectiveState(
            campaign_id="camp-1", primary_objective="x", created_at=_NOW
        ),
        failure_attribution=failure,
        actions=actions or [ActionSpec(name="explore", kind="experiment", required_capabilities=["heater"])],
        available_capabilities=available or ["heater"],
        value_signals=value_signals or [],
        now=_NOW,
    )


def _plan(action: CampaignDecisionAction) -> CampaignDecisionPlan:
    return CampaignDecisionPlan(action_type=action, rationale="t")


def _codes(findings):
    return {f.code for f in findings}


# --- equivalence / agreement ---------------------------------------------


def test_optimization_tracks_agree():
    cmp = compare_shadow_tracks(
        _plan(CampaignDecisionAction.PROPOSE_CANDIDATES), _snap(), now=_NOW
    )

    assert cmp.agree is True
    assert cmp.decision_class == ShadowEquivalenceClass.OPTIMIZATION
    assert cmp.substrate_class == ShadowEquivalenceClass.OPTIMIZATION
    assert cmp.divergences == []


def test_failure_handling_tracks_agree():
    snap = _snap(failure=_attr("temp overshoot exceeded", "heat"))  # -> CALIBRATION
    cmp = compare_shadow_tracks(
        _plan(CampaignDecisionAction.RECOVER_FAILURE), snap, now=_NOW
    )

    assert cmp.substrate_mode == CampaignMode.CALIBRATION
    assert cmp.agree is True
    assert cmp.decision_class == ShadowEquivalenceClass.FAILURE_HANDLING


def test_class_mismatch_is_a_divergence():
    cmp = compare_shadow_tracks(
        _plan(CampaignDecisionAction.QUERY_LITERATURE), _snap(), now=_NOW
    )

    assert cmp.agree is False
    assert _codes(cmp.divergences) == {"class_mismatch"}
    assert cmp.decision_class == ShadowEquivalenceClass.CONTEXT_SEEKING
    assert cmp.substrate_class == ShadowEquivalenceClass.OPTIMIZATION


def test_tighten_constraints_flags_substrate_missing_mode():
    cmp = compare_shadow_tracks(
        _plan(CampaignDecisionAction.TIGHTEN_CONSTRAINTS), _snap(), now=_NOW
    )

    assert cmp.agree is False
    assert cmp.decision_class == ShadowEquivalenceClass.CONSTRAINT
    divergence = cmp.divergences[0]
    assert divergence.code == "substrate_missing_constraint_mode"
    assert divergence.severity == "info"


# --- sanity checks --------------------------------------------------------


def test_sanity_voi_recommends_disabled_action():
    # needs_mass_spec is proposed_disabled (missing capability) yet given the
    # highest value signal, so the advisory VoI ranking puts it on top.
    snap = _snap(
        actions=[ActionSpec(name="needs_mass_spec", kind="experiment", required_capabilities=["mass_spec"])],
        available=["heater"],
        value_signals=[ActionValueSignals(name="needs_mass_spec", immediate_reward=1.0)],
    )
    cmp = compare_shadow_tracks(_plan(CampaignDecisionAction.PROPOSE_CANDIDATES), snap, now=_NOW)

    assert "voi_recommends_disabled" in _codes(cmp.sanity_findings)


def test_sanity_failure_ignored_by_mode():
    snap = _snap(failure=_attr("temp overshoot exceeded", "heat"))
    # Inject the pathology: keep the confident failure but force optimization mode.
    snap.campaign_mode_decision.mode = CampaignMode.BO_OPTIMIZATION

    cmp = compare_shadow_tracks(_plan(CampaignDecisionAction.PROPOSE_CANDIDATES), snap, now=_NOW)

    assert "failure_ignored_by_mode" in _codes(cmp.sanity_findings)


def test_sanity_calibration_without_instrument_failure():
    snap = _snap(failure=_attr("no tips available", "robot.pick_up_tip"))  # execution
    snap.campaign_mode_decision.mode = CampaignMode.CALIBRATION  # inject pathology

    cmp = compare_shadow_tracks(_plan(CampaignDecisionAction.RECOVER_FAILURE), snap, now=_NOW)

    assert "calibration_without_instrument_failure" in _codes(cmp.sanity_findings)


def test_sanity_diagnosis_without_failure():
    snap = _snap()  # no failure
    snap.campaign_mode_decision.mode = CampaignMode.FAILURE_DIAGNOSIS  # inject

    cmp = compare_shadow_tracks(_plan(CampaignDecisionAction.RECOVER_FAILURE), snap, now=_NOW)

    assert "diagnosis_without_failure" in _codes(cmp.sanity_findings)


# --- calibration flags ----------------------------------------------------


def test_calibration_capability_inconsistency():
    snap = _snap()
    # Inject: an action lists a missing capability that is actually available.
    snap.dynamic_action_space_snapshot.assessments[0].missing_capabilities = ["heater"]

    cmp = compare_shadow_tracks(_plan(CampaignDecisionAction.PROPOSE_CANDIDATES), snap, now=_NOW)

    assert "capability_mapping_inconsistency" in _codes(cmp.calibration_flags)


def test_calibration_experiment_without_capability():
    snap = _snap(actions=[ActionSpec(name="lonely", kind="experiment", required_capabilities=[])])

    cmp = compare_shadow_tracks(_plan(CampaignDecisionAction.PROPOSE_CANDIDATES), snap, now=_NOW)

    assert "experiment_without_capability" in _codes(cmp.calibration_flags)


def test_calibration_attribution_known_type_as_external():
    snap = _snap(failure=_attr("temp overshoot exceeded", "heat"))
    # Inject: dominant external_context_missing while the failure type is known.
    snap.failure_attribution = dataclasses.replace(
        snap.failure_attribution,
        dominant_category=FailureAttributionCategory.EXTERNAL_CONTEXT_MISSING,
    )

    cmp = compare_shadow_tracks(_plan(CampaignDecisionAction.RECOVER_FAILURE), snap, now=_NOW)

    assert "attribution_known_type_as_external" in _codes(cmp.calibration_flags)


# --- multi-round summary --------------------------------------------------


def test_summarize_comparisons_reports_agreement_rate_and_histograms():
    agree = compare_shadow_tracks(_plan(CampaignDecisionAction.PROPOSE_CANDIDATES), _snap(), now=_NOW)
    disagree = compare_shadow_tracks(_plan(CampaignDecisionAction.QUERY_LITERATURE), _snap(), now=_NOW)

    summary = summarize_comparisons([agree, disagree])

    assert summary.total == 2
    assert summary.agreement_count == 1
    assert summary.agreement_rate == 0.5
    assert summary.divergence_histogram.get("class_mismatch") == 1


# --- log parsing / serialization -----------------------------------------


def test_parse_substrate_log_line_round_trips():
    snap = _snap()
    line = "adaptive_campaign_substrate_snapshot " + json.dumps(
        snap.model_dump(mode="json"), sort_keys=True
    )

    parsed = parse_substrate_log_line(line)

    assert parsed is not None
    assert parsed.campaign_id == "camp-1"
    assert parsed.campaign_mode_decision.mode == snap.campaign_mode_decision.mode


def test_parse_substrate_log_line_returns_none_on_garbage():
    assert parse_substrate_log_line("something else entirely") is None


def test_comparison_is_json_safe():
    cmp = compare_shadow_tracks(_plan(CampaignDecisionAction.QUERY_LITERATURE), _snap(), now=_NOW)

    json.dumps(cmp.model_dump(mode="json"))
    assert cmp.created_at == _NOW


def test_import_smoke():
    import app.services.shadow_trace_comparison  # noqa: F401
